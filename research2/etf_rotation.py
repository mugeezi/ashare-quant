"""W01 ETF 轮动（稳健档，规格见 hypotheses.md 第二轮 W01）。

用法：python research2/etf_rotation.py train [k]     # k 默认 3；2、4 只是稳健性报告项（主变体达标后才运行）
      python research2/etf_rotation.py valid
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import r2
from etf_download import ETFS

DATA = r2.R2 / "data2" / "etf"
CASH = "sh511880"
EQUITY = [c for c in ETFS if c not in ("sh511880", "sh511010", "sh511360")]
END = pd.Timestamp("2023-12-31")
LOOK, MIN_DAYS, MIN_AMT = 120, 120, 1e7
COMM, MIN_COMM, SLIP = 0.00025, 5.0, 0.001


def load_index(name: str) -> pd.Series:
    d = pd.read_parquet(r2.R2 / "data" / "index" / f"{name}.parquet")
    d = d[pd.to_datetime(d["date"]) <= END]                       # 只读研究期
    return d.set_index(pd.to_datetime(d["date"]))["close"].astype(float)


def adjusted(df: pd.DataFrame) -> pd.DataFrame:
    """复权价：不复权收盘价日收益；分红 / 折算日（后复权与不复权日收益差 > 0.3%）改用后复权日收益。"""
    df = df.sort_values("date").set_index("date")
    df = df[df.raw_close.notna() & (df.raw_volume > 0)]
    rr = df.raw_close.pct_change()
    hr = df.hfq_close.pct_change()
    ev = (hr - rr).abs() > 0.003
    r = rr.where(~ev, hr).fillna(0.0)
    out = pd.DataFrame(index=df.index)
    out["close"] = (1 + r).cumprod()
    out["open"] = df.raw_open * out["close"] / df.raw_close
    out["raw_open"] = df.raw_open
    out["amount"] = df.amount
    out["event"] = ev
    return out


def load_all(cal: pd.DatetimeIndex) -> dict:
    data = {}
    for c in EQUITY + [CASH]:
        f = DATA / f"{c}.parquet"
        if not f.exists():
            continue
        a = adjusted(pd.read_parquet(f))
        data[c] = a.reindex(cal)                                    # 上市前与停牌日为 NaN
    return data


def regime_weak(zz: pd.Series) -> pd.Series:
    """与第一轮 RegimeStrategy 相同：收盘 < 60 日均线 且 20 日波动 > 过去 250 日中位数（前 270 日不判定）。"""
    r = np.log(zz).diff()
    vol20 = r.rolling(20).std()
    med = vol20.rolling(250, min_periods=1).median()
    weak = (zz < zz.rolling(60).mean()) & (vol20 > med)
    weak.iloc[:270] = False
    return weak


def targets(data: dict, weak: pd.Series, cal: pd.DatetimeIndex, k: int) -> dict:
    """每月最后一个交易日收盘的目标权重 {信号日: {代码: 权重}}，只用当日及以前数据。"""
    me = pd.Series(cal, index=cal).groupby([cal.year, cal.month]).max().values
    close = pd.DataFrame({c: d["close"] for c, d in data.items() if c != CASH})
    amt = pd.DataFrame({c: d["amount"] for c, d in data.items() if c != CASH})
    nvalid = close.notna().cumsum()
    mom = close / close.shift(LOOK) - 1                              # 日历对齐；ETF 停牌日为 NaN
    ma = close.rolling(LOOK, min_periods=LOOK).mean()
    a20 = amt.rolling(20, min_periods=15).mean()
    out = {}
    for t in me:
        if weak.get(t, False):
            out[t] = {CASH: 1.0}
            continue
        ok = (nvalid.loc[t] >= MIN_DAYS) & (a20.loc[t] >= MIN_AMT) & close.loc[t].notna() & mom.loc[t].notna() \
            & (close.loc[t] > ma.loc[t])
        cand = mom.loc[t][ok].sort_values(ascending=False)
        pick = list(cand.index[:k])
        w = {c: 1.0 / k for c in pick}
        if len(pick) < k:
            w[CASH] = (k - len(pick)) / k
        out[t] = w
    return out


def simulate(data: dict, tg: dict, cal: pd.DatetimeIndex, start, end, init=200_000.0):
    """信号日次一交易日开盘按开盘价成交；先卖后买；偏离目标不超过 2% 净值的持仓不动；100 份整数倍。"""
    cal = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    sig = sorted(t for t in tg if t >= cal[0] - pd.Timedelta(days=40) and t < cal[-1])
    exec_day = {}
    for t in sig:
        nxt = cal[cal > t]
        if len(nxt):
            exec_day[nxt[0]] = tg[t]
    cash, units = init, {}
    navs, trades = [], []
    for d in cal:
        if d in exec_day:
            w = exec_day[d]
            px = {c: data[c].at[d, "open"] for c in set(w) | set(units)}
            nav_o = cash + sum(u * px[c] for c, u in units.items() if np.isfinite(px[c]))
            want = {c: w.get(c, 0.0) * nav_o for c in set(w) | set(units)}
            order = sorted(want, key=lambda c: want[c] - units.get(c, 0.0) * (px[c] if np.isfinite(px[c]) else 0))
            for c in order:                                         # 先卖后买
                p = px[c]
                if not np.isfinite(p):                              # 当日无成交：本月不调整该 ETF
                    continue
                cur = units.get(c, 0.0) * p
                diff = want[c] - cur
                if abs(diff) < 0.02 * nav_o and want[c] > 0:
                    continue
                lot = 100 * data[c].at[d, "raw_open"]                # 100 份的实际金额
                if diff < 0:                                        # 卖出
                    amt = cur if want[c] == 0 else min(cur, np.floor(-diff / lot) * lot)
                    if amt <= 0:
                        continue
                    fee = max(MIN_COMM, COMM * amt) + SLIP * amt
                    units[c] = units.get(c, 0.0) - amt / p
                    if units[c] <= 1e-9:
                        units.pop(c)
                    cash += amt - fee
                    trades.append(dict(date=d, code=c, side="sell", amount=amt, fee=fee))
                else:                                               # 买入
                    budget = min(diff, cash)
                    n_lot = np.floor(budget / (lot * (1 + SLIP + COMM)))
                    amt = n_lot * lot
                    if amt <= 0:
                        continue
                    fee = max(MIN_COMM, COMM * amt) + SLIP * amt
                    if amt + fee > cash:
                        amt -= lot
                        fee = max(MIN_COMM, COMM * amt) + SLIP * amt
                        if amt <= 0:
                            continue
                    units[c] = units.get(c, 0.0) + amt / p
                    cash -= amt + fee
                    trades.append(dict(date=d, code=c, side="buy", amount=amt, fee=fee))
        val = cash
        for c, u in units.items():
            pc = data[c]["close"].loc[:d].dropna()
            val += u * pc.iloc[-1]
        navs.append((d, val))
    nav = pd.Series(dict(navs), name="nav")
    return nav, pd.DataFrame(trades)


def leak_check(data, zz, cal, k, seed=0, cuts=3):
    """前视检测：随机切点之后的全部 ETF 价格、成交额与指数乘 U(0.5,1.5)，比较切点及以前所有信号日的目标权重。"""
    rng = np.random.default_rng(seed)
    base = targets(data, regime_weak(zz), cal, k)
    lo, hi = 300, int(cal.searchsorted(pd.Timestamp("2020-12-31"))) - 30
    bad = []
    for tc in rng.choice(np.arange(lo, hi), cuts, replace=False):
        cut = cal[tc]
        d2 = {}
        for c, d in data.items():
            x = d.copy()
            m = x.index > cut
            for col in ("close", "open", "raw_open", "amount"):
                x.loc[m, col] = x.loc[m, col] * rng.uniform(0.5, 1.5, m.sum())
            d2[c] = x
        z2 = zz.copy()
        m = z2.index > cut
        z2[m] = z2[m] * rng.uniform(0.5, 1.5, m.sum())
        t2 = targets(d2, regime_weak(z2), cal, k)
        for t in base:
            if t <= cut and base[t] != t2[t]:
                bad.append(str(t.date()))
    return {"passed": not bad, "n_checks": cuts, "changed": bad}


def main(split: str, k: int = 3):
    r2.check_pause()
    r2.register_pid("etf_rotation")
    zz = load_index("zzqz")
    cal = zz.index
    data = load_all(cal)
    weak = regime_weak(zz)
    tg = targets(data, weak, cal, k)
    hyp, var = "W01", ("base" if k == 3 else f"k{k}")
    if split == "train":
        lk = leak_check(data, zz, cal, k)
        if not lk["passed"]:
            raise RuntimeError(f"W01 前视检测未通过：{lk['changed']}")
        leak = f"pass {lk['n_checks']}"
    else:
        leak = "见训练集"
    s0, s1 = r2.SPLITS[split]
    nav, tr = simulate(data, tg, cal, s0, s1)
    idx = pd.DataFrame({"zzqz": zz, "hs300": load_index("hs300")})
    smb = pd.read_parquet(r2.R2 / "data" / "derived" / "smb_daily.parquet")["smb"]
    smb = smb[smb.index <= END]
    params = dict(k=k, look=LOOK, trend=f"MA{LOOK}", cash=CASH, pool=len(EQUITY), weak="第一轮弱市定义")
    row, st = r2.log_run(hyp, var, split, "稳健", nav, tr, idx, smb, params, notes=f"W01 {var}", leak=leak,
                         n_target=k, hold_days=21)
    held = pd.Series({t: ",".join(sorted(w)) for t, w in tg.items() if pd.Timestamp(s0) - pd.Timedelta(days=40) <= t <= pd.Timestamp(s1)})
    st["weak_months"] = int(sum(1 for t, w in tg.items() if w == {CASH: 1.0} and pd.Timestamp(s0) <= t <= pd.Timestamp(s1)))
    st["months"] = int(sum(1 for t in tg if pd.Timestamp(s0) <= t <= pd.Timestamp(s1)))
    st["fees"] = float(tr.fee.sum()) if len(tr) else 0.0
    st["passed"], st["trial_no"] = row["passed"], row["trial_no"]
    held.to_csv(r2.OUT / f"{split}_W01_{var}_targets.csv")
    r2.save_json(f"{split}_W01_{var}.json", st)
    print("W01", var, split, {k_: st[k_] for k_ in ("cagr", "max_drawdown", "excess_hs300", "ann_vol", "weak_months", "months")},
          "passed", row["passed"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
