"""阶段 2 引擎正确性检验（只用研究期 2014-01-01 ~ 2023-12-31，不触及测试集）。

C1 随机选股 vs 等权基准：20 个随机种子，分别不计成本 / 计成本；等权基准 = 每天把所有"可入选且次日开盘可买"
   的股票等权买入、持有同样天数（开盘到开盘、后复权），H 个错开的子组合平均。
   预期：随机（不计成本）均值 ≈ 等权基准；随机（计成本）≈ 等权基准 − 成本拖累。
C2 未来数据对照：按 t+1 之后的真实收益选股（绕过 View 直接读完整数据），应得到明显异常的高收益；
   另做"扰动未来数据"的前视检测：把 t 日之后的数据打乱，诚实策略的信号不变，作弊策略的信号改变。
C3 追涨停：A = t 日收盘涨停、t+1 开盘买；B = 偷看 t+1 开盘、只挑开盘即涨停的股票（必须 0 成交）。
C4 路径检验：偷看未来挑"买入后即停牌"和"即将退市"的股票，检验停牌锁定与退市/换股/现金选择权处理。
C5 压力：滑点 0.2%、停牌期间按中证全指折算估值。
输出 reports/phase2_engine_checks.md
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import Backtest, Config
from data import View, load_market
from metrics import perf
from test_strategies import (DelistCheat, FutureCheat, LimitUpChase, LimitUpOpenCheat, RandomPick,
                             SuspendCheat)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else None
H, N_SEEDS = 5, 20


def ew_benchmark(md, H, start, end, min_listed=60, min_amt=5e7):
    """等权基准：H 个错开的子组合，第 k 个在 t ≡ k (mod H) 的日子把全部可入选股票等权买入，持有 H 天。"""
    a, dates = md.a, md.dates
    t0 = int(dates.searchsorted(pd.Timestamp(start)))
    t1 = int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    dl = pd.to_datetime(md.universe.reindex(md.codes)["delist_date"]).values
    hopen = a["open"] * a["adj_factor"]
    ts = a["tradestatus"]
    # 每只股票"从第 i 天起第一个可成交开盘"的后复权价（处理卖出日停牌）
    T, N = ts.shape
    nxt = np.full((T, N), np.nan)
    last = np.full(N, np.nan)
    for i in range(T - 1, -1, -1):
        ok = (ts[i] == 1) & np.isfinite(hopen[i])
        last = np.where(ok, hopen[i], last)
        nxt[i] = last
    sub = np.ones(H)
    navs = []
    rets = []
    for t in range(t0 - 1, t1 - H):
        d = np.datetime64(dates[t])
        elig = ((ts[t] == 1) & (a["isST"][t] == 0) & (a["isS"][t] == 0) & (a["delist_period"][t] == 0)
                & (a["days_listed"][t] >= min_listed) & (np.nan_to_num(a["amt20"][t]) >= min_amt) & ~(dl <= d))
        o = a["open"][t + 1]
        buy = elig & (ts[t + 1] == 1) & ((a["no_limit"][t + 1] == 1) | (o < a["up_limit"][t + 1] - 0.005))
        r = nxt[t + 1 + H, buy] / hopen[t + 1, buy] - 1
        r = r[np.isfinite(r)]
        rets.append((dates[t + 1], r.mean() if len(r) else 0.0, len(r)))
    df = pd.DataFrame(rets, columns=["date", "ret", "n"]).set_index("date")
    # 子组合 k 依次复利；总净值 = 各子组合净值平均
    nav = np.ones(H)
    out = []
    for i, (d, row) in enumerate(df.iterrows()):
        k = i % H
        nav[k] *= 1 + row.ret
        out.append((d, nav.mean()))
    s = pd.Series(dict(out))
    years = len(s) / 242
    return s, (s.iloc[-1] / 1.0) ** (1 / years) - 1, df


def honest_reversal(t, v):
    r5 = v.close[-1] * v.adj_factor[-1] / (v.close[-6] * v.adj_factor[-6]) - 1
    r5 = np.where(np.isfinite(r5), r5, np.inf)
    return np.argsort(r5)[:100]


class Honest:
    def select(self, t, v):
        return honest_reversal(t, v)


def leak_test(md, strategy_factory, n=30, seed=0):
    """把 t 日之后的价格打乱，比较 t 日信号是否改变。返回改变的日期数。"""
    rng = np.random.default_rng(seed)
    ts = rng.choice(np.arange(300, md.T - 40), n, replace=False)
    changed = 0
    for t in ts:
        s1 = strategy_factory(md).select(t, View(md, t))
        saved = {k: md.a[k][t + 1:].copy() for k in ("open", "close", "adj_factor", "up_limit", "tradestatus")}
        for k in ("open", "close"):
            md.a[k][t + 1:] = md.a[k][t + 1:] * rng.uniform(0.5, 1.5, size=md.a[k][t + 1:].shape)
        s2 = strategy_factory(md).select(t, View(md, t))
        for k, v in saved.items():
            md.a[k][t + 1:] = v
        changed += int(not np.array_equal(np.asarray(s1)[:50], np.asarray(s2)[:50]))
    return changed, n


def run(md, strat, **kw):
    cfg = Config(**kw)
    t0 = time.time()
    r = Backtest(md, strat, cfg).run()
    p = perf(r.nav.nav, {"zzqz": md.index_close.zzqz, "zz1000": md.index_close.zz1000}, r.roundtrips, r.trades)
    p["secs"] = time.time() - t0
    return r, p


def fmt(x, pct=True):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.2%}" if pct else f"{x:,.2f}")


def main():
    md = pickle.load(open(CACHE, "rb")) if CACHE and CACHE.exists() else load_market()
    START, END = "2014-01-01", "2023-12-31"
    L = ["# 阶段 2：回测引擎正确性检验", "",
         f"区间 {START} ~ {END}（研究期，未触及测试集）。初始资金 20 万，目标持仓 10 只，持有 {H} 个交易日；"
         "成本：佣金万 2.5（最低 5 元）、卖出印花税（2023-08-28 前千 1、之后万 5）、单边滑点千 1。", ""]
    out = {}

    # ---------------- C1
    ew_nav, ew_cagr, ew_df = ew_benchmark(md, H, START, END)
    rows = []
    for seed in range(N_SEEDS):
        _, pg = run(md, RandomPick(seed), costs=False, hold_days=H)
        rn, pn = run(md, RandomPick(seed), costs=True, hold_days=H)
        rows.append(dict(seed=seed, gross=pg["cagr"], net=pn["cagr"], fees=pn["fees_total"], turnover=pn["turnover_annual"],
                         n_buys=pn["n_buys"], susp_locked=int((rn.roundtrips.susp_days > 0).sum()),
                         susp_days=rn.roundtrips.loc[rn.roundtrips.susp_days > 0, "susp_days"].mean(),
                         xr_buy=rn.stat["exright_extra_buy"], xr_sell=rn.stat["exright_extra_sell"],
                         lu_block=rn.stat["buy_block"].get("limit_up_open", 0), attempts=rn.stat["buy_attempts"],
                         ld_days=rn.stat["sell_block_days"].get("limit_down_open", 0),
                         insufficient=rn.stat["insufficient_lot"]))
        print("seed", seed, round(pg["cagr"], 4), round(pn["cagr"], 4), flush=True)
    R = pd.DataFrame(rows)
    yearly_ew = ew_nav.groupby(ew_nav.index.year).last()
    yearly_ew = (yearly_ew / yearly_ew.shift(1).fillna(1.0) - 1)
    gross_m, net_m = R.gross.mean(), R.net.mean()
    se = R.gross.std() / np.sqrt(len(R))
    out["C1"] = dict(ew=ew_cagr, gross=gross_m, net=net_m, se=se)
    L += ["## C1 随机选股 vs 等权基准", "",
          "| 项目 | 年化收益 |", "|---|---|",
          f"| 等权基准（全部可入选股票，持有 {H} 日，不计成本） | {fmt(ew_cagr)} |",
          f"| 随机选股 20 个种子均值，不计成本 | {fmt(gross_m)}（标准误 {fmt(se)}，最低 {fmt(R.gross.min())}，最高 {fmt(R.gross.max())}） |",
          f"| 随机选股 20 个种子均值，计成本 | {fmt(net_m)} |",
          f"| 成本拖累（不计成本 − 计成本） | {fmt(gross_m - net_m)} |",
          f"| 随机（不计成本）− 等权基准 | {fmt(gross_m - ew_cagr)}（约 {abs(gross_m - ew_cagr) / se:.1f} 个标准误） |", "",
          f"年均换手 {R.turnover.mean():.1f} 倍（单边），平均每年买入 {R.n_buys.mean() / 10:.0f} 笔，"
          f"10 年累计费用（佣金 + 印花税）平均 {R.fees.mean():,.0f} 元；另有滑点。",
          "", "等权基准分年度：" + "，".join(f"{y} {v:+.1%}" for y, v in yearly_ew.items()), ""]

    # ---------------- C2
    _, pc = run(md, FutureCheat(md, H), hold_days=H)
    ch_h, n_h = leak_test(md, lambda m: Honest())
    ch_c, n_c = leak_test(md, lambda m: FutureCheat(m, H))
    out["C2"] = dict(cheat_cagr=pc["cagr"], honest_changed=ch_h, cheat_changed=ch_c)
    L += ["## C2 未来数据对照", "",
          f"作弊策略（按 t+1 开盘到 t+{1 + H} 开盘的真实收益选股，计成本）：年化 {fmt(pc['cagr'])}，"
          f"夏普 {pc['sharpe']:.1f}，最大回撤 {fmt(pc['max_drawdown'])}，胜率 {fmt(pc['win_rate'])}。结果明显异常，引擎没有把它"
          "\"修正\"成正常结果，说明引擎本身不会掩盖前视问题；前视只能靠数据接口和检测来防。", "",
          f"前视检测（把 t 日之后的价格随机扰动，比较 t 日信号）：诚实策略（5 日反转排序）{n_h} 个日期中信号改变 {ch_h} 个；"
          f"作弊策略 {n_c} 个日期中改变 {ch_c} 个。阶段 3 的每个策略在回测前都先跑这个检测。", ""]

    # ---------------- C3
    ra, pa = run(md, LimitUpChase(0), hold_days=H)
    sa = ra.stat
    ra0, _ = run(md, LimitUpChase(0), hold_days=H, exright_conservative=False)
    rb, pb = run(md, LimitUpOpenCheat(md), hold_days=H)
    sb = rb.stat
    blk_a = sa["buy_block"]
    out["C3"] = dict(A_attempts=sa["buy_attempts"], A_limit=blk_a.get("limit_up_open", 0), A_buys=sa["buys"],
                     A_cagr=pa["cagr"], B_attempts=sb["buy_attempts"], B_buys=sb["buys"])
    L += ["## C3 追涨停检验", "",
          "| 检验 | 尝试买入 | 开盘涨停买不进 | 除权日保守口径挡住 | 停牌 | 成交 | 年化（计成本） |", "|---|---|---|---|---|---|---|",
          f"| A：t 日收盘涨停，t+1 开盘买 | {sa['buy_attempts']:,} | {blk_a.get('limit_up_open', 0):,}（{blk_a.get('limit_up_open', 0) / max(sa['buy_attempts'], 1):.1%}） | "
          f"{sa['exright_extra_buy']} | {blk_a.get('suspended', 0)} | {sa['buys']:,} | {fmt(pa['cagr'])} |",
          f"| B：偷看 t+1 开盘，只挑开盘即涨停 | {sb['buy_attempts']:,} | {sb['buy_block'].get('limit_up_open', 0):,} | "
          f"{sb['exright_extra_buy']} | {sb['buy_block'].get('suspended', 0)} | **{sb['buys']}** | — |", "",
          f"B 的成交数必须为 0：{'通过' if sb['buys'] == 0 else '不通过（引擎涨跌停逻辑有漏洞）'}。",
          f"A 在关闭除权日保守口径时成交 {ra0.stat['buys']:,} 笔，开启时 {sa['buys']:,} 笔。", ""]

    # ---------------- C4
    rs, ps = run(md, SuspendCheat(md), hold_days=2, filters=False)
    rt = rs.roundtrips
    lk = rt[rt.susp_days > 0]
    rd, pdl = run(md, DelistCheat(md), hold_days=10, filters=False, n_target=10)
    ex = rd.stat["exits"]
    rtd = rd.roundtrips
    hows = rtd.how.value_counts().to_dict()
    out["C4"] = dict(susp_locked=len(lk), susp_avg=lk.susp_days.mean() if len(lk) else 0, delist_exits=ex, hows=hows)
    L += ["## C4 停牌与退市路径检验", "",
          f"停牌：偷看未来、专挑买入后即停牌的股票（持有 2 日），共 {len(rt):,} 次持仓，其中 {len(lk):,} 次因停牌被锁定，"
          f"平均锁定 {lk.susp_days.mean():.1f} 个交易日（中位数 {lk.susp_days.median():.0f}，最长 {lk.susp_days.max():.0f}）；"
          f"复牌开盘跌停继续顺延的天数合计 {rs.stat['sell_block_days'].get('limit_down_open', 0)}。", "",
          f"退市：偷看未来、专挑 3–30 个交易日内退市的股票（不做 ST/流动性过滤），共 {len(rtd):,} 次持仓。"
          f"结束方式：{', '.join(f'{k} {v}' for k, v in hows.items())}。", "",
          "说明：hold_days = 正常持有到期卖出；delist_period = 进入退市整理期后强制卖出；writeoff_forced_* = 到退市日仍未卖出、按 0 计；"
          "cash_option = 主动退市按现金选择权/要约价格兑现；merger_convert = 换股后以存续公司股票卖出。", ""]
    mg = rtd[rtd.via.astype(str) != ""]
    if len(mg):
        L += ["换股样例：", "", mg[["code", "entry_date", "exit_date", "ret", "how", "locked_days"]].head(10).to_markdown(index=False), ""]

    # ---------------- C5
    _, p2 = run(md, RandomPick(0), hold_days=H, slippage=0.002)
    _, p1 = run(md, RandomPick(0), hold_days=H)
    _, p3 = run(md, RandomPick(0), hold_days=H, suspension_stress=True)
    L += ["## C5 压力测试（随机种子 0）", "",
          "| 设定 | 年化 | 最大回撤 |", "|---|---|---|",
          f"| 基准设定（滑点千 1） | {fmt(p1['cagr'])} | {fmt(p1['max_drawdown'])} |",
          f"| 滑点千 2 | {fmt(p2['cagr'])} | {fmt(p2['max_drawdown'])} |",
          f"| 停牌期间按中证全指折算估值 | {fmt(p3['cagr'])} | {fmt(p3['max_drawdown'])} |", ""]

    # ---------------- 汇总统计
    L += ["## 补充统计（随机选股 20 个种子合计）", "",
          f"- 因停牌被锁定的持仓：{R.susp_locked.sum():,} 次，平均锁定 {np.nanmean(R.susp_days):.1f} 个交易日",
          f"- 卖出遇开盘跌停顺延：{R.ld_days.sum():,} 个持仓日",
          f"- 买入遇开盘涨停：{R.lu_block.sum():,} 次（占尝试买入 {R.lu_block.sum() / R.attempts.sum():.2%}）",
          f"- 除权日保守口径额外挡住：买入 {R.xr_buy.sum()} 次、卖出 {R.xr_sell.sum()} 次",
          f"- 资金不足一手：{R.insufficient.sum():,} 次", ""]
    (REPORTS / "phase2_engine_checks.md").write_text("\n".join(L), encoding="utf-8")
    json.dump(out, open(REPORTS / "phase2_engine_checks.json", "w"), default=str, ensure_ascii=False, indent=1)
    R.to_csv(REPORTS / "phase2_random_seeds.csv", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
