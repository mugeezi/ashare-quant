"""D03 自适应短线档（设定见 hypotheses.md "D03" 一节，运行前已登记）。
用法：python d03.py train          训练期：行情只用到 2020-12-31，输出 2016–2020 结果
      python d03.py valid          验证期：只有在训练期判定允许后才运行（行情用到 2023-12-31）
"""
import json
import math
import sys
import time

import numpy as np
import pandas as pd

from framework import CACHE, DAYS, ROOT, Lab, eligible_matrix, excess_stats, hfq, read_exp
from signals import TOL, limit_up_close

SLIP, COMM = 0.001, 0.00025
SEED = 20261017
METHODS = ["S1", "S2", "S3", "S4", "S5"]
# 卖出规则：(止损 k, 止盈 k 或 "trail", 最长持有日)；ORIG 为用户原始规则（固定百分比）
RULES = {}
for ks in (0.5, 1.0):
    for kt in (1.5, 3.0, "trail"):
        for n in (2, 5):
            RULES[f"SL{ks}_{'TR' if kt == 'trail' else 'TP' + str(kt)}_N{n}"] = (ks, kt, n)
RULES["ORIG"] = ("pct", "pct", 5)
ASSUME = ["cons", "opt", "neut"]
TOPK = 10
W = 26                                    # 买入后最多观察 25 个交易日（含顺延）
OUT = ROOT / "reports" / "d03"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- 数据与信号
def prepare(md, t_end):
    a = md.a
    T = t_end + 1
    el = eligible_matrix(md)[:T]
    ts = (a["tradestatus"][:T] == 1)
    Ch = hfq(md, "close")[:T]
    with np.errstate(invalid="ignore", divide="ignore"):
        ret20 = Ch / np.vstack([np.full((20, md.N), np.nan), Ch[:-20]]) - 1
        ret2 = Ch / np.vstack([np.full((2, md.N), np.nan), Ch[:-2]]) - 1
        r1 = a["close"][:T] / a["preclose"][:T] - 1
        rng = np.where(ts, (a["high"][:T] - a["low"][:T]) / a["preclose"][:T], np.nan)
    ma5 = pd.DataFrame(Ch).rolling(5, min_periods=5).mean().values
    ma20 = pd.DataFrame(Ch).rolling(20, min_periods=20).mean().values
    A = pd.DataFrame(rng).rolling(20, min_periods=10).mean().values
    amt = np.where(ts, a["amount"][:T].astype(float), np.nan)
    amt_prev20 = pd.DataFrame(amt).shift(1).rolling(20, min_periods=15).mean().values
    lu = limit_up_close(md)[:T]
    lu_prev = np.vstack([np.zeros((1, md.N), bool), lu[:-1]])
    up, o, h, l_ = a["up_limit"][:T], a["open"][:T], a["high"][:T], a["low"][:T]
    yizi = (o >= up - TOL) & (h <= up + TOL) & (l_ >= up - TOL)
    board = (a["limit_pct"][:T] >= 0.15).astype(int)          # 1 = B20
    # 大盘状态
    c = md.index_close["zzqz"].ffill().values[:T]
    ma60 = pd.Series(c).rolling(60, min_periods=60).mean().values
    trend = (c > ma60).astype(int)
    with np.errstate(invalid="ignore"):
        prem = np.nanmean(np.where(lu_prev & ts, r1, np.nan), axis=1)
    prem5 = pd.Series(prem).rolling(5, min_periods=3).mean()
    med = prem5.rolling(250, min_periods=120).median()
    emo = (prem5 > med).astype(int).values
    state_ok = np.isfinite(ma60) & med.notna().values
    # 截面分位（只在池内）
    r20p = pd.DataFrame(np.where(el, ret20, np.nan)).rank(axis=1, pct=True).values
    cand = {}
    with np.errstate(invalid="ignore"):
        cand["S1"] = (el & (r20p >= 0.95) & (Ch > ma5), ret20)
        cand["S2"] = (el & lu & ~lu_prev & ~yizi, amt)
        cand["S3"] = (el & (r1 >= 0.05) & (a["close"][:T] < up - TOL) & (amt >= 2 * amt_prev20), amt / amt_prev20)
        cand["S4"] = (el & (r20p >= 0.90) & (ret2 <= -0.03) & (ret2 >= -0.08) & (Ch >= ma20), ret20)
    sc = np.load(CACHE.parent / f"score_H67_nf_base_{'train' if t_end <= int(md.dates.searchsorted(pd.Timestamp('2021-01-01'))) else 'valid'}.npy")[:T]
    s5 = np.where(el & np.isfinite(sc), sc, np.nan)
    rk = pd.DataFrame(s5).rank(axis=1, ascending=False, method="first").values
    cand["S5"] = (rk <= 10, s5)
    return dict(el=el, A=A, board=board, trend=trend, emo=emo, state_ok=state_ok, cand=cand)


def signals(md, P, t_end):
    """每种方式每天每个板块取排序前 10 名，去掉 t+1 开盘涨停或停牌、A 缺失的。"""
    a = md.a
    rows = []
    for m in METHODS:
        mask, key = P["cand"][m]
        tt, jj = np.where(mask & np.isfinite(key) & np.isfinite(P["A"]))
        if not len(tt):
            continue
        df = pd.DataFrame(dict(t=tt, j=jj, key=key[tt, jj], board=P["board"][tt, jj]))
        df = df[(df.t + 1 <= t_end) & P["state_ok"][df.t]]
        df = df.sort_values(["t", "board", "key"], ascending=[True, True, False])
        df["rank"] = df.groupby(["t", "board"]).cumcount()
        df = df[df["rank"] < TOPK]
        e = df.t.values + 1
        ok = (a["tradestatus"][e, df.j.values] == 1) & ~((a["no_limit"][e, df.j.values] == 0)
                                                            & (a["open"][e, df.j.values] >= a["up_limit"][e, df.j.values] - TOL))
        df = df[ok].copy()
        df["method"] = m
        rows.append(df)
    s = pd.concat(rows, ignore_index=True)
    s["cell"] = s.board * 4 + P["trend"][s.t] * 2 + P["emo"][s.t]
    s["A"] = P["A"][s.t, s.j]
    s["e"] = s.t + 1
    return s.reset_index(drop=True)


# ---------------------------------------------------------------- 逐笔卖出模拟
def simulate(md, s, t_end):
    """对每个信号、每种卖出规则、每种双触假设，算出卖出日、卖出价（后复权）、原因、是否双触。"""
    a = md.a
    n = len(s)
    e = s.e.values
    j = s.j.values
    off = np.arange(W)
    D = e[:, None] + off[None, :]
    valid = D <= t_end
    Dc = np.minimum(D, t_end)
    adj = a["adj_factor"][Dc, j[:, None]]
    O = a["open"][Dc, j[:, None]] * adj
    H = a["high"][Dc, j[:, None]] * adj
    L = a["low"][Dc, j[:, None]] * adj
    C = a["close"][Dc, j[:, None]] * adj
    DL = a["down_limit"][Dc, j[:, None]] * adj
    ts = (a["tradestatus"][Dc, j[:, None]] == 1) & valid
    nol = a["no_limit"][Dc, j[:, None]] == 1
    sealed = ~nol & (O <= DL + TOL * adj) & (H <= DL + TOL * adj)
    open_at_dl = ~nol & (O <= DL + TOL * adj)
    P0 = O[:, 0]
    A = s.A.values
    # 中性假设的随机数：按（信号日、股票、卖出规则）确定，与样本截止日无关（splitmix64 哈希）
    def u01(ri):
        x = (s.t.values.astype(np.uint64) * np.uint64(1000003) + j.astype(np.uint64) * np.uint64(10007)
             + np.uint64(ri * 101 + SEED))
        with np.errstate(over="ignore"):
            x = (x + np.uint64(0x9E3779B97F4A7C15))
            x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            x = x ^ (x >> np.uint64(31))
        return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    u = np.column_stack([u01(ri) for ri in range(len(RULES))])
    res = {}
    for ri, (rname, (ks, kt, N)) in enumerate(RULES.items()):
        if ks == "pct":
            stop, tp = P0 * 0.99, P0 * 1.03
        else:
            stop = P0 * (1 - ks * A)
            tp = P0 * (1 + kt * A) if kt != "trail" else np.full(n, np.inf)
        trail = kt == "trail"
        for asm in ASSUME:
            done = np.zeros(n, bool)
            xo = np.full(n, -1)
            xp = np.full(n, np.nan)
            at_open = np.zeros(n, bool)
            dbl = np.zeros(n, bool)
            why = np.zeros(n, np.int8)       # 1 止损/移动止盈 2 止盈 3 到期 4 最后收盘
            maxH = H[:, 0].copy()
            for i in range(1, W):
                act = ~done & ts[:, i]
                if not act.any():
                    maxH = np.where(ts[:, i], np.maximum(maxH, H[:, i]), maxH)
                    continue
                if trail:
                    lvl = np.maximum(stop, maxH * (1 - A))
                else:
                    lvl = stop
                forced = act & (i >= N)
                # 到期：开盘卖（开盘跌停封死顺延；开盘跌停但盘中打开按跌停价）
                f_ok = forced & ~sealed[:, i]
                px = np.where(open_at_dl[:, i], DL[:, i], O[:, i])
                xo[f_ok], xp[f_ok], at_open[f_ok], why[f_ok] = i, px[f_ok], ~open_at_dl[:, i][f_ok], 3
                done |= f_ok
                live = act & ~forced & ~sealed[:, i]
                # 开盘越过
                o_dn = live & (O[:, i] <= lvl)
                pxd = np.where(open_at_dl[:, i], DL[:, i], O[:, i])
                xo[o_dn], xp[o_dn], at_open[o_dn], why[o_dn] = i, pxd[o_dn], ~open_at_dl[:, i][o_dn], 1
                o_up = live & ~o_dn & (O[:, i] >= tp)
                xo[o_up], xp[o_up], at_open[o_up], why[o_up] = i, O[:, i][o_up], True, 2
                done |= o_dn | o_up
                live &= ~(o_dn | o_up)
                hd = live & (L[:, i] <= lvl)
                hu = live & (H[:, i] >= tp)
                both = hd & hu
                dbl |= both
                if asm == "cons":
                    pick_dn = hd
                elif asm == "opt":
                    pick_dn = hd & ~hu
                else:
                    pick_dn = hd & (~hu | (u[:, ri] < 0.5))
                pick_up = hu & ~pick_dn
                xo[pick_dn], xp[pick_dn], why[pick_dn] = i, lvl[pick_dn], 1
                xo[pick_up], xp[pick_up], why[pick_up] = i, tp[pick_up], 2
                done |= pick_dn | pick_up
                # 盘中先触及后封跌停的情形：触发价不低于最低价即可成交（已由 L <= lvl 保证）
                maxH = np.where(ts[:, i], np.maximum(maxH, H[:, i]), maxH)
            # 25 个交易日内仍未卖出：若窗口完整（不超过数据截止），按最后一个正常交易日收盘价计；否则记为未完成
            last_ok = valid[:, -1]
            rem = ~done & last_ok
            if rem.any():
                lastc = np.where(ts, C, np.nan)
                idx = np.where(ts, off[None, :], -1).max(axis=1)
                xo[rem] = idx[rem]
                xp[rem] = lastc[np.arange(n), np.maximum(idx, 0)][rem]
                why[rem] = 4
                done |= rem
            res[(rname, asm)] = dict(off=xo, px=xp, at_open=at_open, why=why, dbl=dbl, done=done)
    return res, P0


def net_ret(md, P0, r, e):
    d = e + np.maximum(r["off"], 0)
    dates = md.dates[np.minimum(d, md.T - 1)]
    stamp = np.where(dates >= pd.Timestamp("2023-08-28"), 0.0005, 0.001)
    g = r["px"] / P0
    return g * (1 - SLIP) * (1 - COMM - stamp) / ((1 + SLIP) * (1 + COMM)) - 1


# ---------------------------------------------------------------- 逐年选择
def choose(tr, cells=range(8)):
    """tr：可用交易（保守假设），列 method、rule、cell、ret。返回 {cell: (method, rule) 或 None}，以及明细。"""
    g = tr.groupby(["method", "rule"]).ret.agg(["mean", "std", "count"])
    if g.empty:
        return {c: None for c in cells}, None, {}
    best = g["mean"].idxmax()
    glob = best if g.loc[best, "mean"] > 0 else None
    cg = tr.groupby(["cell", "method", "rule"]).ret.agg(["mean", "std", "count"])
    out, info = {}, {}
    for c in cells:
        x = cg.loc[c] if c in cg.index.get_level_values(0) else None
        big = x[x["count"] >= 200] if x is not None else None
        if big is not None and len(big) and (big["mean"] < 0).all():
            out[c], info[c] = None, "该格所有方案（≥200 笔）均为负 → 不交易"
            continue
        if glob is not None and big is not None and len(big):
            b = big["mean"].idxmax()
            gb = x.loc[glob] if glob in x.index else None
            if gb is not None and gb["count"] > 1:
                se = math.sqrt((big.loc[b, "std"] ** 2) / big.loc[b, "count"] + (gb["std"] ** 2) / gb["count"])
                if big.loc[b, "mean"] - gb["mean"] > 2 * se and b != glob:
                    out[c], info[c] = b, f"专门方案（格内 {big.loc[b, 'mean']:.4f} vs 全局 {gb['mean']:.4f}，2SE={2 * se:.4f}）"
                    continue
        out[c], info[c] = glob, "全局方案" if glob is not None else "全局为负 → 不交易"
    return out, glob, info


# ---------------------------------------------------------------- 组合模拟
def portfolio(md, s, res, P0, tables, asm, d0, d1):
    """tables：{year: {cell: (method, rule) 或 None}}。返回净值序列与成交明细。"""
    a = md.a
    adjA = a["adj_factor"]
    t0 = int(md.dates.searchsorted(pd.Timestamp(d0)))
    t1 = int(md.dates.searchsorted(pd.Timestamp(d1), side="right")) - 1
    cash, nav_prev = 200_000.0, 200_000.0
    pos = []                                   # dict(j, sh, adj_e, exit_t, exit_px_h, at_open, cost, e)
    by_t = {t: g for t, g in s.groupby("t")}
    navs, trades, empty = [], [], 0
    ysig = md.dates.year.values

    def sell(p):
        nonlocal cash
        amt = p["sh"] * p["exit_px_h"] / p["adj_e"] * (1 - SLIP)
        stamp = amt * (0.0005 if md.dates[p["exit_t"]] >= pd.Timestamp("2023-08-28") else 0.001)
        com = max(5.0, amt * COMM)
        cash += amt - com - stamp
        trades.append(dict(code=md.codes[p["j"]], buy=md.dates[p["e"]], sell=md.dates[p["exit_t"]], cost=p["cost"],
                           proceeds=amt - com - stamp, pnl=amt - com - stamp - p["cost"], fees=p["fees"] + com + stamp,
                           why=int(p["why"]), dbl=bool(p["dbl"]), A=p["A"], gross_ret=p["exit_px_h"] / p["p0h"] - 1,
                           cell=int(p["cell"]), scheme=p["scheme"]))

    for t in range(t0, t1 + 1):
        # 开盘卖出
        for p in [p for p in pos if p["exit_t"] == t and p["at_open"]]:
            sell(p)
            pos.remove(p)
        # 开盘买入（t−1 的信号）
        slots = 5 - len(pos)
        g = by_t.get(t - 1)
        if slots > 0 and g is not None:
            tab = tables.get(int(ysig[t - 1]), {})
            lists = []
            for b in (0, 1):
                gb = g[g.board == b]
                if not len(gb):
                    lists.append([])
                    continue
                cell = int(gb.cell.iloc[0])
                sch = tab.get(cell)
                if sch is None:
                    lists.append([])
                    continue
                m, rname = sch
                lists.append([(i, rname, cell) for i in gb[gb.method == m].sort_values("rank").index])
            order = []
            for k in range(max(len(lists[0]), len(lists[1]))):
                for b in (0, 1):
                    if k < len(lists[b]):
                        order.append(lists[b][k])
            held = {p["j"] for p in pos}
            alloc = nav_prev / 5
            for i, rname, cell in order:
                if slots <= 0:
                    break
                j = int(s.j[i])
                if j in held:
                    continue
                r = res[(rname, asm)]
                o = a["open"][t, j]
                px = o * (1 + SLIP)
                sh = math.floor(min(alloc, cash) / (px * (1 + COMM)) / 100) * 100
                if sh < 100:
                    continue
                amt = sh * px
                com = max(5.0, amt * COMM)
                cash -= amt + com
                ex_t = t + int(r["off"][i]) if r["done"][i] else 10 ** 9      # 截止日前未卖出的持仓按收盘价估值到最后
                pos.append(dict(j=j, sh=sh, adj_e=adjA[t, j], e=t, exit_t=ex_t, exit_px_h=r["px"][i],
                                at_open=bool(r["at_open"][i]), why=r["why"][i], dbl=r["dbl"][i], cost=amt + com,
                                fees=com + sh * o * SLIP * 2, A=s.A[i], p0h=P0[i], cell=cell, scheme=f"{s.method[i]}|{rname}"))
                held.add(j)
                slots -= 1
        # 盘中卖出（收盘后入账）
        for p in [p for p in pos if p["exit_t"] == t and not p["at_open"]]:
            sell(p)
            pos.remove(p)
        mv = 0.0
        for p in pos:
            c = a["close"][t, p["j"]]
            if not (c > 0):
                k = t
                while k > 0 and not (a["close"][k, p["j"]] > 0):
                    k -= 1
                c = a["close"][k, p["j"]]
                mv += p["sh"] * c * adjA[k, p["j"]] / p["adj_e"]
            else:
                mv += p["sh"] * c * adjA[t, p["j"]] / p["adj_e"]
        nav = cash + mv
        navs.append((md.dates[t], nav, len(pos)))
        empty += len(pos) == 0
        nav_prev = nav
    nav = pd.DataFrame(navs, columns=["date", "nav", "n_pos"]).set_index("date")
    return nav, pd.DataFrame(trades)


def report(nav, trades, bench, smb):
    st = excess_stats(nav.nav, bench, smb)
    yrs = len(nav) / DAYS
    tr = trades
    win = tr[tr.pnl > 0]
    loss = tr[tr.pnl <= 0]
    stop = tr[tr.why == 1]
    return dict(年化=round(st["cagr"], 4), 超额=round(st["excess_cagr"], 4), 最大回撤=round(st["max_drawdown"], 4),
                分年超额={k: round(v, 3) for k, v in st["yearly_excess"].items()},
                胜率=round(len(win) / max(len(tr), 1), 3), 平均盈利=round(win.pnl.mean() / win.cost.mean(), 4) if len(win) else None,
                平均亏损=round(loss.pnl.mean() / loss.cost.mean(), 4) if len(loss) else None,
                年交易次数=round(len(tr) / yrs, 1), 成本占比=round(tr.fees.sum() / yrs / nav.nav.mean(), 4),
                空仓天数占比=round(float((nav.n_pos == 0).mean()), 3), 双触比例=round(float(tr.dbl.mean()), 4) if len(tr) else None,
                止损亏损超2A比例=round(float((stop.gross_ret < -2 * stop.A).mean()), 4) if len(stop) else None,
                交易笔数=int(len(tr)))


def main(phase):
    OUT.mkdir(parents=True, exist_ok=True)
    lab = Lab()
    md = lab.md
    end = "2020-12-31" if phase == "train" else "2023-12-31"
    t_end = int(md.dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    log("准备数据，截止", md.dates[t_end].date())
    P = prepare(md, t_end)
    s = signals(md, P, t_end)
    log("信号", len(s), s.groupby("method").size().to_dict())
    res, P0 = simulate(md, s, t_end)
    log("逐笔模拟完成")
    # 交易表（保守假设，用于选择）
    rows = []
    for rname in RULES:
        r = res[(rname, "cons")]
        ret = net_ret(md, P0, r, s.e.values)
        rows.append(pd.DataFrame(dict(i=np.arange(len(s)), method=s.method, rule=rname, cell=s.cell, t=s.t,
                                      exit_t=s.e.values + r["off"], done=r["done"], ret=ret)))
    T = pd.concat(rows, ignore_index=True)
    T = T[T.done & (s.t.values[T.i] >= int(md.dates.searchsorted(pd.Timestamp("2014-01-01"))))]
    # 65 种组合的整体统计（训练期：卖出日 ≤ 2020-12-31）
    first = lambda y: int(md.dates.searchsorted(pd.Timestamp(f"{y}-01-01")))  # noqa: E731
    years = range(2016, 2021) if phase == "train" else range(2016, 2024)
    tables, infos = {}, {}
    for Y in years:
        avail = T[T.exit_t < first(Y)]
        tab, glob, info = choose(avail)
        tables[Y] = tab
        infos[Y] = dict(全局=None if glob is None else "|".join(glob),
                        各格={str(c): ("不交易" if v is None else "|".join(v)) + "；" + info[c] for c, v in tab.items()})
    static_pool = T[T.exit_t <= int(md.dates.searchsorted(pd.Timestamp("2020-12-31"), side="right")) - 1]
    st_g = static_pool.groupby(["method", "rule"]).ret.agg(["mean", "std", "count"])
    static = st_g["mean"].idxmax()
    st_g.reset_index().to_csv(OUT / f"combos_2014_2020_{phase}.csv", index=False)
    static_tab = {Y: {c: static for c in range(8)} for Y in years}
    orig_tab = {Y: {c: ("S1", "ORIG") for c in range(8)} for Y in years}
    (OUT / f"tables_{phase}.json").write_text(json.dumps(dict(自适应=infos, 静态最优="|".join(static)), ensure_ascii=False, indent=1))
    bench = md.index_close["zzqz"]
    d0, d1 = ("2016-01-01", "2020-12-31") if phase == "train" else ("2021-01-01", "2023-12-31")
    out = {}
    for name, tab in (("自适应", tables), ("静态最优", static_tab), ("用户原始规则", orig_tab)):
        for asm in ASSUME:
            nav, tr = portfolio(md, s, res, P0, tab, asm, d0, d1)
            out[f"{name}|{asm}"] = report(nav, tr, bench, lab.smb)
            nav.to_csv(OUT / f"nav_{phase}_{name}_{asm}.csv")
            tr.to_csv(OUT / f"trades_{phase}_{name}_{asm}.csv", index=False)
            log(name, asm, out[f"{name}|{asm}"])
    (OUT / f"result_{phase}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    # 试验记录
    if phase == "train":
        for (m, rname), r in st_g.iterrows():
            lab.log_trial("D03", f"{m}_{rname}", "train", {"method": m, "rule": rname},
                          f"D03 组合评估：2014–2020 保守假设每笔平均收益 {r['mean']:.4%}（{int(r['count'])} 笔）")
        for name in ("自适应", "静态最优", "用户原始规则"):
            x = out[f"{name}|cons"]
            lab.log_trial("D03", f"portfolio_{name}", "train", {"assume": "cons"},
                          f"D03 组合模拟 2016–2020 保守假设：年化 {x['年化']:.2%}，超额 {x['超额']:.2%}，回撤 {x['最大回撤']:.2%}",
                          excess_cagr=x["超额"], max_drawdown=x["最大回撤"], cagr=x["年化"])
    log("完成")


if __name__ == "__main__":
    main(sys.argv[1])
