"""第五批：大范围试错的策略族。每个函数返回 T×N 打分矩阵（越大越好，NaN = 不入选），只用 t 日及以前的数据。
每个族背后是一个独立的想法，不是同一信号换参数。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import _days_since, _roll
from framework import hfq
from signals import (TOL, buyback_events, event_rows, gdhs_decline, lag, limit_up_close, peer_mean, yjyg_positive)


def _lr(md):
    C = hfq(md, "close")
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.log(C / lag(C, 1))
    r[md.a["tradestatus"] != 1] = np.nan
    return r


def _mkt_ret(md):
    c = md.index_close["zzqz"].ffill().values
    r = np.full(len(c), np.nan)
    r[1:] = np.log(c[1:] / c[:-1])
    return r


# F01 复牌补涨：长期停牌（≥ 10 日）复牌后，停牌期间大盘涨跌幅远大于复牌后已走出的幅度 → 补涨空间
def f01_resume_catchup(md, min_susp=10, W=10):
    a = md.a
    ts = a["tradestatus"] == 1
    susp_run = np.zeros(md.N)
    s = np.full((md.T, md.N), np.nan)
    mc = np.nancumsum(np.nan_to_num(_mkt_ret(md)))
    C = hfq(md, "close")
    last_close_t = np.full(md.N, -1)
    resume_t = np.full(md.N, -10 ** 9)
    resume_from = np.full(md.N, -1)
    for t in range(1, md.T):
        just = ts[t] & (susp_run >= min_susp)
        resume_t = np.where(just, t, resume_t)
        resume_from = np.where(just, last_close_t, resume_from)
        susp_run = np.where(ts[t], 0, susp_run + 1)
        last_close_t = np.where(ts[t], t, last_close_t)
        act = (t - resume_t < W) & (resume_from >= 0) & ts[t]
        if act.any():
            j = np.where(act)[0]
            mkt_move = mc[t] - mc[resume_from[j]]
            own = np.log(C[t, j] / C[resume_from[j], j])
            s[t, j] = mkt_move - own
    return s


# F02 次新股换手衰减：上市 60–250 个交易日、换手从高位持续回落（筹码沉淀），价格未破位
def f02_subnew_settle(md):
    a = md.a
    dl = a["days_listed"]
    t5 = _roll(np.where(a["tradestatus"] == 1, a["turn"].astype(float), np.nan), 5)
    t60 = _roll(np.where(a["tradestatus"] == 1, a["turn"].astype(float), np.nan), 60)
    with np.errstate(invalid="ignore", divide="ignore"):
        decay = -np.log(t5 / t60)
    C = hfq(md, "close")
    pos = (C - _roll(C, 60, "min")) / (_roll(C, 60, "max") - _roll(C, 60, "min"))
    ok = (dl >= 60) & (dl <= 250) & (pos > 0.3)
    return np.where(ok, decay, np.nan)


# F03 送转填权：送转比例 ≥ 30% 的除权日后 20 日内，价格尚未填权（距除权前价格仍有空间）
def f03_exright_fill(md, min_ratio=1.3, W=20):
    adj = md.a["adj_factor"]
    with np.errstate(invalid="ignore", divide="ignore"):
        jump = adj / lag(adj, 1)
    big = (jump >= min_ratio) & (md.a["tradestatus"] == 1)
    C = hfq(md, "close")
    s = np.full((md.T, md.N), np.nan)
    tt, jj = np.where(big)
    for t, j in zip(tt, jj):
        base = C[t - 1, j]                     # 除权前一日的后复权收盘（后复权价连续）
        for k in range(t, min(t + W, md.T)):
            s[k, j] = -(C[k, j] / base - 1)    # 跌得越多、离"填权"越远，得分越高
    return s


# F04 下跌日抗跌：过去 20 日中大盘下跌的日子里，个股平均跑赢大盘
def f04_down_day_resilience(md, w=20):
    r = _lr(md)
    m = _mkt_ret(md)[:, None]
    down = (m < 0)
    x = np.where(down, r - m, np.nan)
    cnt = _roll((~np.isnan(x)).astype(float), w, "sum", 1)
    s = _roll(np.nan_to_num(x), w, "sum", 1) / np.maximum(cnt, 1)
    return np.where(cnt >= 5, s, np.nan)


# F05 收盘强度持续：过去 20 日收盘位置（收在当日区间的高位）平均值高 = 尾盘持续有人承接
def f05_close_strength(md):
    a = md.a
    H, L, C = a["high"], a["low"], a["close"]
    with np.errstate(invalid="ignore", divide="ignore"):
        clv = (C - L) / (H - L)
    clv[(a["tradestatus"] != 1) | ~np.isfinite(clv)] = np.nan
    s = _roll(clv, 20)
    lu = _roll(limit_up_close(md).astype(float), 20, "sum", 1)
    return np.where(lu == 0, s, np.nan)           # 排除有涨停的（涨停收盘位置天然为 1）


# F06 波动收敛 + 接近前高：20 日振幅 / 120 日振幅低，且收盘在 120 日区间高位
def f06_squeeze(md):
    a = md.a
    C = hfq(md, "close")
    rng = (a["high"] - a["low"]) / a["close"]
    rng[a["tradestatus"] != 1] = np.nan
    ratio = _roll(rng, 20) / _roll(rng, 120)
    pos = (C - _roll(C, 120, "min")) / (_roll(C, 120, "max") - _roll(C, 120, "min"))
    return np.where(pos >= 0.8, -ratio, np.nan)


# F07 缺口守住：过去 10 日内出现 ≥ 3% 的向上跳空，此后最低价从未回补缺口
def f07_gap_hold(md, gap=0.03, W=10):
    a = md.a
    O, H, L = a["open"], a["high"], a["low"]
    prev_h = lag(a["high"].astype(float), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = O / a["preclose"] - 1
    is_gap = (g >= gap) & (a["tradestatus"] == 1) & (L > prev_h) & ~limit_up_close(md)
    s = np.full((md.T, md.N), np.nan)
    tt, jj = np.where(is_gap)
    adj = a["adj_factor"]
    for t, j in zip(tt, jj):
        line = prev_h[t, j] * adj[t - 1, j]                     # 缺口上沿（后复权）
        for k in range(t, min(t + W, md.T)):
            if L[k, j] * adj[k, j] <= line:
                break
            s[k, j] = g[t, j] - 0.001 * (k - t)
    return s


# F08 解禁后错杀：解禁占流通股 ≥ 5%，解禁后 10 日内股价较解禁前下跌 ≥ 8% → 抛压释放后反弹
def f08_unlock_oversold(md, ext, min_ratio=0.05, drop=-0.08, W=10):
    df = ext["unlock"]
    t, j, df = event_rows(md, df)
    D = t + 3
    C = hfq(md, "close")
    fs = md.a["fshares"]
    s = np.full((md.T, md.N), np.nan)
    for tt, jj, d, sh in zip(t, j, D, df["shares_actual"].values.astype(float)):
        if d >= md.T or not np.isfinite(sh) or not (fs[tt, jj] > 0) or sh / fs[tt, jj] < min_ratio:
            continue
        base = C[d - 1, jj]
        for k in range(d, min(d + W, md.T)):
            r = C[k, jj] / base - 1
            if r <= drop:
                s[k, jj] = -r
    return s


# F09 业绩预告利空出尽：首亏 / 预减公告后 5–20 日内，累计跌幅 ≥ 15% 且最近 3 日止跌
def f09_bad_news_exhaust(md, ext, W=20):
    x = ext["yjyg"]
    x = x[x["metric"].astype(str).str.contains("归属于上市公司股东的净利润", na=False) & x["type"].isin(["首亏", "预减", "增亏"])]
    t, j, _ = event_rows(md, x, after_close_shift=True)
    C = hfq(md, "close")
    s = np.full((md.T, md.N), np.nan)
    for tt, jj in zip(t, j):
        base = C[max(tt - 2, 0), jj]
        for k in range(tt + 5, min(tt + W, md.T)):
            dd = C[k, jj] / base - 1
            rebound = C[k, jj] / np.nanmin(C[k - 3:k + 1, jj]) - 1
            if dd <= -0.15 and rebound >= 0:
                s[k, jj] = -dd
    return s


# F10 筹码集中且价格滞涨：股东户数下降 ≥ 10% 的公告后 20 日内，公告以来涨幅 < 5%
def f10_concentration_lag(md, ext, W=20):
    ev = gdhs_decline(ext["gdhs"])
    t, j, ev = event_rows(md, ev, after_close_shift=True)
    C = hfq(md, "close")
    s = np.full((md.T, md.N), np.nan)
    for tt, jj, st in zip(t, j, ev["strength"].values):
        base = C[tt, jj]
        for k in range(tt, min(tt + W, md.T)):
            if C[k, jj] / base - 1 < 0.05:
                s[k, jj] = st
    return s


# F11 同伴群体动量：PIT 相关性同伴的 20 日平均收益（群体趋势），个股自身 20 日收益不过高
def f11_cluster_momentum(md):
    C = hfq(md, "close")
    with np.errstate(invalid="ignore", divide="ignore"):
        r20 = C / lag(C, 20) - 1
    pm = peer_mean(md, r20)
    return np.where(r20 < pm + 0.10, pm, np.nan)


# F12 资金流向：20 日（成交额 × 当日阴阳方向）/ 流通市值
def f12_money_flow(md):
    a = md.a
    sign = np.sign(a["close"] - a["open"])
    mf = np.where(a["tradestatus"] == 1, sign * a["amount"].astype(float), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _roll(mf, 20, "sum", 10) / a["fmv"]


# F13 量价背离：20 日价格跌、但上涨日成交额占比上升（买盘占优）
def f13_up_volume_share(md):
    a = md.a
    up = (a["close"] > a["preclose"]) & (a["tradestatus"] == 1)
    amt = np.where(a["tradestatus"] == 1, a["amount"].astype(float), 0.0)
    share = _roll(np.where(up, amt, 0.0), 20, "sum", 10) / _roll(amt, 20, "sum", 10)
    C = hfq(md, "close")
    with np.errstate(invalid="ignore", divide="ignore"):
        r20 = C / lag(C, 20) - 1
    return np.where(r20 < 0, share, np.nan)


# F14 春节前效应择时 + 回购/业绩事件：作为择时叠加使用（见 run_batch5）


# F15 涨停后缩量回踩：20 日内有涨停（非一字），之后连续缩量回调到涨停当日开盘价附近（±3%）
def f15_limit_pullback(md, W=20):
    a = md.a
    lu = limit_up_close(md) & (a["low"] < a["high"])
    s = np.full((md.T, md.N), np.nan)
    tt, jj = np.where(lu)
    adj = a["adj_factor"]
    amt = a["amount"].astype(float)
    for t, j in zip(tt, jj):
        level = a["open"][t, j] * adj[t, j]
        a_lu = amt[t, j]
        for k in range(t + 3, min(t + W, md.T)):
            c = a["close"][k, j] * adj[k, j]
            if not np.isfinite(c):
                continue
            if abs(c / level - 1) <= 0.03 and amt[k, j] < 0.5 * a_lu:
                s[k, j] = -(amt[k, j] / a_lu)
    return s


# F16 回购护盘线：回购预案后 60 日内，股价跌至公告前收盘价下方 5% 以内（公司有动力在此价位买入）
def f16_buyback_floor(md, ext, W=60):
    ev = buyback_events(ext["buyback"])
    t, j, _ = event_rows(md, ev, after_close_shift=True)
    C = hfq(md, "close")
    s = np.full((md.T, md.N), np.nan)
    for tt, jj in zip(t, j):
        base = C[max(tt - 2, 0), jj]
        for k in range(tt, min(tt + W, md.T)):
            r = C[k, jj] / base - 1
            if -0.10 <= r <= 0.0:
                s[k, jj] = -r
    return s


# F17 业绩预增 + 股东户数下降双重确认
def f17_dual_confirm(md, ext, W=60):
    a1 = np.zeros((md.T, md.N))
    for t, j in zip(*event_rows(md, yjyg_positive(ext["yjyg"]), True)[:2]):
        a1[t:min(t + W, md.T), j] = 1
    a2 = np.zeros((md.T, md.N))
    for t, j in zip(*event_rows(md, gdhs_decline(ext["gdhs"]), True)[:2]):
        a2[t:min(t + W, md.T), j] = 1
    both = (a1 > 0) & (a2 > 0)
    return np.where(both, 1.0 - _roll(_lr(md), 20, "std") * 10, np.nan)


def _xrank(s):
    """每日截面百分位排名（NaN 保留），只用当日数据。"""
    return pd.DataFrame(s).rank(axis=1, pct=True).values


# F18 多重避雷：把第一、五批中前 10 名显著跑输的几类"受关注"特征合起来，选同时最不具备这些特征的股票（事后）
def f18_avoid_all(md, el):
    bad = [f04_down_day_resilience(md), f05_close_strength(md), f12_money_flow(md), f13_up_volume_share(md)]
    a = md.a
    lu20 = _roll(limit_up_close(md).astype(float), 20, "sum", 1)
    C = hfq(md, "close")
    with np.errstate(invalid="ignore", divide="ignore"):
        r5 = C / lag(C, 5) - 1
    bad += [lu20, np.abs(r5)]
    tot = np.zeros((md.T, md.N))
    cnt = np.zeros((md.T, md.N))
    for b in bad:
        b = np.where(el, b, np.nan)
        r = _xrank(b)
        tot += np.nan_to_num(r)
        cnt += np.isfinite(r)
    return np.where(cnt >= 4, -tot / np.maximum(cnt, 1), np.nan)


# 用缓存的模型打分做组合构建上的变化（F22、F23）
def f22_large_only(md, score, top_frac=0.4):
    fmv = md.a["fmv"]
    r = _xrank(fmv)
    return np.where(r >= 1 - top_frac, score, np.nan)


def f23_model_avoid(md, score, el):
    av = f18_avoid_all(md, el)
    ra = _xrank(av)
    return np.where(ra >= 0.3, score, np.nan)          # 剔除"避雷分"最差的 30%


# F24 同伴信息溢出：PIT 相关性同伴中，近 10 日发布正面业绩预告的比例高，而本股近 60 日没有发布 → 同伴的好消息尚未反映到本股
def f24_peer_info_transfer(md, ext, W=10):
    ev = yjyg_positive(ext["yjyg"])
    t, j, _ = event_rows(md, ev, True)
    recent = np.zeros((md.T, md.N))
    own60 = np.zeros((md.T, md.N), bool)
    for tt, jj in zip(t, j):
        recent[tt:min(tt + W, md.T), jj] = 1.0
        own60[tt:min(tt + 60, md.T), jj] = True
    frac = peer_mean(md, recent)
    C = hfq(md, "close")
    with np.errstate(invalid="ignore", divide="ignore"):
        r10 = C / lag(C, 10) - 1
    return np.where(~own60 & (frac >= 0.15) & (r10 < 0.1), frac, np.nan)


# ================================================================ 第七批：基本面与换手结构
def _fund(md, ext):
    from fundamentals import fundamental_features
    return dict(fundamental_features(md, ext))


# G01 业绩超预告：正式财报归母净利润高于本公司此前业绩预告的预测值 ≥ 10%（A 股强制预告制度特有）
def g01_beat_own_guidance(md, fd):
    s = fd["fc_surprise"]
    fresh = fd["days_since_report"] <= 20
    return np.where(fresh & (s >= 0.10), s, np.nan)


# G02 相对自身历史便宜 + 业绩加速
def g02_self_cheap_accel(md, fd):
    ok = (fd["ep_self_pct"] >= 0.8) & (fd["ni_yoy"] > 20) & (fd["ep_ttm"] > 0)
    return np.where(ok, fd["ep_self_pct"] + fd["ni_yoy"] / 1000, np.nan)


# G03 同伴相对低估 + 盈利质量：比 PIT 相关性同伴便宜，ROE（年化）> 8%
def g03_peer_cheap_quality(md, fd):
    ok = (fd["roe_ann"] > 8) & (fd["ep_ttm"] > 0)
    return np.where(ok, fd["ep_peer_rel"], np.nan)


# G04 毛利率改善尚未被定价：毛利率同比提升 ≥ 2 个百分点，财报后 20 日内股价涨幅 < 5%
def g04_margin_unpriced(md, fd):
    C = hfq(md, "close")
    with np.errstate(invalid="ignore", divide="ignore"):
        r20 = C / lag(C, 20) - 1
    ok = (fd["gm_chg"] >= 2) & (fd["days_since_report"] <= 40) & (r20 < 0.05)
    return np.where(ok, fd["gm_chg"], np.nan)


# G05 换手平稳度：60 日换手中最大 5 天占比低（成交平稳，没有脉冲式炒作），且 60 日价格为正
def g05_turnover_steady(md):
    a = md.a
    turn = np.where(a["tradestatus"] == 1, a["turn"].astype(float), np.nan)
    tot = _roll(turn, 60, "sum", 40)
    from numpy.lib.stride_tricks import sliding_window_view
    top5 = np.full(turn.shape, np.nan)
    z = np.nan_to_num(turn)
    for c0 in range(0, turn.shape[1], 300):                  # 分块，控制内存；窗口 [t−59, t]
        w = sliding_window_view(z[:, c0:c0 + 300], 60, axis=0)          # (T−59, n, 60)
        part = np.partition(w, 55, axis=2)[:, :, 55:].sum(axis=2)
        top5[59:, c0:c0 + 300] = part
    C = hfq(md, "close")
    with np.errstate(invalid="ignore", divide="ignore"):
        r60 = C / lag(C, 60) - 1
        share = top5 / tot
    return np.where(r60 > 0, -share, np.nan)


# G06 现金含量高的增长：每股经营现金流 / 每股收益 > 1.2，净利润同比 > 0，E/P > 0
def g06_cash_backed_growth(md, fd):
    ok = (fd["cfo_ratio"] > 1.2) & (fd["ni_yoy"] > 0) & (fd["ep_ttm"] > 0) & (fd["days_since_report"] <= 60)
    return np.where(ok, fd["cfo_ratio"] + fd["ni_yoy"] / 100, np.nan)


# G07 早预约：首次预约披露日在同一报告期内排在最早 30% 的公司，在其预约日前 10 个交易日内买入（好消息倾向早披露）
def g07_early_schedule(md, ext, pct=0.3, lead=10):
    df = ext["yysj"].copy()
    df["rk"] = df.groupby("period")["first_sched"].rank(pct=True)
    df = df[df["rk"] <= pct]
    cm = {c.split(".")[1]: i for i, c in enumerate(md.codes)}
    s = np.full((md.T, md.N), np.nan)
    for c6, av, sd, rk in zip(df["code6"].astype(str).str.zfill(6), df["avail_date"], df["first_sched"], df["rk"]):
        j = cm.get(c6)
        if j is None or pd.isna(sd):
            continue
        ta = int(md.dates.searchsorted(av))                     # 首次预约日程公开（保守）之后
        td = int(md.dates.searchsorted(sd))                     # 预约披露日
        for k in range(max(ta, td - lead), min(td, md.T)):
            s[k, j] = 1 - rk
    return s


# ================================================================ 第九批：经验规则型（趋势、回踩、突破、反转），中等复杂度
def _ma(x, w):
    return _roll(x, w, "mean", w)


def _base(md):
    a = md.a
    C = hfq(md, "close")
    amt = np.where(a["tradestatus"] == 1, a["amount"].astype(float), np.nan)
    return a, C, amt


# T01 上升趋势中缩量回踩 20 日线：MA20 > MA60 且 MA60 向上，收盘在 MA20 ±2% 内，5 日均额 < 0.7 × 20 日均额
def t01_trend_pullback(md):
    a, C, amt = _base(md)
    m20, m60 = _ma(C, 20), _ma(C, 60)
    up = (m20 > m60) & (m60 > lag(m60, 20)) & (C > m60)
    near = np.abs(C / m20 - 1) <= 0.02
    shrink = _roll(amt, 5) < 0.7 * _roll(amt, 20)
    slope = m60 / lag(m60, 20) - 1
    return np.where(up & near & shrink, slope, np.nan)


# T02 平台突破：收盘创 60 日新高，突破前 60 日振幅（最高/最低−1）< 30%，当日成交额 ≥ 1.5 × 20 日均额，非涨停
def t02_box_breakout(md):
    a, C, amt = _base(md)
    hi60 = lag(_roll(C, 60, "max"), 1)
    lo60 = lag(_roll(C, 60, "min"), 1)
    tight = hi60 / lo60 - 1
    brk = (C > hi60) & (tight < 0.30) & (amt >= 1.5 * _roll(amt, 20)) & ~limit_up_close(md)
    return np.where(brk, -tight, np.nan)


# T03 底部反转：120 日跌幅 ≥ 30%，近 3 日内 5 日线上穿 20 日线，5 日均额 > 1.2 × 20 日均额
def t03_bottom_reversal(md):
    a, C, amt = _base(md)
    r120 = C / lag(C, 120) - 1
    m5, m20 = _ma(C, 5), _ma(C, 20)
    cross = (m5 > m20) & (lag(m5, 1) <= lag(m20, 1))
    recent = _roll(cross.astype(float), 3, "sum", 1) > 0
    vol_up = _roll(amt, 5) > 1.2 * _roll(amt, 20)
    return np.where((r120 <= -0.30) & recent & vol_up & (m5 > m20), -r120, np.nan)


# T04 强势股短暂回调：60 日涨幅处于截面前 20%，近 5 日回调 3%–10%，10 日内无跌停
def t04_strong_pullback(md, el=None):
    a, C, amt = _base(md)
    r60 = C / lag(C, 60) - 1
    r5 = C / lag(C, 5) - 1
    rk = _xrank(np.where(el, r60, np.nan) if el is not None else r60)
    ld = (a["tradestatus"] == 1) & (a["no_limit"] == 0) & (a["close"] <= a["down_limit"] + TOL)
    no_ld = _roll(ld.astype(float), 10, "sum", 1) == 0
    ok = (rk >= 0.8) & (r5 <= -0.03) & (r5 >= -0.10) & no_ld
    return np.where(ok, rk - r5, np.nan)


# T05 稳健上升趋势：收盘在 20、60 日线之上且两线都向上，按 20 日波动从低到高排序
def t05_smooth_trend(md):
    a, C, amt = _base(md)
    m20, m60 = _ma(C, 20), _ma(C, 60)
    up = (C > m20) & (m20 > m60) & (m20 > lag(m20, 5)) & (m60 > lag(m60, 20))
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.log(C / lag(C, 1))
    v20 = _roll(lr, 20, "std")
    return np.where(up, -v20, np.nan)


# T06 周线向上、日线回调：20 日线连续 20 日上升（周线级别上升），近 3 日累计下跌 ≥ 3%
def t06_weekly_up_daily_dip(md):
    a, C, amt = _base(md)
    m20 = _ma(C, 20)
    rising = _roll((m20 > lag(m20, 1)).astype(float), 20, "sum", 20) >= 18
    r3 = C / lag(C, 3) - 1
    return np.where(rising & (r3 <= -0.03) & (C > _ma(C, 60)), -r3, np.nan)


def market_uptrend(md, w=60):
    """中证全指收盘在 w 日均线之上（只用当日及以前）。"""
    c = md.index_close["zzqz"].ffill().values
    ma = pd.Series(c).rolling(w, min_periods=w).mean().values
    return c > ma


# T08 市场状态切换：大盘在 60 日线上方时用 T04（强势回调），下方时用 T05（稳健趋势），两边都满足条件的股票才入选
def t08_regime_switch(md, el):
    up = market_uptrend(md)[:, None]
    return np.where(up, t04_strong_pullback(md, el), t05_smooth_trend(md))


# ================================================================ 逆势型经验规则（事后）
def r01_oversold_dryup(md, el):
    a, C, amt = _base(md)
    r20 = C / lag(C, 20) - 1
    ld = (a["tradestatus"] == 1) & (a["no_limit"] == 0) & (a["close"] <= a["down_limit"] + TOL)
    no_ld = _roll(ld.astype(float), 3, "sum", 1) == 0
    big = _xrank(np.where(el, a["fmv"], np.nan)) >= 0.5
    ok = (r20 <= -0.15) & (_roll(amt, 5) < 0.7 * _roll(amt, 20)) & no_ld & big
    return np.where(ok, -r20, np.nan)


def r02_long_up_short_down(md):
    a, C, amt = _base(md)
    m250, m60 = _roll(C, 250, "mean", 200), _ma(C, 60)
    ok = (m250 > lag(m250, 20)) & (C / m60 - 1 <= -0.15)
    return np.where(ok, -(C / m60 - 1), np.nan)


def r03_down_streak_dryup(md):
    a, C, amt = _base(md)
    down = (a["close"] < a["open"]) & (a["tradestatus"] == 1)
    streak = _roll(down.astype(float), 5, "sum", 5) == 5
    dec = _roll((amt < lag(amt, 1)).astype(float), 4, "sum", 4) >= 3
    r5 = C / lag(C, 5) - 1
    return np.where(streak & dec, -r5, np.nan)


def r04_panic_buy(md, el, W=3):
    a, C, amt = _base(md)
    m = md.index_close["zzqz"].ffill().values
    mret = np.r_[np.nan, m[1:] / m[:-1] - 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        r1 = a["close"] / a["preclose"] - 1
    breadth = np.nanmean(np.where(el, r1 < 0, np.nan), axis=1)
    panic = (mret <= -0.02) & (breadth >= 0.8)
    big = _xrank(np.where(el, a["fmv"], np.nan)) >= 0.5
    s = np.full((md.T, md.N), np.nan)
    for t in np.where(panic)[0]:
        sc = np.where(big[t] & el[t], -r1[t], np.nan)
        for k in range(t, min(t + W, md.T)):
            s[k] = np.where(np.isfinite(sc), sc, s[k])
    return s


def r05_floor_volume(md, el):
    a, C, amt = _base(md)
    r60 = C / lag(C, 60) - 1
    weak = _xrank(np.where(el, r60, np.nan)) <= 0.2
    turn = np.where(a["tradestatus"] == 1, a["turn"].astype(float), np.nan)
    pct = pd.DataFrame(turn).rolling(120, min_periods=100).rank(pct=True).values
    return np.where(weak & (pct <= 0.1), -r60, np.nan)


# ================================================================ 第十批：打分卡
SCORECARD = [("n_touchu20", -1), ("days_since_lu", 1), ("max20", -1), ("turn20", -1), ("gap20", -1), ("in20", -1),
             ("ma60", -1), ("ep_self_pct", 1), ("ep_ttm", 1), ("gm_chg", 1), ("fc_surprise", 1), ("ni_yoy", 1)]
PILLARS = {"anti_hype": ["n_touchu20", "days_since_lu", "max20", "turn20", "gap20", "in20", "ma60"],
           "cheap": ["ep_self_pct", "ep_ttm"], "improve": ["gm_chg", "fc_surprise", "ni_yoy"]}


def scorecard(md, ext, el):
    """返回 (S01 等权打分, S03 三支柱打分)。逐个特征累加，控制内存。"""
    import gc
    import features as FE
    want = dict(SCORECARD)
    tot = np.zeros((md.T, md.N), np.float32)
    pil = {p: np.zeros((md.T, md.N), np.float32) for p in PILLARS}
    for name, arr in FE.build(md, ext):
        if name not in want:
            del arr
            continue
        r = _xrank(np.where(el, arr * want[name], np.nan)).astype(np.float32)
        r = np.where(np.isfinite(r), r, np.float32(0.5))            # 缺失记中性
        del arr
        tot += r
        for p, v in PILLARS.items():
            if name in v:
                pil[p] += r / len(v)
        del r
        gc.collect()
    s1 = np.where(el, tot / len(want), np.nan)
    s3 = np.where(el, sum(pil.values()) / len(pil), np.nan)
    return s1, s3


def s04_filter_then_rank(md, ext, el):
    import features as FE
    need = {"n_touchu20", "turn20", "max20", "gm_chg", "fc_surprise", "ni_yoy"}
    v = {}
    for name, arr in FE.build(md, ext):
        if name in need:
            v[name] = arr
        else:
            del arr
    hype = (v["n_touchu20"] > 0) | (_xrank(np.where(el, v["turn20"], np.nan)) >= 0.7) | \
           (_xrank(np.where(el, v["max20"], np.nan)) >= 0.7)
    imp = [np.where(np.isfinite(r), r, 0.5) for r in (_xrank(np.where(el, v[k], np.nan)) for k in ("gm_chg", "fc_surprise", "ni_yoy"))]
    return np.where(el & ~hype, np.mean(imp, axis=0), np.nan)
