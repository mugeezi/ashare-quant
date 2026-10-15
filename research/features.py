"""截面特征（全部只用 t 日及以前的数据），供机器学习打分模型使用。

每个特征是 T×N 的 float32 矩阵。滚动统计一律用 pandas rolling（向过去看），不使用任何向前移位。
外部事件特征按 avail_date（盘后处理顺延一日）对齐，计算"距最近一次事件的天数"等。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from framework import hfq
from signals import (TOL, buyback_events, event_rows, gdhs_decline, lag, limit_up_close, yjyg_positive,
                     zengchi_events)

F32 = np.float32


def _roll(x, w, fn="mean", minp=None):
    r = pd.DataFrame(x).rolling(w, min_periods=minp or max(2, int(w * 0.8)))
    return getattr(r, fn)().values.astype(F32)


def price_features(md):
    """逐个产出 (名称, T×N 矩阵)，用完即可释放，控制内存。"""
    a = md.a
    ts = a["tradestatus"] == 1
    C, O, H, L = hfq(md, "close"), hfq(md, "open"), hfq(md, "high"), hfq(md, "low")
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.log(C / lag(C, 1))
        on = np.log(O / lag(C, 1))
        intra = np.log(C / O)
    for x in (lr, on, intra):
        x[~ts] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        for n in (1, 3, 5, 10, 20, 60):
            yield f"ret{n}", (C / lag(C, n) - 1).astype(F32)
        for w in (5, 20):
            yield f"on{w}", _roll(np.nan_to_num(on), w, "sum", w)
            yield f"in{w}", _roll(np.nan_to_num(intra), w, "sum", w)
        for w in (5, 20, 60):
            yield f"vol{w}", _roll(lr, w, "std")
        yield "skew20", _roll(lr, 20, "skew")
        yield "max20", _roll(lr, 20, "max")
        yield "min20", _roll(lr, 20, "min")
        turn = np.where(ts, a["turn"].astype(np.float64), np.nan)
        t5, t20, t60 = _roll(turn, 5), _roll(turn, 20), _roll(turn, 60)
        yield "turn5", t5
        yield "turn20", t20
        yield "turn60", t60
        yield "turn_5_60", (t5 / t60).astype(F32)
        yield "turn_1_20", (turn / t20).astype(F32)
        del t5, t20, t60, turn
        amt = np.where(ts, a["amount"].astype(np.float64), np.nan)
        a20 = _roll(amt, 20)
        yield "log_amt20", np.log(a20).astype(F32)
        yield "amt_1_20", (amt / a20).astype(F32)
        yield "amihud20", np.log1p(_roll(np.abs(lr) / amt * 1e9, 20)).astype(F32)
        del amt, a20
        rng = (H - L) / C
        yield "range5", _roll(rng, 5)
        yield "range20", _roll(rng, 20)
        clv = (2 * C - H - L) / (H - L)
        clv[~np.isfinite(clv)] = 0.0
        yield "clv5", _roll(np.where(ts, clv, np.nan), 5)
        del rng, clv
        for w in (20, 60):
            hi, lo = _roll(C, w, "max"), _roll(C, w, "min")
            yield f"pos{w}", ((C - lo) / (hi - lo)).astype(F32)
        for w in (5, 20, 60):
            yield f"ma{w}", (C / _roll(C, w) - 1).astype(F32)
        yield "gap20", _roll(np.abs(on), 20)
        lu = limit_up_close(md)
        ld = (a["tradestatus"] == 1) & (a["no_limit"] == 0) & (a["close"] <= a["down_limit"] + TOL)
        tu = (a["tradestatus"] == 1) & (a["no_limit"] == 0) & (a["high"] >= a["up_limit"] - TOL)
        yield "n_lu20", _roll(lu.astype(F32), 20, "sum", 1)
        yield "n_ld20", _roll(ld.astype(F32), 20, "sum", 1)
        yield "n_touchu20", _roll(tu.astype(F32), 20, "sum", 1)
        yield "days_since_lu", _days_since(lu, cap=120)
        yield "log_fmv", np.log(a["fmv"]).astype(F32)
        yield "log_price", np.log(a["close"]).astype(F32)
        yield "log_listed", np.log1p(a["days_listed"].astype(F32))


def _days_since(mask: np.ndarray, cap: int = 120) -> np.ndarray:
    """距最近一次 mask 为真的交易日数（含当日 = 0），从未发生或超过 cap 记为 cap。只向过去看。"""
    T, N = mask.shape
    out = np.full((T, N), cap, dtype=F32)
    last = np.full(N, -10 ** 9)
    for t in range(T):
        last = np.where(mask[t], t, last)
        out[t] = np.minimum(t - last, cap)
    return out


def _event_mask(md, t, j) -> np.ndarray:
    m = np.zeros((md.T, md.N), dtype=bool)
    m[t, j] = True
    return m


def event_features(md, ext: dict):
    f = {}
    if "yjyg" in ext:
        ev = yjyg_positive(ext["yjyg"])
        t, j, ev = event_rows(md, ev, after_close_shift=True)
        f["d_yjyg_pos"] = _days_since(_event_mask(md, t, j), 60)
        s = np.full((md.T, md.N), np.nan)
        s[t, j] = ev["strength"].values
        f["yjyg_strength"] = pd.DataFrame(s).ffill(limit=60).values.astype(F32)
        x = ext["yjyg"]
        neg = x[x["metric"].astype(str).str.contains("归属于上市公司股东的净利润", na=False)
                & x["type"].isin(["首亏", "预减", "增亏", "续亏"])]
        t2, j2, _ = event_rows(md, neg, after_close_shift=True)
        f["d_yjyg_neg"] = _days_since(_event_mask(md, t2, j2), 60)
    if "buyback" in ext:
        t, j, _ = event_rows(md, buyback_events(ext["buyback"]), after_close_shift=True)
        f["d_buyback"] = _days_since(_event_mask(md, t, j), 60)
    if "zengchi" in ext:
        t, j, _ = event_rows(md, zengchi_events(ext["zengchi"]), after_close_shift=True)
        f["d_zengchi"] = _days_since(_event_mask(md, t, j), 60)
    if "gdhs" in ext:
        ev = gdhs_decline(ext["gdhs"], lo=-90.0, hi=1e9)          # 全部记录（含上升），取变动比例
        t, j, ev = event_rows(md, ev, after_close_shift=True)
        s = np.full((md.T, md.N), np.nan)
        s[t, j] = -ev["strength"].values                          # = 变动比例 / 100
        f["gdhs_chg"] = pd.DataFrame(s).ffill(limit=120).values.astype(F32)
    if "lhb" in ext:
        df = ext["lhb"]
        t, j, df = event_rows(md, df)
        s = np.zeros((md.T, md.N))
        np.add.at(s, (t, j), np.sign(df["inst_net"].values.astype(float)))
        f["lhb_inst20"] = _roll(s, 20, "sum", 1)
    if "dzjy" in ext:
        df = ext["dzjy"]
        t, j, df = event_rows(md, df)
        s = np.zeros((md.T, md.N))
        np.add.at(s, (t, j), np.where(df["premium"].values >= 0, 1.0, -1.0))
        f["dzjy_net20"] = _roll(s, 20, "sum", 1)
    if "reports" in ext:
        df = ext["reports"]
        t, j, df = event_rows(md, df, after_close_shift=True)
        cnt = np.zeros((md.T, md.N))
        np.add.at(cnt, (t, j), 1.0)
        newc = np.zeros((md.T, md.N))
        isnew = (df["lastEmRatingName"].astype(str).str.len() == 0) & (df["emRatingName"].astype(str).str.len() > 0)
        np.add.at(newc, (t[isnew.values], j[isnew.values]), 1.0)
        start = int(md.dates.searchsorted(pd.to_datetime(df["avail_date"]).min()))   # 数据起点之前记为缺失
        r30, n30 = _roll(cnt, 30, "sum", 1), _roll(newc, 30, "sum", 1)
        r30[:start + 30] = np.nan
        n30[:start + 30] = np.nan
        f["rep30"], f["newcov30"] = r30, n30
    if "unlock" in ext:
        df = ext["unlock"]
        t, j, df = event_rows(md, df)                  # avail_date = 解禁日前 3 个交易日（见 unlock_ext）
        D = t + 3
        sh = df["shares_actual"].values.astype(float)
        up = np.zeros((md.T, md.N))
        past = np.zeros((md.T, md.N))
        with np.errstate(invalid="ignore", divide="ignore"):
            for tt, jj, d, s_ in zip(t, j, D, sh):
                if not np.isfinite(s_):
                    continue
                for k in range(tt, min(d, md.T)):     # 解禁前 3 个交易日：即将解禁
                    up[k, jj] += s_
                for k in range(d, min(d + 10, md.T)):  # 解禁后 10 个交易日
                    past[k, jj] += s_
            fs = md.a["fshares"]
            f["unlock_next"] = (up / fs).astype(F32)
            f["unlock_past10"] = (past / fs).astype(F32)
    yield from f.items()
    if "fin" in ext:
        from fundamentals import fundamental_features
        yield from fundamental_features(md, ext)
    if "margin" in ext:
        yield from margin_features(md, ext)
    for key, pat, excl in (("wenxun", "问询函", "审核|回复|答复|延期|核查意见"), ("lian", "立案", "进展|结案|撤销|终止")):
        if key in ext:
            df = ext[key]
            tt = df["title"].astype(str)
            ev = df[tt.str.contains(pat) & ~tt.str.contains(excl)]
            t, j, _ = event_rows(md, ev, after_close_shift=True)
            m = np.zeros((md.T, md.N), bool)
            m[t, j] = True
            yield f"d_{key}", _days_since(m, 250)


def build(md, ext: dict | None = None):
    """逐个产出 (名称, 矩阵)。"""
    yield from price_features(md)
    if ext:
        yield from event_features(md, ext)


def margin_features(md, ext):
    """融资融券特征（交易所次日公布，统一顺延一个交易日：t 日只用到 t−1 日的数据）。非两融标的记为缺失。
    数据来自 data/derived/margin.npz（由 build_margin.py 从明细压缩而成，按数据日期对齐）。
    前视检测时 ext["margin"] 为扰动后的同结构字典。"""
    mg = ext["margin"]
    def get(k):
        m = np.asarray(mg[k], dtype=np.float64)[: md.T]
        return lag(m, 1)                                    # 顺延一个交易日
    rzye, rzjme, rqyl = get("rzye"), get("rzjme"), get("rqyl")
    fmv = md.a["fmv"]
    with np.errstate(invalid="ignore", divide="ignore"):
        yield "rz_ratio", (rzye / fmv).astype(F32)
        yield "rz_chg5", ((rzye - lag(rzye, 5)) / fmv).astype(F32)
        yield "rz_net20", (_roll(np.nan_to_num(rzjme), 20, "sum", 10) / fmv).astype(F32)
        yield "rq_ratio", (rqyl * lag(md.a["close"], 1) / fmv).astype(F32)
