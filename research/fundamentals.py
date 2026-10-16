"""基本面特征（时点 = 首次公告日之后的第一个交易日，按盘后处理），全部只用当时已公开的财报。

由业绩报表（累计值）推出单季与滚动四季（TTM）数值；估值用后复权口径，避免送转造成的跳变：
  后复权每股收益 = 每股收益 × 公告日复权因子；E/P = TTM 后复权每股收益 / 当日后复权收盘价。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from framework import hfq
from signals import event_rows, peer_mean

F32 = np.float32


def quarterly(fin: pd.DataFrame) -> pd.DataFrame:
    """严格时点版：每一行的单季、TTM、去年同期毛利率，只用同一公司**公告日不晚于本行公告日**的财报计算。
    （个别记录的公告日远晚于报告期，若直接按报告期排序做差，会用到当时尚未"公开"的记录。）"""
    df = fin.copy()
    df["code6"] = df["code6"].astype(str).str.zfill(6)
    df = df.sort_values(["code6", "notice_date", "period"]).drop_duplicates(["code6", "period"], keep="first")
    df["q"] = df["period"].dt.quarter
    out = {k: np.full(len(df), np.nan) for k in ("basic_eps_ttm", "parent_netprofit_ttm", "total_operate_income_ttm",
                                                    "gm_ly")}
    cols = ("basic_eps", "parent_netprofit", "total_operate_income")
    vals = {c: df[c].astype(float).values for c in cols}
    gm = df["xsmll"].astype(float).values
    per = df["period"].values
    codes = df["code6"].values
    known = {}
    cur = None
    for i in range(len(df)):
        if codes[i] != cur:
            cur, known = codes[i], {}
        p = pd.Timestamp(per[i])
        known[p] = {c: vals[c][i] for c in cols} | {"gm": gm[i]}

        def single(pp, c):
            v = known.get(pp, {}).get(c, np.nan)
            if pp.quarter == 1:
                return v
            prev = pp - pd.offsets.QuarterEnd(1)
            return v - known.get(prev, {}).get(c, np.nan)

        qs = [p - pd.offsets.QuarterEnd(k) for k in range(4)]
        for c in cols:
            out[f"{c}_ttm"][i] = sum(single(pp, c) for pp in qs)
        out["gm_ly"][i] = known.get(p - pd.DateOffset(years=1) + pd.offsets.QuarterEnd(0), {}).get("gm", np.nan)
    for k, v in out.items():
        df[k] = v
    df["roe_ann"] = df["weightavg_roe"].astype(float) * 4 / df["q"]
    df["gm"] = gm
    df["gm_chg"] = df["gm"] - df["gm_ly"]
    with np.errstate(invalid="ignore", divide="ignore"):
        df["cfo_ratio"] = np.clip(df["mgjyxjje"].astype(float) / df["basic_eps"].astype(float).abs(), -5, 5)
    df["ni_yoy"] = np.clip(df["sjltz"].astype(float), -300, 300)
    df["rev_yoy"] = np.clip(df["ystz"].astype(float), -300, 300)
    return df


def _fill(md, t, j, v, limit=260):
    s = np.full((md.T, md.N), np.nan)
    order = np.argsort(t, kind="stable")
    s[t[order], j[order]] = v[order]
    return pd.DataFrame(s).ffill(limit=limit).values


def surprise_vs_forecast(fin_q: pd.DataFrame, yjyg: pd.DataFrame) -> pd.Series:
    """正式财报归母净利润相对业绩预告预测值（首条）的偏离：(实际 − 预测) / |预测|。只用财报公告日之前发布的预告。"""
    y = yjyg[yjyg["metric"].astype(str).str.contains("归属于上市公司股东的净利润", na=False)].copy()
    y["code6"] = y["code6"].astype(str).str.zfill(6)
    y = y.sort_values("avail_date").drop_duplicates(["code6", "report_period"], keep="first")
    m = fin_q.merge(y[["code6", "report_period", "forecast", "avail_date"]].rename(
        columns={"report_period": "period", "avail_date": "fc_date"}), on=["code6", "period"], how="left")
    ok = m["fc_date"] < m["notice_date"]
    with np.errstate(invalid="ignore", divide="ignore"):
        sur = (m["parent_netprofit"].astype(float) - m["forecast"].astype(float)) / m["forecast"].astype(float).abs()
    return pd.Series(np.where(ok, np.clip(sur, -2, 2), np.nan), index=fin_q.index)


def report_days_only(md, ext: dict):
    """只用财报公告日（不用任何数值）：距最近一次财报公告的交易日数。与 fundamental_features 中的 days_since_report 逐值相同。
    2026-10-16 数据修复：东方财富业绩报表数值存在追溯调整后记在首次公告日的问题，_nf 版本只保留这一个特征。"""
    q = quarterly(ext["fin"])
    t, j, _ = event_rows(md, q, after_close_shift=True)
    days = np.full((md.T, md.N), np.nan)
    ev = np.zeros((md.T, md.N), bool)
    ev[t, j] = True
    last = np.full(md.N, -10 ** 9)
    for k in range(md.T):
        last = np.where(ev[k], k, last)
        days[k] = np.minimum(k - last, 260)
    yield "days_since_report", days.astype(F32)


def fundamental_features(md, ext: dict):
    """逐个产出 (名称, T×N 矩阵)。"""
    q = quarterly(ext["fin"])
    t, j, qq = event_rows(md, q, after_close_shift=True)
    adj = md.a["adj_factor"]
    C = hfq(md, "close")
    adj_n = adj[np.minimum(t, md.T - 1), j]
    with np.errstate(invalid="ignore", divide="ignore"):
        eps_h = _fill(md, t, j, qq["basic_eps_ttm"].values.astype(float) * adj_n)
        bps_h = _fill(md, t, j, qq["bps"].values.astype(float) * adj_n)
        ep = (eps_h / C).astype(F32)
        bp = (bps_h / C).astype(F32)
    yield "ep_ttm", ep
    yield "bp", bp
    for c in ("roe_ann", "gm", "gm_chg", "cfo_ratio", "ni_yoy", "rev_yoy"):
        yield c, _fill(md, t, j, qq[c].values.astype(float)).astype(F32)
    if "yjyg" in ext:
        sur = surprise_vs_forecast(q, ext["yjyg"]).loc[qq.index].values
        yield "fc_surprise", _fill(md, t, j, sur, limit=60).astype(F32)
    # 估值相对自身历史：当前 E/P 在自身过去 750 个交易日中的分位
    epd = pd.DataFrame(ep)
    yield "ep_self_pct", epd.rolling(750, min_periods=250).rank(pct=True).values.astype(F32)
    # 估值相对同伴（PIT 相关性同伴）
    yield "ep_peer_rel", (ep - peer_mean(md, ep)).astype(F32)
    days = np.full((md.T, md.N), np.nan)
    ev = np.zeros((md.T, md.N), bool)
    ev[t, j] = True
    last = np.full(md.N, -10 ** 9)
    for k in range(md.T):
        last = np.where(ev[k], k, last)
        days[k] = np.minimum(k - last, 260)
    yield "days_since_report", days.astype(F32)
