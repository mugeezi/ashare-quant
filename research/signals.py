"""阶段 3 信号定义。每个 factory(md, ext) 返回 SignalStrategy；只允许用 lag（向过去取值），不允许任何向前移位。
所有信号均在 t 日收盘后计算，t+1 开盘成交（引擎负责）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from framework import SignalStrategy, eligible_matrix, hfq

TOL = 0.005


def lag(x: np.ndarray, k: int = 1) -> np.ndarray:
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if k < x.shape[0]:
        out[k:] = x[:-k]
    return out


def rsum(x: np.ndarray, w: int) -> np.ndarray:
    """过去 w 日（含当日）滚动和；窗口内有 NaN 则为 NaN。"""
    return pd.DataFrame(x).rolling(w, min_periods=w).sum().values


def ret_n(md, n: int, field="close") -> np.ndarray:
    p = hfq(md, field)
    with np.errstate(invalid="ignore", divide="ignore"):
        return p / lag(p, n) - 1


def limit_up_close(md) -> np.ndarray:
    a = md.a
    return (a["tradestatus"] == 1) & (a["no_limit"] == 0) & (a["close"] >= a["up_limit"] - TOL)


def touched_down(md) -> np.ndarray:
    a = md.a
    return (a["tradestatus"] == 1) & (a["no_limit"] == 0) & (a["low"] <= a["down_limit"] + TOL)


def _score(cond: np.ndarray, value: np.ndarray) -> np.ndarray:
    s = np.where(cond, value, np.nan)
    s[~np.isfinite(s)] = np.nan
    return s


# ---------------------------------------------------------------- H01 撬板反弹
def h01(md, ext=None, rebound=0.03):
    a = md.a
    with np.errstate(invalid="ignore", divide="ignore"):
        rb = (a["close"] - a["low"]) / a["preclose"]
    return SignalStrategy(_score(touched_down(md) & (rb >= rebound), rb))


# ---------------------------------------------------------------- H02 低开高走放量
def h02(md, ext=None, gap=-0.03, up=0.01, vol_mult=1.5):
    a = md.a
    with np.errstate(invalid="ignore", divide="ignore"):
        g = a["open"] / a["preclose"] - 1
        r = a["close"] / a["preclose"] - 1
        intr = a["close"] / a["open"] - 1
    cond = (a["tradestatus"] == 1) & (g <= gap) & (r >= up) & (a["amount"] >= vol_mult * a["amt20"])
    return SignalStrategy(_score(cond, intr))


# ---------------------------------------------------------------- 同伴股（收益相关性，按月重算，只用过去数据）
def peer_sets(md, k=20, window=250, min_obs=200) -> tuple[np.ndarray, np.ndarray]:
    """返回 (peer_idx, valid_from)：peer_idx[t] 为 N×k 同伴下标矩阵所在的重算批次编号。
    每月第一个交易日 t_m 用 [t_m−window+1, t_m] 的日收益（后复权收盘）计算相关系数，自 t_m 当日收盘起使用。"""
    cache = md.__dict__.setdefault("_peer_cache", {})      # 缓存挂在 md 对象上，扰动后的副本各算各的
    key = (k, window, min_obs)
    if key in cache:
        return cache[key]
    p = hfq(md, "close")
    with np.errstate(invalid="ignore", divide="ignore"):
        r = p[1:] / p[:-1] - 1
    r = np.vstack([np.full((1, md.N), np.nan), r])
    r[(md.a["tradestatus"] == 0)] = np.nan
    months = pd.Series(md.dates).dt.to_period("M")
    starts = np.where(months.ne(months.shift()).values)[0]
    starts = starts[starts >= window]
    batches = []
    batch_of_t = np.full(md.T, -1)
    for bi, tm in enumerate(starts):
        w = r[tm - window + 1: tm + 1]
        obs = np.isfinite(w).sum(0)
        ok = obs >= min_obs
        x = w[:, ok]
        mu = np.nanmean(x, 0)
        sd = np.nanstd(x, 0)
        z = np.nan_to_num((x - mu) / np.where(sd > 0, sd, np.nan))
        c = (z.T @ z) / np.maximum(1, (np.isfinite(x).astype(float).T @ np.isfinite(x).astype(float)))
        np.fill_diagonal(c, -np.inf)
        kk = min(k, c.shape[0] - 1)
        top = np.argpartition(-c, kk, axis=1)[:, :kk]
        full = np.full((md.N, kk), -1, dtype=np.int64)
        oki = np.where(ok)[0]
        full[oki] = oki[top]
        batches.append(full)
        end = starts[bi + 1] if bi + 1 < len(starts) else md.T
        batch_of_t[tm:end] = bi
    cache[key] = (batches, batch_of_t)
    return batches, batch_of_t


def peer_mean(md, x: np.ndarray, k=20) -> np.ndarray:
    """每个 t：同伴（当期批次）在 t 日的 x 均值。"""
    batches, bot = peer_sets(md, k=k)
    out = np.full(x.shape, np.nan)
    for t in range(md.T):
        b = bot[t]
        if b < 0:
            continue
        P = batches[b]
        has = P[:, 0] >= 0
        vals = x[t][np.where(P >= 0, P, 0)]
        vals = np.where(P >= 0, vals, np.nan)
        with np.errstate(invalid="ignore"):
            m = np.nanmean(vals, axis=1)
        out[t] = np.where(has, m, np.nan)
    return out


# ---------------------------------------------------------------- H03 同伴补涨
def h03(md, ext=None, n=5, k=20):
    r = ret_n(md, n)
    pm = peer_mean(md, r, k=k)
    gap = pm - r
    cond = (pm > 0) & np.isfinite(gap) & (md.a["tradestatus"] == 1)
    return SignalStrategy(_score(cond, gap))


# ---------------------------------------------------------------- H04 主题扩散：同伴集体涨停、本股未跟
def h04(md, ext=None, k=20, min_cnt=3, own_max=0.05):
    lu = limit_up_close(md).astype(float)
    frac = peer_mean(md, lu, k=k)                     # 同伴中涨停的比例
    cnt = np.round(frac * k)
    with np.errstate(invalid="ignore", divide="ignore"):
        own = md.a["close"] / md.a["preclose"] - 1
    cond = (cnt >= min_cnt) & (own < own_max) & (md.a["tradestatus"] == 1)
    return SignalStrategy(_score(cond, cnt + 1e-3 * np.nan_to_num(-own)))


# ---------------------------------------------------------------- H05 ST 摘帽首日
def h05(md, ext=None):
    st = md.a["isST"].astype(float)
    cond = (lag(st, 1) == 1) & (st == 0) & (md.a["tradestatus"] == 1) & (md.a["delist_period"] == 0)
    return SignalStrategy(_score(cond, np.nan_to_num(md.a["amt20"]).astype(float)))


# ---------------------------------------------------------------- H08 隔夜−日内拉锯
def h08(md, ext=None, w=20, band=0.10):
    o, c = hfq(md, "open"), hfq(md, "close")
    with np.errstate(invalid="ignore", divide="ignore"):
        on = np.log(o / lag(c, 1))
        intra = np.log(c / o)
    on[md.a["tradestatus"] == 0] = 0.0
    intra[md.a["tradestatus"] == 0] = 0.0
    S_on, S_in = rsum(on, w), rsum(intra, w)
    tot = S_on + S_in
    cond = (np.abs(tot) <= np.log1p(band)) & (md.a["tradestatus"] == 1)
    return SignalStrategy(_score(cond, S_in - S_on))


# ---------------------------------------------------------------- H09 断板反包
def h09(md, ext=None, drop=-0.03):
    a = md.a
    lu = limit_up_close(md)
    first = lu & ~np.vstack([np.zeros((1, md.N), bool), lu[:-1]]) & (a["low"] < a["high"])   # 首板、非一字
    first_y = np.vstack([np.zeros((1, md.N), bool), first[:-1]])                              # t−1 首板
    with np.errstate(invalid="ignore", divide="ignore"):
        r = a["close"] / a["preclose"] - 1
        g = a["open"] / a["preclose"] - 1
    amt_y = lag(a["amount"].astype(float), 1)
    cond = first_y & (a["tradestatus"] == 1) & (g > 0) & (r <= drop) & (a["amount"] > amt_y)
    return SignalStrategy(_score(cond, -r))


# ---------------------------------------------------------------- 外部事件信号的公共工具
def code_map(md) -> dict:
    return {c.split(".")[1]: i for i, c in enumerate(md.codes)}


def event_rows(md, df: pd.DataFrame, after_close_shift: bool = False):
    """把事件表映射到 (t, j)。avail_date 为交易日时 t = 该日；非交易日顺延到下一交易日。
    after_close_shift=True：公告时刻不确定是否盘后，信号再顺延到 avail_date 之后的第一个交易日。"""
    cm = code_map(md)
    j = df["code6"].astype(str).str.zfill(6).map(cm)
    d = pd.to_datetime(df["avail_date"]).values
    t = md.dates.searchsorted(d, side="right" if after_close_shift else "left")
    ok = j.notna().values & (t < md.T)
    return t[ok], j[ok].astype(int).values, df[ok]


# ---------------------------------------------------------------- H06 龙虎榜机构净买入
def h06(md, ext, min_inst=2, min_ratio=5.0):
    df = ext["lhb"]
    df = df[(df["n_buy_inst"] >= min_inst) & (df["inst_net"] > 0) & (df["net_ratio_pct"] >= min_ratio)]
    t, j, df = event_rows(md, df)
    s = np.full((md.T, md.N), np.nan)
    v = df["net_ratio_pct"].values.astype(float)
    order = np.argsort(v)                       # 同一格多条：后写入的更大，保留最大值
    s[t[order], j[order]] = v[order]
    return SignalStrategy(s)


# ---------------------------------------------------------------- H07 回购预案公告
BB_EXCL = ("进展", "实施", "完成", "结果", "注销", "补充", "更正", "问询", "法律意见", "独立意见", "股权激励", "限制性",
           "核查", "回复", "提示性", "债", "摘要", "取消", "终止", "调整", "说明", "延期", "期满", "购买理财", "回售", "赎回",
           "股东大会", "决议", "会议", "报告书", "授权", "通知")


def buyback_events(df: pd.DataFrame) -> pd.DataFrame:
    tt = df["title"].astype(str)
    m = tt.str.contains("回购") & tt.str.contains("预案|方案") & ~tt.str.contains("|".join(BB_EXCL))
    ev = df[m].sort_values("avail_date")
    keep, last = [], {}
    for idx, c, d in zip(ev.index, ev["code6"].astype(str), pd.to_datetime(ev["avail_date"])):
        if c in last and (d - last[c]).days < 90:
            continue
        last[c] = d
        keep.append(idx)
    return ev.loc[keep]


def h07(md, ext):
    ev = buyback_events(ext["buyback"])
    t, j, _ = event_rows(md, ev, after_close_shift=True)
    s = np.full((md.T, md.N), np.nan)
    s[t, j] = np.nan_to_num(md.a["amount"][t, j].astype(float)) + 1.0
    return SignalStrategy(s)


# ---------------------------------------------------------------- H10 大宗交易溢价成交
def h10(md, ext, min_ratio=0.001):
    df = ext["dzjy"]
    df = df[df["premium"] >= 0].copy()
    df["w"] = df["amount"] * df["premium"]
    g = df.groupby(["avail_date", "code6"], as_index=False).agg(amount=("amount", "sum"), w=("w", "sum"))
    g["wprem"] = g["w"] / g["amount"]
    t, j, g = event_rows(md, g)
    fmv = md.a["fmv"][t, j]
    ratio = g["amount"].values / fmv
    ok = np.isfinite(ratio) & (ratio >= min_ratio)
    s = np.full((md.T, md.N), np.nan)
    s[t[ok], j[ok]] = ratio[ok] * (1 + g["wprem"].values[ok])
    return SignalStrategy(s)


# ================================================================ 第二批：事件窗口、关注度排除
def window_score(md, t, j, base, W=20):
    """事件在信号日 t 起 W 个交易日内都是候选。得分 = base × 1000 + 剩余天数（同强度时越新越靠前）。
    同一格多个事件取最大值。只向后（未来方向）填充事件本身的有效期，不使用事件之后的任何数据。"""
    s = np.full((md.T, md.N), np.nan)
    base = np.asarray(base, dtype=float)
    for tt, jj, b in zip(t, j, base):
        hi = min(md.T, tt + W)
        v = b * 1000 + (W - np.arange(hi - tt))
        cur = s[tt:hi, jj]
        s[tt:hi, jj] = np.where(np.isnan(cur) | (v > cur), v, cur)
    return s


def quiet_mask(md, w=10, max_abs=0.07, turn_mult=2.0):
    """关注度排除：过去 w 个交易日（含 t）内没有收盘涨停或跌停、没有单日涨跌幅绝对值 ≥ max_abs，
    且 t 日换手率不超过过去 60 日换手率中位数的 turn_mult 倍。全部只用 t 日及以前的数据。"""
    a = md.a
    nl = a["no_limit"] == 0
    lim = ((a["close"] >= a["up_limit"] - TOL) | (a["close"] <= a["down_limit"] + TOL)) & nl & (a["tradestatus"] == 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.abs(a["close"] / a["preclose"] - 1)
    big = (r >= max_abs) & (a["tradestatus"] == 1) & nl
    bad = (lim | big).astype(float)
    recent = pd.DataFrame(bad).rolling(w, min_periods=1).max().values > 0
    turn = np.where(a["tradestatus"] == 1, a["turn"].astype(float), np.nan)
    med = pd.DataFrame(turn).rolling(60, min_periods=20).median().values
    hot = turn > turn_mult * med
    return ~recent & ~hot


def _apply_quiet(md, s, quiet):
    if quiet:
        s = np.where(quiet_mask(md), s, np.nan)
    return s


# ---------------------------------------------------------------- H11 回购预案窗口
def h11(md, ext, W=20, quiet=False):
    ev = buyback_events(ext["buyback"])
    t, j, _ = event_rows(md, ev, after_close_shift=True)
    return SignalStrategy(_apply_quiet(md, window_score(md, t, j, np.ones(len(t)), W), quiet))


# ---------------------------------------------------------------- 业绩预告正面事件
def yjyg_positive(df: pd.DataFrame, min_chg=50.0) -> pd.DataFrame:
    x = df[df["metric"].astype(str).str.contains("归属于上市公司股东的净利润", na=False)].copy()
    x = x.sort_values("avail_date").drop_duplicates(["code6", "report_period"], keep="first")
    pos = (x["type"] == "扭亏") | ((x["type"] == "预增") & (x["chg_pct"] >= min_chg))
    x = x[pos].copy()
    x["strength"] = np.where(x["type"] == "扭亏", np.log1p(2.0), np.log1p(np.clip(x["chg_pct"].astype(float), 0, 1000) / 100))
    return x


def h12(md, ext, W=20, quiet=False):
    ev = yjyg_positive(ext["yjyg"])
    t, j, ev = event_rows(md, ev, after_close_shift=True)
    return SignalStrategy(_apply_quiet(md, window_score(md, t, j, ev["strength"].values, W), quiet))


# ---------------------------------------------------------------- H13 业绩预告正面且未被追涨
def h13(md, ext, W=20, max_react=0.03):
    ev = yjyg_positive(ext["yjyg"])
    t, j, ev = event_rows(md, ev, after_close_shift=True)
    d0 = md.dates.searchsorted(pd.to_datetime(ev["avail_date"]).values, side="left")   # 公告日（或其后首个交易日）
    c = hfq(md, "close")
    base = np.maximum(d0 - 1, 0)                                                       # 公告日前一交易日收盘
    with np.errstate(invalid="ignore", divide="ignore"):
        react = c[t, j] / c[base, j] - 1
    lu = limit_up_close(md)[t, j]
    ok = np.isfinite(react) & (react < max_react) & ~lu
    return SignalStrategy(window_score(md, t[ok], j[ok], ev["strength"].values[ok], W))


# ---------------------------------------------------------------- 增持计划
ZC_EXCL = ("减持", "进展", "完成", "完毕", "结果", "实施情况", "届满", "终止", "补充", "更正", "问询", "回复", "法律意见", "核查",
           "延期", "调整", "取消", "股东大会", "决议", "会议", "员工持股", "债", "提示性公告的更正")


def zengchi_events(df: pd.DataFrame) -> pd.DataFrame:
    tt = df["title"].astype(str)
    m = tt.str.contains("增持") & tt.str.contains("计划|首次增持|增持公司股份的公告|增持股份的公告") & \
        ~tt.str.contains("|".join(ZC_EXCL))
    ev = df[m].sort_values("avail_date")
    keep, last = [], {}
    for idx, c, d in zip(ev.index, ev["code6"].astype(str), pd.to_datetime(ev["avail_date"])):
        if c in last and (d - last[c]).days < 90:
            continue
        last[c] = d
        keep.append(idx)
    return ev.loc[keep]


def h14(md, ext, W=20, quiet=False):
    ev = zengchi_events(ext["zengchi"])
    t, j, _ = event_rows(md, ev, after_close_shift=True)
    return SignalStrategy(_apply_quiet(md, window_score(md, t, j, np.ones(len(t)), W), quiet))


# ---------------------------------------------------------------- H15 信息事件合成
def h15(md, ext, W=20, quiet=True):
    parts = []
    ev = buyback_events(ext["buyback"]); t, j, _ = event_rows(md, ev, True); parts.append((t, j))
    ev = yjyg_positive(ext["yjyg"]); t, j, _ = event_rows(md, ev, True); parts.append((t, j))
    ev = zengchi_events(ext["zengchi"]); t, j, _ = event_rows(md, ev, True); parts.append((t, j))
    cnt = np.zeros((md.T, md.N))
    rec = np.full((md.T, md.N), np.nan)
    for t, j in parts:
        w = window_score(md, t, j, np.zeros(len(t)), W)          # 0×1000 + 剩余天数
        has = np.isfinite(w)
        cnt += has
        rec = np.where(has & (np.isnan(rec) | (w > rec)), w, rec)
    s = np.where(cnt > 0, cnt * 1000 + rec, np.nan)
    return SignalStrategy(_apply_quiet(md, s, quiet))


# ---------------------------------------------------------------- H17 关注度排除本身（随机打分基准）
def h17(md, ext=None, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.Series(md.dates).dt.to_period("M")
    mid = months.factorize()[0]
    draws = rng.random((mid.max() + 1, md.N))
    s = draws[mid]                                               # 每只股票每月一个固定随机分，便于续持
    return SignalStrategy(np.where(quiet_mask(md), s, np.nan))


# ---------------------------------------------------------------- H18 龙虎榜机构净买、当日未大涨
def h18(md, ext, max_ret=0.03, W=5):
    df = ext["lhb"]
    df = df[(df["n_buy_inst"] >= 2) & (df["inst_net"] > 0) & (df["net_ratio_pct"] >= 5.0)]
    t, j, df = event_rows(md, df)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = md.a["close"][t, j] / md.a["preclose"][t, j] - 1
    ok = np.isfinite(r) & (r < max_ret)
    return SignalStrategy(window_score(md, t[ok], j[ok], df["net_ratio_pct"].values[ok] / 100, W))


# ---------------------------------------------------------------- H19 大宗交易：机构专用席位溢价买入
def h19(md, ext, W=10):
    df = ext["dzjy"]
    df = df[(df["premium"] >= 0) & df["buyer"].astype(str).str.contains("机构专用")]
    g = df.groupby(["avail_date", "code6"], as_index=False).agg(amount=("amount", "sum"))
    t, j, g = event_rows(md, g)
    ratio = g["amount"].values / md.a["fmv"][t, j]
    ok = np.isfinite(ratio) & (ratio >= 0.001)
    return SignalStrategy(window_score(md, t[ok], j[ok], np.log1p(ratio[ok] * 100), W))
