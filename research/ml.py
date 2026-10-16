"""滚动训练的截面打分模型（LightGBM）。

- 样本：每个交易日 t 的可入选股票（与引擎同口径过滤）。
- 标签：t+1 开盘到 t+1+H 开盘的后复权收益，在当日截面内转成分位数（0–1）再减 0.5。
- 滚动（walk-forward）：预测第 Y 年时，只用标签窗口在 Y 年第一个交易日之前**结束**的样本训练
  （t + 1 + H < Y 年首日），即在训练集和预测期之间留出 H+1 天的隔离带，防止标签重叠泄漏。
  每年重训一次（扩张窗口）。训练集评估用 2016–2020 的样本外预测；验证集运行时 2021、2022、2023
  分别用截至上一年末的数据重训。
- 超参数事先固定，不调参（避免增加隐性试验次数）。
"""
from __future__ import annotations

import gc

import lightgbm as lgb
import numpy as np
import pandas as pd

import features as FE
from framework import SignalStrategy, eligible_matrix, hfq

PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=2000,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
              num_threads=2, seed=7, deterministic=True, force_row_wise=True, verbose=-1)
N_ROUNDS = 300
WARM = "2013-10-01"


def _rank_inputs(X, ti):
    """每个特征在当日截面内转成百分位（0–1），缺失记 0.5。消除不同市场阶段特征水平的漂移。"""
    out = np.empty_like(X)
    order = np.argsort(ti, kind="stable")
    tis = ti[order]
    bounds = np.flatnonzero(np.diff(tis)) + 1
    starts = np.r_[0, bounds]
    ends = np.r_[bounds, len(tis)]
    for k in range(X.shape[1]):
        col = X[order, k]
        res = np.empty(len(col), dtype=np.float32)
        for a, b in zip(starts, ends):
            v = col[a:b]
            r = pd.Series(v).rank(pct=True).values
            res[a:b] = np.where(np.isnan(r), 0.5, r)
        out[order, k] = res
    return out


def make_panel(md, ext, H, max_year=None, keep=None, rank_inputs=False):
    el = eligible_matrix(md)
    t0 = int(md.dates.searchsorted(pd.Timestamp(WARM)))
    el[:t0] = False
    if max_year is not None:                                 # 只保留到 max_year 年末（前视检测用的轻量版）
        el[int(md.dates.searchsorted(pd.Timestamp(f"{max_year + 1}-01-01"))):] = False
    ti, ji = np.where(el)
    names = []
    cap = 96
    X = np.empty((len(ti), cap), dtype=np.float32, order="F")   # 列优先预分配：未用到的列不占物理内存
    for name, arr in FE.build(md, ext):
        if keep is not None and name not in keep:
            del arr
            continue
        if len(names) == cap:
            raise RuntimeError("特征数超过预分配上限")
        X[:, len(names)] = arr[ti, ji]
        names.append(name)
        del arr
        gc.collect()
    X = X[:, :len(names)]                                    # 视图，不复制
    print(f"特征 {len(names)} 个：{','.join(names)}", flush=True)
    gc.collect()
    if rank_inputs:
        X = _rank_inputs(X, ti)
    O = hfq(md, "open")
    fwd = np.full(len(ti), np.nan)
    ok = ti + 1 + H < md.T
    a, b = O[ti[ok] + 1, ji[ok]], O[ti[ok] + 1 + H, ji[ok]]
    with np.errstate(invalid="ignore", divide="ignore"):
        fwd[ok] = b / a - 1
    y = pd.Series(fwd).groupby(ti).rank(pct=True).values - 0.5
    return names, X, y, ti, ji


def walk_forward_scores(md, ext=None, H=10, first_year=2016, last_year=2023, stride=2, max_year=None, keep=None,
                        seeds=(7,), model="lgbm", rank_inputs=False, window_years=None, label="rank", free_md=False,
                        universe="all"):
    """model：lgbm（回归）| rank（LightGBM lambdarank，标签按截面分成 5 档，偏重前排）| ridge（线性，需 rank_inputs）。"""
    names, X, y, ti, ji = make_panel(md, ext, H, max_year=max_year, keep=keep, rank_inputs=rank_inputs or model == "ridge")
    if label == "riskadj":                                   # 风险调整标签：收益 / 过去 20 日波动，再转截面分位
        vi = names.index("vol20")
        O = hfq(md, "open")
        fwd = np.full(len(ti), np.nan)
        ok = ti + 1 + H < md.T
        with np.errstate(invalid="ignore", divide="ignore"):
            fwd[ok] = (O[ti[ok] + 1 + H, ji[ok]] / O[ti[ok] + 1, ji[ok]] - 1) / np.maximum(X[ok, vi], 0.005)
        y = pd.Series(fwd).groupby(ti).rank(pct=True).values - 0.5
    if label == "sizeneutral":                              # 标签在当日流通市值五分组内做截面分位，剥离市值效应
        vi = names.index("log_fmv")
        O = hfq(md, "open")
        fwd = np.full(len(ti), np.nan)
        ok = ti + 1 + H < md.T
        with np.errstate(invalid="ignore", divide="ignore"):
            fwd[ok] = O[ti[ok] + 1 + H, ji[ok]] / O[ti[ok] + 1, ji[ok]] - 1
        dfl = pd.DataFrame({"t": ti, "s": X[:, vi], "f": fwd})
        dfl["q"] = dfl.groupby("t")["s"].rank(pct=True).mul(5).clip(upper=4.999).fillna(2).astype(int)
        y = dfl.groupby(["t", "q"])["f"].rank(pct=True).values - 0.5
        del dfl
    if label == "pathadj":                                  # 路径调整标签：持有期收益 − 0.5 × 持有期内最大不利偏移（买入后最低价相对买入价的跌幅）
        O, Lw = hfq(md, "open"), hfq(md, "low")
        fmin = pd.DataFrame(Lw[::-1]).rolling(H, min_periods=1).min().values[::-1]   # fmin[s] = min(L[s..s+H-1])
        fwd = np.full(len(ti), np.nan)
        ok = ti + 1 + H < md.T
        with np.errstate(invalid="ignore", divide="ignore"):
            buy = O[ti[ok] + 1, ji[ok]]
            ret = O[ti[ok] + 1 + H, ji[ok]] / buy - 1
            mae = np.minimum(fmin[ti[ok] + 1, ji[ok]] / buy - 1, 0)
            fwd[ok] = ret + 0.5 * mae
        y = pd.Series(fwd).groupby(ti).rank(pct=True).values - 0.5
        del fmin, Lw
    if label in ("on1", "cc1"):                              # 超短线标签：on1 = t 日收盘→t+1 开盘（隔夜）；cc1 = t 收盘→t+1 收盘
        Cc = hfq(md, "close")
        nxt = hfq(md, "open") if label == "on1" else Cc
        fwd = np.full(len(ti), np.nan)
        ok = ti + 2 < md.T
        with np.errstate(invalid="ignore", divide="ignore"):
            fwd[ok] = nxt[ti[ok] + 1, ji[ok]] / Cc[ti[ok], ji[ok]] - 1
        y = pd.Series(fwd).groupby(ti).rank(pct=True).values - 0.5
        del Cc, nxt
    if label == "top10":                                # 二分类：是否进入当日截面前 10%
        y = np.where(np.isfinite(y), (y >= 0.4).astype(float), np.nan)
    if universe == "large":                                  # 只在当日流通市值前 50% 的股票中训练和选股
        vi = names.index("log_fmv")
        big = pd.Series(X[:, vi]).groupby(ti).rank(pct=True).values >= 0.5
        y = np.where(big, y, np.nan)
    else:
        big = None
    T, N = md.T, md.N
    if free_md:                                              # 仅供单独打分进程使用：面板建好后释放行情数组，省内存
        md.a.clear()
        md.__dict__.pop("_peer_cache", None)
        gc.collect()
    score = np.full((T, N), np.nan)
    years = md.dates.year.values
    last = min(last_year, max_year or last_year, int(years.max()))
    for Y in range(first_year, last + 1):
        tY = int(np.searchsorted(years, Y))                 # Y 年第一个交易日
        if tY >= T:
            break
        tr = (ti + 1 + H < tY) & np.isfinite(y) & (ti % stride == 0)
        if window_years:                                      # 滚动窗口：只用最近 window_years 年的样本
            tr &= ti >= tY - int(242 * window_years)
        pr = years[ti] == Y
        if tr.sum() < 10000 or not pr.any():
            continue
        pred = np.zeros(int(pr.sum()))
        if model == "ridge":
            from sklearn.linear_model import Ridge
            m = Ridge(alpha=1000.0).fit(X[tr] - 0.5, y[tr])
            pred = m.predict(X[pr] - 0.5)
        elif model == "tree":
            from sklearn.tree import DecisionTreeRegressor
            Xt = np.nan_to_num(X[tr], nan=0.5)
            m = DecisionTreeRegressor(max_depth=4, min_samples_leaf=20000, random_state=0).fit(Xt, y[tr])
            pred = m.predict(np.nan_to_num(X[pr], nan=0.5))
        elif model == "small":
            for sd in seeds:
                m = lgb.train(dict(PARAMS, seed=sd, num_leaves=7, min_data_in_leaf=5000),
                              lgb.Dataset(X[tr], y[tr], feature_name=names), 100)
                pred += m.predict(X[pr], num_threads=2) / len(seeds)
                del m
        elif model == "rank":
            idx = np.where(tr)[0]
            idx = idx[np.argsort(ti[idx], kind="stable")]
            grp = np.diff(np.r_[0, np.flatnonzero(np.diff(ti[idx])) + 1, len(idx)])
            rel = np.clip(np.floor((y[idx] + 0.5) * 5), 0, 4).astype(int)
            for sd in seeds:
                m = lgb.train(dict(PARAMS, objective="lambdarank", seed=sd, lambdarank_truncation_level=50,
                                   label_gain=[0, 1, 3, 7, 15], min_data_in_leaf=2000),
                              lgb.Dataset(X[idx], rel, group=grp, feature_name=names), N_ROUNDS)
                pred += m.predict(X[pr], num_threads=2) / len(seeds)
                del m
        else:
            prm = dict(PARAMS, objective="binary") if label == "top10" else PARAMS
            for sd in seeds:                                   # 多个随机种子取平均（集成）
                m = lgb.train(dict(prm, seed=sd), lgb.Dataset(X[tr], y[tr], feature_name=names, free_raw_data=True),
                              N_ROUNDS)
                pred += m.predict(X[pr], num_threads=2) / len(seeds)
                del m
        if big is not None:
            pred = np.where(big[pr], pred, np.nan)
        score[ti[pr], ji[pr]] = pred
        gc.collect()
    return score


class RegimeStrategy(SignalStrategy):
    """仓位控制：中证全指收盘低于 60 日均线、且 20 日波动高于过去 250 日中位数时，目标持仓降为 n_low。
    只用 t 日及以前的指数数据。"""

    def __init__(self, score, n_full=10, n_low=5):
        super().__init__(score)
        self.n_full, self.n_low = n_full, n_low

    def target_n(self, t, v):
        c = v.index_close["zzqz"].values
        if len(c) < 270:
            return self.n_full
        r = np.diff(np.log(c[-271:]))
        vol20 = pd.Series(r).rolling(20).std().values
        risk_off = c[-1] < c[-60:].mean() and vol20[-1] > np.nanmedian(vol20[-250:])
        return self.n_low if risk_off else self.n_full


EVENT_ONLY_KEEP = {"d_yjyg_pos", "yjyg_strength", "d_yjyg_neg", "d_buyback", "d_zengchi", "gdhs_chg", "lhb_inst20",
                   "dzjy_net20", "log_fmv", "log_price", "vol20", "turn20", "ret5", "ret20", "amihud20", "log_listed"}


def unlock_ext(md, df: pd.DataFrame) -> pd.DataFrame:
    """解禁表加 avail_date = 解禁日前 3 个交易日（实际解禁数量的提示性公告一般在此之前发布）。"""
    df = df.copy()
    ti = md.dates.searchsorted(pd.to_datetime(df["unlock_date"]).values)
    ok = (ti >= 3) & (ti < md.T)
    df = df[ok].copy()
    df["avail_date"] = md.dates[ti[ok] - 3]
    return df


class DiversifiedRegimeStrategy(RegimeStrategy):
    """在 H42 的排序上做分散：按得分依次挑选，若某只股票的 PIT 相关性同伴（20 只）中已有 2 只被选中，就跳过。"""

    def __init__(self, score, md, max_peers=2):
        super().__init__(score)
        from signals import peer_sets
        self.batches, self.bot = peer_sets(md)
        self.max_peers = max_peers

    def select(self, t, v):
        order = super().select(t, v)
        b = self.bot[t]
        if b < 0 or not len(order):
            return order
        P = self.batches[b]
        chosen, out = [], []
        cs = set()
        for c in order:
            peers = set(P[c][P[c] >= 0].tolist())
            if len(peers & cs) >= self.max_peers:
                continue
            out.append(c)
            cs.add(c)
            if len(out) >= self.top:
                break
        return np.array(out, dtype=int)


class StopLossRegimeStrategy(RegimeStrategy):
    """H42 + 止损：持仓收盘价较买入价下跌 ≥ stop 时，次日开盘卖出（只用 t 日收盘价）。"""

    def __init__(self, score, stop=0.10):
        super().__init__(score)
        self.stop = stop

    def should_exit(self, t, v, p):
        c = v.close[-1][p.c] * v.adj_factor[-1][p.c]
        adj0 = p.adj_entry if p.adj_entry > 0 else v.adj_factor[-1][p.c]
        return bool(np.isfinite(c) and c / (p.entry_px * adj0) - 1 <= -self.stop)


def factory(H=10, use_events=False, max_year=None, last_year=2020, keep=None, seeds=(7,), regime=False,
            model="lgbm", rank_inputs=False, window_years=None, label="rank", universe="all", construct=None):
    """返回 factory(md, ext)。use_events=False 时忽略外部数据（只用行情特征）。
    训练集运行时 last_year=2020：不训练、不预测验证集年份，验证集标签在验证集运行之前完全不碰。"""
    def f(md, ext):
        sc = walk_forward_scores(md, ext if use_events else None, H=H, max_year=max_year, last_year=last_year,
                                 keep=keep, seeds=seeds, model=model, rank_inputs=rank_inputs,
                                 window_years=window_years, label=label, universe=universe)
        if construct == "div":
            return DiversifiedRegimeStrategy(sc, md)
        if construct == "stop":
            return StopLossRegimeStrategy(sc)
        return RegimeStrategy(sc) if regime else SignalStrategy(sc)
    return f


def ensemble_factory(H=20, use_events=True, max_year=None, last_year=2020, **kw):
    """H34：H27（行情 + 事件）与 H31（再加基本面）两个打分各自转截面百分位后取平均。
    ext 需要包含财报（fin）；H27 部分会忽略 fin。"""
    def f(md, ext):
        e27 = {k: v for k, v in ext.items() if k != "fin"}
        s27 = walk_forward_scores(md, e27, H=H, max_year=max_year, last_year=last_year)
        s31 = walk_forward_scores(md, ext, H=H, max_year=max_year, last_year=last_year)
        r = (pd.DataFrame(s27).rank(axis=1, pct=True).values + pd.DataFrame(s31).rank(axis=1, pct=True).values) / 2
        return SignalStrategy(r)
    return f


def _labels(md, ti, ji, H):
    O = hfq(md, "open")
    fwd = np.full(len(ti), np.nan)
    ok = ti + 1 + H < md.T
    with np.errstate(invalid="ignore", divide="ignore"):
        fwd[ok] = O[ti[ok] + 1 + H, ji[ok]] / O[ti[ok] + 1, ji[ok]] - 1
    return pd.Series(fwd).groupby(ti).rank(pct=True).values - 0.5


def multi_scores(md, ext, configs, first_year=2016, last_year=2023, max_year=None, stride=2):
    """只构建一次特征面板，按多组 (特征子集, 标签周期) 分别滚动训练，返回各自的打分（省内存）。"""
    names, X, _, ti, ji = make_panel(md, ext, 20, max_year=max_year)
    years = md.dates.year.values
    last = min(last_year, max_year or last_year, int(years.max()))
    out = []
    for cols, H in configs:
        idx = [i for i, c in enumerate(names) if cols is None or (cols(c) if callable(cols) else c in cols)]
        sub_names = [names[i] for i in idx]
        y = _labels(md, ti, ji, H)
        score = np.full((md.T, md.N), np.nan, dtype=np.float32)
        for Y in range(first_year, last + 1):
            tY = int(np.searchsorted(years, Y))
            if tY >= md.T:
                break
            tr = (ti + 1 + H < tY) & np.isfinite(y) & (ti % stride == 0)
            pr = years[ti] == Y
            if tr.sum() < 10000 or not pr.any():
                continue
            tri, pri = np.where(tr)[0], np.where(pr)[0]
            mdl = lgb.train(PARAMS, lgb.Dataset(X[np.ix_(tri, idx)], y[tri], feature_name=sub_names), N_ROUNDS)
            score[ti[pri], ji[pri]] = mdl.predict(X[np.ix_(pri, idx)], num_threads=2)
            del mdl
            gc.collect()
        out.append(score)
        del y
    del X
    gc.collect()
    return out


FIN_FEATURES = {"ep_ttm", "bp", "roe_ann", "gm", "gm_chg", "cfo_ratio", "ni_yoy", "rev_yoy", "fc_surprise",
                "ep_self_pct", "ep_peer_rel", "days_since_report"}


def multi_factory(max_year=None, last_year=2020, regime=True, **kw):
    """H37：三个子模型的截面百分位平均 —— H27 口径（20 日标签，不含财报特征）、H31 口径（20 日，含财报）、
    H32 口径（40 日，含财报）。特征面板只构建一次。"""
    def f(md, ext):
        drop = lambda n: n not in FIN_FEATURES                      # noqa: E731
        parts = multi_scores(md, ext, [(drop, 20), (None, 20), (None, 40)], max_year=max_year, last_year=last_year)
        r = np.nanmean(np.stack([pd.DataFrame(p).rank(axis=1, pct=True).values.astype(np.float32) for p in parts]),
                       axis=0)
        return RegimeStrategy(r) if regime else SignalStrategy(r)
    return f


def build_strategy(sc, sp, md):
    """由打分矩阵和登记的规格构建策略对象（缓存打分复用时使用，与 factory 的构建方式一致）。"""
    c = sp.get("construct")
    if c == "div":
        return DiversifiedRegimeStrategy(sc, md)
    if c == "stop":
        return StopLossRegimeStrategy(sc)
    return RegimeStrategy(sc) if sp.get("regime") else SignalStrategy(sc)
