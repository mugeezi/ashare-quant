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


def make_panel(md, ext, H, max_year=None, keep=None):
    el = eligible_matrix(md)
    t0 = int(md.dates.searchsorted(pd.Timestamp(WARM)))
    el[:t0] = False
    if max_year is not None:                                 # 只保留到 max_year 年末（前视检测用的轻量版）
        el[int(md.dates.searchsorted(pd.Timestamp(f"{max_year + 1}-01-01"))):] = False
    ti, ji = np.where(el)
    names, cols = [], []
    for name, arr in FE.build(md, ext):
        if keep is not None and name not in keep:
            del arr
            continue
        names.append(name)
        cols.append(arr[ti, ji].astype(np.float32))
        del arr
    X = np.column_stack(cols)
    del cols
    gc.collect()
    O = hfq(md, "open")
    fwd = np.full(len(ti), np.nan)
    ok = ti + 1 + H < md.T
    a, b = O[ti[ok] + 1, ji[ok]], O[ti[ok] + 1 + H, ji[ok]]
    with np.errstate(invalid="ignore", divide="ignore"):
        fwd[ok] = b / a - 1
    y = pd.Series(fwd).groupby(ti).rank(pct=True).values - 0.5
    return names, X, y, ti, ji


def walk_forward_scores(md, ext=None, H=10, first_year=2016, last_year=2023, stride=2, max_year=None, keep=None,
                        seeds=(7,)):
    names, X, y, ti, ji = make_panel(md, ext, H, max_year=max_year, keep=keep)
    score = np.full((md.T, md.N), np.nan)
    years = md.dates.year.values
    last = min(last_year, max_year or last_year, int(years.max()))
    for Y in range(first_year, last + 1):
        tY = int(np.searchsorted(years, Y))                 # Y 年第一个交易日
        if tY >= md.T:
            break
        tr = (ti + 1 + H < tY) & np.isfinite(y) & (ti % stride == 0)
        pr = years[ti] == Y
        if tr.sum() < 10000 or not pr.any():
            continue
        pred = np.zeros(int(pr.sum()))
        for sd in seeds:                                   # 多个随机种子取平均（集成）
            m = lgb.train(dict(PARAMS, seed=sd), lgb.Dataset(X[tr], y[tr], feature_name=names, free_raw_data=True),
                          N_ROUNDS)
            pred += m.predict(X[pr], num_threads=2) / len(seeds)
            del m
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


def factory(H=10, use_events=False, max_year=None, last_year=2020, keep=None, seeds=(7,), regime=False):
    """返回 factory(md, ext)。use_events=False 时忽略外部数据（只用行情特征）。
    训练集运行时 last_year=2020：不训练、不预测验证集年份，验证集标签在验证集运行之前完全不碰。"""
    def f(md, ext):
        sc = walk_forward_scores(md, ext if use_events else None, H=H, max_year=max_year, last_year=last_year,
                                 keep=keep, seeds=seeds)
        return RegimeStrategy(sc) if regime else SignalStrategy(sc)
    return f
