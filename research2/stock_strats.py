"""第二轮股票类策略：A01 进攻档集中版、W02 低波动高股息、W03 大盘增强（规格见 hypotheses.md 第二轮登记）。

用法：python research2/stock_strats.py A01 train|valid
      python research2/stock_strats.py W02 train|valid
      python research2/stock_strats.py W03 train|valid
"""
from __future__ import annotations

import gc
import math
import sys

import numpy as np
import pandas as pd

import r2
from r2 import Backtest, Config, F, SCORES
import ml  # noqa: E402  (round2 工作目录下的 research/ml.py，与 main 相同)
import dividends as DV

WEAK = "弱市（与第一轮相同）：中证全指收盘 < 60 日均线 且 20 日波动 > 过去 250 日中位数"


def cached(name: str, split: str) -> np.ndarray:
    return np.load(SCORES / f"score_{name}_base_{split}.npy").astype(np.float64)


def size_mask(md, top_frac: float) -> np.ndarray:
    """当日可交易股票（引擎同口径过滤）中流通市值排名前 top_frac 的布尔矩阵，只用当日数据。"""
    el = F.eligible_matrix(md)
    fmv = np.where(el & np.isfinite(md.a["fmv"]), md.a["fmv"], np.nan)
    pct = pd.DataFrame(fmv).rank(axis=1, pct=True, ascending=True).values
    return pct > 1 - top_frac


# ---------------------------------------------------------------- 工厂
def a01_factory(n: int, split: str):
    sc = cached("H73_nf", split)
    n_low = math.ceil(n / 2)
    return lambda md, ext=None: ml.RegimeStrategy(sc, n_full=n, n_low=n_low)


def w03_factory(split: str):
    sc = cached("H73_nf", split)

    def f(md, ext=None):
        s = np.where(size_mask(md, 0.20), sc[:md.T], np.nan)   # 前视检测时行情被截短
        return ml.RegimeStrategy(s, n_full=20, n_low=10)
    return f


def w02_score(md) -> np.ndarray:
    ev = DV.events(md)
    dy = DV.ttm_yield(md, ev)
    px = pd.DataFrame(md.a["close"] * md.a["adj_factor"]).ffill()
    ok = md.a["tradestatus"] == 1
    r = px.pct_change().where(ok)
    vol = r.rolling(120, min_periods=100).std().values
    uni = size_mask(md, 0.30) & np.isfinite(dy) & (dy > 0) & np.isfinite(vol)
    a = pd.DataFrame(np.where(uni, dy, np.nan)).rank(axis=1, pct=True).values
    b = pd.DataFrame(np.where(uni, -vol, np.nan)).rank(axis=1, pct=True).values
    return np.where(uni, 0.5 * a + 0.5 * b, np.nan)


def w02_factory(n: int, regime: bool):
    def f(md, ext=None):
        s = w02_score(md)
        return ml.RegimeStrategy(s, n_full=n, n_low=n // 2) if regime else ml.SignalStrategy(s)
    return f


# ---------------------------------------------------------------- 运行
SPEC = {
    # hyp: (tier, start_train, [(variant, factory_builder(split), cfg, params)])
    "A01_n3": ("进攻", "2016-01-01", [("base", lambda sp: a01_factory(3, sp), dict(n_target=3, renew=True, hold_days=10),
                                       dict(score="H73_nf缓存", n=3, n_low=2))]),
    "A01_n5": ("进攻", "2016-01-01", [("base", lambda sp: a01_factory(5, sp), dict(n_target=5, renew=True, hold_days=10),
                                       dict(score="H73_nf缓存", n=5, n_low=3))]),
    "W03": ("稳健", "2016-01-01", [("base", lambda sp: w03_factory(sp), dict(n_target=20, renew=True, hold_days=20),
                                    dict(score="H73_nf缓存", universe="流通市值前20%", n=20, n_low=10))]),
    "W02": ("稳健", "2014-01-01", [
        ("full", lambda sp: w02_factory(25, False), dict(n_target=25, renew=True, hold_days=20),
         dict(universe="流通市值前30%", n=25, dy="TTM复权因子推算", vol="120日")),
        ("weak_half", lambda sp: w02_factory(25, True), dict(n_target=25, renew=True, hold_days=20),
         dict(universe="流通市值前30%", n=25, n_low=12, dy="TTM复权因子推算", vol="120日")),
    ]),
}
LEAK_TRUNC = "2020-12-31"


def run(hyp: str, split: str, variants=None, lab=None):
    r2.check_pause()
    tier, start, vs = SPEC[hyp]
    lab = lab or F.Lab()
    md = lab.md
    out = []
    for var, fb, cfg, prm in vs:
        if variants and var not in variants:
            continue
        r2.check_pause()
        fac = fb(split)
        if split == "train":
            lk = F.leak_test(md, fac, truncate=LEAK_TRUNC, t_range=(300, int(md.dates.searchsorted(pd.Timestamp(LEAK_TRUNC))) - 30))
            gc.collect()
            if not lk["passed"]:
                raise RuntimeError(f"{hyp}/{var} 前视检测未通过：{lk['changed']}")
            leak = f"pass {lk['n_checks']}"
        else:
            leak = "见训练集"
        s0, s1 = r2.SPLITS[split]
        if split == "train":
            s0 = start
        strat = fac(md)
        res = Backtest(md, strat, Config(start=s0, end=s1, **cfg)).run()
        nav = res.nav.nav
        row, st = r2.log_run(hyp, var, split, tier, nav, res.trades, md.index_close, lab.smb, dict(prm, **cfg),
                             notes=f"{hyp} {var}", leak=leak, n_target=cfg["n_target"], hold_days=cfg["hold_days"])
        conc = r2.concentration(md, res.roundtrips, nav, lab.bench, lab.smb, fmv_pct=r2.fmv_percentile(md))
        st.update(conc=conc, passed=row["passed"], trial_no=row["trial_no"])
        out.append((var, st))
        print(hyp, var, split, {k: st[k] for k in ("cagr", "excess_cagr", "max_drawdown", "excess_hs300", "ann_vol")},
              "passed", row["passed"], flush=True)
        r2.save_json(f"{split}_{hyp}_{var}.json", st)
        del strat, res
        gc.collect()
    return out


if __name__ == "__main__":
    r2.register_pid("stock_strats")
    hyp, split = sys.argv[1], sys.argv[2]
    variants = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    run(hyp, split, variants)
