"""修订口径的紧缩夏普：样本外记录 2016–2023（训练期样本外 + 验证集），V[SR] = 1/T，N = 全部试验次数。
训练期净值用缓存的训练集打分重跑引擎（同一策略的复核，不是新试验）；验证集净值读 reports/valid_*_nav.csv。"""
import math
import sys

import numpy as np
import pandas as pd
from scipy import stats

import ml
from framework import CACHE, ROOT, Backtest, Config, Lab, read_exp

G = 0.5772156649015329
CANDS = {  # 名称: (训练集打分文件, 是否仓位控制, 验证集净值文件)
    "H27": ("score_H27_base_train.npy", False, "valid_H27_base_nav.csv"),
    "H31": ("score_H31_base_train.npy", False, "valid_H31_base_nav.csv"),
    "H33": ("score_H31_base_train.npy", True, "valid_H33_base_nav.csv"),
    "H40": ("score_H40_base_train.npy", True, "valid_H40_base_nav.csv"),
    "H42": ("score_H42_base_train.npy", True, "valid_H42_base_nav.csv"),
}


def dsr(e, N):
    T = len(e)
    sr = e.mean() / e.std()
    sk, ku = stats.skew(e), stats.kurtosis(e, fisher=False)
    sr0 = math.sqrt(1 / T) * ((1 - G) * stats.norm.ppf(1 - 1 / N) + G * stats.norm.ppf(1 - 1 / (N * math.e)))
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    return float(stats.norm.cdf(z)), sr, sr0, T


def excess(nav, bench):
    r = nav.pct_change().dropna()
    b = bench.reindex(nav.index).ffill().pct_change().reindex(r.index).fillna(0)
    return r - b


def _auto(nm):
    """自动流程的新候选：训练集打分 score_{nm}_base_train.npy，验证集净值 valid_{nm}_base_nav.csv。"""
    from run_batch3 import SPECS
    return (f"score_{nm}_base_train.npy", SPECS[(nm, "base")], f"valid_{nm}_base_nav.csv")


def main(names):
    lab = Lab()
    md = lab.md
    N = len(read_exp())
    rows = []
    for nm in names:
        if nm in CANDS:
            f, reg, vf = CANDS[nm]
            sc = np.load(CACHE.parent / f).astype(np.float64)
            strat = ml.RegimeStrategy(sc) if reg else ml.SignalStrategy(sc)
        else:
            f, sp, vf = _auto(nm)
            sc = np.load(CACHE.parent / f).astype(np.float64)
            strat = ml.build_strategy(sc, sp, md)
        res = Backtest(md, strat, Config(start="2016-01-01", end="2020-12-31", n_target=10, renew=True, hold_days=10)).run()
        e_tr = excess(res.nav.nav, lab.bench)
        vn = pd.read_csv(ROOT / "reports" / vf, index_col=0, parse_dates=True).nav
        e_va = excess(vn, lab.bench)
        e = pd.concat([e_tr, e_va])
        p, sr, sr0, T = dsr(e, N)
        pv, srv, sr0v, Tv = dsr(e_va, N)
        rows.append(dict(策略=nm, 样本外日数=T, 年化IR=round(sr * math.sqrt(242), 2), SR0年化=round(sr0 * math.sqrt(242), 2),
                         DSR_2016_2023=round(p, 3), 验证集IR=round(srv * math.sqrt(242), 2), DSR_仅验证集=round(pv, 3)))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reports" / "dsr_revised.csv", index=False)
    print(f"N = {N}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or list(CANDS))
