import numpy as np, pandas as pd
from families import _xrank
from framework import CACHE, Lab, eligible_matrix
from screen import Screener
P = CACHE.parent
lab = Lab(); md = lab.md; sc = Screener(lab); el = eligible_matrix(md)
c = md.index_close["zzqz"].ffill().values
r = np.r_[np.nan, np.diff(np.log(c))]
vol20 = pd.Series(r).rolling(20).std(); med = vol20.rolling(250, min_periods=200).median(); ma60 = pd.Series(c).rolling(60).mean()
off = ((c < ma60) & (vol20 > med)).values
e5, e3 = np.where(off, 0.5, 1.0), np.where(off, 0.3, 1.0)
L = lambda n: np.load(P / n).astype(np.float64)
h27, h31, h32, s04 = L("score_H27_base_train.npy"), L("score_H31_base_train.npy"), L("score_H32_base_train.npy"), L("score_S04.npy")
nohype = np.isfinite(s04)
sc.run("H35", "base", np.where(nohype, h31, np.nan), H=10, start="2016-01-01", exposure=e5, notes="H31 非炒作股 + 仓位控制")
sc.run("H36", "base", np.nanmean(np.stack([_xrank(h31), _xrank(s04)]), axis=0), H=10, start="2016-01-01", exposure=e5, notes="H31+S04 百分位平均 + 仓位控制")
sc.run("H37", "base", np.nanmean(np.stack([_xrank(h27), _xrank(h31), _xrank(h32)]), axis=0), H=10, start="2016-01-01", exposure=e5, notes="多周期集成 + 仓位控制")
sc.run("H38", "base", h31, H=10, start="2016-01-01", exposure=e3, notes="H31 + 弱势3只")
