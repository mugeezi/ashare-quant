import sys
import numpy as np
import families as F
from families import _xrank
from framework import CACHE, Lab
from run_batch3 import load_ext
from screen import Screener
import pandas as pd
lab = Lab(); md = lab.md; sc = Screener(lab)
ext = {"yysj": pd.read_parquet(lab.md and (CACHE.parents[0]) / "x" if False else "../data/events/yysj.parquet")}
sc.run("G07", "h10", F.g07_early_schedule(md, ext), H=10, notes="早预约")
h27 = np.load(CACHE.parent / "score_H27_base_train.npy").astype(np.float64)
h31 = np.load(CACHE.parent / "score_H31_base_train.npy").astype(np.float64)
sc.run("H34", "base", np.nanmean(np.stack([_xrank(h27), _xrank(h31)]), axis=0), H=10, start="2016-01-01", notes="H27+H31 集成")
