import numpy as np, pandas as pd, gc
import families as F
from framework import Lab, eligible_matrix
from run_batch3 import load_ext
from screen import Screener
lab = Lab(); md = lab.md; el = eligible_matrix(md)
ext = load_ext(md, plus="fin")
s4 = F.s04_filter_then_rank(md, ext, el); del ext; gc.collect()
np.save("/tmp/claude-0/-home-claude/47e34d9e-ea3d-5e5a-bd04-7c13816a5033/scratchpad/score_S04.npy", s4.astype(np.float32))
sc = Screener(lab)
c = md.index_close["zzqz"].ffill().values
r = np.r_[np.nan, np.diff(np.log(c))]
vol20 = pd.Series(r).rolling(20).std(); med = vol20.rolling(250, min_periods=200).median(); ma60 = pd.Series(c).rolling(60).mean()
expo = np.where(((c < ma60) & (vol20 > med)).values, 0.5, 1.0)
sc.run("S04", "regime", s4, H=10, notes="S04 + 仓位控制", exposure=expo)
