import numpy as np, pandas as pd
import families as F
from framework import Lab, eligible_matrix
from run_batch3 import load_ext
from screen import Screener
lab = Lab(); md = lab.md; el = eligible_matrix(md)
ext = load_ext(md, plus="fin")
s1, s3 = F.scorecard(md, ext, el)
del ext
import gc; gc.collect()
sc = Screener(lab)
c = md.index_close["zzqz"].ffill().values
r = np.r_[np.nan, np.diff(np.log(c))]
vol20 = pd.Series(r).rolling(20).std(); med = vol20.rolling(250, min_periods=200).median(); ma60 = pd.Series(c).rolling(60).mean()
expo = np.where(((c < ma60) & (vol20 > med)).values, 0.5, 1.0)
np.save("/tmp/claude-0/-home-claude/47e34d9e-ea3d-5e5a-bd04-7c13816a5033/scratchpad/score_S01.npy", s1.astype(np.float32))
sc.run("S01", "base", s1, H=10, notes="反炒作+便宜+改善 打分卡（等权 12 信号）")
sc.run("S02", "base", s1, H=10, notes="S01 + 仓位控制", exposure=expo)
sc.run("S03", "base", s3, H=10, notes="三支柱等权")
sc.run("S01", "since2016", s1, H=10, start="2016-01-01", notes="S01 在 2016–2020（与模型同期对照）")
