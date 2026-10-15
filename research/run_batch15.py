import numpy as np, pandas as pd
from framework import CACHE, Lab
from screen import Screener
lab = Lab(); md = lab.md; sc = Screener(lab)
c = md.index_close["zzqz"].ffill().values
r = np.r_[np.nan, np.diff(np.log(c))]
vol20 = pd.Series(r).rolling(20).std(); med = vol20.rolling(250, min_periods=200).median(); ma60 = pd.Series(c).rolling(60).mean()
e5 = np.where(((c < ma60) & (vol20 > med)).values, 0.5, 1.0)
for h, note in [("H44", "H42+监管事件")]:
    s = np.load(CACHE.parent / f"score_{h}_base_train.npy").astype(np.float64)
    sc.run(h, "base", s, H=10, start="2016-01-01", exposure=e5, notes=note)
