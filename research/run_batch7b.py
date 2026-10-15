import numpy as np
from framework import CACHE, Lab
from screen import Screener
lab = Lab(); sc = Screener(lab)
for h, note in [("H31", "H27 + 基本面特征"), ("H32", "H31 但 40 日标签")]:
    s = np.load(CACHE.parent / f"score_{h}_base_train.npy").astype(np.float64)
    sc.run(h, "base", s, H=10, start="2016-01-01", notes=note)
