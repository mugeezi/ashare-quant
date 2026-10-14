import sys
import numpy as np
from framework import CACHE, Lab
from screen import Screener
lab = Lab(); sc = Screener(lab)
for v in sys.argv[1:]:
    s = np.load(CACHE.parent / f"score_H27_{v}_train.npy").astype(np.float64)
    sc.run("H27", f"seed_{v}", s, H=10, start="2016-01-01", notes="种子敏感性（训练集）")
