import numpy as np, gc
import families as F
from framework import CACHE, Lab, eligible_matrix
from run_batch3 import load_ext
from screen import Screener
lab = Lab(); md = lab.md; el = eligible_matrix(md)
ext = load_ext(md, plus="fin")
s4 = F.s04_filter_then_rank(md, ext, el); del ext; gc.collect()
sc = Screener(lab)
sc.run("S04", "base", s4, H=10, notes="先排除炒作，再按基本面改善排序")
for h in ("S05", "S06"):
    s = np.load(CACHE.parent / f"score_{h}_base_train.npy").astype(np.float64)
    sc.run(h, "base", s, H=10, start="2016-01-01", notes={"S05": "深度4决策树", "S06": "小型提升树"}[h])
