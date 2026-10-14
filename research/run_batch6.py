"""第六批：模型结构变体，用筛选器在训练集 2016–2020 评估。"""
import numpy as np
import pandas as pd

from families import _xrank
from framework import CACHE, Lab
from screen import Screener

lab = Lab()
sc = Screener(lab)
S = {h: np.load(CACHE.parent / f"score_{h}_base_train.npy").astype(np.float64) for h in ["H22", "H26", "H27", "H28", "H30"]}
names = {"H26": "岭回归 + 截面排名输入", "H27": "20 日标签", "H28": "lambdarank 排序目标", "H30": "截面排名输入 LGBM"}
res = {}
for h in ["H26", "H27", "H28", "H30"]:
    res[h] = sc.run(h, "base", S[h], H=10, start="2016-01-01", notes=names[h])
ens = np.nanmean(np.stack([_xrank(S[h]) for h in ["H22", "H26", "H27", "H30"]]), axis=0)
res["H29"] = sc.run("H29", "base", ens, H=10, start="2016-01-01", notes="H22/H26/H27/H30 截面百分位平均")
print("\n2019–2020 两年超额：", {h: (r["yearly"].get(2019), r["yearly"].get(2020)) for h, r in res.items()})
