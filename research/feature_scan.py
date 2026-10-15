"""训练集（标签在 2020 年底前结束）上的特征扫描：每个特征的日度截面秩相关（与 20 日后收益）均值、ICIR，
以及 LightGBM（2013-10 ~ 2019 训练）的增益重要性。每个特征的 IC 计 1 次探索性试验。"""
import json
import lightgbm as lgb
import numpy as np
import pandas as pd

import ml
from framework import Lab
from run_batch3 import load_ext

lab = Lab(); md = lab.md
ext = load_ext(md, plus="fin")
names, X, y, ti, ji = ml.make_panel(md, ext, 20, max_year=2020)
H = 20
t_end = int(md.dates.searchsorted(pd.Timestamp("2021-01-01")))
ok = np.isfinite(y) & (ti + 1 + H < t_end)
d = pd.DataFrame(X[ok], columns=names); d["y"] = y[ok]; d["t"] = ti[ok]
d = d[d.t % 5 == 0]                                     # 每 5 天取一个截面，降低重叠
rows = []
for c in names:
    ic = d.groupby("t").apply(lambda g: g[c].rank().corr(g["y"].rank()) if g[c].notna().sum() > 50 else np.nan).dropna()
    rows.append((c, ic.mean(), ic.mean() / ic.std() if ic.std() > 0 else np.nan, len(ic)))
ic = pd.DataFrame(rows, columns=["feature", "ic", "icir", "n"]).set_index("feature")
tY = int(md.dates.searchsorted(pd.Timestamp("2020-01-01")))
tr = np.isfinite(y) & (ti + 1 + H < tY) & (ti % 2 == 0)
m = lgb.train(ml.PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=names), ml.N_ROUNDS)
imp = pd.Series(m.feature_importance("gain"), index=names)
ic["gain_share"] = (imp / imp.sum()).reindex(ic.index)
ic = ic.sort_values("gain_share", ascending=False)
ic.to_csv("../reports/feature_scan_train.csv")
print(ic.round(4).to_string())
for f, r in ic.iterrows():
    lab.log_trial("S00", f"ic:{f}", "train", {"label_H": 20}, f"特征扫描：训练集秩 IC {r.ic:.4f}，ICIR {r.icir:.3f}")
