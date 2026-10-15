"""筛选器评估一个缓存打分（训练集 2016–2020），按规格决定是否加仓位控制。输出 JSON 一行。"""
import json, sys
import numpy as np, pandas as pd
from framework import CACHE, Lab
from run_batch3 import SPECS
from screen import Screener
hyp, var = sys.argv[1], sys.argv[2]
sp = SPECS[(hyp, var)]
lab = Lab(); md = lab.md; sc = Screener(lab)
c = md.index_close["zzqz"].ffill().values
r = np.r_[np.nan, np.diff(np.log(c))]
vol20 = pd.Series(r).rolling(20).std(); med = vol20.rolling(250, min_periods=200).median(); ma60 = pd.Series(c).rolling(60).mean()
e5 = np.where(((c < ma60) & (vol20 > med)).values, 0.5, 1.0) if sp.get("regime") else None
s = np.load(CACHE.parent / f"score_{hyp}_{var}_train.npy").astype(np.float64)
st = sc.run(hyp, var, s, H=10, start="2016-01-01", exposure=e5, notes="自动流程筛选")
print("RESULT " + json.dumps({"excess": st["excess_cagr"], "dd": st["max_drawdown"], "years": st["years_pos"]}))
