import numpy as np, pandas as pd
import families as F
from framework import Lab, eligible_matrix
from screen import Screener
lab = Lab(); md = lab.md; sc = Screener(lab); el = eligible_matrix(md)
c = md.index_close["zzqz"].ffill().values
r = np.r_[np.nan, np.diff(np.log(c))]
vol20 = pd.Series(r).rolling(20).std(); med = vol20.rolling(250, min_periods=200).median(); ma60 = pd.Series(c).rolling(60).mean()
expo = np.where(((c < ma60) & (vol20 > med)).values, 0.5, 1.0)
for k, title, fn in [("R01", "超跌缩量", lambda: F.r01_oversold_dryup(md, el)), ("R02", "长期向上短期超跌", lambda: F.r02_long_up_short_down(md)),
                     ("R03", "连阴缩量", lambda: F.r03_down_streak_dryup(md)), ("R04", "恐慌日抄底", lambda: F.r04_panic_buy(md, el)),
                     ("R05", "地量地价", lambda: F.r05_floor_volume(md, el))]:
    s = fn()
    sc.run(k, "full", s, H=10, notes=title)
    sc.run(k, "regime", s, H=10, notes=title + " + 仓位控制", exposure=expo)
