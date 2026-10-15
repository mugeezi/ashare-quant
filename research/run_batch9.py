"""第九批：经验规则型策略，快速筛选（满仓 / 仓位控制两个变体）。"""
import numpy as np
import pandas as pd

import families as F
from framework import CACHE, Lab, eligible_matrix
from screen import Screener

lab = Lab(); md = lab.md; sc = Screener(lab); el = eligible_matrix(md)
c = md.index_close["zzqz"].ffill().values
r = np.r_[np.nan, np.diff(np.log(c))]
vol20 = pd.Series(r).rolling(20).std()
med = vol20.rolling(250, min_periods=200).median()
ma60 = pd.Series(c).rolling(60).mean()
risk_off = ((c < ma60) & (vol20 > med)).values
expo = np.where(risk_off, 0.5, 1.0)

RULES = {"T01": ("缩量回踩20日线", lambda: F.t01_trend_pullback(md)),
         "T02": ("平台突破", lambda: F.t02_box_breakout(md)),
         "T03": ("底部反转", lambda: F.t03_bottom_reversal(md)),
         "T04": ("强势股短暂回调", lambda: F.t04_strong_pullback(md, el)),
         "T05": ("稳健上升趋势", lambda: F.t05_smooth_trend(md)),
         "T06": ("周线向上日线回调", lambda: F.t06_weekly_up_daily_dip(md)),
         "T08": ("市场状态切换", lambda: F.t08_regime_switch(md, el))}
for k, (title, fn) in RULES.items():
    s = fn()
    sc.run(k, "full", s, H=10, notes=title)
    sc.run(k, "regime", s, H=10, notes=title + " + 仓位控制", exposure=expo)
h31 = np.load(CACHE.parent / "score_H31_base_train.npy").astype(np.float64)
C = F.hfq(md, "close")
above = C > F._ma(C, 20)
s7 = np.where(above, h31, np.nan)
sc.run("T07", "full", s7, H=10, start="2016-01-01", notes="H31 + 收盘在20日线上方")
sc.run("T07", "regime", s7, H=10, start="2016-01-01", notes="H31 + 20日线上方 + 仓位控制", exposure=expo)
sc.run("H33", "screen", h31, H=10, start="2016-01-01", notes="对照：H31 打分 + 仓位控制（即 H33 的筛选器版本）", exposure=expo)
