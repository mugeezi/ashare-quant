"""第七批筛选。用法：python run_batch7.py G05 | fund"""
import sys
import time

import families as F
from framework import Lab
from run_batch3 import load_ext
from screen import Screener

lab = Lab()
md = lab.md
sc = Screener(lab)
if "G05" in sys.argv:
    t = time.time()
    s = F.g05_turnover_steady(md)
    print("G05 打分", round(time.time() - t), "s", flush=True)
    sc.run("G05", "h10", s, H=10, notes="换手平稳度")
if "fund" in sys.argv:
    ext = load_ext(md, plus="fin")
    t = time.time()
    fd = F._fund(md, ext)
    print("基本面特征", round(time.time() - t), "s", flush=True)
    for name, fn, note in [("G01", F.g01_beat_own_guidance, "业绩超预告"), ("G02", F.g02_self_cheap_accel, "自身历史便宜+业绩加速"),
                           ("G03", F.g03_peer_cheap_quality, "同伴相对低估+质量"), ("G04", F.g04_margin_unpriced, "毛利率改善未定价"),
                           ("G06", F.g06_cash_backed_growth, "现金含量高的增长")]:
        sc.run(name, "h10", fn(md, fd), H=10, notes=note)
