"""第五批：用快速筛选器在训练集上大范围试错。用法：python run_batch5.py [族名 ...]"""
import sys
import time

import families as F
import signals as S
from framework import Lab
from run_batch3 import load_ext
from screen import Screener

FAM = {
    "F01": ("复牌补涨", lambda md, e: F.f01_resume_catchup(md)),
    "F02": ("次新股换手衰减", lambda md, e: F.f02_subnew_settle(md)),
    "F03": ("送转填权", lambda md, e: F.f03_exright_fill(md)),
    "F04": ("下跌日抗跌", lambda md, e: F.f04_down_day_resilience(md)),
    "F05": ("收盘强度持续", lambda md, e: F.f05_close_strength(md)),
    "F06": ("波动收敛近前高", lambda md, e: F.f06_squeeze(md)),
    "F07": ("缺口守住", lambda md, e: F.f07_gap_hold(md)),
    "F08": ("解禁后错杀", lambda md, e: F.f08_unlock_oversold(md, e)),
    "F09": ("利空出尽", lambda md, e: F.f09_bad_news_exhaust(md, e)),
    "F10": ("筹码集中滞涨", lambda md, e: F.f10_concentration_lag(md, e)),
    "F11": ("同伴群体动量", lambda md, e: F.f11_cluster_momentum(md)),
    "F12": ("资金流向", lambda md, e: F.f12_money_flow(md)),
    "F13": ("下跌中买盘占优", lambda md, e: F.f13_up_volume_share(md)),
    "F15": ("涨停后缩量回踩", lambda md, e: F.f15_limit_pullback(md)),
    "F16": ("回购护盘线", lambda md, e: F.f16_buyback_floor(md, e)),
    "F17": ("预增+筹码集中", lambda md, e: F.f17_dual_confirm(md, e)),
}


def main(names):
    lab = Lab()
    md = lab.md
    ext = load_ext(md, plus=True)
    sc = Screener(lab)
    if not names:
        names = list(FAM)
    if "CAL" in names:   # 筛选器校准：与引擎回测过的 H13 对照（引擎训练集超额 −1.9%）
        sc.run("H13", "calib", S.h13(md, ext).score, H=10, notes="筛选器校准，对照引擎 E0012")
        names = [n for n in names if n != "CAL"]
    for n in names:
        title, fn = FAM[n]
        t = time.time()
        s = fn(md, ext)
        print(f"-- {n} {title}（打分 {time.time() - t:.0f}s）", flush=True)
        sc.run(n, "h10", s, H=10, notes=title)


if __name__ == "__main__":
    main(sys.argv[1:])
