"""第一批假设：训练集运行。用法：python run_batch1.py H01:b5 H02:b5 ..."""
import json
import sys
import time

import pandas as pd

import signals as S
from framework import ROOT, Lab

EV = ROOT / "data" / "events"
EXT = {"H06": lambda: {"lhb": pd.read_parquet(EV / "lhb_jg.parquet")},
       "H07": lambda: {"buyback": pd.read_parquet(EV / "buyback.parquet")},
       "H10": lambda: {"dzjy": pd.read_parquet(EV / "dzjy.parquet")}}

CFG = dict(n_target=10, renew=True)
SPECS = {
    "H01": (S.h01, {"b5": 5, "b10": 10}),
    "H02": (S.h02, {"b5": 5, "b10": 10}),
    "H03": (S.h03, {"b5": 5, "b10": 10}),
    "H04": (S.h04, {"b5": 5, "b3": 3}),
    "H08": (S.h08, {"b5": 5, "b10": 10}),
    "H09": (S.h09, {"b5": 5, "b3": 3}),
    "H06": (S.h06, {"b5": 5, "b10": 10}),
    "H07": (S.h07, {"b10": 10, "b5": 5}),
    "H10": (S.h10, {"b5": 5, "b10": 10}),
}


def main(items):
    lab = Lab()
    for it in items:
        hyp, var = it.split(":")
        fac, vars_ = SPECS[hyp]
        t0 = time.time()
        ext = EXT[hyp]() if hyp in EXT else None
        rr = lab.run(hyp, var, fac, split="train", ext=ext, hold_days=vars_[var], **CFG)
        r, st = rr.row, rr.stats
        print(f"{hyp}/{var}: 年化 {r['cagr']:+.1%}  超额 {r['excess_cagr']:+.1%}  回撤 {r['max_drawdown']:.1%}  "
              f"正超额年份 {r['years_excess_pos']}  SMB β {r['smb_beta']:+.2f}  α 占比 {r['alpha_share']}  "
              f"换手 {r['turnover_annual']}  买入 {r['n_buys']}  达标 {r['passed']}  ({time.time() - t0:.0f}s)", flush=True)
        print("   分年超额:", json.dumps({k: round(v, 3) for k, v in st["yearly_excess"].items()}), flush=True)
        nav = rr.result.nav
        print(f"   平均持仓数 {nav.n_pos.mean():.1f}，续持 {rr.result.stat.get('renewals', 0)} 次", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
