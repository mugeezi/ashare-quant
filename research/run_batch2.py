"""第二批假设：训练集运行。用法：python run_batch2.py H11:base H11:q ..."""
import json
import sys
import time
from functools import partial

import pandas as pd

import signals as S
from framework import ROOT, Lab

EV = ROOT / "data" / "events"
LOAD = {"buyback": "buyback.parquet", "yjyg": "yjyg.parquet", "zengchi": "zengchi.parquet", "lhb": "lhb_jg.parquet",
        "dzjy": "dzjy.parquet", "jgdy": "jgdy.parquet", "gdhs": "gdhs.parquet"}
CFG = dict(n_target=10, renew=True, hold_days=10)
SPECS = {
    "H11": (["buyback"], {"base": partial(S.h11), "q": partial(S.h11, quiet=True)}),
    "H12": (["yjyg"], {"base": partial(S.h12), "q": partial(S.h12, quiet=True)}),
    "H13": (["yjyg"], {"base": partial(S.h13)}),
    "H14": (["zengchi"], {"base": partial(S.h14), "q": partial(S.h14, quiet=True)}),
    "H15": (["buyback", "yjyg", "zengchi"], {"base": partial(S.h15, quiet=True), "noq": partial(S.h15, quiet=False)}),
    "H17": ([], {"base": lambda md, ext: S.h17(md)}),
    "H18": (["lhb"], {"base": partial(S.h18)}),
    "H19": (["dzjy"], {"base": partial(S.h19)}),
    "H20": (["gdhs"], {"base": partial(S.h20), "q": partial(S.h20, quiet=True)}),
}


def main(items):
    lab = Lab()
    for it in items:
        hyp, var = it.split(":")
        names, vars_ = SPECS[hyp]
        ext = {n: pd.read_parquet(EV / LOAD[n]) for n in names} or None
        t0 = time.time()
        rr = lab.run(hyp, var, vars_[var], split="train", ext=ext, **CFG)
        r, st = rr.row, rr.stats
        print(f"{hyp}/{var}: 年化 {r['cagr']:+.1%}  超额 {r['excess_cagr']:+.1%}  回撤 {r['max_drawdown']:.1%}  "
              f"正超额年份 {r['years_excess_pos']}  SMB β {r['smb_beta']:+.2f}  α {r['smb_alpha']:+.1%}  α 占比 {r['alpha_share']}  "
              f"换手 {r['turnover_annual']}  买入 {r['n_buys']}  达标 {r['passed']}  ({time.time() - t0:.0f}s)", flush=True)
        print("   分年超额:", json.dumps({k: round(v, 3) for k, v in st["yearly_excess"].items()}), flush=True)
        nav = rr.result.nav
        print(f"   平均持仓数 {nav.n_pos.mean():.1f}，续持 {rr.result.stat.get('renewals', 0)} 次", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
