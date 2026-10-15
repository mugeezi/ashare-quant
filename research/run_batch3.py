"""第三批：滚动训练打分模型。用法：python run_batch3.py H21:base H21:h5 H22:base"""
import json
import sys
import time

import pandas as pd

import ml
from framework import ROOT, Lab

EV = ROOT / "data" / "events"
EVENTS = {"buyback": "buyback.parquet", "yjyg": "yjyg.parquet", "zengchi": "zengchi.parquet",
          "lhb": "lhb_jg.parquet", "dzjy": "dzjy.parquet", "gdhs": "gdhs.parquet"}
EVENTS_PLUS = dict(EVENTS, reports="reports.parquet", unlock="unlock.parquet")
EVENTS_FIN = dict(EVENTS, fin="fin.parquet")


def load_ext(md, plus=False):
    src = EVENTS_FIN if plus == "fin" else (EVENTS_PLUS if plus else EVENTS)
    ext = {k: pd.read_parquet(EV / v) for k, v in src.items()}
    if "unlock" in ext:
        ext["unlock"] = ml.unlock_ext(md, ext["unlock"])
    return ext
SPECS = {
    ("H21", "base"): dict(H=10, use_events=False, hold=10),
    ("H21", "h5"): dict(H=5, use_events=False, hold=5),
    ("H22", "base"): dict(H=10, use_events=True, hold=10),
    ("H23", "base"): dict(H=10, use_events=True, hold=10, keep=ml.EVENT_ONLY_KEEP),
    ("H24", "base"): dict(H=10, use_events=True, hold=10, regime=True),
    ("H25", "base"): dict(H=10, use_events=True, hold=10, plus=True, seeds=(7, 11, 13, 17, 19)),
    ("H26", "base"): dict(H=10, use_events=True, hold=10, model="ridge", rank_inputs=True),
    ("H27", "base"): dict(H=20, use_events=True, hold=10),
    ("H28", "base"): dict(H=10, use_events=True, hold=10, model="rank"),
    ("H30", "base"): dict(H=10, use_events=True, hold=10, rank_inputs=True),
    ("H27", "s11"): dict(H=20, use_events=True, hold=10, seeds=(11,)),
    ("H31", "base"): dict(H=20, use_events=True, hold=10, plus="fin"),
    ("H32", "base"): dict(H=40, use_events=True, hold=10, plus="fin"),
    ("H33", "base"): dict(H=20, use_events=True, hold=10, plus="fin", regime=True),
    ("S05", "base"): dict(H=20, use_events=True, hold=10, plus="fin", model="tree", rank_inputs=True,
                          keep={k for k, _ in __import__("families").SCORECARD}),
    ("S06", "base"): dict(H=20, use_events=True, hold=10, plus="fin", model="small",
                          keep={k for k, _ in __import__("families").SCORECARD}),
    ("H34", "base"): dict(H=20, use_events=True, hold=10, plus="fin", ensemble=True),
    ("H27", "s13"): dict(H=20, use_events=True, hold=10, seeds=(13,)),
}
LEAK_RANGE = None


def main(items):
    lab = Lab()
    md = lab.md
    t_lo = int(md.dates.searchsorted(pd.Timestamp("2016-02-01")))
    t_hi = int(md.dates.searchsorted(pd.Timestamp("2017-12-01")))
    for it in items:
        hyp, var = it.split(":")
        sp = SPECS[(hyp, var)]
        ext = load_ext(md, sp.get("plus", False)) if sp["use_events"] else None
        t0 = time.time()
        extra = {k: sp[k] for k in ("keep", "seeds", "regime", "model", "rank_inputs") if k in sp}
        mk = ml.ensemble_factory if sp.get("ensemble") else ml.factory
        fac = mk(H=sp["H"], use_events=sp["use_events"], last_year=2020, **extra)
        leak_fac = mk(H=sp["H"], use_events=sp["use_events"], max_year=2017, last_year=2017, **extra)
        rr = lab.run(hyp, var, fac, split="train", ext=ext, start="2016-01-01", leak_factory=leak_fac,
                     leak_t_range=(t_lo, t_hi), leak_truncate="2018-12-31", params={"H": sp["H"], "model": "lgbm-fixed"},
                     n_target=10, renew=True, hold_days=sp["hold"])
        r, st = rr.row, rr.stats
        print(f"{hyp}/{var}: 年化 {r['cagr']:+.1%}  超额 {r['excess_cagr']:+.1%}  回撤 {r['max_drawdown']:.1%}  "
              f"正超额年份 {r['years_excess_pos']}  SMB β {r['smb_beta']:+.2f}  α {r['smb_alpha']:+.1%}  α 占比 {r['alpha_share']}  "
              f"换手 {r['turnover_annual']}  买入 {r['n_buys']}  达标 {r['passed']}  ({time.time() - t0:.0f}s)", flush=True)
        print("   分年超额:", json.dumps({k: round(v, 3) for k, v in st["yearly_excess"].items()}), flush=True)
        nav = rr.result.nav
        print(f"   平均持仓数 {nav.n_pos.mean():.1f}，续持 {rr.result.stat.get('renewals', 0)} 次", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
