"""验证集运行（每个假设只一次）。用法：python run_valid.py H21 base"""
import json
import sys

import pandas as pd

import ml
from framework import Lab
from run_batch3 import EV, EVENTS, SPECS, load_ext


def main(hyp, var):
    sp = SPECS[(hyp, var)]
    lab = Lab()
    md = lab.md
    ext = load_ext(md, sp.get("plus", False)) if sp["use_events"] else None
    extra = {k: sp[k] for k in ("keep", "seeds", "regime") if k in sp}
    fac = ml.factory(H=sp["H"], use_events=sp["use_events"], last_year=2023, **extra)
    leak_fac = ml.factory(H=sp["H"], use_events=sp["use_events"], max_year=2017, last_year=2017, **extra)
    t_lo = int(md.dates.searchsorted(pd.Timestamp("2016-02-01")))
    t_hi = int(md.dates.searchsorted(pd.Timestamp("2017-12-01")))
    rr = lab.run(hyp, var, fac, split="valid", ext=ext, leak_factory=leak_fac, leak_t_range=(t_lo, t_hi),
                 params={"H": sp["H"], "model": "lgbm-fixed", "refit": "yearly, data to prior year-end"},
                 n_target=10, renew=True, hold_days=sp["hold"])
    r, st = rr.row, rr.stats
    print(json.dumps({k: (float(v) if hasattr(v, "item") else v) for k, v in r.items()}, ensure_ascii=False, indent=1))
    print("分年超额:", json.dumps({k: round(v, 3) for k, v in st["yearly_excess"].items()}))
    print(f"DSR {st['dsr']:.3f}；SMB α {st['smb_alpha']:+.2%}，β {st['smb_beta']:+.2f}，算术超额 {st['arith_excess']:+.2%}")
    rr.result.nav.to_csv(f"../reports/valid_{hyp}_{var}_nav.csv")
    rr.result.trades.to_csv(f"../reports/valid_{hyp}_{var}_trades.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
