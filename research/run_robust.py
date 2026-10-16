"""H21 稳健性：同一打分，只改执行参数。用法：python run_robust.py H21 base"""
import json
import sys
import time

import pandas as pd

import ml
from framework import Lab, leak_test
from run_batch3 import EV, EVENTS, SPECS, load_ext, load_ext_plus

VARS = {"hold8": dict(hold_days=8), "hold12": dict(hold_days=12), "k24": dict(renew_k=24), "k36": dict(renew_k=36),
        "n8": dict(n_target=8), "n12": dict(n_target=12)}


def main(hyp, base_var):
    sp = SPECS[(hyp, base_var)]
    H, use_events = sp["H"], sp["use_events"]
    extra = {k: sp[k] for k in ("keep", "seeds", "regime", "model", "rank_inputs", "window_years", "label", "universe", "construct") if k in sp}
    lab = Lab()
    md = lab.md
    mk = ml.multi_factory if sp.get("multi") else (ml.ensemble_factory if sp.get("ensemble") else ml.factory)
    fac = mk(H=H, use_events=use_events, last_year=2020, **extra)
    leak_fac = mk(H=H, use_events=use_events, max_year=2017, last_year=2017, **extra)
    t_lo = int(md.dates.searchsorted(pd.Timestamp("2016-02-01")))
    t_hi = int(md.dates.searchsorted(pd.Timestamp("2017-12-01")))
    ext = load_ext_plus(md, sp) if use_events else None
    lf = __import__("framework").ROOT / "reports" / f"leak_{hyp}_{base_var}.json"
    lk = __import__("json").loads(lf.read_text()) if lf.exists() else \
        leak_test(md, leak_fac, ext, t_range=(t_lo, t_hi), truncate="2018-12-31")
    print("前视检测", lk, flush=True)
    cf = __import__("framework").CACHE.parent / f"score_{hyp}_{base_var}_train.npy"
    if cf.exists():
        import numpy as np
        sc_ = np.load(cf).astype(np.float64)
        strat = ml.build_strategy(sc_, sp, md)
    else:
        strat = fac(md, ext)
    for v, kw in VARS.items():
        name = f"{base_var}_{v}"
        lab._leak_cache[(hyp, name)] = lk
        cfg = dict(n_target=10, renew=True, hold_days=sp["hold"])
        cfg.update(kw)
        rr = lab.run(hyp, name, fac, split="train", start="2016-01-01", strategy=strat,
                     params={"H": H, "model": "lgbm-fixed", "robust_of": base_var}, notes="稳健性检验" + ("；数据修复重跑" if sp.get("nofin") else ""), **cfg)
        r, st = rr.row, rr.stats
        print(f"{hyp}/{name}: 年化 {r['cagr']:+.1%}  超额 {r['excess_cagr']:+.1%}  回撤 {r['max_drawdown']:.1%}  "
              f"正超额年份 {r['years_excess_pos']}  α {r['smb_alpha']:+.1%}  换手 {r['turnover_annual']}  达标 {r['passed']}",
              flush=True)
        print("   分年超额:", json.dumps({k: round(v, 3) for k, v in st["yearly_excess"].items()}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
