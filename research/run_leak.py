"""单独进程做前视检测（省内存），结果写到 reports/leak_{hyp}_{var}.json，供后续运行读取。"""
import json
import sys

import pandas as pd

import ml
from framework import ROOT, Lab, leak_test
from run_batch3 import SPECS, load_ext


def main(hyp, var):
    sp = SPECS[(hyp, var)]
    lab = Lab()
    md = lab.md
    ext = load_ext(md, sp.get("plus", False)) if sp["use_events"] else None
    extra = {k: sp[k] for k in ("keep", "seeds", "regime", "model", "rank_inputs", "window_years", "label", "universe", "construct") if k in sp}
    mk = ml.multi_factory if sp.get("multi") else (ml.ensemble_factory if sp.get("ensemble") else ml.factory)
    leak_fac = mk(H=sp["H"], use_events=sp["use_events"], max_year=2017, last_year=2017, **extra)
    t_lo = int(md.dates.searchsorted(pd.Timestamp("2016-02-01")))
    t_hi = int(md.dates.searchsorted(pd.Timestamp("2017-12-01")))
    lk = leak_test(md, leak_fac, ext, t_range=(t_lo, t_hi), truncate="2018-12-31")
    (ROOT / "reports" / f"leak_{hyp}_{var}.json").write_text(json.dumps(lk, ensure_ascii=False))
    print(lk)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
