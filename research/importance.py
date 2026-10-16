"""原版候选模型中各特征按 gain 的重要性（2016–2023 每年重训的模型，种子 7），重点看业绩报表数值特征的占比。
只训练、不回测、不写 experiments.csv。用法：IMP_OUT=1 python importance.py H67 H68 H60 H53 H71"""
import json
import os
import sys

import ml
from framework import ROOT, Lab
from run_batch3 import SPECS, load_ext_plus

FIN11 = {"ep_ttm", "bp", "roe_ann", "gm", "gm_chg", "cfo_ratio", "ni_yoy", "rev_yoy", "fc_surprise", "ep_self_pct", "ep_peer_rel"}

os.environ["IMP_OUT"] = "1"
_f = ROOT / "reports/importance_original.json"
out = json.loads(_f.read_text()) if _f.exists() else {}
for h in sys.argv[1:]:
    sp = SPECS[(h, "base")]
    lab = Lab()
    md = lab.md
    ext = load_ext_plus(md, sp)
    extra = {k: sp[k] for k in ("keep", "model", "rank_inputs", "window_years", "label", "universe") if k in sp}
    ml._IMP.clear()
    ml.walk_forward_scores(md, ext, H=sp["H"], last_year=2023, max_year=2023, free_md=True, seeds=(7,), **extra)
    years = {}
    for Y, d in ml._IMP.items():
        tot = sum(d.values())
        years[int(Y)] = dict(fin_share=round(sum(v for k, v in d.items() if k in FIN11) / tot, 4),
                             top10=sorted(((k, round(v / tot, 4)) for k, v in d.items()), key=lambda x: -x[1])[:10],
                             fin_each={k: round(d.get(k, 0) / tot, 4) for k in sorted(FIN11)})
    out[h] = years
    print(h, {y: v["fin_share"] for y, v in years.items()}, flush=True)
    (ROOT / "reports/importance_original.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    del lab, md, ext
