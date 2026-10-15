"""S04r（先排除炒作、再按基本面改善排序 + 仓位控制）：引擎训练集 → 稳健性 → （满足规则时）验证集。
用法：python run_s04.py train | robust | valid"""
import json
import sys

import pandas as pd

import families as F
import ml
from framework import Lab, eligible_matrix, leak_test
from run_batch3 import load_ext


N_LOW = int(__import__("os").environ.get("S04_NLOW", "5"))
VAR = "regime" if N_LOW == 5 else f"regime{N_LOW}"


def fac(md, ext):
    return ml.RegimeStrategy(F.s04_filter_then_rank(md, ext, eligible_matrix(md)), n_low=N_LOW)


def show(rr):
    r, st = rr.row, rr.stats
    print(f"{r['hyp_id']}/{r['variant']} [{r['split']}]: 年化 {r['cagr']:+.1%}  超额 {r['excess_cagr']:+.1%}  回撤 {r['max_drawdown']:.1%}  "
          f"正超额年份 {r['years_excess_pos']}  α {r['smb_alpha']:+.1%}  α 占比 {r['alpha_share']}  换手 {r['turnover_annual']}  "
          f"达标 {r['passed']}  {r['notes']}", flush=True)
    print("   分年超额:", json.dumps({k: round(v, 3) for k, v in st["yearly_excess"].items()}), flush=True)


def main(mode):
    lab = Lab()
    md = lab.md
    ext = load_ext(md, plus="fin")
    t_lo = int(md.dates.searchsorted(pd.Timestamp("2016-02-01")))
    t_hi = int(md.dates.searchsorted(pd.Timestamp("2017-12-01")))
    lk = leak_test(md, fac, ext, t_range=(t_lo, t_hi), truncate="2018-12-31")
    print("前视检测", lk, flush=True)
    if not lk["passed"]:
        return
    base = dict(n_target=10, renew=True, hold_days=10)
    if mode == "train":
        lab._leak_cache[("S04", VAR)] = lk
        show(lab.run("S04", VAR, fac, split="train", ext=ext, params={"rule": "S04r"}, **base))
    elif mode == "robust":
        strat = fac(md, ext)
        for v, kw in {"hold8": dict(hold_days=8), "hold12": dict(hold_days=12), "k24": dict(renew_k=24),
                      "k36": dict(renew_k=36), "n8": dict(n_target=8), "n12": dict(n_target=12)}.items():
            lab._leak_cache[("S04", f"{VAR}_{v}")] = lk
            show(lab.run("S04", f"{VAR}_{v}", fac, split="train", ext=ext, strategy=strat,
                         params={"rule": "S04r", "robust_of": VAR}, notes="稳健性检验", **dict(base, **kw)))
    elif mode == "valid":
        lab._leak_cache[("S04", VAR)] = lk
        rr = lab.run("S04", VAR, fac, split="valid", ext=ext, params={"rule": "S04r"}, **base)
        show(rr)
        rr.result.nav.to_csv(f"../reports/valid_S04_{VAR}_nav.csv")


if __name__ == "__main__":
    main(sys.argv[1])
