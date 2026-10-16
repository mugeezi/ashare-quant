"""第三批：滚动训练打分模型。用法：python run_batch3.py H21:base H21:h5 H22:base"""
import json
import sys
import time

import numpy as np
import pandas as pd

import ml
from framework import CACHE, ROOT, Lab

EV = ROOT / "data" / "events"
EVENTS = {"buyback": "buyback.parquet", "yjyg": "yjyg.parquet", "zengchi": "zengchi.parquet",
          "lhb": "lhb_jg.parquet", "dzjy": "dzjy.parquet", "gdhs": "gdhs.parquet"}
EVENTS_PLUS = dict(EVENTS, reports="reports.parquet", unlock="unlock.parquet")
EVENTS_FIN = dict(EVENTS, fin="fin.parquet")
EVENTS_MARGIN = dict(EVENTS_FIN)
EVENTS_REG = dict(EVENTS_FIN, wenxun="wenxun.parquet", lian="lian.parquet")


def load_ext(md, plus=False):
    src = EVENTS_REG if plus == "reg" else EVENTS_MARGIN if plus == "margin" else (EVENTS_FIN if plus == "fin" else (EVENTS_PLUS if plus else EVENTS))
    ext = {k: pd.read_parquet(EV / v) for k, v in src.items()}
    if plus in ("margin", "reg"):                           # 融资融券用压缩后的矩阵（按数据日期对齐）
        z = np.load(ROOT / "data/derived/margin.npz")
        ext["margin"] = {k: z[k] for k in ("rzye", "rzjme", "rqyl")}
        ext["margin"]["_dates"] = z["dates"]
    if "unlock" in ext:
        ext["unlock"] = ml.unlock_ext(md, ext["unlock"])
    return ext


def load_ext_plus(md, sp):
    """按规格加载外部数据，并加上市场状态 / 同伴环境特征开关。"""
    ext = load_ext(md, sp.get("plus", False))
    if sp.get("mkt"):
        ext["_mkt"] = True
    if sp.get("peer"):
        ext["_peer"] = True
    if sp.get("nofin"):
        ext["_nofin"] = True
    for k in ("chip", "emo", "cal", "peer2", "emo2"):
        if sp.get(k):
            ext["_" + k] = True
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
    ("H40", "base"): dict(H=20, use_events=True, hold=10, plus="fin", regime=True, window_years=3),
    ("H41", "base"): dict(H=20, use_events=True, hold=10, plus="fin", regime=True, label="riskadj"),
    ("H42", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True),
    ("H43", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, window_years=3),
    ("H44", "base"): dict(H=20, use_events=True, hold=10, plus="reg", regime=True),
    ("H45", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, label="sizeneutral"),
    ("H46", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, universe="large"),
    ("H47", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, construct="div"),
    ("H49", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, label="top10"),
    ("H51", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, construct="stop"),
    ("H53", "base"): dict(H=30, use_events=True, hold=10, plus="margin", regime=True),
    ("H57", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, mkt=True),
    ("H60", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, peer=True),
    ("H61", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, chip=True),
    ("H62", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, chip=True, mkt=True, peer=True),
    ("H63", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, combo=["H42", "H27"]),
    ("H64", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, combo=["H42", "H53", "H43"]),
    ("H65", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, combo=["H42", "H61"]),
    ("H66", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, label="pathadj"),
    ("H67", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, emo=True),
    ("H68", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, cal=True),
    ("H69", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, seeds=(7, 11, 13)),
    ("H70", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, combo=["H42", "H60"]),
    ("H71", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, peer=True, emo=True),
    ("H72", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, peer=True, peer2=True),
    ("H73", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, combo=["H60", "H53"]),
    ("H60s", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, peer=True, seeds=(7, 11, 13)),
    ("H67s", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, emo=True, seeds=(7, 11, 13)),
    ("H78", "base"): dict(H=20, use_events=True, hold=10, plus="margin", regime=True, combo=["H67", "H60"]),
    ("H79", "base"): dict(H=30, use_events=True, hold=10, plus="margin", regime=True, emo=True),
    ("H80", "base"): dict(H=1, use_events=True, hold=1, plus="margin", emo=True, label="on1"),
    ("H81", "base"): dict(H=1, use_events=True, hold=1, plus="margin", emo=True, label="cc1"),
    ("H37", "base"): dict(H=20, use_events=True, hold=10, plus="fin", multi=True),
    ("S05", "base"): dict(H=20, use_events=True, hold=10, plus="fin", model="tree", rank_inputs=True,
                          keep={k for k, _ in __import__("families").SCORECARD}),
    ("S06", "base"): dict(H=20, use_events=True, hold=10, plus="fin", model="small",
                          keep={k for k, _ in __import__("families").SCORECARD}),
    ("H34", "base"): dict(H=20, use_events=True, hold=10, plus="fin", ensemble=True),
    ("H27", "s13"): dict(H=20, use_events=True, hold=10, seeds=(13,)),
}
# 数据修复重跑（2026-10-16）：9 个候选去掉全部业绩报表数值特征的版本（_nf），其余完全不变
for _h in ("H67", "H67s", "H68", "H60", "H60s", "H53", "H71"):
    SPECS[(f"{_h}_nf", "base")] = dict(SPECS[(_h, "base")], nofin=True)
SPECS[("H73_nf", "base")] = dict(SPECS[("H73", "base")], combo=["H60_nf", "H53_nf"], nofin=True)
# 冻结后探索（2026-10-16 用户要求继续尝试）：只用训练集，不上验证集，全部不含业绩报表数值特征
_B = dict(H=20, use_events=True, hold=10, plus="margin", regime=True, nofin=True)
SPECS[("E01", "base")] = dict(_B, emo=True, cal=True)
SPECS[("E02", "base")] = dict(_B, emo=True, emo2=True)
SPECS[("E03", "base")] = dict(_B, emo=True, H=30)
SPECS[("E04", "base")] = dict(_B, emo=True, emo2=True, cal=True)
SPECS[("E05", "base")] = dict(_B, emo=True, peer=True, cal=True)
SPECS[("E06", "base")] = dict(_B, emo=True, seeds=(7, 11, 13, 17, 19))
SPECS[("E07", "base")] = dict(_B, emo=True, H=10)
SPECS[("E08", "base")] = dict(_B, emo=True, window_years=4)
LEAK_RANGE = None


def main(items):
    lab = Lab()
    md = lab.md
    t_lo = int(md.dates.searchsorted(pd.Timestamp("2016-02-01")))
    t_hi = int(md.dates.searchsorted(pd.Timestamp("2017-12-01")))
    for it in items:
        hyp, var = it.split(":")
        sp = SPECS[(hyp, var)]
        ext = load_ext_plus(md, sp) if sp["use_events"] else None
        t0 = time.time()
        extra = {k: sp[k] for k in ("keep", "seeds", "regime", "model", "rank_inputs", "window_years", "label", "universe", "construct") if k in sp}
        mk = ml.multi_factory if sp.get("multi") else (ml.ensemble_factory if sp.get("ensemble") else ml.factory)
        fac = mk(H=sp["H"], use_events=sp["use_events"], last_year=2020, **extra)
        leak_fac = mk(H=sp["H"], use_events=sp["use_events"], max_year=2017, last_year=2017, **extra)
        lf = ROOT / "reports" / f"leak_{hyp}_{var}.json"
        if lf.exists():                                   # 已在单独进程做过前视检测
            import json as _j
            lab._leak_cache[(hyp, var)] = _j.loads(lf.read_text())
        cf = CACHE.parent / f"score_{hyp}_{var}_train.npy"
        strat = None
        if cf.exists():                                   # 训练集打分已由 cache_scores.py 用同一工厂算好，直接复用（省内存）
            sc_ = np.load(cf).astype(np.float64)
            strat = ml.build_strategy(sc_, sp, md)
        rr = lab.run(hyp, var, fac, split="train", strategy=strat, ext=ext, start="2016-01-01", leak_factory=leak_fac,
                     leak_t_range=(t_lo, t_hi), leak_truncate="2018-12-31", params={"H": sp["H"], "model": "lgbm-fixed"},
                     notes="数据修复重跑：去掉业绩报表数值特征（2026-10-16）" if sp.get("nofin") else "", n_target=10, renew=True, hold_days=sp["hold"])
        r, st = rr.row, rr.stats
        print(f"{hyp}/{var}: 年化 {r['cagr']:+.1%}  超额 {r['excess_cagr']:+.1%}  回撤 {r['max_drawdown']:.1%}  "
              f"正超额年份 {r['years_excess_pos']}  SMB β {r['smb_beta']:+.2f}  α {r['smb_alpha']:+.1%}  α 占比 {r['alpha_share']}  "
              f"换手 {r['turnover_annual']}  买入 {r['n_buys']}  达标 {r['passed']}  ({time.time() - t0:.0f}s)", flush=True)
        print("   分年超额:", json.dumps({k: round(v, 3) for k, v in st["yearly_excess"].items()}), flush=True)
        nav = rr.result.nav
        print(f"   平均持仓数 {nav.n_pos.mean():.1f}，续持 {rr.result.stat.get('renewals', 0)} 次", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
