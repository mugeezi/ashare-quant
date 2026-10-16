"""汇总所有用过验证集的候选，按当前全部试验数 N 统一重算修订口径 DSR，并列出五项标准是否通过。"""
import math
import numpy as np
import pandas as pd
import ml
import run_exec as RX
from dsr_eval import dsr, excess, _auto
from framework import CACHE, ROOT, Backtest, Config, Lab, read_exp

lab = Lab()
md = lab.md
exp = read_exp()
N = len(exp)
v = exp[exp.split == "valid"].copy()
rows = []
for _, r in v.iterrows():
    h, var = r.hyp_id, r.variant
    if h.startswith(("X10_", "X12_", "X13_")) and not str(var).endswith("_fix"):
        continue                                       # 修复前结果含前视漏洞，已作废，以 _fix 重跑结果为准
    vf = ROOT / "reports" / f"valid_{h}_{var}_nav.csv"
    if not vf.exists():
        continue
    try:
        if h in RX.X or h.startswith("X"):
            b, vs = RX.X[h]
            kind, kw = vs[var[:-4] if str(var).endswith("_fix") else var]
            sc = np.load(CACHE.parent / f"score_{b}_base_train.npy").astype(np.float64)
            strat, cfg = RX.build(kind, sc, md), {**RX.BASE, **kw}
        else:
            f, sp, _ = _auto(h)
            sc = np.load(CACHE.parent / f).astype(np.float64)
            strat, cfg = ml.build_strategy(sc, sp, md), dict(n_target=10, renew=True, hold_days=10)
    except Exception as e:  # noqa: BLE001
        print("跳过", h, var, e)
        continue
    res = Backtest(md, strat, Config(start="2016-01-01", end="2020-12-31", **cfg)).run()
    vn = pd.read_csv(vf, index_col=0, parse_dates=True).nav
    e = pd.concat([excess(res.nav.nav, lab.bench), excess(vn, lab.bench)])
    p, sr, sr0, T = dsr(e, N)
    ok = (r.excess_cagr >= 0.08 and int(str(r.years_excess_pos)[0]) >= 2 and r.max_drawdown >= -0.35
          and (r.alpha_share or 0) >= 0.5 and p >= 0.95)
    rows.append(dict(策略=f"{h}/{var}" if var != "base" else h, 验证集超额=r.excess_cagr, 正年=r.years_excess_pos,
                     回撤=r.max_drawdown, α占比=r.alpha_share, 样本外IR=round(sr * math.sqrt(242), 2), DSR=round(p, 3), 五项全过=ok))
    del sc
df = pd.DataFrame(rows).sort_values("DSR", ascending=False)
df.to_csv(ROOT / "reports" / "leaderboard.csv", index=False)
print(f"N = {N}")
print(df.to_string(index=False))
