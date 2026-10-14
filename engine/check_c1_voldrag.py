"""C1 差距分解：随机选股（不计成本）与等权基准年化差 -2.8% 的来源。"""
import pickle, sys
import numpy as np, pandas as pd
from backtest import Backtest, Config
from test_strategies import RandomPick
from run_checks import ew_benchmark

md = pickle.load(open(sys.argv[1], "rb"))
H, START, END = 5, "2014-01-01", "2023-12-31"
ew_nav, ew_cagr, ew_df = ew_benchmark(md, H, START, END)
ew_r = ew_nav.pct_change().dropna()
print(f"EW: CAGR {ew_cagr:.4f}  arith {ew_r.mean()*242:.4f}  vol {ew_r.std()*np.sqrt(242):.4f}  "
      f"cohort mean 5d ret {ew_df.ret.mean():.5f}  avg n {ew_df.n.mean():.0f}")
rows = []
for seed in range(int(sys.argv[2]) if len(sys.argv) > 2 else 6):
    r = Backtest(md, RandomPick(seed), Config(start=START, end=END, costs=False, hold_days=H)).run()
    nav = r.nav
    d = nav.nav.pct_change().dropna()
    yrs = len(nav) / 242
    rt = r.roundtrips
    rows.append(dict(seed=seed, cagr=(nav.nav.iloc[-1] / nav.nav.iloc[0]) ** (1 / yrs) - 1,
                     arith=d.mean() * 242, vol=d.std() * np.sqrt(242),
                     cash_frac=(nav.cash / nav.nav).mean(),
                     rt_mean=rt.ret.mean(), rt_n=len(rt),
                     rt_mean_normal=rt.loc[rt.how == "hold_days", "ret"].mean(),
                     hows=rt.how.value_counts().to_dict()))
    print(rows[-1], flush=True)
df = pd.DataFrame(rows)
print(df.drop(columns="hows").describe().T[["mean", "std"]])
df.to_csv("../reports/phase2_c1_voldrag.csv", index=False)
