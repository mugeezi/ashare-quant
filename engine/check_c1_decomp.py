"""C1 分解：随机 − 同日等权 = 选股抽样误差（理想退出）+ 交易摩擦（跌停顺延、停牌锁定）。"""
import pickle, sys
import numpy as np, pandas as pd
from backtest import Backtest, Config
from test_strategies import RandomPick
from run_checks import ew_benchmark

md = pickle.load(open(sys.argv[1], "rb"))
H, START, END = 5, "2014-01-01", "2023-12-31"
_, _, ew_df = ew_benchmark(md, H, START, END)
a = md.a
hopen = a["open"] * a["adj_factor"]; ts = a["tradestatus"]
T, N = ts.shape
nxt = np.full((T, N), np.nan); last = np.full(N, np.nan)
for i in range(T - 1, -1, -1):
    ok = (ts[i] == 1) & np.isfinite(hopen[i]); last = np.where(ok, hopen[i], last); nxt[i] = last
ci = {c: i for i, c in enumerate(md.codes)}
out = []
for seed in range(int(sys.argv[2])):
    r = Backtest(md, RandomPick(seed), Config(start=START, end=END, costs=False, hold_days=H)).run()
    rt = r.roundtrips[r.roundtrips.how == "hold_days"].copy()
    tb = md.dates.get_indexer(pd.to_datetime(rt.entry_date)); j = rt.code.map(ci).values
    rt["ideal"] = nxt[np.minimum(tb + H, T - 1), j] / hopen[tb, j] - 1
    rt["ew"] = ew_df.ret.reindex(pd.to_datetime(rt.entry_date)).values
    rt = rt.dropna(subset=["ew", "ideal"])
    delayed = (pd.to_datetime(rt.exit_date).values != md.dates[np.minimum(tb + H, T - 1)].values)
    out.append(dict(seed=seed, total=(rt.ret - rt.ew).mean(), selection=(rt.ideal - rt.ew).mean(),
                    friction=(rt.ret - rt.ideal).mean(), n_delayed=int(delayed.sum()),
                    friction_on_delayed=(rt.ret - rt.ideal)[delayed].mean()))
    print(out[-1], flush=True)
df = pd.DataFrame(out)
print(df.mean().to_string()); print("跨种子标准差\n", df.std().to_string())
df.to_csv("../reports/phase2_c1_decomp.csv", index=False)
