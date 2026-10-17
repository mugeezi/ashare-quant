"""候选诊断（2026-10-16 第 2 项）：只用 2016–2020 训练期样本外与 2021–2023 验证集，不读测试集。
对修复后五项全过的候选：
  1) 日超额收益相关矩阵与分族（平均连接，相关系数 ≥ 0.7 视为同族）；
  2) 风格：SMB 回归、"国证 2000 − 中证全指"日收益差回归的 α 占比；持仓流通市值在全部 A 股中的平均分位数；
  3) 收益集中度：每年贡献最大的 5 笔交易占全年盈亏的比例；剔除收益最高的 1% 交易后的超额。
各期净值用缓存打分重跑引擎得到（与已记录的运行逐笔一致，不是新试验，不写 experiments.csv）。"""
import json

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

import ml
import run_exec as RX
from dsr_eval import _auto
from framework import CACHE, DAYS, ROOT, Backtest, Config, Lab, excess_stats

CANDS = ["H67", "H68", "X08_H60s/vk3", "X12_H68/dip1zb_fix", "X08_H71/vk3", "X08_H53/zb", "X12_H67s/dip1zb_fix",
         "H73", "H60s"]
PERIODS = {"train": ("2016-01-01", "2020-12-31"), "valid": ("2021-01-01", "2023-12-31")}
END = pd.Timestamp("2023-12-31")


def builder(name, md):
    if name.startswith("X"):
        h, var = name.split("/")
        b, vs = RX.X[h]
        kind, kw = vs[var[:-4] if var.endswith("_fix") else var]
        return b, (lambda sc: RX.build(kind, sc, md)), {**RX.BASE, **kw}
    f, sp, _ = _auto(name)
    return name, (lambda sc: ml.build_strategy(sc, sp, md)), dict(n_target=10, renew=True, hold_days=10)


def regress(e, x):
    m = x.reindex(e.index).notna()
    X = np.column_stack([np.ones(m.sum()), x.reindex(e.index)[m].values])
    coef, *_ = np.linalg.lstsq(X, e[m].values, rcond=None)
    ar = e.mean() * DAYS
    return coef[0] * DAYS / ar if ar > 0 else np.nan, coef[1]


def main(cands=None, tag=""):
    cands = cands or CANDS
    lab = Lab()
    md = lab.md
    bench = lab.bench
    gz = pd.read_parquet(ROOT / "data/index/gz2000.parquet")
    gz = gz[pd.to_datetime(gz["date"]) <= END]                     # 只用到 2023 年末
    gz = gz.set_index(pd.to_datetime(gz["date"]))["close"]
    zz = bench[bench.index <= END]
    size_spread = (gz.pct_change() - zz.pct_change()).dropna()     # 国证 2000 − 中证全指 日收益差
    # 全部 A 股（当日正常交易且有流通市值）中的流通市值分位
    fmv = np.where((md.a["tradestatus"] == 1) & np.isfinite(md.a["fmv"]), md.a["fmv"], np.nan)
    pct = pd.DataFrame(fmv).rank(axis=1, pct=True).values.astype(np.float32)
    col = {c: i for i, c in enumerate(md.codes)}
    ex_all, rows, conc_rows = {}, [], []
    for name in cands:
        h0 = name.split("/")[0]
        v0 = (name.split("/") + ["base"])[1]
        if not (ROOT / "reports" / f"valid_{h0}_{v0}_nav.csv").exists():
            print(f"{name}：没有验证集结果，跳过", flush=True)
            continue
        b, mk, cfg = builder(name, md)
        per = {}
        for sp, (s0, s1) in PERIODS.items():
            sc = np.load(CACHE.parent / f"score_{b}_base_{sp}.npy").astype(np.float64)
            res = Backtest(md, mk(sc), Config(start=s0, end=s1, **cfg)).run()
            per[sp] = res
            del sc
        # 核对：验证集净值与已保存的一致
        h, v = (name.split("/") + ["base"])[:2]
        vf = ROOT / "reports" / f"valid_{h}_{v}_nav.csv"
        saved = pd.read_csv(vf, index_col=0, parse_dates=True).nav
        match = abs(saved.iloc[-1] / per["valid"].nav.nav.iloc[-1] - 1) < 1e-9
        navs = {sp: per[sp].nav.nav for sp in per}
        e = {}
        for sp, nav in navs.items():
            r = nav.pct_change().dropna()
            bb = bench.reindex(nav.index).ffill().pct_change().reindex(r.index).fillna(0)
            e[sp] = r - bb
        ex_all[name] = pd.concat([e["train"], e["valid"]])
        row = dict(策略=name, 验证集净值核对一致=bool(match))
        for sp in ("train", "valid"):
            st = excess_stats(navs[sp], bench, lab.smb)
            a_gz, b_gz = regress(e[sp], size_spread)
            row[f"{sp}_超额"] = round(st["excess_cagr"], 4)
            row[f"{sp}_SMB_α占比"] = round(st["alpha_share"], 3)
            row[f"{sp}_SMB_β"] = round(st["smb_beta"], 3)
            row[f"{sp}_国证2000差_α占比"] = round(a_gz, 3)
            row[f"{sp}_国证2000差_β"] = round(b_gz, 3)
            rt = per[sp].roundtrips
            vals = []
            for _, x in rt.iterrows():
                j = col.get(x.code)
                if j is None:
                    continue
                t0, t1 = md.dates.get_loc(x.entry_date), md.dates.get_loc(x.exit_date)
                vals.extend(pct[t0:max(t1, t0 + 1), j].tolist())
            row[f"{sp}_持仓市值分位均值"] = round(float(np.nanmean(vals)), 3)
            row[f"{sp}_持仓市值分位中位数"] = round(float(np.nanmedian(vals)), 3)
            # 集中度
            rt = rt.copy()
            rt["year"] = pd.to_datetime(rt.exit_date).dt.year
            for y, g in rt.groupby("year"):
                tot = g.pnl.sum()
                top5 = g.pnl.nlargest(5).sum()
                conc_rows.append(dict(策略=name, 区间=sp, 年份=int(y), 交易笔数=len(g), 全年盈亏=round(tot),
                                      前5笔盈亏=round(top5), 前5笔占比=round(top5 / tot, 3) if tot > 0 else None))
            k = max(1, int(np.ceil(len(rt) * 0.01)))
            top = rt.nlargest(k, "ret")
            adj = pd.Series(0.0, index=navs[sp].index)
            for _, x in top.iterrows():                               # 视为这些交易没有发生：从卖出日起扣除其盈利
                adj[adj.index >= pd.Timestamp(x.exit_date)] += x.pnl
            nav2 = navs[sp] - adj
            st2 = excess_stats(nav2, bench, lab.smb)
            row[f"{sp}_剔除前1%交易笔数"] = k
            row[f"{sp}_剔除前1%后超额"] = round(st2["excess_cagr"], 4)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    E = pd.DataFrame(ex_all)
    corr = E.corr()
    corr_v = E[E.index >= "2021-01-01"].corr()
    d = squareform(1 - corr.values, checks=False)
    fam = fcluster(linkage(d, "average"), t=0.3, criterion="distance")
    out = pd.DataFrame(rows)
    out["相关性族"] = [f"F{f}" for f in fam]
    out.to_csv(ROOT / f"reports/diag_candidates{tag}.csv", index=False)
    corr.round(3).to_csv(ROOT / f"reports/diag_corr_2016_2023{tag}.csv")
    corr_v.round(3).to_csv(ROOT / f"reports/diag_corr_valid{tag}.csv")
    pd.DataFrame(conc_rows).to_csv(ROOT / f"reports/diag_concentration{tag}.csv", index=False)
    print(corr.round(2).to_string())
    print(out[["策略", "相关性族"]].to_string(index=False))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "nf":
        main(["H67_nf", "H68_nf", "H60s_nf", "H73_nf", "X08_H60s_nf/vk3", "X08_H71_nf/vk3", "X08_H53_nf/zb",
              "X12_H68_nf/dip1zb", "X12_H67s_nf/dip1zb"], tag="_nf")
    elif len(sys.argv) > 1 and sys.argv[1] == "pool":            # 最终测试候选池：9 个 _nf 版本 + E01
        main(["H67_nf", "H68_nf", "H60s_nf", "H73_nf", "X08_H60s_nf/vk3", "X08_H71_nf/vk3", "X08_H53_nf/zb",
              "X12_H68_nf/dip1zb", "X12_H67s_nf/dip1zb", "E01"], tag="_pool")
    else:
        main()
