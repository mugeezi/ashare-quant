"""最终测试运行器（FINAL_TEST_PLAN.md 第 5 节）。

正式运行的前提（缺一不可，否则直接退出）：
  1) 仓库根目录有 FINAL_TEST_APPROVED，第一行是获批时的提交哈希；
  2) 该哈希等于标签 forward-start 指向的提交；
  3) 当前工作区中 engine/、research/ 下的代码与 forward-start 完全一致（策略代码与参数冻结）。
--dry-run 例外：只用研究期数据，把 2023 年当作"测试年"（模型用截至 2022 年末的数据训练），
检验数据合并、逐年重训、组合打分、回测、指标计算的整条流程和内存峰值，不输出任何收益指标。

步骤（每步单独进程，控制内存）：
  python run_final_test.py prep  --cutoff 2026-10-16   加载全期行情（写测试集访问日志）、合并研究期与测试期事件、重建两融矩阵
  python run_final_test.py score H60_nf                逐年重训并预测测试期（2024、2025、2026 各用截至上一年末的数据）
  python run_final_test.py eval                        组合打分、回测、计算通过标准与报告项，写 reports/FINAL_TEST_RESULT.md
"""
import argparse
import json
import math
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import ml
from framework import CACHE, DAYS, ROOT, Backtest, Config, eligible_matrix, excess_stats, hfq
from run_batch3 import EV, EVENTS_MARGIN, SPECS

# ---- 入选策略（FINAL_TEST_PLAN.md 第 3 节；冻结后不得修改）
MAIN, SECOND, CONTROL = "H73_nf", "H67_nf", "H42_nf"
STRATS = [MAIN, SECOND, CONTROL]
COMPONENTS = {"H73_nf": ["H60_nf", "H53_nf"], "H67_nf": ["H67_nf"], "H42_nf": ["H42_nf"]}
ENGINE = dict(n_target=10, renew=True, hold_days=10)      # 其余为引擎默认值（滑点 0.1%、成本、过滤条件等）
TEST_START = "2024-01-01"
CRITERIA = dict(excess=0.05, ir=0.5, max_dd=-0.35, years_pos=2, alpha_share=0.5)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def guard():
    f = ROOT / "FINAL_TEST_APPROVED"
    if not f.exists():
        raise SystemExit("未找到 FINAL_TEST_APPROVED：最终测试未获批")
    approved = f.read_text().split()[0]
    tag = git("rev-parse", "forward-start^{commit}")
    if approved != tag:
        raise SystemExit(f"FINAL_TEST_APPROVED 中的提交 {approved[:10]} 与 forward-start {tag[:10]} 不一致")
    if git("diff", "--name-only", "forward-start", "--", "engine", "research"):
        raise SystemExit("engine/ 或 research/ 的代码与 forward-start 不一致，冻结被破坏")


def workdir(dry):
    d = (CACHE.parent / "final_dry") if dry else (ROOT / "data" / "final_run")
    d.mkdir(parents=True, exist_ok=True)
    return d


def years(dry, cutoff):
    return (2023, 2023) if dry else (2024, pd.Timestamp(cutoff).year)


# ---------------------------------------------------------------- 数据
def prep(dry, cutoff):
    from data import add_derived, load_market
    wd = workdir(dry)
    if dry:
        md = add_derived(load_market(end="2023-12-31"))
    else:
        md = add_derived(load_market(end=cutoff, final_test=True, purpose="最终测试：加载全期行情"))
    pickle.dump(md, open(wd / "md.pkl", "wb"), protocol=5)
    T, N, dates, codes = md.T, md.N, md.dates, md.codes
    first, last = str(md.dates[0].date()), str(md.dates[-1].date())
    del md                                                     # 先落盘再释放行情，降低内存峰值
    import gc
    gc.collect()
    ext = {}
    for k, v in EVENTS_MARGIN.items():                         # 研究期 + 测试期（测试期只取公告日 ≥ 2024-01-01）
        parts = [pd.read_parquet(EV / v)]
        f2 = ROOT / "data" / "events_final" / v
        if not dry and f2.exists():
            parts.append(pd.read_parquet(f2))
        ext[k] = pd.concat(parts, ignore_index=True)
    cm = {c.split(".")[1]: i for i, c in enumerate(codes)}
    m = {c: np.full((T, N), np.nan, dtype=np.float32) for c in ("rzye", "rzjme", "rqyl")}
    srcs = [EV / "margin.parquet"] + ([ROOT / "data" / "events_final" / "margin.parquet"] if not dry else [])
    for f in srcs:                                             # 与 build_margin.py 同口径（按数据日期对齐，使用时再顺延一日）
        if not f.exists():
            continue
        mg = pd.read_parquet(f, columns=["date", "code6", "rzye", "rzjme", "rqyl"])
        j = mg["code6"].astype(str).str.zfill(6).map(cm)
        t = dates.get_indexer(pd.to_datetime(mg["date"]))
        ok = j.notna().values & (t >= 0)
        for c in m:
            m[c][t[ok], j[ok].astype(int).values] = pd.to_numeric(mg[c], errors="coerce").values[ok].astype(np.float32)
        del mg, j, t, ok
        gc.collect()
    m["_dates"] = dates.values.astype("datetime64[D]").astype(str)
    ext["margin"] = m
    pickle.dump(ext, open(wd / "ext.pkl", "wb"), protocol=5)
    print("prep 完成", T, N, first, "~", last, flush=True)


def load_ext(wd, sp):
    ext = pickle.load(open(wd / "ext.pkl", "rb"))
    if sp.get("nofin"):
        ext["_nofin"] = True
    for k in ("emo", "cal", "peer", "emo2", "chip", "peer2"):
        if sp.get(k):
            ext["_" + k] = True
    if sp.get("mkt"):
        ext["_mkt"] = True
    return ext


# ---------------------------------------------------------------- 打分
def score(dry, cutoff, h):
    import os
    from signals import peer_sets
    wd = workdir(dry)
    os.environ["PANEL_MEMMAP_DIR"] = str(wd)                  # 面板放磁盘映射文件，数值不变
    fy, ly = years(dry, cutoff)
    sp = SPECS[(h, "base")]
    md = pickle.load(open(wd / "md.pkl", "rb"))
    peer_sets(md)
    ext = load_ext(wd, sp)
    extra = {k: sp[k] for k in ("keep", "seeds", "model", "rank_inputs", "window_years", "label", "universe") if k in sp}
    sc = ml.walk_forward_scores(md, ext, H=sp["H"], first_year=fy, last_year=ly, max_year=ly, free_md=True, **extra)
    np.save(wd / f"score_{h}.npy", sc.astype(np.float32))
    print("score 完成", h, "非空比例", round(float(np.isfinite(sc).mean()), 4), flush=True)


# ---------------------------------------------------------------- 评估
def smb_from_md(md):
    """与 framework.smb_series 同一公式，但不读研究期缓存（测试期需要重新计算）。"""
    elig = eligible_matrix(md)
    px = hfq(md, "close")
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = px[1:] / px[:-1] - 1
    out = []
    for t in range(1, md.T):
        ok = elig[t - 1] & np.isfinite(md.a["fmv"][t - 1]) & np.isfinite(ret[t - 1])
        mv, r = md.a["fmv"][t - 1][ok], ret[t - 1][ok]
        if len(mv) < 100:
            out.append(np.nan)
            continue
        lo, hi = np.quantile(mv, [0.3, 0.7])
        out.append(r[mv <= lo].mean() - r[mv >= hi].mean())
    return pd.Series(out, index=md.dates[1:], name="smb")


def combo_score(wd, comps):
    mats = [pd.DataFrame(np.load(wd / f"score_{c}.npy").astype(np.float64)).rank(axis=1, pct=True).values for c in comps]
    return np.nanmean(np.stack(mats), axis=0) if len(mats) > 1 else np.load(wd / f"score_{comps[0]}.npy").astype(np.float64)


def evaluate(dry, cutoff):
    wd = workdir(dry)
    need = sorted({c for h in STRATS for c in COMPONENTS[h]})
    miss = [c for c in need if not (wd / f"score_{c}.npy").exists()]
    if miss:
        raise SystemExit(f"打分未全部完成，缺 {miss}；eval 必须在全部打分完成后一次性运行")
    if not dry:                                                # eval 只允许运行一次（FINAL_TEST_PLAN.md 4.2）
        mark = wd / "EVAL_STARTED"
        if mark.exists() or (ROOT / "reports" / "FINAL_TEST_RESULT.json").exists():
            raise SystemExit("eval 已运行过，测试视为已完成，不得再次运行")
        mark.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + f" cutoff={cutoff} commit={git('rev-parse', 'HEAD')}\n")
    md = pickle.load(open(wd / "md.pkl", "rb"))
    from framework import add_derived
    md = add_derived(md)
    bench = md.index_close["zzqz"]
    smb = smb_from_md(md)
    s0 = "2023-01-01" if dry else TEST_START
    s1 = "2023-12-31" if dry else cutoff
    gz = pd.read_parquet(ROOT / "data/index/gz2000.parquet")
    gz = gz.set_index(pd.to_datetime(gz["date"]))["close"]
    gz = gz[gz.index <= pd.Timestamp(s1)]
    spread = (gz.pct_change() - bench.reindex(gz.index).ffill().pct_change()).dropna()
    res_all = {}
    for h in STRATS:
        sc = combo_score(wd, COMPONENTS[h])
        strat = ml.build_strategy(sc, SPECS[(h, "base")], md)
        res = Backtest(md, strat, Config(start=s0, end=s1, **ENGINE)).run()
        nav = res.nav.nav
        st = excess_stats(nav, bench, smb)
        r = nav.pct_change().dropna()
        e = r - bench.reindex(nav.index).ffill().pct_change().reindex(r.index).fillna(0)
        ir = e.mean() / e.std() * math.sqrt(DAYS) if e.std() > 0 else float("nan")
        m = spread.reindex(e.index).notna()
        X = np.column_stack([np.ones(m.sum()), spread.reindex(e.index)[m].values])
        coef, *_ = np.linalg.lstsq(X, e[m].values, rcond=None)
        ar = e.mean() * DAYS
        a_gz = coef[0] * DAYS / ar if ar > 0 else float("nan")
        rt = res.roundtrips
        k = max(1, int(math.ceil(len(rt) * 0.01)))
        adj = pd.Series(0.0, index=nav.index)
        for _, x in rt.nlargest(k, "ret").iterrows():
            adj[adj.index >= pd.Timestamp(x.exit_date)] += x.pnl
        ex_trim = excess_stats(nav - adj, bench, smb)["excess_cagr"]
        ypos = sum(v > 0 for v in st["yearly_excess"].values())
        out = dict(策略=h, 起=s0, 止=str(nav.index[-1].date()), 年化=st["cagr"], 年化超额=st["excess_cagr"], 信息比率=ir,
                   最大回撤=st["max_drawdown"], 分年超额=st["yearly_excess"], 正超额年数=ypos,
                   SMB_α占比=st["alpha_share"], SMB_β=st["smb_beta"], 国证2000差_α占比=a_gz, 国证2000差_β=coef[1],
                   剔除前1pct交易后超额=ex_trim, 剔除笔数=k, 买入笔数=int((res.trades.side == "buy").sum()))
        out["通过"] = bool(out["年化超额"] >= CRITERIA["excess"] and out["信息比率"] >= CRITERIA["ir"]
                         and out["最大回撤"] >= CRITERIA["max_dd"] and ypos >= CRITERIA["years_pos"]
                         and out["SMB_α占比"] >= CRITERIA["alpha_share"] and a_gz >= CRITERIA["alpha_share"])
        res_all[h] = out
        if not dry:
            res.nav.to_csv(ROOT / "reports" / f"final_{h}_nav.csv")
            res.trades.to_csv(ROOT / "reports" / f"final_{h}_trades.csv", index=False)
    if dry:
        print("eval 演练完成：", {h: (v["买入笔数"], v["起"], v["止"]) for h, v in res_all.items()}, flush=True)
        return
    meta = dict(截止日=cutoff, 实际最后交易日=res_all[MAIN]["止"], forward_start=git("rev-parse", "forward-start^{commit}"),
                运行时间=time.strftime("%Y-%m-%d %H:%M:%S"), 主策略=MAIN, 第二策略=SECOND, 对照=CONTROL, 通过标准=CRITERIA)
    (ROOT / "reports" / "FINAL_TEST_RESULT.json").write_text(json.dumps(dict(meta=meta, results=res_all), ensure_ascii=False,
                                                                          indent=1, default=str))
    write_md(meta, res_all)
    verdict = "通过" if res_all[MAIN]["通过"] else "未通过"
    print(f"主策略 {MAIN}：{verdict}", flush=True)


def write_md(meta, res):
    f = lambda x: f"{x:+.2%}" if isinstance(x, (int, float)) and x == x else str(x)  # noqa: E731
    m = res[MAIN]
    checks = [("扣费后相对中证全指年化超额 ≥ 5%", f(m["年化超额"]), m["年化超额"] >= CRITERIA["excess"]),
              ("年化信息比率 ≥ 0.5", f"{m['信息比率']:.2f}", m["信息比率"] >= CRITERIA["ir"]),
              ("最大回撤 ≤ 35%", f"{m['最大回撤']:.2%}", m["最大回撤"] >= CRITERIA["max_dd"]),
              ("2024/2025/2026 至少两个年度超额为正", "、".join(f"{y} {f(v)}" for y, v in m["分年超额"].items()), m["正超额年数"] >= CRITERIA["years_pos"]),
              ("SMB 回归 α 占比 ≥ 0.5", f"{m['SMB_α占比']:.2f}", m["SMB_α占比"] >= CRITERIA["alpha_share"]),
              ("国证 2000 − 中证全指 回归 α 占比 ≥ 0.5", f"{m['国证2000差_α占比']:.2f}", m["国证2000差_α占比"] >= CRITERIA["alpha_share"])]
    L = ["# 最终测试结果", "",
         f"测试区间：2024-01-01 至 {meta['截止日']}（实际最后交易日 {meta['实际最后交易日']}）。代码：forward-start = `{meta['forward_start']}`。运行时间 {meta['运行时间']}。", "",
         f"## 判定：主策略 {MAIN} **{'通过' if m['通过'] else '未通过'}**", "",
         "| 通过标准（主策略） | 结果 | 是否满足 |", "|---|---|---|"]
    L += [f"| {a} | {b} | {'是' if c else '否'} |" for a, b, c in checks]
    L += ["", "决策规则（事先固定）：" + ("通过 → 先做至少 3 个月前向模拟盘，之后由用户决定是否实盘。" if m["通过"] else
                                    "未通过 → 这条研究线到此为止，不修改策略重测，不再使用测试集。第二策略结果只作参考，不得替换主策略。"), "",
          "## 全部策略", "",
          "| 策略 | 角色 | 年化 | 年化超额 | 信息比率 | 最大回撤 | 分年超额 | SMB α 占比 | 国证2000差 α 占比 | 剔除前 1% 交易后超额 | 买入笔数 | 五项 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    role = {MAIN: "主策略", SECOND: "第二策略（参考）", CONTROL: "对照（参考）"}
    for h, v in res.items():
        L.append(f"| {h} | {role[h]} | {f(v['年化'])} | {f(v['年化超额'])} | {v['信息比率']:.2f} | {v['最大回撤']:.2%} | "
                 + "、".join(f"{y} {f(x)}" for y, x in v["分年超额"].items())
                 + f" | {v['SMB_α占比']:.2f} | {v['国证2000差_α占比']:.2f} | {f(v['剔除前1pct交易后超额'])}（{v['剔除笔数']} 笔） | {v['买入笔数']} | {'满足' if v['通过'] else '不满足'} |")
    (ROOT / "reports" / "FINAL_TEST_RESULT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "score", "eval"])
    ap.add_argument("hyp", nargs="?")
    ap.add_argument("--cutoff")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.dry_run:
        guard()
        if not a.cutoff:
            raise SystemExit("必须给出 --cutoff")
    {"prep": lambda: prep(a.dry_run, a.cutoff), "score": lambda: score(a.dry_run, a.cutoff, a.hyp),
     "eval": lambda: evaluate(a.dry_run, a.cutoff)}[a.step]()
