"""第二轮研究公共模块：运行登记、分档门槛、集中度诊断、暂停检查。

- 试验登记沿用 experiments.csv（round2 分支内的副本，试验号从 938 继续累加），字段与第一轮相同。
- 验证集：同一假设同一变体训练集达标才能运行；每个假设只用一次验证集。
- 分档门槛（hypotheses.md 第二轮登记）：
    进攻档 训练：超额 ≥ 5%、最大回撤 ≤ 35%、正超额年份 ≥ 4/7 折算、SMB α 占比 ≥ 0.5
           验证：超额 ≥ 8%、最大回撤 ≤ 35%、正超额年份 ≥ 2/3、SMB α 占比 ≥ 0.5；紧缩夏普只报告
    稳健档 训练：年化 ≥ 8%、最大回撤 ≤ 15%
           验证：年化 ≥ 8%、最大回撤 ≤ 15%；紧缩夏普只报告
- 最终测试优先：工作目录下出现 PAUSED_FOR_FINAL_TEST 时拒绝运行。
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

R2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R2 / "research"))
sys.path.insert(0, str(R2 / "research2"))
import framework as F  # noqa: E402
from framework import DAYS, Backtest, Config, excess_stats, read_exp, append_exp, deflated_sharpe  # noqa: E402

assert F.ROOT == R2, "必须在 round2 工作目录下运行"
OUT = R2 / "reports" / "round2"
SCORES = F.CACHE.parent
SPLITS = {"train": ("2014-01-01", "2020-12-31"), "valid": ("2021-01-01", "2023-12-31")}
END_ALLOWED = pd.Timestamp("2023-12-31")


def check_pause():
    if (R2 / "PAUSED_FOR_FINAL_TEST").exists():
        raise SystemExit("最终测试进行中，第二轮实验暂停")


def register_pid(name: str):
    (R2 / "logs").mkdir(exist_ok=True)
    (R2 / "logs" / f"{name}.pid").write_text(str(os.getpgid(0)))


def gate(tier: str, split: str, st: dict, ypos: int, nyears: int) -> bool:
    if tier == "进攻":
        if split == "train":
            need = math.ceil(4 / 7 * nyears - 1e-9)
            return (st["excess_cagr"] >= 0.05 and st["max_drawdown"] >= -0.35 and ypos >= need
                    and (st["alpha_share"] or 0) >= 0.5)
        return (st["excess_cagr"] >= 0.08 and st["max_drawdown"] >= -0.35 and ypos >= 2
                and (st["alpha_share"] or 0) >= 0.5)
    if tier == "稳健":
        return st["cagr"] >= 0.08 and st["max_drawdown"] >= -0.15
    raise ValueError(tier)


def yearly(nav: pd.Series) -> dict:
    r = nav.pct_change().dropna()
    return {int(y): float(np.prod(1 + g) - 1) for y, g in r.groupby(r.index.year)}


def nav_stats(nav: pd.Series, md_index: pd.DataFrame, smb: pd.Series) -> dict:
    st = excess_stats(nav, md_index["zzqz"], smb)
    hs = md_index["hs300"].reindex(nav.index).ffill()
    r = nav.pct_change().dropna()
    st["ann_vol"] = float(r.std() * math.sqrt(DAYS))
    st["calmar"] = st["cagr"] / abs(st["max_drawdown"]) if st["max_drawdown"] < 0 else np.nan
    st["yearly_ret"] = yearly(nav)
    years = len(r) / DAYS
    st["hs300_cagr"] = float((hs.iloc[-1] / hs.iloc[0]) ** (1 / years) - 1)
    st["zzqz_cagr"] = float((md_index["zzqz"].reindex(nav.index).ffill().iloc[-1]
                             / md_index["zzqz"].reindex(nav.index).ffill().iloc[0]) ** (1 / years) - 1)
    eh = r - hs.pct_change().reindex(r.index).fillna(0)
    st["excess_hs300"] = float(np.prod(1 + eh) ** (1 / years) - 1)
    return st


def log_run(hyp, variant, split, tier, nav, trades, md_index, smb, params, notes, leak, n_target=None, hold_days=None,
            exp=None):
    """统一登记一次回测（训练集或验证集）。返回 (row, stats)。"""
    exp = read_exp() if exp is None else exp
    if nav.index.max() > END_ALLOWED:
        raise PermissionError("第二轮不得使用 2024-01-01 以后的数据")
    if split == "valid":
        tr = exp[(exp.hyp_id == hyp) & (exp.variant == variant) & (exp.split == "train")]
        if not len(tr) or not tr.passed.astype(str).str.lower().eq("true").any():
            raise PermissionError(f"{hyp}/{variant} 训练集未达标，不能运行验证集")
        if ((exp.hyp_id == hyp) & (exp.split == "valid")).any():
            raise PermissionError(f"{hyp} 已用过验证集")
    st = nav_stats(nav, md_index, smb)
    ypos = sum(v > 0 for v in st["yearly_excess"].values())
    passed = gate(tier, split, st, ypos, len(st["yearly_excess"]))
    years = len(nav) / DAYS
    if trades is not None and len(trades):
        amt = trades.loc[trades.side.isin(["buy", "sell"]), "amount"].sum()
        turnover = amt / 2 / nav.mean() / years
        n_buys = int((trades.side == "buy").sum())
    else:
        turnover, n_buys = 0.0, 0
    note = notes
    if split == "valid":
        srs = exp["sharpe_daily"].dropna().astype(float).tolist() + [st["sharpe_daily"]]
        dsr, sr0 = deflated_sharpe(st["sharpe_daily"], st["n_days"], st["skew"], st["kurt"], srs, n_trials=len(exp) + 1)
        st["dsr"] = dsr
        note = (note + f"; DSR={dsr:.3f}（只报告）, SR0={sr0:.4f}, N={len(exp) + 1}").strip("; ")
    row = dict(exp_id=f"E{len(exp) + 1:04d}", hyp_id=hyp, variant=variant, split=split,
               run_time=time.strftime("%Y-%m-%d %H:%M"), params=json.dumps(dict(params, tier=tier), ensure_ascii=False),
               hold_days=hold_days, n_target=n_target, cagr=round(st["cagr"], 4), excess_cagr=round(st["excess_cagr"], 4),
               sharpe_daily=round(st["sharpe_daily"], 5), max_drawdown=round(st["max_drawdown"], 4),
               turnover_annual=round(turnover, 2), n_buys=n_buys, years_excess_pos=f"{ypos}/{len(st['yearly_excess'])}",
               smb_alpha=round(st["smb_alpha"], 4), smb_beta=round(st["smb_beta"], 3),
               alpha_share=round(st["alpha_share"], 3) if st["alpha_share"] == st["alpha_share"] else None,
               leak_test=leak, passed=bool(passed), trial_no=len(exp) + 1,
               valid_use_no=int(((exp.hyp_id == hyp) & (exp.split == "valid")).sum()) + (split == "valid"),
               notes=("第二轮；" + note).strip("；"))
    append_exp(row)
    OUT.mkdir(parents=True, exist_ok=True)
    nav.to_frame("nav").to_csv(OUT / f"{split}_{hyp}_{variant}_nav.csv")
    if trades is not None and len(trades):
        trades.to_csv(OUT / f"{split}_{hyp}_{variant}_trades.csv", index=False)
    return row, st


def concentration(md, rt: pd.DataFrame, nav: pd.Series, bench: pd.Series, smb: pd.Series, fmv_pct=None) -> dict:
    """收益集中度：每年前 5 笔盈利占比、剔除收益最高 1% 交易后的超额、单票盈亏占总盈亏最大比例、持仓市值分位。"""
    out = {}
    rt = rt.copy()
    rt["year"] = pd.to_datetime(rt.exit_date).dt.year
    yrs = {}
    for y, g in rt.groupby("year"):
        tot = g.pnl.sum()
        yrs[int(y)] = round(float(g.pnl.nlargest(5).sum() / tot), 3) if tot > 0 else None
    out["前5笔盈利占全年盈亏"] = yrs
    k = max(1, int(np.ceil(len(rt) * 0.01)))
    top = rt.nlargest(k, "ret")
    adj = pd.Series(0.0, index=nav.index)
    for _, x in top.iterrows():
        adj[adj.index >= pd.Timestamp(x.exit_date)] += x.pnl
    st2 = excess_stats(nav - adj, bench, smb)
    out["剔除前1%交易笔数"] = k
    out["剔除前1%后超额"] = round(st2["excess_cagr"], 4)
    out["剔除前1%后年化"] = round(st2["cagr"], 4)
    by = rt.groupby("code").pnl.sum()
    tot = rt.pnl.sum()
    out["单票最大盈亏占总盈亏"] = round(float(by.max() / tot), 3) if tot > 0 else None
    out["交易笔数"] = int(len(rt))
    out["胜率"] = round(float((rt.pnl > 0).mean()), 3)
    if fmv_pct is not None:
        col = {c: i for i, c in enumerate(md.codes)}
        vals = []
        for _, x in rt.iterrows():
            j = col.get(x.code)
            if j is None:
                continue
            t0, t1 = md.dates.get_loc(x.entry_date), md.dates.get_loc(x.exit_date)
            vals.extend(fmv_pct[t0:max(t1, t0 + 1), j].tolist())
        out["持仓市值分位均值"] = round(float(np.nanmean(vals)), 3)
    return out


def fmv_percentile(md) -> np.ndarray:
    fmv = np.where((md.a["tradestatus"] == 1) & np.isfinite(md.a["fmv"]), md.a["fmv"], np.nan)
    return pd.DataFrame(fmv).rank(axis=1, pct=True).values.astype(np.float32)


def save_json(name: str, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str))
