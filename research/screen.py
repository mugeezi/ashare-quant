"""快速筛选器：在训练集上用向量化方式近似引擎，几秒钟评估一个打分矩阵，用来大范围试错。

近似方式：H 个错开的子组合，子组合 k 在 t ≡ k (mod H) 的日子收盘按打分取前 N 只
（可入选、次日开盘非涨停、次日交易），t+1 开盘买，t+1+H 开盘卖（卖出日停牌则顺延到下一个可交易开盘）。
成本：每次调仓按"换掉的股票比例"计 —— 买入滑点 0.1% + 佣金 0.025%，卖出滑点 0.1% + 佣金 0.025% + 印花税
（2023-08-28 前 0.1%，之后 0.05%）。若新组合与上一期重合（相当于续持），重合部分不计成本。
没有模拟：开盘跌停卖不出、整手、资金不足、最低佣金。因此比引擎略乐观，入围者必须再用引擎正式回测。

每次评估都写入 experiments.csv（split = train，variant 前缀 "screen:"），计入试验次数。
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from framework import DAYS, Lab, eligible_matrix, hfq

STAMP_CUT = pd.Timestamp("2023-08-28")


class Screener:
    def __init__(self, lab: Lab, start="2014-01-01", end="2020-12-31"):
        self.lab, self.md = lab, lab.md
        md = self.md
        a = md.a
        self.el = eligible_matrix(md)
        T, N = a["close"].shape
        buy = np.zeros((T, N), bool)
        buy[:-1] = (a["tradestatus"][1:] == 1) & ((a["no_limit"][1:] == 1) | (a["open"][1:] < a["up_limit"][1:] - 0.005))
        self.buyable = buy
        O = hfq(md, "open")
        ts = a["tradestatus"]
        nxt = np.full((T, N), np.nan)
        last = np.full(N, np.nan)
        for i in range(T - 1, -1, -1):
            ok = (ts[i] == 1) & np.isfinite(O[i])
            last = np.where(ok, O[i], last)
            nxt[i] = last
        self.O, self.nxt = O, nxt
        self.t0 = int(md.dates.searchsorted(pd.Timestamp(start)))
        self.t1 = int(md.dates.searchsorted(pd.Timestamp(end), side="right")) - 1
        self.bench = md.index_close["zzqz"]

    def evaluate(self, score: np.ndarray, H: int = 10, n: int = 10, start=None):
        md = self.md
        t0 = self.t0 if start is None else int(md.dates.searchsorted(pd.Timestamp(start)))
        sub_nav = np.ones(H)
        prev = [set() for _ in range(H)]
        navs, n_hold, turn = [], [], []
        for t in range(t0 - 1, self.t1 - H):
            k = (t - t0 + 1) % H
            row = score[t]
            ok = self.el[t] & self.buyable[t] & np.isfinite(row)
            idx = np.where(ok)[0]
            if len(idx):
                sel = idx[np.argsort(-row[idx], kind="stable")[:n]]
            else:
                sel = np.array([], dtype=int)
            r = self.nxt[t + 1 + H, sel] / self.O[t + 1, sel] - 1 if len(sel) else np.array([])
            r = np.where(np.isfinite(r), r, -1.0)                      # 缺价（退市）按 −100%
            w = len(sel) / n                                           # 不足 n 只的部分空仓
            gross = r.mean() * w if len(sel) else 0.0
            new = set(sel.tolist())
            changed = len(new - prev[k]) / n
            stamp = 0.001 if md.dates[t] < STAMP_CUT else 0.0005
            cost = changed * (0.00125 + 0.00125 + stamp) + (len(prev[k] - new) / n) * 0.0   # 卖出成本已含在往返里
            prev[k] = new
            sub_nav[k] *= (1 + gross - cost)
            navs.append((md.dates[t + 1 + H], sub_nav.mean()))   # 记在该子组合收益实现的日期
            n_hold.append(len(sel))
            turn.append(changed)
        nav = pd.Series(dict(navs))
        r = nav.pct_change().dropna()
        b = self.bench.reindex(nav.index).ffill().pct_change().reindex(r.index).fillna(0)
        e = r - b
        years = len(r) / DAYS
        ex_cagr = float(np.prod(1 + e) ** (1 / years) - 1)
        cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1)
        yr = {int(y): round(float(np.prod(1 + r[i]) - np.prod(1 + b[i])), 3) for y, i in r.groupby(r.index.year).groups.items()}
        dd = float((nav / nav.cummax() - 1).min())
        smb = self.lab.smb.reindex(e.index)
        m = smb.notna()
        X = np.column_stack([np.ones(m.sum()), smb[m].values])
        coef, *_ = np.linalg.lstsq(X, e[m].values, rcond=None)
        return dict(cagr=cagr, excess_cagr=ex_cagr, max_drawdown=dd, yearly=yr,
                    years_pos=sum(v > 0 for v in yr.values()), n_years=len(yr),
                    alpha=float(coef[0] * DAYS), beta=float(coef[1]), sharpe_daily=float(e.mean() / e.std()),
                    avg_hold=float(np.mean(n_hold)), turnover=float(np.mean(turn) * DAYS / H))

    def run(self, hyp, variant, score, H=10, n=10, notes="", start=None):
        t = time.time()
        st = self.evaluate(score, H=H, n=n, start=start)
        self.lab.log_trial(hyp, f"screen:{variant}", "train", {"H": H, "n": n, "start": start or "2014-01-01"}, notes,
                           cagr=round(st["cagr"], 4), excess_cagr=round(st["excess_cagr"], 4),
                           sharpe_daily=round(st["sharpe_daily"], 5), max_drawdown=round(st["max_drawdown"], 4),
                           turnover_annual=round(st["turnover"], 1),
                           years_excess_pos=f"{st['years_pos']}/{st['n_years']}", smb_alpha=round(st["alpha"], 4),
                           smb_beta=round(st["beta"], 3), passed=None)
        print(f"{hyp}/{variant:<14} 超额 {st['excess_cagr']:+7.1%}  年化 {st['cagr']:+7.1%}  回撤 {st['max_drawdown']:7.1%}  "
              f"α {st['alpha']:+6.1%} β {st['beta']:+.2f}  正年 {st['years_pos']}/{st['n_years']}  持仓 {st['avg_hold']:4.1f}  "
              f"换手 {st['turnover']:4.1f}  {json.dumps(st['yearly'])}  ({time.time() - t:.0f}s)", flush=True)
        return st
