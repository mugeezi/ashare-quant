"""由复权因子跳变推算每股现金分红（第二轮 W02 用；不使用业绩报表）。

数据来源：研究期行情（data/raw、md.pkl，2023-12-29 截止）中的复权因子、收盘价与流通股本（成交量 ÷ 换手率）。
原计划用新浪"分红送配"页面作为分红记录核对，但该页面对本环境返回空内容（HTTP 200、0 字节），因此改为：
  1) 除权除息日 = 复权因子发生变化的交易日 t；
  2) 除权比 R = adj_t / adj_{t-1}，交易所口径下 R = P / [(P − D) / (1 + s)]，P 为前一有效收盘价，D 为每股现金分红，s 为每股送转股；
  3) 送转比例 s 由流通股本在除权日前后的变化估计（前后各 5 个交易日中位数之比），取最近的 0.1 的整数倍，变化小于 5% 视为无送转；
  4) D = P × (1 − (1 + s) / R)；D < 0（配股等）记为 0 并标记，D / P > 20% 标记为可疑并剔除；
  5) 只用除权除息日当日及以前的数据：分红在除息日计入，此时方案早已公告。
股息率（TTM）= 过去 250 个交易日内各次现金分红折算到当前股本后的合计 ÷ 当日收盘价，
即 Σ d_k × hfq_{k−1} / hfq_t，其中 d_k = D_k / P_k 为该次分红相对除息前收盘价的比例。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def events(md) -> pd.DataFrame:
    a = md.a
    adj, close, fs = a["adj_factor"], a["close"], a["fshares"]
    T, N = adj.shape
    rows = []
    for j in range(N):
        f = adj[:, j]
        ok = np.isfinite(f) & (f > 0)
        idx = np.where(ok)[0]
        if len(idx) < 2:
            continue
        fv = f[idx]
        chg = np.where(np.abs(fv[1:] / fv[:-1] - 1) > 1e-6)[0] + 1
        for k in chg:
            t, tp = idx[k], idx[k - 1]
            R = fv[k] / fv[k - 1]
            P = close[tp, j]
            if not (np.isfinite(P) and P > 0) or R <= 1:
                rows.append(dict(t=t, j=j, R=R, P=P, s=np.nan, D=0.0, flag="R<=1或无收盘价"))
                continue
            pre = fs[max(0, tp - 4):tp + 1, j]
            post = fs[t:t + 5, j]
            pre, post = pre[np.isfinite(pre) & (pre > 0)], post[np.isfinite(post) & (post > 0)]
            q = np.median(post) / np.median(pre) if len(pre) and len(post) else 1.0
            s = round((q - 1) / 0.1) * 0.1 if abs(q - 1) >= 0.05 else 0.0
            s = max(s, 0.0)
            D = P * (1 - (1 + s) / R)
            flag = ""
            if D < 0:
                # 送转估计偏大：在 0.1 网格上找使 D 非负且最接近 q 的 s
                cands = [x / 10 for x in range(0, 31) if P * (1 - (1 + x / 10) / R) >= 0]
                if cands:
                    s = min(cands, key=lambda x: abs(1 + x - q))
                    D = P * (1 - (1 + s) / R)
                    flag = "s修正"
                else:
                    D, flag = 0.0, "D<0(配股等)"
            if D / P > 0.20:
                D, flag = 0.0, "D/P>20%剔除"
            rows.append(dict(t=t, j=j, R=R, P=P, q=q, s=s, D=D, flag=flag))
    ev = pd.DataFrame(rows)
    ev["date"] = md.dates[ev.t.values]
    ev["code"] = np.asarray(md.codes)[ev.j.values]
    ev["d"] = ev.D / ev.P
    return ev


def ttm_yield(md, ev: pd.DataFrame, window: int = 250) -> np.ndarray:
    """T×N 股息率矩阵（只用当日及以前的数据）。"""
    T, N = md.a["close"].shape
    hfq = md.a["close"] * md.a["adj_factor"]
    hfq = pd.DataFrame(hfq).ffill().values
    acc = np.zeros((T, N))                      # 每个除息日在 t 加入 d_k × hfq_{k−1}
    e = ev[ev.D > 0]
    for t, j, d in zip(e.t.values, e.j.values, e.d.values):
        tp = t - 1
        while tp >= 0 and not np.isfinite(hfq[tp, j]):
            tp -= 1
        if tp >= 0:
            acc[t, j] += d * hfq[tp, j]
    cs = np.cumsum(acc, axis=0)
    roll = cs - np.vstack([np.zeros((window, N)), cs[:-window]])
    with np.errstate(invalid="ignore", divide="ignore"):
        y = roll / hfq
    y[~np.isfinite(y)] = np.nan
    return y.astype(np.float32)
