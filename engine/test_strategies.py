"""检验用策略（不是研究策略）。select(t, view) 返回按优先级排序的股票列号；引擎再做统一过滤。"""
import numpy as np


class RandomPick:
    """随机选股：每天把当日可交易股票随机排序。"""
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def select(self, t, v):
        cand = np.where(v.tradestatus[-1] == 1)[0]
        return self.rng.permutation(cand)


class FutureCheat:
    """故意使用未来数据：按 t+1 开盘到 t+1+h 开盘的后复权收益排序（直接读完整数据，绕过 View）。"""
    def __init__(self, md, h=5):
        self.md, self.h = md, h

    def select(self, t, v):
        a = self.md.a
        T = self.md.T
        if t + 1 + self.h >= T:
            return []
        o1 = a["open"][t + 1] * a["adj_factor"][t + 1]
        o2 = a["open"][t + 1 + self.h] * a["adj_factor"][t + 1 + self.h]
        r = o2 / o1 - 1
        r = np.where(np.isfinite(r), r, -np.inf)
        return np.argsort(-r)[:200]


class LimitUpChase:
    """检验 A：t 日收盘涨停（非新股期）的股票，t+1 开盘买入。"""
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def select(self, t, v):
        c, ul, nl, ts = v.close[-1], v.up_limit[-1], v.no_limit[-1], v.tradestatus[-1]
        hit = np.where((ts == 1) & (nl == 0) & (c >= ul - 0.005))[0]
        return self.rng.permutation(hit)


class LimitUpOpenCheat:
    """检验 B：偷看 t+1 开盘价，只挑 t+1 开盘即涨停的股票。引擎应一只都买不进。"""
    def __init__(self, md):
        self.md = md

    def select(self, t, v):
        a = self.md.a
        if t + 1 >= self.md.T:
            return []
        o, ul, nl, ts = a["open"][t + 1], a["up_limit"][t + 1], a["no_limit"][t + 1], a["tradestatus"][t + 1]
        return np.where((ts == 1) & (nl == 0) & (o >= ul - 0.005))[0]


class SuspendCheat:
    """路径检验：偷看未来，挑 t+2 起停牌的股票（t+1 开盘买入后即停牌），检验停牌锁定。"""
    def __init__(self, md):
        self.md = md

    def select(self, t, v):
        a = self.md.a
        if t + 2 >= self.md.T:
            return []
        return np.where((a["tradestatus"][t + 1] == 1) & (a["tradestatus"][t + 2] == 0))[0]


class DelistCheat:
    """路径检验：偷看未来，挑 3–30 个交易日内退市的股票，检验退市/换股/现金选择权处理。
    引擎的统一过滤会剔除 ST 和整理期股票，所以这里关掉过滤（见 run_checks.py 中的配置）。"""
    def __init__(self, md):
        self.md = md
        self.dl = md.universe.reindex(md.codes)["delist_date"].values

    def select(self, t, v):
        d = self.md.dates
        if t + 30 >= self.md.T:
            return []
        lo, hi = np.datetime64(d[t + 3]), np.datetime64(d[t + 30])
        return np.where((self.dl >= lo) & (self.dl <= hi) & (self.md.a["tradestatus"][t] == 1))[0]
