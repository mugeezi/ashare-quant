"""日线回测引擎（A 股，t 日收盘出信号、t+1 日开盘成交）。

一个交易日的处理顺序：
  开盘  1) 退市处理：持仓股票到达终止上市日 —— 换股吸收合并按比例转为存续公司股票；
           主动退市按现金选择权/要约价格兑现；强制退市（含无整理期的交易类退市）按 0 计。
        2) 卖出：上一日收盘时标记要卖的持仓，以开盘价 ×(1−滑点) 成交。停牌、开盘即跌停、
           （除权日保守口径）开盘一字封在跌幅限制附近 → 卖不出，顺延到下一个交易日。
        3) 买入：上一日收盘生成的候选，按排序依次尝试，直到持仓数达到目标。开盘即涨停、
           停牌、（除权日保守口径）开盘一字封在涨幅限制附近 → 买不进，换下一个候选。
           每只目标金额 = 上一日收盘净值 / 目标持仓数，按 100 股整手向下取整，不足一手不买。
  收盘  4) 估值：持仓按收盘价计市值（复权因子换算送转/分红后的股数）；停牌按最后收盘价，
           可选"压力估值"：停牌期间按同期中证全指涨跌折算。
        5) 当日候选排名：调用 strategy.select(t, view)；view 只含 t 日及以前的数据。
           引擎再统一过滤：当日正常交易、非 ST、非 S 股、不在退市整理期、上市满 60 个交易日、
           前 20 日平均成交额 ≥ 5000 万、尚未到退市日。
        6) 退出决策：持有满 hold_days 个交易日、进入退市整理期、或策略要求退出 → 次日开盘卖出。
           续持（renew=True）：到期时若该股仍在当日候选前 renew_k 名（滞后带），则不卖出，
           持有期从次日重新计算；换股而来的持仓不续持。
        7) 次日买入候选：候选前 buy_top 名（续持模式下默认 = n_target），依次尝试。

红利税（持有不足 1 个月按 20%）未单独扣除：复权因子把现金分红视为再投资，2–10 日持仓
跨除息日的情形很少，影响很小，报告中注明。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from costs import COMMISSION_RATE, commission, stamp_tax
from data import MarketData, View

TOL = 0.005          # 价格比较容差（元）：开盘价与涨跌停价相差不到半分即视为封板


@dataclass
class Config:
    start: str = "2014-01-01"
    end: str = "2023-12-31"
    init_cash: float = 200_000.0
    n_target: int = 10                  # 目标持仓数（任务要求 8–12）
    hold_days: int = 5                  # 持有交易日数（2–10）：第 t 日开盘买入，第 t+hold_days 日开盘卖出
    slippage: float = 0.001             # 单边滑点；压力测试用 0.002
    costs: bool = True                  # False：不计佣金、印花税、滑点（用于拆分成本）
    exright_conservative: bool = True   # 除权日保守口径
    suspension_stress: bool = False     # 停牌期间按指数折算估值
    max_candidates: int = 60            # 每日候选上限（买不进时依次替补）
    min_listed_days: int = 60
    min_amt20: float = 5e7
    filters: bool = True                # False 仅供路径检验：不做 ST/新股/流动性过滤
    renew: bool = False                 # 续持：到期时若仍在当日候选前 renew_k 名，则不卖出、重新计算持有期
    renew_k: int | None = None          # 滞后带宽度，默认 3 × n_target
    buy_top: int | None = None          # 新买入只从候选前 buy_top 名中选；默认 renew 时 = n_target，否则 = max_candidates

    def __post_init__(self):
        if self.renew_k is None:
            self.renew_k = 3 * self.n_target
        if self.buy_top is None:
            self.buy_top = self.n_target if self.renew else self.max_candidates


@dataclass
class Position:
    c: int
    shares: float           # 入场时的股数（整手）
    adj_entry: float        # 入场日复权因子；当前股数 = shares × adj_t / adj_entry
    entry_t: int
    entry_px: float
    cost: float             # 买入总支出（含佣金）
    pending: bool = False
    reason: str = ""
    last_value: float = 0.0
    last_trade_value: float = 0.0
    lock_idx: float = math.nan
    locked_days: int = 0
    susp_days: int = 0
    origin: str = ""        # 换股而来时记录原股票代码
    entry_code: str = ""
    hold_start: int = -1    # 当前持有期的起点（续持后更新为续持决定日的下一交易日）
    renewals: int = 0


class Backtest:
    def __init__(self, md: MarketData, strategy, cfg: Config):
        self.md, self.s, self.cfg = md, strategy, cfg
        a = md.a
        self.O, self.H, self.L, self.C = a["open"], a["high"], a["low"], a["close"]
        self.PC, self.UL, self.DL, self.LP = a["preclose"], a["up_limit"], a["down_limit"], a["limit_pct"]
        self.ADJ, self.TS, self.NL, self.XR = a["adj_factor"], a["tradestatus"], a["no_limit"], a["exright_day"]
        self.slip = cfg.slippage if cfg.costs else 0.0
        u = md.universe.reindex(md.codes)
        self.delist = pd.to_datetime(u["delist_date"]).values
        ex = md.exits.reindex(md.codes)
        self.exit_class = ex["exit_class"].values
        self.exit_ratio = ex["ratio"].values
        self.exit_cash = ex["cash"].values
        self.exit_target = ex["target_code"].values
        self.idx = md.index_close["zzqz"].ffill().values

    # ------------------------------------------------------------------ 规则判断
    def eligible(self, t: int) -> np.ndarray:
        a = self.md.a
        d = np.datetime64(self.md.dates[t])
        if not self.cfg.filters:
            return (a["tradestatus"][t] == 1) & ~(self.delist <= d)
        ok = ((a["tradestatus"][t] == 1) & (a["isST"][t] == 0) & (a["isS"][t] == 0) & (a["delist_period"][t] == 0)
              & (a["days_listed"][t] >= self.cfg.min_listed_days) & (np.nan_to_num(a["amt20"][t]) >= self.cfg.min_amt20))
        ok &= ~(self.delist <= d)
        return ok

    def buy_block(self, t, c) -> str:
        o = self.O[t, c]
        if self.TS[t, c] != 1 or not (o > 0):
            return "suspended"
        if self.cfg.filters and (self.md.a["isST"][t, c] == 1 or self.md.a["delist_period"][t, c] == 1):
            return "became_st"
        if self.NL[t, c] == 0:
            if o >= self.UL[t, c] - TOL:
                return "limit_up_open"
            if (self.cfg.exright_conservative and self.XR[t, c] == 1 and abs(o - self.H[t, c]) < 1e-6
                    and o / self.PC[t, c] - 1 >= self.LP[t, c] - 0.01):
                return "exright_conservative"
        return ""

    def sell_block(self, t, c) -> str:
        o = self.O[t, c]
        if self.TS[t, c] != 1 or not (o > 0):
            return "suspended"
        if self.NL[t, c] == 0:
            if o <= self.DL[t, c] + TOL:
                return "limit_down_open"
            if (self.cfg.exright_conservative and self.XR[t, c] == 1 and abs(o - self.L[t, c]) < 1e-6
                    and o / self.PC[t, c] - 1 <= -(self.LP[t, c] - 0.01)):
                return "exright_conservative"
        return ""

    def cur_shares(self, p: Position, t: int) -> float:
        adj = self.ADJ[t, p.c]
        if not (adj > 0) or not (p.adj_entry > 0):
            return p.shares
        return p.shares * adj / p.adj_entry

    # ------------------------------------------------------------------ 主循环
    def run(self):
        md, cfg = self.md, self.cfg
        dates = md.dates
        t0 = int(dates.searchsorted(pd.Timestamp(cfg.start)))
        t1 = int(dates.searchsorted(pd.Timestamp(cfg.end), side="right")) - 1
        cash, positions = cfg.init_cash, []
        nav_prev = cfg.init_cash
        cands = np.array([], dtype=int)
        cur_target = cfg.n_target          # 策略可通过 target_n(t, view) 动态降低持仓数（仓位控制），每只金额不变
        trades, rts, navs = [], [], []
        stat = {"buy_block": {}, "sell_block_days": {}, "insufficient_lot": 0, "buy_attempts": 0, "buys": 0,
                "exits": {}, "exright_extra_buy": 0,
                "exright_extra_sell": 0}

        def bump(dct, k, n=1):
            dct[k] = dct.get(k, 0) + n

        def close_rt(p, t, proceeds, how):
            rts.append(dict(code=p.entry_code or md.codes[p.c], entry_date=dates[p.entry_t], exit_date=dates[t],
                            hold_days=t - p.entry_t, cost=p.cost, proceeds=proceeds, pnl=proceeds - p.cost,
                            ret=proceeds / p.cost - 1 if p.cost > 0 else np.nan, how=how,
                            locked_days=p.locked_days, susp_days=p.susp_days, via=p.origin, renewals=p.renewals))

        for t in range(t0, t1 + 1):
            date = dates[t]
            d64 = np.datetime64(date)
            # 1) 退市处理
            for p in list(positions):
                dl = self.delist[p.c]
                if not pd.isna(dl) and d64 >= dl:
                    cls = self.exit_class[p.c]
                    sh = self.cur_shares(p, t - 1 if t > 0 else t)
                    if cls == "merger" and isinstance(self.exit_target[p.c], str):
                        tc = md.col(self.exit_target[p.c])
                        npos = Position(c=tc, shares=sh * self.exit_ratio[p.c], adj_entry=math.nan, entry_t=p.entry_t,
                                        entry_px=p.entry_px, cost=p.cost, pending=True, reason="merger_convert",
                                        last_value=p.last_value, last_trade_value=p.last_value,
                                        locked_days=p.locked_days, susp_days=p.susp_days,
                                        origin=md.codes[p.c], entry_code=md.codes[p.c])
                        positions.remove(p)
                        positions.append(npos)
                        bump(stat["exits"], "merger_convert")
                    elif cls == "voluntary" and self.exit_cash[p.c] > 0:
                        proceeds = sh * float(self.exit_cash[p.c])
                        cash += proceeds
                        trades.append(dict(date=date, code=md.codes[p.c], side="cash_option", price=float(self.exit_cash[p.c]),
                                           shares=sh, amount=proceeds, commission=0.0, stamp=0.0, reason="voluntary_delist"))
                        close_rt(p, t, proceeds, "cash_option")
                        positions.remove(p)
                        bump(stat["exits"], "cash_option")
                    else:
                        trades.append(dict(date=date, code=md.codes[p.c], side="writeoff", price=0.0, shares=sh,
                                           amount=0.0, commission=0.0, stamp=0.0, reason=f"delisted_{cls}"))
                        close_rt(p, t, 0.0, f"writeoff_{cls}")
                        positions.remove(p)
                        bump(stat["exits"], f"writeoff_{cls}")
            # 2) 卖出
            for p in list(positions):
                if not p.pending:
                    continue
                why = self.sell_block(t, p.c)
                if why:
                    bump(stat["sell_block_days"], why)
                    if why == "exright_conservative":
                        stat["exright_extra_sell"] += 1
                    p.locked_days += 1
                    if why == "suspended":
                        p.susp_days += 1
                    continue
                if not (p.adj_entry > 0):                 # 换股而来、首次可交易：以今日因子为基准
                    p.adj_entry = self.ADJ[t, p.c] if self.ADJ[t, p.c] > 0 else 1.0
                sh = self.cur_shares(p, t)
                px = self.O[t, p.c] * (1 - self.slip)
                amt = sh * px
                com = commission(amt) if cfg.costs else 0.0
                stp = stamp_tax(amt, date) if cfg.costs else 0.0
                proceeds = amt - com - stp
                cash += proceeds
                trades.append(dict(date=date, code=md.codes[p.c], side="sell", price=px, shares=sh, amount=amt,
                                   commission=com, stamp=stp, reason=p.reason))
                close_rt(p, t, proceeds, p.reason)
                positions.remove(p)
            # 3) 买入
            slots = cur_target - len(positions)
            if slots > 0 and len(cands):
                alloc = nav_prev / cfg.n_target
                held = {p.c for p in positions}
                for c in cands:
                    if slots <= 0:
                        break
                    if c in held:
                        continue
                    stat["buy_attempts"] += 1
                    why = self.buy_block(t, c)
                    if why:
                        bump(stat["buy_block"], why)
                        if why == "exright_conservative":
                            stat["exright_extra_buy"] += 1
                        continue
                    px = self.O[t, c] * (1 + self.slip)
                    budget = min(alloc, cash)
                    rate = COMMISSION_RATE if cfg.costs else 0.0
                    shares = math.floor(budget / (px * (1 + rate)) / 100) * 100
                    amt = shares * px
                    com = commission(amt) if cfg.costs else 0.0
                    if shares < 100 or amt + com > cash:
                        stat["insufficient_lot"] += 1
                        continue
                    cash -= amt + com
                    positions.append(Position(c=c, shares=shares, adj_entry=self.ADJ[t, c], entry_t=t, entry_px=px,
                                              cost=amt + com, entry_code=md.codes[c], hold_start=t))
                    trades.append(dict(date=date, code=md.codes[c], side="buy", price=px, shares=shares, amount=amt,
                                       commission=com, stamp=0.0, reason=""))
                    stat["buys"] += 1
                    held.add(c)
                    slots -= 1
            # 4) 收盘估值
            mv = 0.0
            for p in positions:
                c = p.c
                px = self.C[t, c]
                if px > 0 and p.adj_entry > 0:
                    v = self.cur_shares(p, t) * px
                elif px > 0 and not (p.adj_entry > 0):   # 换股后存续方已有行情但尚未卖出
                    v = p.shares * px
                else:
                    v = p.last_value                    # 无行情（新代码未上市、已摘牌）沿用上一估值
                susp = self.TS[t, c] != 1
                if susp:
                    if math.isnan(p.lock_idx):
                        p.lock_idx = self.idx[t - 1] if t > 0 else self.idx[t]
                        p.last_trade_value = p.last_value if p.last_value > 0 else v
                    if cfg.suspension_stress and self.idx[t] > 0 and p.lock_idx > 0:
                        v = p.last_trade_value * self.idx[t] / p.lock_idx
                else:
                    p.lock_idx = math.nan
                p.last_value = v
                mv += v
            nav = cash + mv
            navs.append((date, nav, cash, len(positions), sum(p.pending for p in positions)))
            nav_prev = nav
            # 5) 当日候选排名（收盘后，只用 t 日及以前数据）
            ranked = np.array([], dtype=int)
            if t < t1:
                raw = np.asarray(self.s.select(t, View(md, t)), dtype=int)
                if len(raw):
                    ok = self.eligible(t)
                    ranked = raw[ok[raw]]
            # 6) 退出决策
            dp = self.md.a["delist_period"]
            topk = set(ranked[: cfg.renew_k].tolist()) if cfg.renew else set()
            if hasattr(self.s, "target_n"):
                cur_target = int(min(cfg.n_target, max(0, self.s.target_n(t, View(md, t)))))
            n_keep = sum(1 for p in positions if not p.pending)
            for p in positions:
                if p.pending:
                    continue
                start = p.hold_start if p.hold_start >= 0 else p.entry_t
                if dp[t, p.c] == 1:
                    p.pending, p.reason = True, "delist_period"
                elif t - start >= cfg.hold_days - 1:      # 开盘买入、第 hold_days 个交易日后开盘卖出
                    if cfg.renew and p.c in topk and not p.origin and n_keep <= cur_target:
                        p.hold_start, p.renewals = t + 1, p.renewals + 1
                        stat["renewals"] = stat.get("renewals", 0) + 1
                    else:
                        p.pending, p.reason = True, "hold_days"
                        n_keep -= 1
                elif hasattr(self.s, "should_exit") and self.s.should_exit(t, View(md, t), p):
                    p.pending, p.reason = True, "strategy_exit"
            # 7) 次日买入候选
            cands = ranked[: cfg.buy_top]
        nav = pd.DataFrame(navs, columns=["date", "nav", "cash", "n_pos", "n_pending"]).set_index("date")
        return Result(nav=nav, trades=pd.DataFrame(trades), roundtrips=pd.DataFrame(rts), stat=stat, cfg=cfg, md=md)


@dataclass
class Result:
    nav: pd.DataFrame
    trades: pd.DataFrame
    roundtrips: pd.DataFrame
    stat: dict
    cfg: Config
    md: MarketData
