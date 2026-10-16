"""引擎单元检验：在人工构造的小数据上逐条验证交易规则。运行：python unit_tests.py"""
import math

import numpy as np
import pandas as pd

from backtest import Backtest, Config
from costs import commission, stamp_tax
from data import MarketData, View


def make_md(T=30, N=4, price=10.0):
    dates = pd.bdate_range("2019-01-02", periods=T)
    codes = np.array([f"sz.00000{i+1}" for i in range(N)])
    f = lambda v: np.full((T, N), v, dtype=float)
    a = {"open": f(price), "high": f(price), "low": f(price), "close": f(price), "preclose": f(price),
         "amount": f(1e8), "volume": f(1e7), "turn": f(1.0), "adj_factor": f(1.0), "limit_pct": f(0.10),
         "up_limit": f(round(price * 1.1, 2)), "down_limit": f(round(price * 0.9, 2)), "amt20": f(1e8)}
    for c in ["tradestatus"]:
        a[c] = np.ones((T, N), dtype=np.int8)
    for c in ["isST", "isS", "no_limit", "delist_period", "exright_day"]:
        a[c] = np.zeros((T, N), dtype=np.int8)
    a["days_listed"] = np.full((T, N), 1000, dtype=np.int32)
    md = MarketData(dates=dates, codes=codes, a=a)
    md.universe = pd.DataFrame({"delist_date": pd.NaT}, index=codes)
    md.exits = pd.DataFrame({"exit_class": None, "ratio": np.nan, "cash": np.nan, "target_code": None}, index=codes)
    md.index_close = pd.DataFrame({"zzqz": 1000.0, "zz1000": 1000.0, "hs300": 1000.0, "zz500": 1000.0}, index=dates)
    return md


class Fixed:
    """每天都把固定的候选（按顺序）交给引擎。"""
    def __init__(self, order, start_t=0, once=False):
        self.order, self.start_t, self.once, self.done = order, start_t, once, False

    def select(self, t, v):
        if t < self.start_t or (self.once and self.done):
            return []
        self.done = True
        return self.order


def cfg(md, **kw):
    base = dict(start=str(md.dates[0].date()), end=str(md.dates[-1].date()), n_target=1, hold_days=3,
                init_cash=100_000, min_listed_days=60, min_amt20=5e7)
    base.update(kw)
    return Config(**base)


RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(("通过 " if cond else "失败 ") + name + (f"：{detail}" if detail else ""))


def main():
    # U1 成本
    check("佣金最低 5 元", commission(10_000) == 5.0 and abs(commission(100_000) - 25.0) < 1e-9)
    check("印花税 2023-08-28 切换", stamp_tax(10_000, pd.Timestamp("2023-08-25")) == 10.0
          and stamp_tax(10_000, pd.Timestamp("2023-08-28")) == 5.0)

    # U2 T+1 开盘成交、持有天数、整手
    md = make_md()
    r = Backtest(md, Fixed([0], once=True), cfg(md, hold_days=3)).run()
    tr = r.trades
    b, s = tr[tr.side == "buy"].iloc[0], tr[tr.side == "sell"].iloc[0]
    bi, si = md.dates.get_loc(b.date), md.dates.get_loc(s.date)
    check("t 日信号、t+1 开盘买入", bi == 1, f"买入日序号 {bi}")
    check("持有 hold_days 个交易日后开盘卖出", si - bi == 3, f"买入 {bi} 卖出 {si}")
    check("按 100 股整手", b.shares % 100 == 0, f"{b.shares} 股")
    check("买入价含滑点", abs(b.price - 10 * 1.001) < 1e-9)

    # U3 资金不足一手不买
    md = make_md(price=3000.0)
    r = Backtest(md, Fixed([0], once=True), cfg(md, init_cash=200_000, n_target=10)).run()
    check("目标金额不足一手不买入", (r.trades.side == "buy").sum() == 0 if len(r.trades) else True,
          f"不足一手次数 {r.stat['insufficient_lot']}")

    # U4 开盘涨停买不进，换下一个候选
    md = make_md()
    md.a["open"][1, 0] = md.a["up_limit"][1, 0]
    r = Backtest(md, Fixed([0, 1], once=True), cfg(md)).run()
    first = r.trades[r.trades.side == "buy"].iloc[0]
    check("开盘涨停买不进并替补下一只", first.code == md.codes[1] and r.stat["buy_block"].get("limit_up_open") == 1)

    # U5 开盘跌停卖不出，顺延
    md = make_md()
    md.a["open"][4, 0] = md.a["down_limit"][4, 0]
    r = Backtest(md, Fixed([0], once=True), cfg(md)).run()
    s = r.trades[r.trades.side == "sell"].iloc[0]
    check("开盘跌停卖不出、次日卖出", md.dates.get_loc(s.date) == 5, f"卖出日序号 {md.dates.get_loc(s.date)}")

    # U6 停牌锁定：持仓第 3 天起停牌 4 天，复牌首日一字跌停，再下一日卖出
    md = make_md()
    md.a["tradestatus"][3:7, 0] = 0
    md.a["open"][7, 0] = md.a["low"][7, 0] = md.a["high"][7, 0] = md.a["close"][7, 0] = md.a["down_limit"][7, 0]
    r = Backtest(md, Fixed([0], once=True), cfg(md, hold_days=2)).run()
    s = r.trades[r.trades.side == "sell"].iloc[0]
    rt = r.roundtrips.iloc[0]
    check("停牌期间卖不出，复牌一字跌停继续顺延", md.dates.get_loc(s.date) == 8,
          f"卖出日序号 {md.dates.get_loc(s.date)}，锁定 {rt.locked_days} 天（停牌 {rt.susp_days} 天）")

    # U7 除权日保守口径：开盘 = 最高价、涨幅 ≥ 涨跌幅限制 − 1 个百分点，即使低于推算涨停价也不买
    md = make_md()
    md.a["exright_day"][1, 0] = 1
    md.a["open"][1, 0] = md.a["high"][1, 0] = 10.95      # +9.5%，推算涨停价 11.00
    r = Backtest(md, Fixed([0, 1], once=True), cfg(md)).run()
    check("除权日保守口径挡住近似封板的买入", r.stat["exright_extra_buy"] == 1
          and r.trades[r.trades.side == "buy"].iloc[0].code == md.codes[1])
    r2 = Backtest(md, Fixed([0, 1], once=True), cfg(md, exright_conservative=False)).run()
    check("关闭保守口径时可以买入", r2.trades[r2.trades.side == "buy"].iloc[0].code == md.codes[0])

    # U8 送转除权后持股数随复权因子增加，市值不变
    md = make_md()
    md.a["adj_factor"][3:, 0] = 2.0
    for k in ["open", "high", "low", "close"]:
        md.a[k][3:, 0] = 5.0
    md.a["preclose"][3:, 0] = 5.0
    md.a["up_limit"][3:, 0], md.a["down_limit"][3:, 0] = 5.5, 4.5
    r = Backtest(md, Fixed([0], once=True), cfg(md, costs=False, hold_days=4)).run()
    check("10 送 10 后收益为 0（不计成本）", abs(r.roundtrips.iloc[0].ret) < 1e-12, f"{r.roundtrips.iloc[0].ret:.2e}")

    # U9 退市：强制退市按 0、主动退市按现金、换股按比例
    md = make_md()
    md.universe.loc[md.codes[0], "delist_date"] = md.dates[5]
    md.exits.loc[md.codes[0], ["exit_class"]] = "forced_no_period"
    md.a["tradestatus"][3:, 0] = 0
    r = Backtest(md, Fixed([0], once=True), cfg(md, hold_days=5)).run()
    check("无整理期的强制退市按 0 计", r.roundtrips.iloc[0].how == "writeoff_forced_no_period"
          and r.roundtrips.iloc[0].proceeds == 0)
    md.exits.loc[md.codes[0], ["exit_class", "cash"]] = ["voluntary", 12.0]
    r = Backtest(md, Fixed([0], once=True), cfg(md, hold_days=5, costs=False)).run()
    check("主动退市按现金选择权价格兑现", r.roundtrips.iloc[0].how == "cash_option"
          and abs(r.roundtrips.iloc[0].ret - 0.2) < 1e-9, f"收益 {r.roundtrips.iloc[0].ret:.4f}")
    md.exits.loc[md.codes[0], ["exit_class", "cash", "ratio", "target_code"]] = ["merger", np.nan, 0.5, md.codes[1]]
    for k in ["open", "high", "low", "close"]:
        md.a[k][:, 1] = 24.0
    r = Backtest(md, Fixed([0], once=True), cfg(md, hold_days=5, costs=False)).run()
    rt = r.roundtrips.iloc[0]
    check("换股吸收合并：按比例转为存续公司股票后卖出", rt.via == md.codes[0] and abs(rt.ret - 0.2) < 1e-9,
          f"收益 {rt.ret:.4f}（10 元买入，1 股换 0.5 股 × 24 元）")

    # U10 策略看不到未来、不能改数据
    md = make_md()
    v = View(md, 5)
    ok_len = v.close.shape[0] == 6
    try:
        v.close[0, 0] = 1.0
        ro = False
    except ValueError:
        ro = True
    check("View 只到 t 日、只读", ok_len and ro)

    # U11 续持 + 滞后带
    class Sched:
        def __init__(self, f):
            self.f = f

        def select(self, t, v):
            return self.f(t)

    md = make_md()
    r = Backtest(md, Sched(lambda t: [0, 1, 2, 3]), cfg(md, hold_days=3, renew=True)).run()
    buys = r.trades[r.trades.side == "buy"]
    check("续持：一直排第一的股票不卖出", len(buys) == 1 and (r.trades.side == "sell").sum() == 0
          and r.stat.get("renewals", 0) >= 5, f"续持 {r.stat.get('renewals', 0)} 次")
    order = lambda t: [0, 1, 2, 3] if t < 6 else [1, 2, 3, 0]          # 第 6 日起股票 0 掉到第 4 名
    r = Backtest(md, Sched(order), cfg(md, hold_days=3, renew=True, renew_k=3)).run()
    s0 = r.trades[(r.trades.side == "sell") & (r.trades.code == md.codes[0])]
    check("续持：掉出前 K 名后到期卖出", len(s0) == 1, f"卖出日序号 {md.dates.get_loc(s0.iloc[0].date) if len(s0) else None}")
    r = Backtest(md, Sched(order), cfg(md, hold_days=3, renew=True, renew_k=4)).run()
    check("续持：K 放宽到 4 时仍在滞后带内，不卖", ((r.trades.side == "sell") & (r.trades.code == md.codes[0])).sum() == 0)
    r = Backtest(md, Sched(lambda t: [0, 1, 2, 3]), cfg(md, hold_days=3, renew=False)).run()
    check("关闭续持时照常到期卖出", (r.trades.side == "sell").sum() >= 5)
    md = make_md()
    md.a["open"][1, 0] = md.a["up_limit"][1, 0]
    r = Backtest(md, Fixed([0, 1], once=True), cfg(md, renew=True)).run()
    check("续持模式：新买入只从前 n_target 名中选（第 1 名涨停买不进时不替补）", (r.trades.side == "buy").sum() == 0
          if len(r.trades) else True)

    # U12 仓位控制：策略的 target_n 降低持仓数，每只金额仍为净值 / n_target
    class Half(Sched):
        def target_n(self, t, v):
            return 2 if t >= 8 else 4

    md = make_md()
    r = Backtest(md, Half(lambda t: [0, 1, 2, 3]), cfg(md, n_target=4, hold_days=3, renew=True)).run()
    npos = r.nav.n_pos.values
    check("仓位控制：target_n 从 4 降到 2 后持仓数随之下降", npos[3] == 4 and npos[-1] == 2,
          f"第 3 日 {npos[3]} 只，最后一日 {npos[-1]} 只")

    # U13 盘中止盈：买入次日最高价触及目标价，按目标价成交；买入当日触及不卖（T+1）
    md = make_md()
    md.a["high"][1, 0] = 11.0                           # 买入当日（序号 1）最高 11：T+1 不能卖
    md.a["high"][3, 0] = 10.6                           # 序号 3 最高 10.6 ≥ 10×1.05
    r = Backtest(md, Fixed([0], once=True), cfg(md, hold_days=10, take_profit=0.05)).run()
    s = r.trades[r.trades.side == "sell"]
    ok = len(s) == 1 and md.dates.get_loc(s.iloc[0].date) == 3 and abs(s.iloc[0].price - 10.5 * 0.999) < 1e-9
    check("止盈：T+1 后最高价触及目标价按目标价成交", ok, f"{s[['date', 'price', 'reason']].values.tolist()}")
    md = make_md()
    md.a["open"][2, 0] = 10.8
    md.a["high"][2, 0] = 10.8
    r = Backtest(md, Fixed([0], once=True), cfg(md, hold_days=10, take_profit=0.05)).run()
    s = r.trades[r.trades.side == "sell"]
    check("止盈：开盘已高于目标价按开盘价成交", len(s) == 1 and abs(s.iloc[0].price - 10.8 * 0.999) < 1e-9)

    # U14 盘中止损：最低价触及止损价按止损价成交；同日既触及止盈又触及止损按止损
    md = make_md()
    md.a["low"][3, 0] = 9.2
    md.a["high"][3, 0] = 10.8
    r = Backtest(md, Fixed([0], once=True), cfg(md, hold_days=10, take_profit=0.05, stop_loss=0.07)).run()
    s = r.trades[r.trades.side == "sell"]
    check("止损：同日触及止盈与止损按止损价", len(s) == 1 and s.iloc[0].reason == "stop_loss"
          and abs(s.iloc[0].price - 9.3 * 0.999) < 1e-9, f"{s[['price', 'reason']].values.tolist()}")

    # U15 收盘买入（尾盘）：t 日收盘买，含额外滑点；hold_days=1 次日开盘卖
    md = make_md()
    r = Backtest(md, Fixed([0], once=True), cfg(md, hold_days=1, buy_at="close", tail_slip=0.002)).run()
    b, s = r.trades[r.trades.side == "buy"], r.trades[r.trades.side == "sell"]
    ok = (len(b) == 1 and md.dates.get_loc(b.iloc[0].date) == 0 and abs(b.iloc[0].price - 10 * 1.003) < 1e-9
          and len(s) == 1 and md.dates.get_loc(s.iloc[0].date) == 1)
    check("尾盘买入：当日收盘买、次日开盘卖", ok, f"买 {b.date.tolist()} 卖 {s.date.tolist()}")
    md = make_md()
    md.a["close"][0, 0] = md.a["up_limit"][0, 0]
    r = Backtest(md, Fixed([0], once=True), cfg(md, hold_days=1, buy_at="close")).run()
    check("尾盘买入：收盘涨停封板买不进", (r.trades.side == "buy").sum() == 0 if len(r.trades) else True)

    # U16 收盘卖出：hold_days=1、收盘买入 → 次日收盘卖（不能当日卖）；盘中止盈优先
    md = make_md()
    r = Backtest(md, Fixed([0], once=True), cfg(md, hold_days=1, buy_at="close", sell_at="close")).run()
    s = r.trades[r.trades.side == "sell"]
    check("收盘卖出：次日收盘卖出", len(s) == 1 and md.dates.get_loc(s.iloc[0].date) == 1)
    md = make_md()
    md.a["high"][1, 0] = 10.4
    r = Backtest(md, Fixed([0], once=True), cfg(md, hold_days=1, buy_at="close", sell_at="close", take_profit=0.03)).run()
    s = r.trades[r.trades.side == "sell"]
    check("收盘卖出：次日盘中触及止盈先按目标价卖", len(s) == 1 and s.iloc[0].reason == "take_profit"
          and abs(s.iloc[0].price - 10.3 * 0.999) < 1e-9)

    # U17 回调限价买入 / 不追高开
    md = make_md()
    md.a["low"][1, 0] = 9.85
    r = Backtest(md, Fixed([0], once=True), cfg(md, hold_days=3, buy_dip=0.01)).run()
    b = r.trades[r.trades.side == "buy"]
    check("回调买入：最低价触及 前收×0.99 按限价成交", len(b) == 1 and abs(b.iloc[0].price - 9.9 * 1.001) < 1e-9)
    md = make_md()
    r = Backtest(md, Fixed([0], once=True), cfg(md, hold_days=3, buy_dip=0.01)).run()
    check("回调买入：未触及限价不成交", (r.trades.side == "buy").sum() == 0 if len(r.trades) else True)
    md = make_md()
    md.a["open"][1, 0] = 10.6
    md.a["high"][1, 0] = 10.6
    r = Backtest(md, Fixed([0], once=True), cfg(md, hold_days=3, max_gap=0.05)).run()
    check("不追高开：高开 6% 放弃买入", (r.trades.side == "buy").sum() == 0 if len(r.trades) else True)

    n_bad = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n共 {len(RESULTS)} 项，失败 {n_bad} 项")
    return RESULTS


if __name__ == "__main__":
    main()
