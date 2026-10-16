"""X 系列：买卖点规则（止盈 / 止损 / 情绪条件止盈 / 尾盘买入）。选股打分沿用已缓存的模型打分（默认 H67）。
用法：python run_exec.py train X01                 # 训练集上跑该假设登记的全部变体
      python run_exec.py valid X01 tp5             # 指定变体上验证集（每个假设只用一次）
"""
import json
import sys
import time

import numpy as np
import pandas as pd

import ml
from framework import CACHE, ROOT, Lab
from run_batch3 import SPECS
from signals import limit_up_close

BASE = dict(n_target=10, renew=True, hold_days=10)


class EmoTP(ml.RegimeStrategy):
    """情绪条件止盈：买入时若市场处于"退潮"（昨日涨停股今日平均涨幅的 5 日均值 < 0），本笔启用止盈 tp_off；
    否则（情绪高潮或平稳）不设止盈，让利润奔跑。只用 t 日及以前数据。"""

    def __init__(self, score, md, tp_off=0.05, tp_on=None):
        super().__init__(score)
        a = md.a
        lu = limit_up_close(md)
        prev = np.vstack([np.zeros((1, md.N), bool), lu[:-1]])
        with np.errstate(invalid="ignore", divide="ignore"):
            r1 = a["close"] / a["preclose"] - 1
            prem = np.nanmean(np.where(prev & (a["tradestatus"] == 1), r1, np.nan), axis=1)
        self.prem5 = pd.Series(prem).rolling(5, min_periods=3).mean().values
        self.tp_off, self.tp_on = tp_off, tp_on

    def tp_for(self, t, c):
        if t < 0 or not np.isfinite(self.prem5[t]):
            return self.tp_on
        return self.tp_off if self.prem5[t] < 0 else self.tp_on


class ExitRule(ml.RegimeStrategy):
    """附加卖出规则（只用 t 日及以前数据，t 日收盘判断、次日开盘卖）：
    ma5：后复权收盘价跌破 5 日均线（趋势破位就走）；zb：当日触及涨停但收盘没封住（炸板）。"""

    def __init__(self, score, md, mode):
        super().__init__(score)
        a = md.a
        self.mode = mode
        if mode == "ma5":
            from framework import hfq
            C = hfq(md, "close")
            self.flag = C < pd.DataFrame(C).rolling(5, min_periods=5).mean().values
        elif mode == "zb":
            from signals import TOL
            self.flag = ((a["tradestatus"] == 1) & (a["no_limit"] == 0) & (a["high"] >= a["up_limit"] - TOL)
                         & (a["close"] < a["up_limit"] - TOL))
        else:
            raise ValueError(mode)

    def should_exit(self, t, v, p):
        return t > p.entry_t and bool(self.flag[t, p.c])


class EmoQ(EmoTP):
    """情绪条件止盈（相对阈值版）：涨停溢价 5 日均值低于其过去 250 日的 30% 分位视为退潮，本笔设止盈 tp_off。"""

    def __init__(self, score, md, tp_off=0.05):
        super().__init__(score, md, tp_off=tp_off)
        q = pd.Series(self.prem5).rolling(250, min_periods=120).quantile(0.3).values
        self.cold = self.prem5 < q

    def tp_for(self, t, c):
        return self.tp_off if t >= 0 and self.cold[t] else None


class Lagged(ml.RegimeStrategy):
    """保守版尾盘买入：t 日收盘买入只用 t−1 日收盘的打分（不用当日收盘信息）。"""

    def __init__(self, score):
        s = np.full_like(score, np.nan)
        s[1:] = score[:-1]
        super().__init__(s)


# 每个假设：底层打分 + 变体 {名称: (策略构建方式, 引擎参数)}
X = {
    "X01": ("H67", {f"tp{int(p * 100)}": ("std", dict(take_profit=p)) for p in (0.03, 0.05, 0.08, 0.12)}),
    "X02": ("H67", {f"vk{k}": ("std", dict(tp_vol_k=k)) for k in (1.0, 2.0, 3.0)}),
    "X03": ("H67", {"emo5": ("emo5", {}), "emo3": ("emo3", {}), "emo8": ("emo8", {})}),
    "X04": ("H67", {"sl5": ("std", dict(stop_loss=0.05)), "sl8": ("std", dict(stop_loss=0.08))}),
    "X05": ("H67", {"tail": ("std", dict(buy_at="close", tail_slip=0.002)),
                    "tail_lag": ("lag", dict(buy_at="close", tail_slip=0.002))}),
    "X06": ("H80", {"on": ("plain", dict(buy_at="close", tail_slip=0.002, hold_days=1, renew=False)),
                    "on_reg": ("std", dict(buy_at="close", tail_slip=0.002, hold_days=1, renew=False))}),
    "X07": ("H81", {"cc": ("plain", dict(buy_at="close", sell_at="close", tail_slip=0.002, hold_days=1, renew=False)),
                    "cc_tp3": ("plain", dict(buy_at="close", sell_at="close", tail_slip=0.002, hold_days=1, renew=False,
                                             take_profit=0.03)),
                    "cc_tp5": ("plain", dict(buy_at="close", sell_at="close", tail_slip=0.002, hold_days=1, renew=False,
                                             take_profit=0.05))}),
}

RULES = {"tp8": ("std", dict(take_profit=0.08)), "vk3": ("std", dict(tp_vol_k=3.0)), "sl8": ("std", dict(stop_loss=0.08)),
         "ma5": ("ma5", {}), "zb": ("zb", {}), "emoq5": ("emoq5", {})}
for _b in ("H27", "H42", "H44", "H53", "H60", "H60s", "H61", "H67s", "H71"):
    X[f"X08_{_b}"] = (_b, dict(RULES))
for _b in ("H67", "H60s", "H42", "H53"):
    X[f"X10_{_b}"] = (_b, {"dip1": ("std", dict(buy_dip=0.01)), "dip2": ("std", dict(buy_dip=0.02))})
    X[f"X11_{_b}"] = (_b, {"gap3": ("std", dict(max_gap=0.03)), "gap5": ("std", dict(max_gap=0.05))})
for _b in ("H42", "H44", "H53", "H60", "H60s", "H67", "H71", "H68", "H27", "H47", "H61", "H64", "H67s"):
    X[f"X12_{_b}"] = (_b, {"dip2zb": ("zb", dict(buy_dip=0.02)), "dip1zb": ("zb", dict(buy_dip=0.01)),
                           "dip2vk3": ("std", dict(buy_dip=0.02, tp_vol_k=3.0)), "dip2sl8": ("std", dict(buy_dip=0.02, stop_loss=0.08))})
for _b in ("H42", "H44", "H60", "H60s"):
    X[f"X13_{_b}"] = (_b, {"dv2zb": ("zb", dict(buy_dip_vol_k=0.2)), "dv4zb": ("zb", dict(buy_dip_vol_k=0.4)),
                           "dv3zb": ("zb", dict(buy_dip_vol_k=0.3))})
X["X03q"] = ("H67", {"emoq3": ("emoq3", {}), "emoq5": ("emoq5", {}), "emoq8": ("emoq8", {})})
X["X09"] = ("H67", {"ma5": ("ma5", {}), "zb": ("zb", {})})


def build(kind, sc, md):
    if kind == "std":
        return ml.RegimeStrategy(sc)
    if kind == "plain":
        from framework import SignalStrategy
        return SignalStrategy(sc)
    if kind == "lag":
        return Lagged(sc)
    if kind in ("ma5", "zb"):
        return ExitRule(sc, md, kind)
    if kind.startswith("emoq"):
        return EmoQ(sc, md, tp_off=int(kind[4:]) / 100)
    if kind.startswith("emo"):
        return EmoTP(sc, md, tp_off=int(kind[3:]) / 100)
    raise ValueError(kind)


def main(split, hyp, only=None, lab=None):
    base_hyp, variants = X[hyp]
    lab = lab or Lab()
    md = lab.md
    lf = ROOT / "reports" / f"leak_{base_hyp}_base.json"
    if lf.exists():
        lk = json.loads(lf.read_text())
    else:                                              # 早期假设的前视检测在进程内完成，结果记录在 experiments.csv
        d = pd.read_csv(ROOT / "experiments.csv")
        r = d[(d.hyp_id == base_hyp) & (d.variant == "base") & (d.split == "train")]
        ok = len(r) > 0 and str(r.iloc[-1].leak_test).startswith("pass")
        lk = {"passed": ok, "n_checks": int(str(r.iloc[-1].leak_test).split()[-1]) if ok else 0, "changed": []}
    sc = np.load(CACHE.parent / f"score_{base_hyp}_base_{split}.npy").astype(np.float64)
    for var, (kind, kw) in variants.items():
        if only and var != only:
            continue
        lab._leak_cache[(hyp, var)] = lk
        t0 = time.time()
        rr = lab.run(hyp, var, None, split=split, strategy=build(kind, sc, md), start="2016-01-01" if split == "train" else None,
                     params={"score": base_hyp, "kind": kind}, notes=f"买卖点规则，打分 {base_hyp}", **{**BASE, **kw})
        r, st = rr.row, rr.stats
        ex = rr.result.stat.get("exits", {})
        print(f"{hyp}/{var}: 年化 {r['cagr']:+.1%}  超额 {r['excess_cagr']:+.1%}  回撤 {r['max_drawdown']:.1%}  "
              f"正年 {r['years_excess_pos']}  α 占比 {r['alpha_share']}  换手 {r['turnover_annual']}  买入 {r['n_buys']}  "
              f"达标 {r['passed']}  退出 {ex}  ({time.time() - t0:.0f}s)", flush=True)
        print("   分年超额:", json.dumps({k: round(v, 3) for k, v in st["yearly_excess"].items()}), flush=True)
        if split == "valid":
            print(f"   DSR(原口径) {st['dsr']:.3f}", flush=True)
            rr.result.nav.to_csv(ROOT / "reports" / f"valid_{hyp}_{var}_nav.csv")
            rr.result.trades.to_csv(ROOT / "reports" / f"valid_{hyp}_{var}_trades.csv", index=False)
            # 修订口径 DSR：训练期样本外（同一规则在训练集打分上重跑，复核而非新试验）+ 验证集，N = 全部试验数
            from dsr_eval import dsr, excess
            from framework import Backtest, Config, read_exp
            sct = np.load(CACHE.parent / f"score_{base_hyp}_base_train.npy").astype(np.float64)
            res = Backtest(md, build(kind, sct, md), Config(start="2016-01-01", end="2020-12-31", **{**BASE, **kw})).run()
            e = pd.concat([excess(res.nav.nav, lab.bench), excess(rr.result.nav.nav, lab.bench)])
            p, sr, sr0, T = dsr(e, len(read_exp()))
            print(f"   修订口径 DSR {p:.3f}（T={T}，年化 IR {sr * 242 ** 0.5:.2f}，SR0 {sr0 * 242 ** 0.5:.2f}，N={len(read_exp())}）", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "robust":                        # 训练集稳健性：持有 8/12、持仓 8/12、续持带宽 24/36
        lab = Lab()
        for it in sys.argv[2:]:
            h, v = it.split(":")
            b, vs = X[h]
            kind, kw = vs[v]
            alt = {f"{v}_hold8": dict(kw, hold_days=8), f"{v}_hold12": dict(kw, hold_days=12),
                   f"{v}_n8": dict(kw, n_target=8), f"{v}_n12": dict(kw, n_target=12),
                   f"{v}_k24": dict(kw, renew_k=24), f"{v}_k36": dict(kw, renew_k=36)}
            X[h] = (b, {k: (kind, a) for k, a in alt.items()})
            main("train", h, None, lab)
    elif sys.argv[1] == "valid-many":                    # 参数形如 X08_H53:zb
        lab = Lab()
        for it in sys.argv[2:]:
            h, v = it.split(":")
            main("valid", h, v, lab)
    elif sys.argv[1] == "train-many":                    # 一个进程依次跑多个假设（共用一次行情加载）
        lab = Lab()
        for h in sys.argv[2:]:
            main("train", h, None, lab)
    else:
        main(sys.argv[1], sys.argv[2], *(sys.argv[3:4] or [None]))
