"""研究框架自检：扩展前视检测必须放过诚实信号、抓住各种偷看方式；SMB 与紧缩夏普的数值合理。"""
import sys
import time

import numpy as np
import pandas as pd

from framework import SignalStrategy, deflated_sharpe, hfq, leak_test, load_md, smb_series

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "engine"))
from backtest import Backtest, Config  # noqa: E402
from test_strategies import RandomPick  # noqa: E402


def shift_back(x, k=1):
    """把 t+k 的值放到 t（偷看未来）。"""
    out = np.full_like(x, np.nan, dtype=np.float64)
    out[:-k] = x[k:]
    return out


def honest(md, ext):
    p = hfq(md, "close")
    s = np.full(p.shape, np.nan)
    s[5:] = -(p[5:] / p[:-5] - 1)
    return SignalStrategy(s)


def cheat_price(md, ext):
    p = hfq(md, "open")
    return SignalStrategy(shift_back(p, 2) / shift_back(p, 1) - 1)


def cheat_turn(md, ext):                    # 偷看次日换手率
    return SignalStrategy(shift_back(md.a["turn"].astype(float), 1))


def cheat_status(md, ext):                  # 偷看次日是否停牌
    return SignalStrategy(shift_back(md.a["tradestatus"].astype(float), 1) + np.random.default_rng(0).random(md.a["close"].shape) * 0)


def cheat_rolling_center(md, ext):          # 常见错误：居中滚动均值
    c = pd.DataFrame(md.a["close"]).rolling(5, center=True, min_periods=1).mean().values
    return SignalStrategy(-c)


def _ext_score(md, ext, cheat):
    s = np.full(md.a["close"].shape, np.nan)
    df = ext["events"]
    di = md.dates.searchsorted(pd.to_datetime(df["avail_date"]).values)
    ci = np.searchsorted(md.codes, df["code"].values)
    for d, c, v in zip(di, ci, df["value"].values):
        t = d - 1 if cheat else d          # cheat：在公开日前一天就使用
        if 0 <= t < md.T:
            s[t, c] = v
    return SignalStrategy(s)


def make_ext(md):
    rng = np.random.default_rng(1)
    n = 20000
    return {"events": pd.DataFrame({"code": rng.choice(md.codes, n),
                                    "avail_date": md.dates[rng.integers(300, md.T - 1, n)],
                                    "value": rng.normal(size=n)})}


def main():
    md = load_md()
    ext = make_ext(md)
    cases = [("诚实：5 日反转", honest, None, True),
             ("偷看：次日开盘价", cheat_price, None, False),
             ("偷看：次日换手率", cheat_turn, None, False),
             ("偷看：次日交易状态", cheat_status, None, False),
             ("偷看：居中滚动均值", cheat_rolling_center, None, False),
             ("诚实：外部事件按公开日使用", lambda m, e: _ext_score(m, e, False), ext, True),
             ("偷看：外部事件提前一天使用", lambda m, e: _ext_score(m, e, True), ext, False)]
    ok_all = True
    for name, f, e, expect in cases:
        t0 = time.time()
        r = leak_test(md, f, e, cuts=3)
        ok = r["passed"] == expect
        ok_all &= ok
        print(f"{'通过' if ok else '失败'} {name}: 检测结果 {'无前视' if r['passed'] else '有前视'}"
              f"（{len(r['changed'])}/{r['n_checks']} 个检查点改变，{time.time() - t0:.0f}s）", flush=True)
    s = smb_series(md)
    print(f"SMB：{s.index[0].date()}–{s.index[-1].date()}，日均 {s.mean() * 1e4:.2f} bp，年化 {s.mean() * 242:.2%}，"
          f"年化波动 {s.std() * np.sqrt(242):.2%}")
    for y, v in s.groupby(s.index.year).apply(lambda x: np.prod(1 + x) - 1).items():
        print(f"  {y}: {v:+.1%}")
    # 紧缩夏普：同样的 SR，试验次数越多概率越低
    rng = np.random.default_rng(0)
    trials = list(rng.normal(0, 0.02, 50))
    p1, _ = deflated_sharpe(0.08, 700, 0, 3, trials[:5], 5)
    p2, _ = deflated_sharpe(0.08, 700, 0, 3, trials, 50)
    print(f"紧缩夏普：SR=0.08（日频），N=5 → {p1:.3f}，N=50 → {p2:.3f}")
    ok_all &= p1 > p2
    # 续持关闭时引擎结果与阶段 2 一致（随机种子 0，不计成本，阶段 2 为 1.7333%）
    r = Backtest(md, RandomPick(0), Config(start="2014-01-01", end="2023-12-31", costs=False, hold_days=5)).run()
    nav = r.nav.nav
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (242 / (len(nav) - 1)) - 1      # 与 metrics.perf 同口径
    same = abs(cagr - 0.017333037109119376) < 1e-9
    ok_all &= same
    print(f"{'通过' if same else '失败'} 引擎改动后（续持关闭）随机种子 0 结果与阶段 2 一致：{cagr:.6%}")
    print("全部通过" if ok_all else "存在失败项")


if __name__ == "__main__":
    main()
