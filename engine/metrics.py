"""绩效指标。年化按每年 242 个交易日；夏普不扣无风险利率。"""
import numpy as np
import pandas as pd

DAYS = 242


def perf(nav: pd.Series, bench: dict | None = None, roundtrips: pd.DataFrame | None = None,
         trades: pd.DataFrame | None = None) -> dict:
    r = nav.pct_change().dropna()
    n = len(r)
    years = n / DAYS
    out = {"start": str(nav.index[0].date()), "end": str(nav.index[-1].date()),
           "total_return": nav.iloc[-1] / nav.iloc[0] - 1,
           "cagr": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan,
           "vol": r.std() * np.sqrt(DAYS),
           "sharpe": r.mean() / r.std() * np.sqrt(DAYS) if r.std() > 0 else np.nan,
           "max_drawdown": (nav / nav.cummax() - 1).min()}
    for k, s in (bench or {}).items():
        b = s.reindex(nav.index).ffill()
        br = b.pct_change().dropna()
        bc = (b.iloc[-1] / b.iloc[0]) ** (1 / years) - 1
        ex = (r - br.reindex(r.index).fillna(0))
        out[f"bench_cagr_{k}"] = bc
        out[f"excess_cagr_{k}"] = out["cagr"] - bc
        out[f"excess_ir_{k}"] = ex.mean() / ex.std() * np.sqrt(DAYS) if ex.std() > 0 else np.nan
    if trades is not None and len(trades):
        tv = trades.loc[trades.side.isin(["buy", "sell"]), "amount"].sum()
        out["turnover_annual"] = tv / 2 / nav.mean() / years
        out["n_buys"] = int((trades.side == "buy").sum())
        out["fees_total"] = float(trades.commission.sum() + trades.stamp.sum())
    if roundtrips is not None and len(roundtrips):
        rt = roundtrips
        win, loss = rt[rt.pnl > 0], rt[rt.pnl <= 0]
        out["n_roundtrips"] = len(rt)
        out["win_rate"] = len(win) / len(rt)
        out["payoff_ratio"] = win.ret.mean() / -loss.ret.mean() if len(win) and len(loss) and loss.ret.mean() < 0 else np.nan
        out["avg_hold_days"] = rt.hold_days.mean()
    out["yearly"] = (nav.groupby(nav.index.year).last() / nav.groupby(nav.index.year).first().shift(1)
                     .fillna(nav.iloc[0]) - 1).to_dict() if n else {}
    yl = nav.groupby(nav.index.year).last()
    prev = yl.shift(1)
    prev.iloc[0] = nav.iloc[0]
    out["yearly"] = (yl / prev - 1).round(4).to_dict()
    return out
