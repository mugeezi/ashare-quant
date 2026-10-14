"""阶段 3 研究框架：信号策略、扩展前视检测、SMB 因子、紧缩夏普、实验登记。

用法（见 research/h*.py）：
    from framework import Lab
    lab = Lab()
    lab.run("H01", "base", factory, split="train", hold_days=5, renew=True)
    lab.run("H01", "base", factory, split="valid", ...)   # 训练集最低条件满足后，每个假设只运行一次

规则（与 hypotheses.md 研究协议一致）：
- 每次 run 都写入 experiments.csv，trial_no 递增；验证集运行同时累计 valid_use_no。
- 训练集运行前先做扩展前视检测，不通过直接拒绝回测。
- 验证集运行要求：同一假设同一变体在训练集上满足最低条件；该假设没有用过验证集
  （确需再次使用时必须传 reason，原因会写入 notes，并在 hypotheses.md 说明）。
"""
from __future__ import annotations

import json
import math
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "scripts"))
from backtest import Backtest, Config  # noqa: E402
from data import MarketData, View, add_derived, load_market  # noqa: E402

EXP_CSV = ROOT / "experiments.csv"
CACHE = Path("/tmp/claude-0/-home-claude/47e34d9e-ea3d-5e5a-bd04-7c13816a5033/scratchpad/md.pkl")
DERIVED = ROOT / "data" / "derived"
SPLITS = {"train": ("2014-01-01", "2020-12-31"), "valid": ("2021-01-01", "2023-12-31")}
DAYS = 242
TRAIN_MIN = dict(excess=0.05, years_pos=4, max_dd=-0.45, alpha_share=0.5)
VALID_MIN = dict(excess=0.08, years_pos=2, max_dd=-0.35, alpha_share=0.5, dsr=0.95)
TOP = 300                                  # 信号策略每日最多返回的候选数


# ---------------------------------------------------------------- 数据
def load_md() -> MarketData:
    if CACHE.exists():
        md = pickle.load(open(CACHE, "rb"))
    else:
        md = load_market()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(md, open(CACHE, "wb"), protocol=5)
    return add_derived(md)


def eligible_matrix(md: MarketData, min_listed=60, min_amt=5e7) -> np.ndarray:
    """与引擎 eligible() 同口径的 T×N 布尔矩阵（只用当日及以前数据）。"""
    a = md.a
    dl = pd.to_datetime(md.universe.reindex(md.codes)["delist_date"]).values
    ok = ((a["tradestatus"] == 1) & (a["isST"] == 0) & (a["isS"] == 0) & (a["delist_period"] == 0)
          & (a["days_listed"] >= min_listed) & (np.nan_to_num(a["amt20"]) >= min_amt))
    ok &= ~(dl[None, :] <= md.dates.values[:, None])
    return ok


def hfq(md: MarketData, field="close") -> np.ndarray:
    return md.a[field] * md.a["adj_factor"]


# ---------------------------------------------------------------- 信号策略
class SignalStrategy:
    """score: T×N 矩阵，越大越好；NaN 表示当日不入选。select 返回当日得分最高的 TOP 只（降序）。"""

    def __init__(self, score: np.ndarray, top: int = TOP):
        self.score, self.top = score, top

    def select(self, t, v):
        row = self.score[t]
        idx = np.where(np.isfinite(row))[0]
        if not len(idx):
            return []
        order = idx[np.argsort(-row[idx], kind="stable")]
        return order[: self.top]


# ---------------------------------------------------------------- 扩展前视检测
def _perturb_md(md: MarketData, t_cut: int, rng) -> MarketData:
    a2 = {}
    for k, arr in md.a.items():
        x = arr.copy()
        tail = x[t_cut + 1:]
        if np.issubdtype(x.dtype, np.floating):
            tail *= rng.uniform(0.5, 1.5, size=tail.shape).astype(x.dtype)
        else:                                          # 整数 / 标记：每行在股票间随机打乱
            for i in range(tail.shape[0]):
                tail[i] = rng.permutation(tail[i])
        a2[k] = x
    md2 = MarketData(dates=md.dates, codes=md.codes, a=a2)
    md2.universe = md.universe.copy()
    d = md.dates[t_cut]
    dl = pd.to_datetime(md2.universe["delist_date"])
    fut = dl > d
    md2.universe.loc[fut, "delist_date"] = d + pd.to_timedelta(rng.integers(1, 2000, fut.sum()), unit="D")
    md2.exits = md.exits
    ic = md.index_close.copy()
    ic.iloc[t_cut + 1:] = ic.iloc[t_cut + 1:].values * rng.uniform(0.5, 1.5, size=ic.iloc[t_cut + 1:].shape)
    md2.index_close = ic
    return md2


def _perturb_ext(ext: dict | None, cut_date, rng) -> dict | None:
    """外部数据：avail_date 晚于 t 的记录，数值列乘随机数、各列在这些行之间打乱，日期随机后移。"""
    if not ext:
        return ext
    out = {}
    for name, df in ext.items():
        df = df.copy()
        m = pd.to_datetime(df["avail_date"]) > cut_date
        n = int(m.sum())
        if n:
            for c in df.columns:
                if c == "avail_date":
                    continue
                if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_float_dtype(df[c]):
                    df[c] = df[c].astype(float)
                vals = df.loc[m, c].to_numpy()
                if pd.api.types.is_numeric_dtype(df[c]):
                    vals = vals * rng.uniform(0.5, 1.5, size=n)
                df.loc[m, c] = rng.permutation(vals)
            df.loc[m, "avail_date"] = pd.to_datetime(df.loc[m, "avail_date"]) + \
                pd.to_timedelta(rng.integers(0, 30, n), unit="D")
        out[name] = df
    return out


def leak_test(md: MarketData, factory, ext: dict | None = None, cuts: int = 3, lags=(0, 1, 3), seed: int = 0,
              t_range=None) -> dict:
    """扩展前视检测：在随机切点 t* 之后扰动全部字段（价格、量额、换手、复权因子、交易状态、ST 与涨跌停标记、
    指数、退市日期）及全部外部数据，重建策略，比较 t*−lag 日的信号排名。全部一致才算通过。"""
    rng = np.random.default_rng(seed)
    lo, hi = t_range or (300, md.T - 30)
    cut_ts = rng.choice(np.arange(lo, hi), cuts, replace=False)
    base = factory(md, ext)
    changed = []
    for tc in cut_ts:
        md2 = _perturb_md(md, int(tc), rng)
        ext2 = _perturb_ext(ext, md.dates[tc], rng)
        s2 = factory(md2, ext2)
        for lag in lags:
            t = int(tc) - lag
            r1 = np.asarray(base.select(t, View(md, t)))
            r2 = np.asarray(s2.select(t, View(md2, t)))
            if not np.array_equal(r1, r2):
                changed.append(str(md.dates[t].date()))
        del md2
    return {"passed": not changed, "n_checks": cuts * len(lags), "changed": changed}


# ---------------------------------------------------------------- SMB 因子
def smb_series(md: MarketData) -> pd.Series:
    DERIVED.mkdir(parents=True, exist_ok=True)
    f = DERIVED / "smb_daily.parquet"
    if f.exists():
        return pd.read_parquet(f)["smb"]
    elig = eligible_matrix(md)
    px = hfq(md, "close")
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = px[1:] / px[:-1] - 1
    out = []
    for t in range(1, md.T):
        ok = elig[t - 1] & np.isfinite(md.a["fmv"][t - 1]) & np.isfinite(ret[t - 1])
        mv = md.a["fmv"][t - 1][ok]
        r = ret[t - 1][ok]
        if len(mv) < 100:
            out.append(np.nan)
            continue
        lo, hi = np.quantile(mv, [0.3, 0.7])
        out.append(r[mv <= lo].mean() - r[mv >= hi].mean())
    s = pd.Series(out, index=md.dates[1:], name="smb")
    s.to_frame().to_parquet(f)
    return s


# ---------------------------------------------------------------- 统计量
def excess_stats(nav: pd.Series, bench: pd.Series, smb: pd.Series) -> dict:
    r = nav.pct_change().dropna()
    b = bench.reindex(nav.index).ffill().pct_change().reindex(r.index).fillna(0)
    e = r - b
    years = len(r) / DAYS
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    ex_cagr = float(np.prod(1 + e) ** (1 / years) - 1)
    # 分年超额 = 当年策略收益 − 当年指数收益
    yr = {}
    for y, idx in r.groupby(r.index.year).groups.items():
        yr[int(y)] = float(np.prod(1 + r[idx]) - np.prod(1 + b[idx]))
    dd = float((nav / nav.cummax() - 1).min())
    s = smb.reindex(e.index)
    m = s.notna()
    X = np.column_stack([np.ones(m.sum()), s[m].values])
    coef, *_ = np.linalg.lstsq(X, e[m].values, rcond=None)
    alpha_ann, beta = coef[0] * DAYS, coef[1]
    arith_ex = e.mean() * DAYS
    sr = e.mean() / e.std() if e.std() > 0 else np.nan
    return dict(cagr=cagr, excess_cagr=ex_cagr, yearly_excess=yr, max_drawdown=dd, smb_alpha=alpha_ann, smb_beta=beta,
                alpha_share=alpha_ann / arith_ex if arith_ex > 0 else np.nan, arith_excess=arith_ex,
                sharpe_daily=sr, skew=float(stats.skew(e)), kurt=float(stats.kurtosis(e, fisher=False)), n_days=len(e))


def deflated_sharpe(sr: float, n_obs: int, skew: float, kurt: float, trial_srs: list[float],
                    n_trials: int | None = None) -> tuple[float, float]:
    """Bailey & López de Prado (2014)。sr、trial_srs 为日频夏普；n_trials 为试验总次数（默认 = len(trial_srs)）。
    返回 (DSR 概率, SR0)。"""
    N = n_trials or len(trial_srs)
    if N < 2 or len(trial_srs) < 2:
        return float("nan"), float("nan")
    v = float(np.var(trial_srs, ddof=1))
    g = 0.5772156649015329
    sr0 = math.sqrt(v) * ((1 - g) * stats.norm.ppf(1 - 1 / N) + g * stats.norm.ppf(1 - 1 / (N * math.e)))
    den = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(n_obs - 1) / den)), sr0


# ---------------------------------------------------------------- 实验登记
COLS = ["exp_id", "hyp_id", "variant", "split", "run_time", "params", "hold_days", "n_target", "cagr", "excess_cagr",
        "sharpe_daily", "max_drawdown", "turnover_annual", "n_buys", "years_excess_pos", "smb_alpha", "smb_beta",
        "alpha_share", "leak_test", "passed", "trial_no", "valid_use_no", "notes"]


def read_exp() -> pd.DataFrame:
    if EXP_CSV.exists() and EXP_CSV.stat().st_size > 0:
        return pd.read_csv(EXP_CSV, dtype={"exp_id": str})
    return pd.DataFrame(columns=COLS)


def append_exp(row: dict):
    df = read_exp()
    df = pd.concat([df, pd.DataFrame([{c: row.get(c) for c in COLS}])], ignore_index=True)
    df.to_csv(EXP_CSV, index=False)


@dataclass
class RunResult:
    row: dict
    stats: dict
    result: object


class Lab:
    def __init__(self, md: MarketData | None = None):
        self.md = md or load_md()
        self.smb = smb_series(self.md)
        self.bench = self.md.index_close["zzqz"]
        self._leak_cache: dict = {}

    def trial_count(self) -> int:
        return len(read_exp())

    def log_trial(self, hyp, variant, split, params, notes, **metrics):
        """不经回测的统计（如训练集分组收益）也计入试验次数。"""
        exp = read_exp()
        if split != "train":
            raise PermissionError("探索性统计只允许在训练集上做")
        row = dict(exp_id=f"E{len(exp) + 1:04d}", hyp_id=hyp, variant=variant, split=split,
                   run_time=time.strftime("%Y-%m-%d %H:%M"), params=json.dumps(params, ensure_ascii=False),
                   trial_no=len(exp) + 1, valid_use_no=self._valid_uses(exp, hyp), notes=notes, **metrics)
        append_exp(row)
        return row

    @staticmethod
    def _valid_uses(exp, hyp):
        return int(((exp.hyp_id == hyp) & (exp.split == "valid")).sum()) if len(exp) else 0

    def run(self, hyp: str, variant: str, factory, split: str = "train", ext: dict | None = None,
            params: dict | None = None, notes: str = "", reason: str | None = None, **cfg_kw) -> RunResult:
        exp = read_exp()
        if split not in SPLITS:
            raise PermissionError(f"未知或不允许的数据集：{split}")
        if split == "valid":
            tr = exp[(exp.hyp_id == hyp) & (exp.variant == variant) & (exp.split == "train")]
            if not len(tr) or not tr.passed.astype(str).str.lower().eq("true").any():
                raise PermissionError(f"{hyp}/{variant} 未满足训练集最低条件，不能运行验证集")
            if self._valid_uses(exp, hyp) >= 1 and not reason:
                raise PermissionError(f"{hyp} 已经用过验证集；再次使用必须给出 reason 并在 hypotheses.md 说明")
        key = (hyp, variant)
        if key not in self._leak_cache:
            self._leak_cache[key] = leak_test(self.md, factory, ext)
        lk = self._leak_cache[key]
        if not lk["passed"]:
            raise RuntimeError(f"{hyp}/{variant} 前视检测未通过：{lk['changed']}")
        start, end = SPLITS[split]
        cfg = Config(start=start, end=end, **cfg_kw)
        t0 = time.time()
        strat = factory(self.md, ext)
        res = Backtest(self.md, strat, cfg).run()
        st = excess_stats(res.nav.nav, self.bench, self.smb)
        tr = res.trades
        years = len(res.nav) / DAYS
        turnover = (tr.loc[tr.side.isin(["buy", "sell"]), "amount"].sum() / 2 / res.nav.nav.mean() / years) if len(tr) else 0
        n_buys = int((tr.side == "buy").sum()) if len(tr) else 0
        ypos = sum(v > 0 for v in st["yearly_excess"].values())
        if split == "train":
            passed = (st["excess_cagr"] >= TRAIN_MIN["excess"] and ypos >= TRAIN_MIN["years_pos"]
                      and st["max_drawdown"] >= TRAIN_MIN["max_dd"] and (st["alpha_share"] or 0) >= TRAIN_MIN["alpha_share"])
            dsr = sr0 = float("nan")
        else:
            srs = exp["sharpe_daily"].dropna().astype(float).tolist() + [st["sharpe_daily"]]
            dsr, sr0 = deflated_sharpe(st["sharpe_daily"], st["n_days"], st["skew"], st["kurt"], srs,
                                       n_trials=len(exp) + 1)
            passed = (st["excess_cagr"] >= VALID_MIN["excess"] and ypos >= VALID_MIN["years_pos"]
                      and st["max_drawdown"] >= VALID_MIN["max_dd"] and (st["alpha_share"] or 0) >= VALID_MIN["alpha_share"]
                      and dsr >= VALID_MIN["dsr"])
        prm = dict(params or {}, **{k: v for k, v in cfg_kw.items()})
        note = notes
        if split == "valid":
            note = (note + f"; DSR={dsr:.3f}, SR0={sr0:.4f}, N={len(exp) + 1}").strip("; ")
            if reason:
                note += f"; 再次使用验证集原因：{reason}"
        row = dict(exp_id=f"E{len(exp) + 1:04d}", hyp_id=hyp, variant=variant, split=split,
                   run_time=time.strftime("%Y-%m-%d %H:%M"), params=json.dumps(prm, ensure_ascii=False),
                   hold_days=cfg.hold_days, n_target=cfg.n_target, cagr=round(st["cagr"], 4),
                   excess_cagr=round(st["excess_cagr"], 4), sharpe_daily=round(st["sharpe_daily"], 5),
                   max_drawdown=round(st["max_drawdown"], 4), turnover_annual=round(turnover, 1), n_buys=n_buys,
                   years_excess_pos=f"{ypos}/{len(st['yearly_excess'])}", smb_alpha=round(st["smb_alpha"], 4),
                   smb_beta=round(st["smb_beta"], 3), alpha_share=round(st["alpha_share"], 3) if st["alpha_share"] == st["alpha_share"] else None,
                   leak_test=f"pass {lk['n_checks']}", passed=bool(passed), trial_no=len(exp) + 1,
                   valid_use_no=self._valid_uses(exp, hyp) + (split == "valid"), notes=note)
        append_exp(row)
        st["secs"] = time.time() - t0
        st["dsr"] = dsr
        return RunResult(row=row, stats=st, result=res)
