"""回测数据层：把按年面板展开成"交易日 × 股票"的宽表（numpy 数组），供引擎和策略使用。

研究阶段默认只加载 2013-06 至 2023-12（common.load_panel 硬性截断测试集）；
final_test=True 时才加载 2024 年以后，且需要 FINAL_TEST_APPROVED 存在并写入访问日志。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import INDEX, META, TEST_START, load_df, load_panel  # noqa: E402

FLOAT_COLS = ["open", "high", "low", "close", "preclose", "amount", "volume", "turn", "adj_factor",
              "limit_pct", "up_limit", "down_limit"]
INT_COLS = ["tradestatus", "isST", "isS", "no_limit", "delist_period", "exright_day", "days_listed"]
LIQ_WINDOW = 20


@dataclass
class MarketData:
    dates: pd.DatetimeIndex
    codes: np.ndarray
    a: dict = field(default_factory=dict)          # 列名 -> (T, N) 数组
    universe: pd.DataFrame | None = None
    exits: pd.DataFrame | None = None
    index_close: pd.DataFrame | None = None        # 列：zzqz, zz1000, hs300, zz500

    @property
    def T(self):
        return len(self.dates)

    @property
    def N(self):
        return len(self.codes)

    def col(self, code: str) -> int:
        return int(self._pos[code])

    def __post_init__(self):
        self._pos = pd.Series(np.arange(len(self.codes)), index=self.codes)


def load_market(start: str = "2013-06-01", end: str = "2023-12-31", final_test: bool = False,
                purpose: str | None = None) -> MarketData:
    cols = ["date", "code"] + FLOAT_COLS + INT_COLS
    if final_test:
        df = load_panel(columns=cols, allow_test=True, purpose=purpose or "最终测试", kind="final_test")
    else:
        if pd.Timestamp(end) >= pd.Timestamp(TEST_START):
            raise PermissionError("研究阶段回测区间不得进入测试集（2024-01-01 起）")
        df = load_panel(columns=cols)
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    dates = pd.DatetimeIndex(np.sort(df["date"].unique()))
    codes = np.sort(df["code"].unique())
    di = dates.searchsorted(df["date"].values)
    ci = np.searchsorted(codes, df["code"].values)
    T, N = len(dates), len(codes)
    a = {}
    for c in FLOAT_COLS:
        arr = np.full((T, N), np.nan, dtype=np.float64 if c in ("open", "high", "low", "close", "preclose",
                                                                "up_limit", "down_limit", "adj_factor") else np.float32)
        arr[di, ci] = df[c].values
        a[c] = arr
    for c in INT_COLS:
        arr = np.zeros((T, N), dtype=np.int32 if c == "days_listed" else np.int8)
        arr[di, ci] = df[c].fillna(0).values
        a[c] = arr
    # 前 20 个交易日平均成交额（含当日；停牌日成交额按 0 计，不在市的日子不计入）
    amt = np.nan_to_num(a["amount"].astype(np.float64), nan=0.0)
    listed = ~np.isnan(a["close"])
    cs = np.cumsum(amt, axis=0)
    cn = np.cumsum(listed, axis=0)
    s20 = cs - np.vstack([np.zeros((LIQ_WINDOW, N)), cs[:-LIQ_WINDOW]])
    n20 = cn - np.vstack([np.zeros((LIQ_WINDOW, N)), cn[:-LIQ_WINDOW]])
    with np.errstate(invalid="ignore", divide="ignore"):
        a["amt20"] = np.where(n20 >= LIQ_WINDOW, s20 / np.maximum(n20, 1), np.nan)
    md = MarketData(dates=dates, codes=codes, a=a)
    md.universe = load_df(META / "universe.parquet").set_index("code")
    md.exits = load_df(META / "corporate_exits.parquet").set_index("code")
    idx = {}
    for name in ["zzqz", "zz1000", "hs300", "zz500"]:
        x = load_df(INDEX / f"{name}.parquet").set_index("date")["close"]
        idx[name] = x.reindex(dates)
    md.index_close = pd.DataFrame(idx)
    return md


class View:
    """策略只能通过 View 读数据：每个数组都切到 [:t+1]（含当日收盘），看不到 t 之后的任何一行。"""
    __slots__ = ("_md", "t", "_cache")

    def __init__(self, md: MarketData, t: int):
        self._md, self.t, self._cache = md, t, {}

    def __getattr__(self, name):
        if name in ("dates", "codes"):
            return self._md.dates[: self.t + 1] if name == "dates" else self._md.codes
        arr = self._md.a.get(name)
        if arr is None:
            raise AttributeError(name)
        v = arr[: self.t + 1].view()
        v.flags.writeable = False            # 策略不能改数据
        return v

    @property
    def index_close(self):
        return self._md.index_close.iloc[: self.t + 1]
