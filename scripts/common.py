"""公共配置与工具函数：路径、股票池过滤、板块与涨跌停幅度规则、读写、下载日志。

所有脚本都从这里取常量，后续回测引擎也复用这里的板块/涨跌停规则。
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 路径
ROOT = Path(os.environ.get("ASQ_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
META = DATA / "meta"
RAW_DAILY = DATA / "raw" / "daily"      # 每只股票一个 parquet，可断点续传
PANEL = DATA / "panel"                  # 按年份切分的合并面板 daily_YYYY.parquet
INDEX = DATA / "index"
REPORTS = ROOT / "reports"
for _p in (META, RAW_DAILY, PANEL, INDEX, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 日期
START_DATE = "2014-01-01"     # 研究区间起点
WARMUP_START = "2013-06-01"   # 多下载半年，用于 2014 年初的 20 日均额、上市天数等滚动指标
TRAIN_END = "2020-12-31"
VALID_END = "2023-12-31"
TEST_START = "2024-01-01"     # 最终测试集，研究阶段的加载函数会硬性截断


# 已知的数据源分歧日（详见 reports/data_quality.md 的"抽样交叉核对"）：面板中 price_flag 标出这些日期的全部行
SPECIAL_PRICE_DATES = {
    pd.Timestamp("2016-01-04"): (1, "熔断日：13:34 触发熔断后提前收市，各数据源收盘价口径不一；本库（东方财富）与新浪一致，腾讯不同"),
    pd.Timestamp("2014-04-23"): (2, "腾讯行情该日收盘/最高/最低与东方财富、新浪不一致；本库与新浪一致"),
}


def today_str() -> str:
    return pd.Timestamp.today().strftime("%Y-%m-%d")


# ---------------------------------------------------------------- 股票池与板块
# 只要沪深 A 股：沪主板 600/601/603/605，科创板 688/689，深主板 000/001/002/003，创业板 300/301/302
# 自然排除：北交所(bj.)、B 股(sh.900 / sz.200)、指数、基金、可转债
_PREFIX_BOARD = {
    "sh.600": "main_sh", "sh.601": "main_sh", "sh.603": "main_sh", "sh.605": "main_sh",
    "sh.688": "star", "sh.689": "star",
    "sz.000": "main_sz", "sz.001": "main_sz", "sz.002": "main_sz", "sz.003": "main_sz",
    "sz.300": "chinext", "sz.301": "chinext", "sz.302": "chinext",
}


def board_of(code: str) -> str | None:
    return _PREFIX_BOARD.get(str(code)[:6])


def is_universe_code(code: str) -> bool:
    return board_of(code) is not None


# ---------------------------------------------------------------- 涨跌停规则
CHINEXT_REFORM = pd.Timestamp("2020-08-24")   # 创业板注册制：10% -> 20%（含 ST）
# 沪深主板风险警示（ST/*ST）股票涨跌幅：2026-07-06 起由 5% 调整为 10%（沪深交易所，2026-07-06 生效）。
# 注：2025-06-27 交易所曾发布同方向的征求意见稿，正式生效日为 2026-07-06。
MAIN_ST_10PCT = pd.Timestamp(os.environ.get("ASQ_MAIN_ST_10PCT", "2026-07-06"))
MAIN_REGISTRATION = pd.Timestamp("2023-04-10")  # 主板注册制首批上市，新股前 5 日不设涨跌幅


def limit_pct(board, is_st, date, is_s=None, delist_period=None) -> np.ndarray:
    """每行的涨跌停幅度（小数）。新股无涨跌幅期、退市整理期首日由 no_limit 单独标记。
    主板：10%；主板 ST/*ST：2026-07-06 前 5%，之后 10%；未完成股改（S 前缀）：5%。
    创业板：2020-08-24 前 10%（ST 5%），之后 20%（ST 同为 20%）。科创板：20%（ST 同为 20%）。
    退市整理期（首日除外）：与本板块普通股票相同（主板 10%，创业板/科创板 20%）。"""
    board = pd.Series(board).astype(str).values
    n = len(board)
    is_st = pd.Series(is_st).fillna(0).astype(bool).values
    is_s = np.zeros(n, bool) if is_s is None else pd.Series(is_s).fillna(0).astype(bool).values
    tui = np.zeros(n, bool) if delist_period is None else pd.Series(delist_period).fillna(0).astype(bool).values
    date = pd.to_datetime(pd.Series(date)).values

    main = np.isin(board, ["main_sh", "main_sz"])
    chinext = board == "chinext"
    star = board == "star"
    after_reform = date >= np.datetime64(CHINEXT_REFORM)

    pct = np.full(n, 0.10)
    pct[star] = 0.20
    pct[chinext & after_reform] = 0.20

    st_main = is_st & main & ~tui
    pct[st_main] = np.where(date[st_main] >= np.datetime64(MAIN_ST_10PCT), 0.10, 0.05)
    pct[is_st & chinext & ~after_reform & ~tui] = 0.05
    pct[is_s & main & ~tui] = 0.05
    return pct


def limit_price(preclose: pd.Series, pct: np.ndarray, side: int) -> np.ndarray:
    """交易所口径：前收 × (1 ± 幅度)，四舍五入到 0.01 元。用整数"分"运算，避免 11.055 被浮点表示成
    11.05499… 而舍错：limit_cents = floor((前收分 × (100 ± 百分比) + 50) / 100)。"""
    pc = np.rint(np.asarray(preclose, dtype=float) * 100)
    pp = np.rint(np.asarray(pct, dtype=float) * 100)
    out = np.floor((pc * (100 + side * pp) + 50) / 100) / 100
    return np.where(np.isnan(pc), np.nan, out)


def no_limit_days(board: str, list_date: pd.Timestamp) -> int:
    """上市初期不设涨跌幅的交易日数（首日 44%/-36% 的老规则也视作"无常规涨跌停"）。"""
    if board == "star":
        return 5
    if board == "chinext":
        return 5 if list_date >= CHINEXT_REFORM else 1
    return 5 if list_date >= MAIN_REGISTRATION else 1


# ---------------------------------------------------------------- 读写
# 正式运行一律 parquet。ASQ_TEST_PICKLE=1 仅供无 pyarrow 的测试环境使用。
_TEST_PICKLE = os.environ.get("ASQ_TEST_PICKLE") == "1"


def save_df(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if _TEST_PICKLE:
        df.to_pickle(tmp)
    else:
        df.to_parquet(tmp, index=False)
    os.replace(tmp, path)   # 原子替换，断电/中断不会留下半个文件


def load_df(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if _TEST_PICKLE:
        df = pd.read_pickle(path)
        return df[columns] if columns else df
    return pd.read_parquet(path, columns=columns)


# ---------------------------------------------------------------- 测试集访问规则
# 2024-01-01 及以后为最终测试集。允许：数据完整性检查（缺失、异常值、规则一致性）。
# 禁止：计算任何与收益相关的统计量（收益率、因子 IC、策略表现等），最终测试获批前一律禁止。
# 每次访问都追加记录到 test_set_access.log（时间、目的、读取内容、调用脚本）。
TEST_LOG = ROOT / "test_set_access.log"
FINAL_TEST_APPROVAL = ROOT / "FINAL_TEST_APPROVED"     # 用户确认通过标准后才创建；存在时才允许 purpose="final_test"


def log_test_access(purpose: str, content: str, kind: str = "integrity") -> None:
    """kind: integrity（完整性检查，允许）| final_test（最终测试，需批准）| backfill（补记）。"""
    import inspect
    if kind not in ("integrity", "final_test", "backfill"):
        raise ValueError("测试集访问类别只能是 integrity / final_test / backfill")
    if kind == "final_test" and not FINAL_TEST_APPROVAL.exists():
        raise PermissionError("最终测试尚未获批：不得计算测试集上的收益类统计量")
    caller = "?"
    for fr in inspect.stack()[1:]:
        if not fr.filename.endswith("common.py"):
            caller = Path(fr.filename).name
            break
    new = not TEST_LOG.exists()
    with open(TEST_LOG, "a", encoding="utf-8") as f:
        if new:
            f.write("时间\t类别\t调用脚本\t目的\t读取内容\n")
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S %z')}\t{kind}\t{caller}\t{purpose}\t{content}\n")


def load_panel(years: list[int] | None = None, columns: list[str] | None = None,
               allow_test: bool = False, purpose: str | None = None, kind: str = "integrity") -> pd.DataFrame:
    """研究阶段统一入口。allow_test=False 时不读取 2024 年及以后的文件，并截断日期。
    allow_test=True 必须给出 purpose，并写入测试集访问日志。"""
    if allow_test:
        if not purpose:
            raise ValueError("读取测试集必须说明目的（purpose），并会记录到 test_set_access.log")
        yrs = "全部年份" if years is None else ",".join(str(y) for y in years if y >= int(TEST_START[:4]))
        log_test_access(purpose, f"panel {yrs}; 列 {columns or '全部'}", kind)
    files = sorted(PANEL.glob("daily_*.parquet"))
    test_year = int(TEST_START[:4])
    out = []
    for f in files:
        y = int(f.stem.split("_")[1])
        if years is not None and y not in years:
            continue
        if not allow_test and y >= test_year:
            continue
        out.append(load_df(f, columns))
    if not out:
        return pd.DataFrame(columns=columns)
    df = pd.concat(out, ignore_index=True)
    if not allow_test and "date" in df:
        df = df[df["date"] < pd.Timestamp(TEST_START)]
    return df


# ---------------------------------------------------------------- 下载日志
LOG_PATH = DATA / "download_log.csv"
_LOG_FIELDS = ["dataset", "source", "params", "rows", "started", "finished", "status", "note"]


def log_download(dataset: str, source: str, params: str, rows: int,
                 started: float, status: str = "ok", note: str = "") -> None:
    new = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
        if new:
            w.writeheader()
        fmt = "%Y-%m-%d %H:%M:%S %z"
        w.writerow({
            "dataset": dataset, "source": source, "params": params, "rows": rows,
            "started": time.strftime(fmt, time.localtime(started)),
            "finished": time.strftime(fmt), "status": status, "note": note,
        })
