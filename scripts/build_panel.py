"""第 4 步：把逐只文件合并为按年切分的面板 data/panel/daily_YYYY.parquet，并补充派生列。

派生列（只用当日及以前的信息）：
  board        板块：main_sh / main_sz / chinext / star
  list_date, delist_date
  days_listed  上市以来第几个交易日（上市首日 = 1）
  limit_pct    当日涨跌停幅度（按板块、ST、日期）
  up_limit, down_limit  当日涨/跌停价（不复权，四舍五入到分）
  no_limit     新股上市初期不设常规涨跌停
  isST         由 download_st.py 重建的风险警示标记（含退市整理期）
  isS          未完成股权分置改革（简称 S 前缀），涨跌幅 5%
  delist_period 退市整理期（首日 no_limit=1，之后按板块普通股票幅度）
  resume_no_limit ST/S 股长期停牌后复牌首日（恢复上市、股改复牌等），视为不设涨跌幅
  exright_day  除权除息日（前收为推算值，个别情形与交易所公布值有差异）
  price_flag   1=2016-01-04 熔断日，2=2014-04-23 数据源分歧日（见 common.SPECIAL_PRICE_DATES）
"""
import numpy as np
import pandas as pd

from common import (log_test_access, META, PANEL, RAW_DAILY, SPECIAL_PRICE_DATES, limit_pct, limit_price, load_df, no_limit_days,
                    save_df)

OLD_LISTING_OFFSET = 100000   # 上市日早于日历起点的股票，days_listed 加这个偏移，保证 >> 60


def load_st_events() -> dict:
    p = META / "st_events.parquet"
    if not p.exists():
        raise FileNotFoundError("缺少 data/meta/st_events.parquet，请先运行 download_st.py")
    return dict(tuple(load_df(p).groupby("code")))


def enrich(df: pd.DataFrame, info: pd.Series, tdays: pd.DatetimeIndex, st_events: dict) -> pd.DataFrame:
    """给单只股票的日线补派生列（含 isST）。质检脚本也调用这个函数，保证口径一致。"""
    from download_st import cninfo_method_flags, flags_for
    df = df.copy()
    df["board"] = info["board"]
    df["list_date"] = info["list_date"]
    df["delist_date"] = info["delist_date"]

    pos_date = tdays.searchsorted(df["date"].values)
    if info["list_date"] < tdays[0]:
        df["days_listed"] = pos_date + OLD_LISTING_OFFSET
    else:
        df["days_listed"] = pos_date - tdays.searchsorted(info["list_date"]) + 1
    df["no_limit"] = (df["days_listed"] <= no_limit_days(info["board"], info["list_date"])).astype("int8")

    ev = st_events.get(info.name)
    if info.name.startswith("sh."):     # 沪市：巨潮公告 + 涨跌幅纠正（2026-07-06 前）/ 简称法（之后）
        df["isST"] = cninfo_method_flags(ev, df)
    else:                               # 深市：深交所简称变更（精确）
        df["isST"] = flags_for(ev, df["date"])

    from download_st import special_flags
    s_flag, tui = special_flags(ev, df["date"])
    df["isS"] = s_flag                       # 未完成股改（S 前缀），涨跌幅 5%
    df["delist_period"] = tui                # 退市整理期
    first_tui = (df["delist_period"] == 1) & (df["tradestatus"] == 1)
    if first_tui.any():                      # 退市整理期首日不设涨跌幅
        df.loc[first_tui.idxmax(), "no_limit"] = 1
    # 不设涨跌幅的复牌首日：(1) 恢复上市首日、(2) 股改实施后复牌首日。
    # 判定依据：未股改 S 股停牌 ≥5 日且复牌时已不是 S；或巨潮相关公告后的首个复牌日。
    ts = df["tradestatus"].values
    run = np.zeros(len(ts), dtype=int)
    for k in range(1, len(ts)):
        run[k] = run[k - 1] + 1 if ts[k - 1] == 0 else 0
    last_s = pd.Series(np.where(ts == 1, df["isS"].values, np.nan)).ffill().shift(1).fillna(0).values
    # S 股：仅当停牌期间完成股改（复牌当日已不是 S）才算股改复牌首日
    resume = (ts == 1) & (last_s == 1) & (df["isS"].values == 0) & (run >= 5)
    # 另据巨潮"股票恢复上市（公告书）""股权分置改革方案实施"公告（排除债券、产品等无关标题，见 download_special.py）：
    # 公告后 90 天内的首个复牌日（此前停牌 ≥1 天）
    rel = RELIST_ANN.get(info.name)
    if rel is not None and len(rel):
        d = df["date"].values
        used_until = np.datetime64("1900-01-01")
        for ad in rel:                       # 90 天内的多条相关公告视为同一事件，只取第一个复牌日
            if ad <= used_until:
                continue
            cand = np.where((ts == 1) & (run >= 1) & (d > ad) & (d <= ad + np.timedelta64(90, "D")))[0]
            if len(cand):
                resume[cand[0]] = True
                used_until = d[cand[0]] + np.timedelta64(90, "D")
    df["resume_no_limit"] = resume.astype("int8")
    df.loc[resume, "no_limit"] = 1
    df["limit_pct"] = limit_pct(df["board"], df["isST"], df["date"], df["isS"], df["delist_period"])
    # 除权除息日：前收由新浪复权因子推出；配股（按实际配售比例）、重整转增（交易所调整参考价）等情形
    # 可能与交易所公布的前收相差（E1a 归因）。除权日事先公告，不构成前视信息。
    df["exright_day"] = (df["adj_factor"] / df["adj_factor"].shift(1) - 1).abs().gt(1e-9).astype("int8")
    df["price_flag"] = df["date"].map({d: v[0] for d, v in SPECIAL_PRICE_DATES.items()}).fillna(0).astype("int8")
    df["up_limit"] = limit_price(df["preclose"], df["limit_pct"].values, +1)
    df["down_limit"] = limit_price(df["preclose"], df["limit_pct"].values, -1)
    return df


def _relist_ann() -> dict:
    p = META / "cninfo_relist_reform_ann.parquet"
    if not p.exists():
        return {}
    a = load_df(p)
    a = a[a["relevant"]]
    return {c: np.sort(g["date"].values) for c, g in a.groupby("code")}


RELIST_ANN = _relist_ann()


def trading_days() -> pd.DatetimeIndex:
    cal = load_df(META / "calendar.parquet")
    return pd.DatetimeIndex(np.sort(cal.loc[cal.is_trading_day == 1, "date"].values))


def main():
    log_test_access("数据构建：合并按年面板（含测试集年份），不做统计", "data/raw/daily → data/panel")
    uni = load_df(META / "universe.parquet").set_index("code")
    tdays = trading_days()
    st_events = load_st_events()

    by_year: dict[int, list[pd.DataFrame]] = {}
    files = sorted(RAW_DAILY.glob("*.parquet"))
    n_rows = 0
    for i, f in enumerate(files, 1):
        if f.stem not in uni.index:      # 已被 finalize_universe 剔除的代码
            continue
        df = load_df(f)
        if df.empty:
            continue
        df = enrich(df, uni.loc[f.stem], tdays, st_events)
        for y, g in df.groupby(df["date"].dt.year):
            by_year.setdefault(int(y), []).append(g)
        n_rows += len(df)
        if i % 500 == 0:
            print(f"  已处理 {i}/{len(files)}")

    for y, parts in sorted(by_year.items()):
        d = pd.concat(parts, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)
        save_df(d, PANEL / f"daily_{y}.parquet")
        print(f"  {y}: {len(d):>8,d} 行  {d.code.nunique():>5d} 只")
    print(f"合计 {n_rows:,d} 行")


if __name__ == "__main__":
    main()
