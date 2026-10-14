"""第 2.5 步：日线下载完成后，确定每只股票的上市日、退市日。

- 上市日：优先东方财富列表的上市日期，缺失时用日线首日；两者不一致的记入报告。
- 退市日：优先交易所终止上市名单（会再次尝试获取上交所名单）；
          名单里没有、但最后成交日距今超过 20 个交易日的，推断为"最后成交日的下一交易日"退市，
          来源记为 inferred。
- 2013-06 之前就已经退市（没有任何 2013-06 之后数据）的剔除出股票池。
输出覆盖 data/meta/universe.parquet，并写 reports/universe_notes.csv。
"""
import pandas as pd

from build_panel import trading_days
from common import META, RAW_DAILY, REPORTS, WARMUP_START, is_universe_code, load_df, save_df
from download_daily import INFER_DELIST_DAYS


def main():
    tdays = trading_days()
    uni = load_df(META / "universe_raw.parquet")   # 总是从 download_meta 的原始输出开始，可重复运行
    meta = pd.read_csv(RAW_DAILY.parent / "daily_meta.csv", parse_dates=["kline_first", "kline_last"])
    meta = meta.drop_duplicates("code", keep="last")
    u = uni.merge(meta[["code", "kline_first", "kline_last", "raw_rows", "filled_rows"]], on="code", how="left")

    # 用（带缓存的）交易所名单补退市日
    from download_meta import delist_lists
    dl = delist_lists()
    dl = dl[dl.code.map(is_universe_code)][["code", "delist_date"]].rename(columns={"delist_date": "ex_delist"})
    u = u.merge(dl, on="code", how="left")
    fill = u.ex_delist.notna() & (u.delist_date.isna() | (u.delist_date != u.ex_delist))
    u.loc[fill, "delist_date"] = u.loc[fill, "ex_delist"]
    u.loc[fill, "source"] = u.loc[fill, "source"] + "+exchange_delist"
    print(f"交易所退市名单：补充退市日 {int(fill.sum())} 只（名单覆盖：{', '.join(sorted(set(dl.code.str[:2])))}）")
    u = u.drop(columns="ex_delist")

    # 推断退市（仅在交易所名单缺失时使用）
    from download_daily import OFFICIAL_LISTS
    nxt = pd.Series(tdays[1:], index=tdays[:-1])
    stale = u.delist_date.isna() & u.kline_last.notna() & (u.kline_last < tdays[-INFER_DELIST_DAYS])
    if OFFICIAL_LISTS:
        long_susp = u[stale]
        if len(long_susp):
            print("长期停牌（>20 个交易日无成交）但不在终止上市名单、按在市处理：",
                  ", ".join(f"{c} {n}" for c, n in zip(long_susp.code, long_susp.code_name)))
        stale = stale & False
    u.loc[stale, "delist_date"] = u.loc[stale, "kline_last"].map(lambda d: nxt.get(d, d + pd.Timedelta(days=1)))
    u.loc[stale, "source"] = u.loc[stale, "source"] + "+inferred_delist"
    print(f"按最后成交日推断退市：{int(stale.sum())} 只")

    # 重复序列：两个代码在相同日期的成交量、成交额完全相同（东方财富偶有把换股前旧代码映射到新代码行情的情况，
    # 如 600849"上药转换"与 601607 上海医药 2013 年数据完全相同）。保留官方仍在市/有官方退市日的一方。
    sig = []
    for c in u.code:
        f = RAW_DAILY / f"{c}.parquet"
        if f.exists():
            x = load_df(f, ["date", "volume", "amount", "tradestatus"])
            x = x[(x.tradestatus == 1) & (x.volume > 0)]
            sig.append(x.assign(code=c)[["code", "date", "volume", "amount"]])
    sig = pd.concat(sig)
    dup = sig[sig.duplicated(["date", "volume", "amount"], keep=False)]
    pairs = dup.groupby(["date", "volume", "amount"])["code"].apply(lambda s: tuple(sorted(s)))
    pc = pairs.value_counts()
    pc = pc[pc >= 20]
    dup_drop = []
    for pair, n in pc.items():
        cand = u[u.code.isin(pair)].sort_values("source")       # 推断退市的一方排在后面
        bad = cand[cand.source.str.contains("inferred")]
        victim = bad.code.iloc[0] if len(bad) else cand.sort_values("list_date").code.iloc[0]
        print(f"重复序列 {pair}：相同 {n} 天，剔除 {victim}")
        dup_drop.append(victim)
    u = u[~u.code.isin(dup_drop)]

    # 上市日
    miss = u.list_date.isna()
    u.loc[miss, "list_date"] = u.loc[miss, "kline_first"]
    diff = u[(u.kline_first.notna()) & (u.list_date.notna()) & (u.kline_first != u.list_date)]

    # 剔除 2013-06 前已退市、或完全没有数据的
    drop = u[(u.delist_date.notna() & (u.delist_date < pd.Timestamp(WARMUP_START))) |
             u.kline_last.isna() | (u.kline_last < pd.Timestamp(WARMUP_START))]
    u = u[~u.code.isin(drop.code)]
    u["is_delisted"] = u.delist_date.notna().astype(int)
    keep = ["code", "code_name", "board", "list_date", "delist_date", "is_delisted", "source"]
    save_df(u[keep].reset_index(drop=True), META / "universe.parquet")

    notes = pd.concat([
        diff.assign(note="上市日期与日线首日不一致")[["code", "code_name", "list_date", "kline_first", "note"]],
        drop.assign(note="2013-06 前已退市或无数据，剔除")[["code", "code_name", "list_date", "kline_first", "note"]],
    ])
    notes.to_csv(REPORTS / "universe_notes.csv", index=False, encoding="utf-8-sig")
    print(f"最终股票池 {len(u)} 只，其中已退市 {u.is_delisted.sum()}；剔除 {len(drop)}；上市日不一致 {len(diff)}")
    print(u.groupby(["board", "is_delisted"]).size().unstack(fill_value=0).to_string())
    print(u.source.value_counts().to_string())


if __name__ == "__main__":
    main()
