"""第 1 步：交易日历、股票池（含已退市）。数据源：东方财富、上交所、深交所。

股票池 = 东方财富当前在市沪深 A 股 ∪ 上交所终止上市名单 ∪ 深交所终止上市名单，
再对代码区间做穷举探测：凡在东方财富 2013-06 以后有日线、却不在上述名单里的代码也补进来，
防止名单遗漏导致幸存者偏差。

输出：
  data/meta/calendar.parquet   date, is_trading_day（来自上证指数与深证成指的交易日并集）
  data/meta/universe.parquet   code, code_name, board, list_date, delist_date, is_delisted, source
  data/meta/delist_lists.parquet 交易所原始退市名单
"""
import time
from concurrent.futures import ThreadPoolExecutor

import akshare as ak
import pandas as pd

from common import META, WARMUP_START, board_of, is_universe_code, load_df, log_download, save_df, today_str
from sources import em_a_share_list, em_kline, em_secid, retry

PROBE_RANGES = [("sh", 600000, 601999), ("sh", 603000, 603999), ("sh", 605000, 605999),
                ("sh", 688000, 689999), ("sz", 1, 3999), ("sz", 300000, 302999)]


def build_calendar():
    t0 = time.time()
    _, a = em_kline("1.000001", 0, beg=WARMUP_START.replace("-", ""))
    _, b = em_kline("0.399001", 0, beg=WARMUP_START.replace("-", ""))
    td = pd.DatetimeIndex(sorted(set(a.date) | set(b.date)))
    diff = set(a.date) ^ set(b.date)
    allday = pd.date_range(WARMUP_START, td.max())
    cal = pd.DataFrame({"date": allday, "is_trading_day": allday.isin(td).astype(int)})
    save_df(cal, META / "calendar.parquet")
    log_download("calendar", "eastmoney kline 1.000001 ∪ 0.399001", f"{WARMUP_START}~", len(td), t0,
                 note=f"沪深交易日不一致 {len(diff)} 天")
    print(f"交易日历: {len(td)} 个交易日，{td.min().date()} ~ {td.max().date()}；沪深不一致 {len(diff)} 天")
    return td


def _fetch_delist(exchange: str) -> pd.DataFrame:
    """单个交易所退市名单；成功即缓存到 data/meta/delist_<ex>.parquet，失败时回退到缓存。"""
    cache = META / f"delist_{exchange}.parquet"
    t0 = time.time()
    try:
        if exchange == "sse":
            d = retry(ak.stock_info_sh_delist, tries=3)
            d = d.rename(columns={"公司代码": "num", "公司简称": "name", "上市日期": "list_date", "暂停上市日期": "delist_date"})
            d["code"] = "sh." + d["num"].astype(str).str.zfill(6)
        else:
            d = retry(lambda: ak.stock_info_sz_delist(symbol="终止上市公司"), tries=3)
            d = d.rename(columns={"证券代码": "num", "证券简称": "name", "上市日期": "list_date", "终止上市日期": "delist_date"})
            d["code"] = "sz." + d["num"].astype(str).str.zfill(6)
        for c in ["list_date", "delist_date"]:
            d[c] = pd.to_datetime(d[c], errors="coerce")
        d = d[["code", "name", "list_date", "delist_date"]].assign(exchange=exchange)
        save_df(d, cache)
        log_download(f"delist_{exchange}", f"akshare delist list ({exchange})", "all", len(d), t0)
        return d
    except Exception as e:
        if cache.exists():
            print(f"  {exchange} 退市名单在线获取失败，使用缓存（{type(e).__name__}）")
            return load_df(cache)
        print(f"  {exchange} 退市名单获取失败且无缓存，由 finalize_universe 按最后成交日推断（{type(e).__name__}）")
        log_download(f"delist_{exchange}", f"akshare delist list ({exchange})", "all", 0, t0, status="failed",
                     note=str(e)[:150])
        return pd.DataFrame(columns=["code", "name", "list_date", "delist_date", "exchange"])


def delist_lists():
    """交易所终止上市名单（上交所/深交所接口在云端时常被重置连接，带缓存回退）。"""
    d = pd.concat([_fetch_delist("sse"), _fetch_delist("szse")], ignore_index=True)
    # 同一代码可能出现多次（如暂停后恢复、再次终止上市），保留最近一次终止上市日期
    d = d.sort_values("delist_date").drop_duplicates("code", keep="last").reset_index(drop=True)
    save_df(d, META / "delist_lists.parquet")
    return d


def probe(code):
    try:
        name, df = em_kline(em_secid(code), 0, beg=WARMUP_START.replace("-", ""))
    except Exception:
        return code, "__ERR__", None, None
    if df.empty:
        return code, None, None, None
    return code, name, df.date.min(), df.date.max()


def main():
    tdays = build_calendar()

    t0 = time.time()
    cur = retry(em_a_share_list)
    cur = cur[cur.code.map(is_universe_code)].assign(delist_date=pd.NaT, source="eastmoney_current")
    log_download("current_list", "eastmoney clist m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23", "snapshot", len(cur), t0)

    # 注意：东方财富这个列表里也包含大量已退市股票（如 退市长油、小天鹅A），退市日期用交易所名单回填，
    # 交易所名单取不到的，由 finalize_universe.py 根据"最后交易日"推断。
    dl = delist_lists()
    dl = dl[dl.code.map(is_universe_code)]
    cur = cur.merge(dl[["code", "delist_date"]].rename(columns={"delist_date": "dl_date"}), on="code", how="left")
    cur["delist_date"] = cur["dl_date"]
    cur = cur.drop(columns="dl_date")
    cur = cur[cur.delist_date.isna() | (cur.delist_date >= pd.Timestamp(WARMUP_START))]
    extra = dl[~dl.code.isin(cur.code) & (dl.delist_date >= pd.Timestamp(WARMUP_START))]
    extra = extra.assign(source="exchange_delist_list")
    uni = pd.concat([cur, extra[["code", "name", "list_date", "delist_date", "source"]]], ignore_index=True)
    print(f"东方财富列表 {len(cur)} 只（其中按交易所名单已退市 {cur.delist_date.notna().sum()}），名单额外补充 {len(extra)} 只")

    # 穷举探测名单之外的代码（--skip-probe 可跳过；2026-10-13 全量探测 7474 个代码，发现 0 个遗漏）
    import sys
    if "--skip-probe" in sys.argv:
        uni["board"] = uni.code.map(board_of)
        uni["is_delisted"] = uni.delist_date.notna().astype(int)
        uni = uni.rename(columns={"name": "code_name"}).sort_values("code").reset_index(drop=True)
        save_df(uni[["code", "code_name", "board", "list_date", "delist_date", "is_delisted", "source"]],
                META / "universe_raw.parquet")
        save_df(load_df(META / "universe_raw.parquet"), META / "universe.parquet")
        print(f"股票池合计 {len(uni)} 只（跳过探测）")
        return
    t0 = time.time()
    known = set(uni.code)
    cands = [f"{m}.{n:06d}" for m, lo, hi in PROBE_RANGES for n in range(lo, hi + 1)]
    cands = [c for c in cands if c not in known and is_universe_code(c)]
    print(f"探测 {len(cands)} 个名单外代码……")
    found, errs = [], []
    with ThreadPoolExecutor(8) as ex:
        for i, (code, name, first, last) in enumerate(ex.map(probe, cands), 1):
            if name == "__ERR__":
                errs.append(code)
            elif name is not None:
                found.append({"code": code, "name": name, "first": first, "last": last})
            if i % 2000 == 0:
                print(f"  {i}/{len(cands)}，发现 {len(found)}", flush=True)
    for code in errs:   # 失败的串行再试一次
        r = probe(code)
        if r[1] not in (None, "__ERR__"):
            found.append({"code": r[0], "name": r[1], "first": r[2], "last": r[3]})
        elif r[1] == "__ERR__":
            print(f"  探测仍失败: {code}")
    fd = pd.DataFrame(found, columns=["code", "name", "first", "last"])
    log_download("probe_codes", "eastmoney kline", f"{len(cands)} codes", len(fd), t0)
    if len(fd):
        print("名单外但有数据的代码：")
        print(fd.to_string(index=False))
        # 最后交易日早于最新交易日 → 视为已退市，退市日取最后交易日的下一个交易日
        nxt = {d: tdays[min(i + 1, len(tdays) - 1)] for i, d in enumerate(tdays)}
        fd["delist_date"] = fd["last"].map(lambda d: nxt.get(d) if d < tdays[-1] else pd.NaT)
        fd["list_date"] = pd.NaT   # 由日线首日回填
        uni = pd.concat([uni, fd.assign(source="probe")[["code", "name", "list_date", "delist_date", "source"]]],
                        ignore_index=True)

    uni["board"] = uni.code.map(board_of)
    uni["is_delisted"] = uni.delist_date.notna().astype(int)
    uni = uni.rename(columns={"name": "code_name"}).sort_values("code").reset_index(drop=True)
    uni = uni[["code", "code_name", "board", "list_date", "delist_date", "is_delisted", "source"]]
    save_df(uni, META / "universe_raw.parquet")
    save_df(load_df(META / "universe_raw.parquet"), META / "universe.parquet")
    print(f"股票池合计 {len(uni)} 只（已退市 {uni.is_delisted.sum()}）")
    print(uni.groupby(["board", "is_delisted"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
