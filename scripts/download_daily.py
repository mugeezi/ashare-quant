"""第 2 步：逐只下载个股日线（东方财富不复权），含已退市股票；复权因子取自新浪，并补齐停牌日。

复权：东方财富的"后复权"价是仿射变换（复权价 = a·原价 + b），日收益与真实收益不一致；它的不复权
"涨跌额"也是相对上一日原始收盘，不含除权，因此推不出交易所前收。这里改用新浪的乘法后复权因子：
    adj_factor_t = 新浪因子（除权除息日起生效）
    preclose_t   = round(close_{t-1} × adj_factor_{t-1} / adj_factor_t, 2)   （除权除息参考价，交易所口径）
    hfq_*        = 不复权价 × adj_factor

东方财富不返回停牌日记录。这里按交易日历在 [上市日, 退市前一交易日 / 最新交易日] 区间内补行：
停牌行 tradestatus=0，开高低收 = 前收 = 上一个收盘价，量额为 0，filled=1。

每只存为 data/raw/daily/<code>.parquet；已存在则跳过（断点续传）。
字段：date code open high low close preclose volume(股) amount(元) turn(%) pctChg(%) amplitude(%)
      tradestatus filled hfq_open hfq_high hfq_low hfq_close adj_factor
用法：python download_daily.py [--workers 6] [--codes sh.600000 ...] [--update]
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from build_panel import trading_days
from common import log_test_access, META, RAW_DAILY, WARMUP_START, load_df, log_download, save_df
from sources import em_kline, em_secid, sina_hfq_factor

INFER_DELIST_DAYS = 20
# 两个交易所的终止上市名单都已取得（含缓存）时，不再按"长期无成交"推断退市：长期停牌股（如主动退市
# 审核中的 *ST康佳A）仍按停牌补行，直到名单上出现终止上市日期。
OFFICIAL_LISTS = (META / "delist_sse.parquet").exists() and (META / "delist_szse.parquet").exists()
OUT_COLS = ["date", "code", "open", "high", "low", "close", "preclose", "volume", "amount", "turn",
            "pctChg", "amplitude", "tradestatus", "filled",
            "hfq_open", "hfq_high", "hfq_low", "hfq_close", "adj_factor"]


def build_one(code: str, info: pd.Series, tdays: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    name, raw = em_kline(em_secid(code), 0)
    meta = {"code": code, "em_name": name, "kline_first": raw.date.min() if len(raw) else pd.NaT,
            "kline_last": raw.date.max() if len(raw) else pd.NaT, "raw_rows": len(raw)}
    if raw.empty:
        return pd.DataFrame(columns=OUT_COLS), meta

    df = raw.sort_values("date").reset_index(drop=True)
    df["code"] = code
    df["volume"] = df["vol_lots"] * 100
    fac = sina_hfq_factor(code)
    fac["date"] = fac["date"].astype(df["date"].dtype)
    df = pd.merge_asof(df, fac.rename(columns={"factor": "adj_factor"}), on="date", direction="backward")
    df["adj_factor"] = df["adj_factor"].fillna(1.0)
    ratio = df["adj_factor"].shift(1) / df["adj_factor"]
    # 交易所口径四舍五入（half-up）；+1e-3 分的容差吸收因子浮点误差（如 12.405 被表示成 12.40499999）
    df["preclose"] = np.floor(df["close"].shift(1) * ratio * 100 + 0.5 + 1e-3) / 100
    df.loc[df.index[0], "preclose"] = round(df["close"].iloc[0] - df["change"].iloc[0], 2)  # 上市首日：相对发行价
    df["pctChg"] = (df["close"] / df["preclose"] - 1) * 100          # 交易所口径（东方财富原值不含除权）
    meta["n_exright"] = int((ratio.iloc[1:] - 1).abs().gt(1e-9).sum())
    meta["n_factor_rows"] = len(fac)
    for c in ["open", "high", "low", "close"]:
        df[f"hfq_{c}"] = df[c] * df["adj_factor"]
    df["tradestatus"] = (df["volume"] > 0).astype("int8")
    df["filled"] = np.int8(0)
    df = df.drop(columns=["vol_lots", "change"])

    # ---- 补停牌日
    list_date = info["list_date"] if pd.notna(info["list_date"]) else df.date.min()
    start = max(list_date, tdays[0])
    if pd.notna(info["delist_date"]):
        before = tdays[tdays < info["delist_date"]]
        end = before.max() if len(before) else tdays[0] - pd.Timedelta(days=1)
    elif not OFFICIAL_LISTS and df.date.max() < tdays[-INFER_DELIST_DAYS]:
        # 名单里没有退市日期，但已超过 INFER_DELIST_DAYS 个交易日没有成交：视为已退市，不再往后补停牌行
        end = df.date.max()
        meta["inferred_delist"] = 1
    else:
        end = tdays[-1]
    end = max(end, df.date.max()) if df.date.max() >= tdays[0] else end
    span = tdays[(tdays >= start) & (tdays <= end)]
    hist = df[df.date < tdays[0]]                      # 2013-06 之前的最后一条，用于停牌前收
    cur = df[df.date >= tdays[0]].set_index("date")
    missing = span.difference(cur.index)
    if len(missing):
        add = pd.DataFrame(index=missing)
        add["code"] = code
        add["tradestatus"] = np.int8(0)
        add["filled"] = np.int8(1)
        for c in ["volume", "amount", "turn", "pctChg", "amplitude"]:
            add[c] = 0.0
        cur = pd.concat([cur, add]).sort_index()
        m = cur["filled"] == 1
        # 停牌行：价格 = 上一条真实收盘（含 2013-06 之前的历史），复权因子沿用
        for c, last in [("close", hist["close"].iloc[-1] if len(hist) else np.nan),
                        ("adj_factor", hist["adj_factor"].iloc[-1] if len(hist) else np.nan)]:
            carried = cur[c].where(~m).ffill()
            carried = carried.fillna(last)
            cur.loc[m, c] = carried[m]
        for c in ["open", "high", "low", "preclose"]:
            cur.loc[m, c] = cur.loc[m, "close"]
        for c in ["open", "high", "low", "close"]:
            cur.loc[m, f"hfq_{c}"] = cur.loc[m, c] * cur.loc[m, "adj_factor"]
    df = cur.reset_index().rename(columns={"index": "date"})
    meta["filled_rows"] = int(df["filled"].sum())
    return df[OUT_COLS], meta


def main():
    log_test_access("数据下载：个股日线全部年份（含测试集）", "东方财富 K 线、新浪复权因子")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--codes", nargs="*")
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args()

    uni = load_df(META / "universe.parquet").set_index("code")
    tdays = trading_days()
    codes = args.codes or list(uni.index)
    if not args.update:
        codes = [c for c in codes if not (RAW_DAILY / f"{c}.parquet").exists()]
    print(f"待下载 {len(codes)} 只，线程 {args.workers}", flush=True)
    if not codes:
        return

    t0 = time.time()
    metas, failed = [], []

    def task(code):
        df, meta = build_one(code, uni.loc[code], tdays)
        save_df(df, RAW_DAILY / f"{code}.parquet")
        return meta

    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(task, c): c for c in codes}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                metas.append(f.result())
            except Exception as e:
                failed.append((futs[f], str(e)[:300]))
            if i % 200 == 0 or i == len(codes):
                el = time.time() - t0
                print(f"  {i}/{len(codes)}  失败 {len(failed)}  已用 {el/60:.1f} 分  "
                      f"预计剩余 {el/i*(len(codes)-i)/60:.1f} 分", flush=True)

    mpath = RAW_DAILY.parent / "daily_meta.csv"
    m = pd.DataFrame(metas)
    if mpath.exists():   # 与已有记录合并（--update 或 --codes 只覆盖本次涉及的代码）
        m = pd.concat([pd.read_csv(mpath, parse_dates=["kline_first", "kline_last"]), m]).drop_duplicates("code", keep="last")
    m.to_csv(mpath, index=False)
    if failed:
        pd.DataFrame(failed, columns=["code", "error"]).to_csv(RAW_DAILY.parent / "failed_codes.csv", index=False)
        print(f"失败 {len(failed)} 只，重跑本脚本即可补下")
    log_download("stock_daily", "eastmoney kline fqt=0 (push2his) + sina hfq factor", f"1990~今, 保存 {WARMUP_START} 起; n={len(codes)}",
                 len(metas), t0, status="ok" if not failed else f"partial({len(failed)} failed)")


if __name__ == "__main__":
    main()
