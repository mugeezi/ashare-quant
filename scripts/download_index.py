"""第 4 步：基准指数日线。中证系列以中证指数官网为主（东方财富交叉核对，发现过东财 2014-12-09
中证全指收盘错误），官网缺的交易日用东方财富补；国证2000 只有东方财富来源。

必需：沪深300、中证500、中证1000、中证全指；参考：国证2000。
"微盘股"基准在研究阶段用全市场市值最小的 400 只等权组合自行构造（无免费的微盘股指数日线）。
"""
import time

import akshare as ak
import pandas as pd

from build_panel import trading_days
from common import log_test_access, INDEX, WARMUP_START, log_download, save_df
from sources import em_kline, retry

INDEXES = {  # name: (东方财富 secid, 中证代码 or None)
    "hs300": ("1.000300", "000300"),
    "zz500": ("1.000905", "000905"),
    "zz1000": ("1.000852", "000852"),
    "zzqz": ("1.000985", "000985"),
    "gz2000": ("0.399303", None),
}


def main():
    log_test_access("数据下载：基准指数全部年份（含测试集）", "中证官网、东方财富")
    tdays = trading_days()
    frames, notes = [], []
    for name, (secid, csi) in INDEXES.items():
        t0 = time.time()
        _, df = em_kline(secid, 0, beg=WARMUP_START.replace("-", ""))
        df = df.rename(columns={"vol_lots": "volume"})[["date", "open", "high", "low", "close", "volume", "amount"]]
        df["index"], df["source"] = name, "eastmoney"
        note = ""
        if csi:
            try:
                c = retry(lambda: ak.stock_zh_index_hist_csindex(
                    symbol=csi, start_date=WARMUP_START.replace("-", ""), end_date="20500101"), tries=3)
                c = c.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close",
                                      "成交量": "volume", "成交金额": "amount"})
                c["date"] = pd.to_datetime(c["date"])
                bad_days = int((~c["date"].isin(tdays)).sum())
                c = c[c["date"].isin(tdays)][["date", "open", "high", "low", "close", "volume", "amount"]]
                m = df.merge(c[["date", "close"]], on="date", how="inner", suffixes=("_em", ""))
                rel = (m["close_em"] / m["close"] - 1).abs()
                bad = m[rel > 1e-3]
                c["index"], c["source"] = name, "csindex"
                fill = df[~df["date"].isin(c["date"])]
                df = pd.concat([c, fill], ignore_index=True).sort_values("date").reset_index(drop=True)
                note = (f"主来源中证官网；与东财收盘差 >0.1% 的 {len(bad)} 天"
                        + (f"（{', '.join(str(d.date()) for d in bad.date[:5])}）" if len(bad) else "")
                        + f"；东财补 {len(fill)} 天；官网含非交易日 {bad_days} 行已剔除")
            except Exception as e:
                note = f"中证官网获取失败，全部用东财: {type(e).__name__}"
        notes.append(f"{name}: {note}" if note else name)
        save_df(df, INDEX / f"{name}.parquet")
        log_download(f"index_{name}", f"eastmoney kline {secid}", f"{WARMUP_START}~", len(df), t0, note=note[:200])
        print(f"  {name:7s} {len(df):5d} 行 {df.date.min().date()} ~ {df.date.max().date()}  {note}", flush=True)
        frames.append(df)
    save_df(pd.concat(frames, ignore_index=True), INDEX / "index_daily.parquet")


if __name__ == "__main__":
    main()
