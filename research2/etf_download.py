"""W01 ETF 日线下载（腾讯 web.ifzq.gtimg.cn fqkline 接口；不使用东方财富）。

口径：
- 每只 ETF 下载两套日线：不复权（day）与后复权（hfqday），区间 2013-01-01 至 2023-12-31（研究期以外的数据不请求）。
  接口返回里附带的当前行情字段（qt 等）直接丢弃，不保存、不读取。
- 字段：date, open, close, high, low, volume（手）；amount 估算 = volume × 100 × 不复权收盘价。
- 收益与信号一律用后复权价；后复权价只依赖上市以来已发生的分红/折算，不含未来信息。
- 上市前没有数据：上市后满 120 个交易日才进入可选范围。
- 每次请求间隔 1.5 秒，按年分段请求；失败重试 2 次，间隔 10 秒。
输出：data2/etf/{code}.parquet（与股票数据分开存放）。
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data2" / "etf"
END = "2023-12-31"
URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sym},day,{s},{e},400,{fq}"

# 事先登记的 ETF 列表（见 hypotheses.md 第二轮 W01）
ETFS = {
    # 宽基
    "sh510050": "上证50", "sh510300": "沪深300", "sh510500": "中证500", "sh512100": "中证1000",
    "sz159915": "创业板", "sz159901": "深证100", "sh588000": "科创50",
    # 行业 / 主题
    "sh512880": "证券", "sh512800": "银行", "sh512010": "医药", "sh512170": "医疗", "sh512690": "酒",
    "sz159928": "消费", "sh512480": "半导体", "sh515030": "新能源车", "sh512660": "军工", "sh512400": "有色金属",
    "sh515220": "煤炭", "sh512980": "传媒", "sh512580": "环保", "sh510880": "红利", "sh512200": "房地产",
    "sh515050": "5G通信", "sz159939": "信息技术", "sh515210": "钢铁", "sz159825": "农业",
    # 防御（货币 / 短债）
    "sh511880": "银华日利（货币）", "sh511010": "国债ETF", "sh511360": "短融ETF",
}


def fetch(sym: str, s: str, e: str, fq: str) -> list:
    key = "hfqday" if fq == "hfq" else "day"
    for k in range(3):
        try:
            req = urllib.request.Request(URL.format(sym=sym, s=s, e=e, fq=fq), headers={"User-Agent": "Mozilla/5.0"})
            d = json.loads(urllib.request.urlopen(req, timeout=20).read().decode())
            node = d["data"][sym]
            return [r[:6] for r in node.get(key, [])]          # 只取日线，丢弃 qt 等当前行情字段
        except Exception as ex:  # noqa: BLE001
            if k == 2:
                raise RuntimeError(f"{sym} {s}~{e} {fq}: {ex}")
            time.sleep(10)
        finally:
            time.sleep(1.5)


def one(sym: str) -> pd.DataFrame:
    out = {}
    for fq in ("", "hfq"):
        rows = []
        for y in range(2013, 2024):
            rows += fetch(sym, f"{y}-01-01", f"{y}-12-31", fq)
        df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates("date").sort_values("date").set_index("date").astype(float)
        out[fq or "raw"] = df
    raw, hfq = out["raw"], out["hfq"]
    df = raw.add_prefix("raw_").join(hfq.add_prefix("hfq_"), how="outer")
    df = df[df.index <= END]
    df["amount"] = df["raw_volume"] * 100 * df["raw_close"]
    df["code"] = sym
    return df.reset_index()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    syms = sys.argv[1:] or list(ETFS)
    for sym in syms:
        f = OUT / f"{sym}.parquet"
        if f.exists():
            continue
        df = one(sym)
        df.to_parquet(f)
        print(sym, ETFS.get(sym), len(df), df.date.min().date() if len(df) else None, df.date.max().date() if len(df) else None, flush=True)


if __name__ == "__main__":
    main()
