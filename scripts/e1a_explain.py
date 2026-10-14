"""E1a（价格超出当日涨跌停价、非新股期、非退市整理期）逐行取证。
对每一行收集：此前连续停牌天数、当日是否除权（复权因子变化）、ST 标记、腾讯行情当日价格、
巨潮公告中该日前后 15 天内含"复牌/恢复上市/股改/重整/转增/除权/涨跌幅"等关键词的标题。
输出 reports/e1a_rows.csv，供人工归因（reports/e1a_explained.md）。"""
import json
import re

import numpy as np
import pandas as pd

from cninfo_names import fetch_stock, orgid_map
from common import log_test_access, META, PANEL, RAW_DAILY, REPORTS, load_df
from crosscheck_sample import tx_kline

KEY = r"复牌|恢复上市|重新上市|股改|股权分置|重整|转增|除权|涨跌幅|首日|吸收合并|换股|上市|停牌"


def main():
    log_test_access("E1a 涨跌停不一致逐行取证", "E1a 涉及股票的日线与公告")
    ex = json.load(open(REPORTS / "data_quality_examples.json", encoding="utf-8"))
    rows = pd.DataFrame(ex["E1a 超出涨跌停价：其他（非新股期）"])
    orgs = orgid_map()
    out = []
    for r in rows.itertuples():
        d = pd.Timestamp(r.date)
        x = load_df(RAW_DAILY / f"{r.code}.parquet").sort_values("date").reset_index(drop=True)
        i = x.index[x.date == d][0]
        prev = x.iloc[:i]
        susp = 0
        for t in prev.tradestatus.values[::-1]:
            if t == 0:
                susp += 1
            else:
                break
        fr = x.adj_factor.iloc[i] / x.adj_factor.iloc[i - 1] if i > 0 else 1.0
        last_close = prev[prev.tradestatus == 1].close.iloc[-1] if (prev.tradestatus == 1).any() else np.nan
        try:
            tx, _ = tx_kline(r.code, d.year, "")
            txr = tx[tx.date == d]
            tx_px = f"{txr.open.iloc[0]}/{txr.high.iloc[0]}/{txr.low.iloc[0]}/{txr.close.iloc[0]}" if len(txr) else "无"
        except Exception:
            tx_px = "取数失败"
        try:
            a = fetch_stock(r.code, orgs.get(r.code[3:], ""), str((d - pd.Timedelta(days=20)).date()),
                            str((d + pd.Timedelta(days=3)).date()))
            a = a[a.title.str.contains(KEY)]
            titles = " | ".join(f"{t.date()} {s}" for t, s in zip(a.date, a.title))[:600]
        except Exception:
            titles = "取数失败"
        out.append({"code": r.code, "date": r.date[:10], "board": r.board, "isST": r.isST, "preclose": r.preclose,
                    "high": r.high, "low": r.low, "close": r.close, "limit_pct": r.limit_pct, "ret": round(float(r.ret), 4),
                    "prior_susp_days": susp, "last_trade_close": last_close, "exright_ratio": round(fr, 4),
                    "tencent_OHLC": tx_px, "cninfo_titles": titles})
        print(r.code, r.date[:10], susp, round(fr, 4), tx_px, titles[:150], flush=True)
    pd.DataFrame(out).to_csv(REPORTS / "e1a_rows.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
