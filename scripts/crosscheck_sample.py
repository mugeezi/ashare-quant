"""质检补充：随机抽样，用腾讯行情独立核对东方财富不复权价格与新浪复权因子。

抽样：随机 80 只股票（含已退市），每只随机 3 个年份。
  1) 不复权开高低收：东财 vs 腾讯（day），差异 > 0.011 元记为不一致；两边交易日集合的差异。
  2) 除权除息参考价：腾讯 qfq 数据附带的分红送转记录（"10派X元送Y股转Z股"）按交易所公式
     (前一日收盘 − X/10) / (1 + (Y+Z)/10) 算出参考前收，与本库由新浪因子推出的前收比对（差 > 1 分为不一致）；
     并统计双方除权日集合的差异。腾讯的后复权价是加法复权，不能直接比日收益。
结果写 reports/crosscheck_sample.md，2024 年以后的年份只比对价格，不涉及任何策略统计。
"""
import random
import re

import numpy as np
import pandas as pd

from common import log_test_access, META, RAW_DAILY, REPORTS, SPECIAL_PRICE_DATES, WARMUP_START, load_df

SPECIAL_DATES = list(SPECIAL_PRICE_DATES)
from sources import get_json

TX = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"


def tx_kline(code, year, fq):
    """返回 (行情 DataFrame, 分红送转事件列表)。"""
    sym = code.replace(".", "")
    j = get_json(TX, params={"param": f"{sym},day,{year}-01-01,{year}-12-31,400,{fq}"})
    d = (j.get("data") or {}).get(sym) or {}
    if not isinstance(d, dict):
        return pd.DataFrame(), []
    rows = d.get({"": "day", "qfq": "qfqday"}[fq]) or d.get("day") or []
    events = [r[6] for r in rows if len(r) > 6 and isinstance(r[6], dict)]
    df = pd.DataFrame([r[:6] for r in rows], columns=["date", "open", "close", "high", "low", "vol"])
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
        for c in ["open", "close", "high", "low"]:
            df[c] = pd.to_numeric(df[c])
    return df, events


def parse_fh(text):
    num = lambda pat: sum(float(x) for x in re.findall(pat, text)) if re.search(pat, text) else 0.0
    cash = num(r"派([\d.]+)元")
    shares = num(r"送([\d.]+)股") + num(r"转(?:增)?([\d.]+)股")
    base = float(re.match(r"([\d.]+)", text).group(1)) if re.match(r"([\d.]+)", text) else 10.0
    return cash / base, shares / base, ("配" in text)


def main(n_stocks=80, n_years=3, seed=7):
    log_test_access("抽样交叉核对：本库价格与腾讯/新浪是否一致、除权参考价是否一致", "抽样股票的随机年份（可能含 2024 年以后）")
    rnd = random.Random(seed)
    uni = load_df(META / "universe.parquet")
    codes = rnd.sample(list(uni.code), n_stocks)
    res, px_bad_examples, ret_bad_examples = [], [], []
    special_cmp = {}
    for code in codes:
        ours = load_df(RAW_DAILY / f"{code}.parquet")
        ours = ours[ours.tradestatus == 1]
        years = sorted(ours.date.dt.year.unique())
        for y in rnd.sample(years, min(n_years, len(years))):
            o = ours[ours.date.dt.year == y]
            raw, _ = tx_kline(code, y, "")
            _, events = tx_kline(code, y, "qfq")
            raw = raw[raw.date >= pd.Timestamp(WARMUP_START)] if len(raw) else raw   # 本库从 2013-06 起
            if raw.empty:
                res.append({"code": code, "year": y, "status": "腾讯无数据"})
                continue
            m = o.merge(raw, on="date", suffixes=("", "_tx"))
            diff = np.maximum.reduce([(m[c] - m[f"{c}_tx"]).abs() for c in ["open", "high", "low", "close"]])
            bad = m[diff > 0.011]
            for b_ in bad.itertuples():   # 第三方（新浪）仲裁
                try:
                    import akshare as ak
                    sn = ak.stock_zh_a_daily(symbol=code.replace(".", ""), start_date=b_.date.strftime("%Y%m%d"),
                                             end_date=b_.date.strftime("%Y%m%d"), adjust="")
                    sina_close = float(sn["close"].iloc[0]) if len(sn) else np.nan
                except Exception:
                    sina_close = np.nan
                px_bad_examples.append({"code": code, "date": b_.date.date(), "本库开": b_.open, "腾讯开": b_.open_tx,
                                        "本库收": b_.close, "腾讯收": b_.close_tx, "新浪收": sina_close,
                                        "裁定": "本库" if abs(sina_close - b_.close) < 0.006 else
                                        ("腾讯" if abs(sina_close - b_.close_tx) < 0.006 else "不确定")})
            for dd in SPECIAL_DATES:
                hit = m[m.date == dd]
                if len(hit):
                    special_cmp.setdefault(dd, [0, 0])
                    special_cmp[dd][0] += len(hit)
                    special_cmp[dd][1] += int((diff[hit.index] > 0.011).sum())
            only_tx = raw[~raw.date.isin(o.date)]
            r = {"code": code, "year": y, "status": "ok", "rows_ours": len(o), "rows_tx": len(raw),
                 "common": len(m), "px_mismatch": len(bad),
                 "only_ours": int((~o.date.isin(raw.date)).sum()), "only_tx": len(only_tx),
                 "only_tx_zero_vol": int((pd.to_numeric(only_tx.vol) == 0).sum())}
            # 除权除息参考价核对
            full = load_df(RAW_DAILY / f"{code}.parquet").sort_values("date")
            full["prev_close"] = full["close"].shift(1)
            ours_ex = set(full.loc[(full.adj_factor / full.adj_factor.shift(1) - 1).abs() > 1e-9, "date"]
                          [lambda x: x.dt.year == y])
            tx_ex, n_chk, n_bad = set(), 0, 0
            for ev in events:
                cqr = pd.Timestamp(ev.get("cqr"))
                tx_ex.add(cqr)
                cash, shares, has_rights = parse_fh(ev.get("FHcontent", ""))
                row = full[full.date == cqr]
                if has_rights or row.empty or pd.isna(row.prev_close.iloc[0]):
                    continue
                ref = round((row.prev_close.iloc[0] - cash) / (1 + shares), 2)
                n_chk += 1
                if abs(ref - row.preclose.iloc[0]) > 0.011:
                    n_bad += 1
                    ret_bad_examples.append({"code": code, "date": cqr.date(), "方案": ev.get("FHcontent"),
                                             "前一日收盘": row.prev_close.iloc[0], "参考价(公式)": ref,
                                             "本库前收": row.preclose.iloc[0]})
            r.update(ex_checked=n_chk, ex_mismatch=n_bad, ex_only_ours=len(ours_ex - tx_ex), ex_only_tx=len(tx_ex - ours_ex))
            res.append(r)
    df = pd.DataFrame(res)
    ok = df[df.status == "ok"]
    L = ["# 抽样交叉核对（腾讯行情）", "",
         f"抽样 {n_stocks} 只 × 最多 {n_years} 个年份 = {len(df)} 个股票-年；腾讯无数据 {int((df.status != 'ok').sum())} 个。", "",
         "| 项目 | 比对行数 | 不一致行数 | 不一致率 |", "|---|---|---|---|",
         f"| 不复权开高低收（差 > 1 分） | {int(ok.common.sum()):,} | {int(ok.px_mismatch.sum()):,} | "
         f"{ok.px_mismatch.sum()/max(ok.common.sum(),1):.4%} |",
         f"| 除权除息参考前收（差 > 1 分） | {int(ok.ex_checked.sum()):,} | {int(ok.ex_mismatch.sum()):,} | "
         f"{ok.ex_mismatch.sum()/max(ok.ex_checked.sum(),1):.2%} |", "",
         f"交易日集合：仅本库有 {int(ok.only_ours.sum())} 行；仅腾讯有 {int(ok.only_tx.sum())} 行"
         f"（其中腾讯成交量为 0、本库按停牌处理的 {int(ok.only_tx_zero_vol.sum())} 行）。",
         f"除权日集合：仅本库（新浪因子）有 {int(ok.ex_only_ours.sum())} 次，仅腾讯有 {int(ok.ex_only_tx.sum())} 次"
         f"（配股不在腾讯分红记录里，会出现在'仅本库'）。", "",
         "价格不一致按日期：" + "；".join(f"{d.date()} 比对 {v[0]} 行、不一致 {v[1]} 行" for d, v in special_cmp.items())
         + f"；其余日期不一致 {int(ok.px_mismatch.sum()) - sum(v[1] for v in special_cmp.values())} 行。", "",
         "价格不一致明细（逐行用新浪历史行情做第三方仲裁）：", "", pd.DataFrame(px_bad_examples).to_markdown(index=False) if px_bad_examples else "无", "",
         "除权参考价不一致样例：", "", pd.DataFrame(ret_bad_examples[:10]).to_markdown(index=False) if ret_bad_examples else "无"]
    (REPORTS / "crosscheck_sample.md").write_text("\n".join(L), encoding="utf-8")
    df.to_csv(REPORTS / "crosscheck_sample.csv", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
