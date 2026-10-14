"""巨潮资讯逐只股票的全部公告列表 → "公告日证券简称"观测序列（每日简称的近似）。

巨潮每条公告记录了发布时的证券简称（secName，如 "*ST天业"），因此一只股票的全部公告构成
一条按日期排列的简称观测序列（活跃公司约每周数条）。用途：
  post_cutoff ：2026-07-06 起主板 ST 也是 10%，涨跌幅纠正失效，改用简称法判定 ST（沪市）；
                同时取深市对应股票，用深交所真值验证简称法。
  validation_sh：沪市 ST 标记验证（2013-2023），取定期报告的发布日简称。只作验证用，不进入 ST 标记算法。
缓存：data/meta/cninfo_names/<set>/<code>.parquet
用法：python cninfo_names.py post_cutoff|validation_sh [--workers 4]
"""
import argparse
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from common import META, load_df, load_panel, log_download, save_df, today_str
from download_st import is_st_name
from sources import get_json

CNINFO_QUERY = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
OUT = META / "cninfo_names"
WINDOWS = {"post_cutoff": ("2026-05-01", today_str()), "validation_sh": ("2013-01-01", "2023-12-31")}


def orgid_map() -> dict:
    p = META / "cninfo_orgid.parquet"
    if not p.exists():
        j = get_json("https://www.cninfo.com.cn/new/data/szse_stock.json")
        save_df(pd.DataFrame(j["stockList"]), p)
    df = load_df(p)
    return dict(zip(df["code"], df["orgId"]))


CATEGORIES = {"post_cutoff": "",   # 全部公告
              # 验证集只取定期报告（年报、半年报、一季报、三季报）：每只股票每年 4 个以上简称观测点，
              # 请求量约为全量公告的十分之一
              "validation_sh": "category_ndbg_szsh;category_bndbg_szsh;category_yjdbg_szsh;category_sjdbg_szsh;"}


def fetch_stock(code: str, org: str, start: str, end: str, category: str = "") -> pd.DataFrame:
    num = code[3:]
    column = "sse" if code.startswith("sh.") else "szse"
    rows, page = [], 1
    while True:
        j = get_json(CNINFO_QUERY, data=dict(
            pageNum=page, pageSize=30, column=column, tabName="fulltext", plate="", stock=f"{num},{org}",
            searchkey="", secid="", category=category, trade="", seDate=f"{start}~{end}",
            sortName="", sortType="", isHLtitle="true"))
        anns = j.get("announcements") or []
        for a in anns:
            if a.get("secCode") != num:
                continue
            rows.append({"date": pd.Timestamp(a["announcementTime"], unit="ms", tz="Asia/Shanghai").tz_localize(None).normalize(),
                         "secName": a.get("secName"), "title": re.sub(r"<[^>]+>", "", a.get("announcementTitle") or ""),
                         "announcementId": a.get("announcementId")})
        if not anns or page >= int(j.get("totalpages") or 0) + 1:
            break
        page += 1
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=["date", "secName", "title", "announcementId"]).drop_duplicates("announcementId")
    df["code"] = code
    df["is_st_name"] = df["secName"].map(is_st_name).astype("int8") if len(df) else pd.Series(dtype="int8")
    return df.sort_values("date").reset_index(drop=True)


def target_codes(which: str) -> list[str]:
    uni = load_df(META / "universe.parquet")
    ann = load_df(META / "cninfo_risk_ann.parquet")
    if which == "validation_sh":
        ev = set(ann[ann.code.fillna("").str.startswith("sh.")].code)
        p = load_panel(columns=["code", "isST"])            # 研究期内（2023 年前）的标记，不触及测试集
        flagged = set(p[(p.isST == 1) & p.code.str.startswith("sh.")].code)
        st_related = sorted((ev | flagged) & set(uni.code))
        others = sorted(set(uni[uni.code.str.startswith("sh.") & (uni.list_date < "2023-01-01")].code) - set(st_related))
        random.Random(11).shuffle(others)
        return st_related + others[:60]
    # post_cutoff：2026 年以来有 ST 迹象的沪深股票（风险警示公告、深交所简称、当前简称）
    recent = set(ann[(ann.date >= "2026-01-01")].code.dropna())
    chg = load_df(META / "sz_name_changes.parquet")
    chg = chg[pd.to_datetime(chg["变更日期"]) >= "2025-01-01"]
    recent |= {"sz." + str(c).zfill(6) for c, a, b in zip(chg["证券代码"], chg["变更前简称"], chg["变更后简称"])
               if is_st_name(a) or is_st_name(b)}
    recent |= set(uni[uni.code_name.map(is_st_name)].code)
    return sorted(recent & set(uni[uni.delist_date.isna() | (uni.delist_date >= "2026-05-01")].code))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=list(WINDOWS))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    start, end = WINDOWS[args.which]
    out = OUT / args.which
    out.mkdir(parents=True, exist_ok=True)
    orgs = orgid_map()
    codes = [c for c in target_codes(args.which) if not (out / f"{c}.parquet").exists()]
    print(f"{args.which}: 待取 {len(codes)} 只，区间 {start}~{end}", flush=True)
    t0, n_rows, missing = time.time(), 0, []

    def task(code):
        num = code[3:]
        org = orgs.get(num) or (("gssh0" if code.startswith("sh.") else "gssz0") + num)
        df = fetch_stock(code, org, start, end, CATEGORIES[args.which])
        save_df(df, out / f"{code}.parquet")
        return code, len(df)

    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(task, c) for c in codes]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                code, n = f.result()
                n_rows += n
                if n == 0:
                    missing.append(code)
            except Exception as e:
                print("  失败:", str(e)[:150], flush=True)
            if i % 25 == 0 or i == len(codes):
                print(f"  {i}/{len(codes)}  公告 {n_rows}  无公告 {len(missing)}  {(time.time()-t0)/60:.1f} 分", flush=True)
    log_download(f"cninfo_names_{args.which}", "cninfo hisAnnouncement/query stock=<code,orgId>",
                 f"{start}~{end}; n={len(codes)}", n_rows, t0, note=f"无公告 {len(missing)} 只")


if __name__ == "__main__":
    main()
