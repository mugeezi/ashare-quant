"""特殊交易日公告：巨潮资讯检索"恢复上市""股权分置改革"，用于识别不设涨跌幅的复牌首日。

relevant=True 的公告：
  - "恢复上市公告书"，或标题含"股票（交易）恢复上市"且不是申请/进展/措施/风险提示/意见/报告/说明/承诺/保荐类；
  - "股权分置改革（股改）…实施"，排除"进展""追送对价"以及股东减持类公告；
  - 一律排除含"债"的标题（公司债券恢复上市与股票无关）。
build_panel.py 取每个公告簇（90 天内）之后的首个复牌日，设 resume_no_limit=1、no_limit=1。
输出 data/meta/cninfo_relist_reform_ann.parquet
"""
import re
import time

import pandas as pd

from common import META, log_download, save_df
from download_st import fetch_cninfo


def relevant(t: str) -> bool:
    if "债" in t:
        return False
    if "恢复上市公告书" in t:
        return True
    if re.search(r"(股票|A股)(交易)?恢复上市", t) and not re.search(
            r"可能|申请|进展|措施|风险提示|意见|报告|说明|承诺|保荐", t):
        return True
    if re.search(r"(股权分置改革|股改).*实施", t) and not re.search(r"进展|追送|减持|股东|持股", t):
        return True
    return False


def main():
    t0 = time.time()
    parts = []
    for kw in ["恢复上市", "股权分置改革"]:
        d = fetch_cninfo(start_year=2013, keyword=kw)
        d["keyword"] = kw
        parts.append(d)
    a = pd.concat(parts).dropna(subset=["code"]).drop_duplicates("announcementId")
    a["relevant"] = a["title"].map(relevant)
    save_df(a, META / "cninfo_relist_reform_ann.parquet")
    log_download("cninfo_relist_reform_ann", "cninfo hisAnnouncement/query searchkey=恢复上市 / 股权分置改革",
                 "2013~今", len(a), t0, note=f"relevant {int(a.relevant.sum())}")
    print(f"公告 {len(a)} 条，其中用于识别复牌首日 {int(a.relevant.sum())} 条")


if __name__ == "__main__":
    main()
