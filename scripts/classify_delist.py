"""退市股分类（回测引擎处理持仓退市时使用）。

对 2013-06 后终止上市的每只股票，查：是否有可交易的退市整理期；退市前 240 天内巨潮公告标题是否涉及
吸收合并/换股、主动终止上市/现金选择权、要约收购。分类：
  forced      强制退市，有退市整理期（持仓在整理期内按可成交价格卖出）
  merger      换股吸收合并（持仓按换股比例转为存续公司股票，见 data/meta/merger_swaps.csv）
  voluntary   主动终止上市（现金选择权/要约收购，按公告价格兑现，见 merger_swaps.csv）
  other       以上都不是（引擎按 0 残值处理并单独报告）
输出 data/meta/delist_class.parquet、reports/delist_class.md
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from cninfo_names import fetch_stock, orgid_map
from common import META, RAW_DAILY, REPORTS, load_df, log_test_access, save_df

KEYS = r"吸收合并|换股|主动终止|终止上市.*现金选择权|现金选择权|要约收购|退市整理"


def one(args):
    code, name, dl, orgs = args
    x = load_df(RAW_DAILY / f"{code}.parquet", ["date", "tradestatus"])
    tr = x[x.tradestatus == 1]
    last = tr.date.max() if len(tr) else pd.NaT
    try:
        a = fetch_stock(code, orgs.get(code[3:], ""), str((dl - pd.Timedelta(days=240)).date()), str(dl.date()))
        a = a[a.title.str.contains(KEYS)]
        titles = " | ".join(f"{d.date()} {t}" for d, t in zip(a.date, a.title))
    except Exception as e:
        titles = f"取数失败 {type(e).__name__}"
    return code, name, dl, last, titles


def main():
    log_test_access("退市股分类（引擎处理持仓退市），只读日期与公告标题", "退市股日线交易状态、退市前公告")
    u = load_df(META / "universe.parquet")
    d = u[u.is_delisted == 1]
    p = []
    for y in range(2013, 2027):
        f = META.parent / "panel" / f"daily_{y}.parquet"
        if f.exists():
            q = load_df(f, ["code", "delist_period", "tradestatus"])
            p.append(q[q.code.isin(d.code) & (q.delist_period == 1) & (q.tradestatus == 1)].groupby("code").size())
    tui_days = pd.concat(p).groupby(level=0).sum() if p else pd.Series(dtype=int)
    orgs = orgid_map()
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, [(r.code, r.code_name, r.delist_date, orgs) for r in d.itertuples()]))
    df = pd.DataFrame(res, columns=["code", "name", "delist_date", "last_trade", "titles"])
    df["tui_trade_days"] = df.code.map(tui_days).fillna(0).astype(int)

    def cls(r):
        t = r.titles
        if re.search(r"吸收合并|换股", t) and r.tui_trade_days == 0:
            return "merger"
        if re.search(r"主动终止|现金选择权|要约收购", t) and r.tui_trade_days == 0:
            return "voluntary"
        if r.tui_trade_days > 0:
            return "forced"
        return "other"
    df["class"] = df.apply(cls, axis=1)
    save_df(df, META / "delist_class.parquet")
    L = ["# 退市股分类", "", df["class"].value_counts().rename_axis("类别").reset_index(name="数量").to_markdown(index=False), "",
         "## 非强制退市明细", "", df[df["class"] != "forced"][["code", "name", "delist_date", "last_trade", "class", "titles"]]
         .assign(titles=lambda x: x.titles.str.slice(0, 200)).to_markdown(index=False)]
    (REPORTS / "delist_class.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
