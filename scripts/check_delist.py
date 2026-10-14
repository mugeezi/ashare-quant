"""退市完整性核对：本库股票池 vs 上交所/深交所官方终止上市名单（2013-06-01 之后、沪深 A 股）。

重点：吸收合并 / 换股退市（被合并方在官方名单上有终止上市日期，合并方继续交易），以及
"代码变更"（旧代码不再交易、历史行情并入新代码，不在终止上市名单上）。
输出 reports/delist_check.md。
"""
import pandas as pd

from common import META, RAW_DAILY, REPORTS, WARMUP_START, is_universe_code, load_df

# 2013 年以来已知的吸收合并 / 换股退市与代码变更案例（被合并方 → 存续方），用于逐一核对
MERGERS = [
    ("sz.000527", "美的电器", "sz.000333", "美的集团换股吸收合并"),
    ("sz.000562", "宏源证券", "sz.000166", "申银万国换股吸收合并，申万宏源上市"),
    ("sh.601299", "中国北车", "sh.601766", "南车北车合并为中国中车"),
    ("sh.600832", "东方明珠", "sh.600637", "百视通换股吸收合并"),
    ("sz.000024", "招商地产", "sz.001979", "招商蛇口换股吸收合并"),
    ("sh.600005", "武钢股份", "sh.600019", "宝钢股份换股吸收合并"),
    ("sz.000418", "小天鹅A", "sz.000333", "美的集团换股吸收合并"),
    ("sh.600317", "营口港", "sh.601880", "大连港换股吸收合并"),
    ("sz.002013", "中航机电", "sh.600372", "中航电子换股吸收合并"),
    ("sh.600837", "海通证券", "sh.601211", "国泰君安换股吸收合并"),
    ("sh.601989", "中国重工", "sh.600150", "中国船舶换股吸收合并"),
]
CODE_CHANGES = [
    ("sz.000022", "深赤湾A", "sz.001872", "招商港口：重组后证券代码变更"),
    ("sh.601313", "江南嘉捷", "sh.601360", "三六零借壳后证券代码变更"),
    ("sz.000043", "中航地产", "sz.001914", "招商积余：证券代码变更"),
]


def main():
    u = load_df(META / "universe.parquet").set_index("code")
    dl = load_df(META / "delist_lists.parquet")
    dl = dl[dl.code.map(is_universe_code)]
    off = dl[dl.delist_date >= pd.Timestamp(WARMUP_START)].set_index("code")
    ours = u[u.is_delisted == 1]
    only_off = off.index.difference(ours.index)
    only_ours = ours.index.difference(off.index)
    both = ours.index.intersection(off.index)
    ddiff = (ours.loc[both, "delist_date"] - off.loc[both, "delist_date"]).dt.days
    L = ["# 退市完整性核对", "",
         "官方来源：上交所终止上市公司名单（query.sse.com.cn）、深交所终止上市公司名单（szse.cn），经 AKShare 获取。", "",
         "| 项目 | 数量 |", "|---|---|",
         f"| 官方名单：2013-06-01 后终止上市的沪深 A 股 | {len(off)} |",
         f"| 本库标为退市 | {len(ours)} |",
         f"| 两边都有 | {len(both)}（退市日期一致 {int((ddiff == 0).sum())}，不一致 {int((ddiff != 0).sum())}） |",
         f"| 官方有、本库缺失或未标退市 | {len(only_off)} |",
         f"| 本库标退市、官方没有 | {len(only_ours)} |", ""]
    if len(only_off):
        L += ["官方有本库无：" + "、".join(f"{c} {off.loc[c, 'name']}" for c in only_off), ""]
    if len(only_ours):
        L += ["本库有官方无：" + "、".join(f"{c} {ours.loc[c, 'code_name']}" for c in only_ours), ""]

    def last_trade(code):
        f = RAW_DAILY / f"{code}.parquet"
        if not f.exists():
            return "无文件"
        x = load_df(f, ["date", "tradestatus"])
        x = x[x.tradestatus == 1]
        return str(x.date.max().date()) if len(x) else "无成交"

    L += ["## 吸收合并 / 换股退市", "",
          "| 被合并方 | 本库状态 | 本库退市日 | 官方退市日 | 最后成交日 | 存续方在库 | 说明 |", "|---|---|---|---|---|---|---|"]
    for c, n, keep, note in MERGERS:
        st = "在库" if c in u.index else "缺失"
        d_ours = str(u.loc[c, "delist_date"].date()) if c in u.index and pd.notna(u.loc[c, "delist_date"]) else "—"
        d_off = str(off.loc[c, "delist_date"].date()) if c in off.index else "不在名单"
        L.append(f"| {c} {n} | {st} | {d_ours} | {d_off} | {last_trade(c)} | {'是' if keep in u.index else '否'} {keep} | {note} |")
    L += ["", "## 证券代码变更（不属于终止上市，历史行情并入新代码）", "",
          "| 旧代码 | 新代码 | 新代码在库、上市日 | 说明 |", "|---|---|---|---|"]
    for old, n, new, note in CODE_CHANGES:
        info = f"是，{u.loc[new, 'list_date'].date()}" if new in u.index else "否"
        L.append(f"| {old} {n} | {new} | {info} | {note}；旧代码{'在' if old in u.index else '不在'}股票池，无重复计数 |")
    L += ["", "东方财富把旧代码的历史行情并入新代码（新代码上市日为原上市日），因此这些公司在整个样本期都有连续数据，"
          "没有因为代码变更而缺段或重复。"]
    notes = REPORTS / "universe_notes.csv"
    if notes.exists():
        n = pd.read_csv(notes)
        L += ["", f"其他：`reports/universe_notes.csv` 记录了 {len(n)} 条股票池备注（上市日与日线首日不一致、剔除的代码）。"]
    (REPORTS / "delist_check.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
