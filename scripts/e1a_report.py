"""E1a 逐行说明报告：读取 e1a_explain.py 留下的证据表（reports/e1a_rows.csv，修复前的 52 行），
按证据归类，并对照最新质检结果（data_quality_examples.json）标注每行是否已修复。输出 reports/e1a_explained.md。"""
import json

import pandas as pd

from common import REPORTS

NOTES = {
    "A": "交易所规则：暂停上市后恢复上市首日不设涨跌幅。已用巨潮“股票恢复上市”公告定位该日，设 no_limit=1。",
    "B": "交易所规则：股权分置改革方案实施后复牌首日不设涨跌幅。已用巨潮“股改方案实施”公告及简称 S 前缀消失定位该日，设 no_limit=1。",
    "C": "破产重整计划资本公积转增：交易所按重整计划调整除权参考价（公司发布“调整除权参考价格”公告），新浪复权因子未反映或按常规公式计算，本库推算前收与交易所前收不同。",
    "D": "*ST 公司重大资产重组长期停牌后复牌首日，价格超出按 5% 计算的涨跌停价。可能是停牌期间 ST 状态已变化，或该日适用特殊安排，未能用公开公告确认。",
    "E": "配股除权：除权参考价 =（前收 + 配股价 × 实际配售比例）/（1 + 实际配售比例），交易所按缴款后的实际配售比例计算；新浪因子与之不一致：要让当日最高价不超过涨停价，交易所前收至少要比本库推算值高 0.1%–0.5%。",
    "F": "海润光伏 2014-06-12 除权：当日发生除权参考价错误，公司随后发布“关于 6 月 12 日除权事宜自查情况的公告”。",
    "G": "送转、派息或股改追送对价的除权日，推算前收比交易所前收至少低 0.1%–1%（个别如 600610 股改追送对价约 10%），按推算前收计算的涨停价因此偏低。",
}


def category(r):
    t = str(r.cninfo_titles)
    if "恢复上市" in t:
        return "A 恢复上市首日（不设涨跌幅）"
    if ("股权分置" in t or "股改" in t) and "追送" not in t and r.exright_ratio != 1.1:
        return "B 股改实施后复牌首日（不设涨跌幅）"
    if r.code == "sh.600401":
        return "F 除权参考价事故（海润光伏 2014-06-12）"
    if "重整" in t:
        return "C 重整计划资本公积转增除权（交易所调整除权参考价）"
    if r.prior_susp_days == 6 and r.exright_ratio != 1:
        return "E 配股除权日（配股缴款期停牌后复牌，按实际配售比例除权）"
    if r.exright_ratio != 1:
        return "G 其他除权除息日：推算前收与交易所口径有差"
    return "D *ST 公司长期停牌后复牌首日"


def main():
    rows = pd.read_csv(REPORTS / "e1a_rows.csv")
    cur = pd.DataFrame(json.load(open(REPORTS / "data_quality_examples.json", encoding="utf-8"))
                       .get("E1a 超出涨跌停价：其他（非新股期）", []))
    remain = set(zip(cur.code, cur.date.str[:10])) if len(cur) else set()
    rows["类别"] = rows.apply(category, axis=1)
    rows["当前状态"] = ["已修复" if (c, d) not in remain else
                    ("仍在 E1a（原因未能完全确认）" if k.startswith("D") else "仍在 E1a，已标 exright_day")
                    for c, d, k in zip(rows.code, rows.date, rows["类别"])]
    rows.to_csv(REPORTS / "e1a_rows.csv", index=False, encoding="utf-8-sig")
    n_fixed = int((rows["当前状态"] == "已修复").sum())
    L = ["# E1a 逐行说明（修复前 52 行）", "",
         "E1a：正常交易日、不在新股无涨跌幅期、不在退市整理期，价格却超出按本库前收和涨跌幅规则算出的当日涨跌停价。",
         "证据来源：巨潮资讯上该股票当日前 20 天至后 3 天的公告标题、本库停牌天数与复权因子变化、腾讯行情当日开高低收。", "",
         "## 归类汇总", "", "| 类别 | 行数 | 当前状态 | 说明 |", "|---|---|---|---|"]
    for k, g in rows.groupby(rows["类别"].str[0]):
        st = "；".join(f"{s} {n}" for s, n in g["当前状态"].value_counts().items())
        L.append(f"| {g['类别'].iloc[0]} | {len(g)} | {st} | {NOTES[k]} |")
    L += ["", f"已修复 {n_fixed} 行：A、B 两类通过识别恢复上市首日、股改复牌首日并设 no_limit=1 消除。",
          f"仍保留 {len(rows) - n_fixed} 行：多数是除权除息日前收推算与交易所不一致（C/E/F/G），这些日子在面板中已用 exright_day=1 标出；"
          "D 类原因未能确认。除权日事先公告，回测引擎可以据此对当日的涨跌停判断做保守处理。", "",
          f"当前 E1a 共 {len(remain)} 行（包括上表仍保留的行）。", "",
          "## 逐行明细", "",
          "| # | 代码 | 日期 | 当日涨跌幅 | 此前停牌天数 | 复权因子变化 | 类别 | 状态 | 关键公告 |", "|---|---|---|---|---|---|---|---|---|"]
    keys = ["恢复上市", "股权分置", "股改", "重整", "配股", "除权", "复牌", "转增"]
    for i, r in enumerate(rows.itertuples(), 1):
        t = " ｜ ".join(x for x in str(r.cninfo_titles).split(" | ") if any(k in x for k in keys))[:160] or "—"
        L.append(f"| {i} | {r.code} | {r.date} | {r.ret:+.1%} | {r.prior_susp_days} | {r.exright_ratio:.4f} | "
                 f"{r.类别[0]} | {r.当前状态} | {t} |")
    (REPORTS / "e1a_explained.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:22]))


if __name__ == "__main__":
    main()
