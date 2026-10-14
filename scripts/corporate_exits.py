"""持仓退市时的处理依据：换股比例、现金选择权/要约价格（从上市公司公告逐条摘录，注明出处）。

class:
  merger      换股吸收合并：持仓按 ratio 转为存续公司（target_code）股票
  voluntary   主动终止上市（现金选择权）或全面要约收购后退市：按 cash 价格兑现
  forced      强制退市且有退市整理期：持仓在整理期内按可成交价格卖出，卖不掉的到退市日按 0 计
  forced_no_period  强制退市但没有可交易的整理期（2020 年后的交易类退市等）：最后成交日后锁定，退市日按 0 计
输出 data/meta/corporate_exits.parquet（含全部 2013-06 后退市股票）。
"""
import pandas as pd

from common import META, REPORTS, load_df, save_df

# code: (class, 存续方代码, 每股换得存续方股数, 每股现金, 出处)
SOURCED = {
    "sz.000527": ("merger", "sz.000333", 0.3447, None, "美的电器《关于美的集团发行股份吸收合并本公司的换股实施公告》2013-09-11"),
    "sh.600253": ("merger", "sh.600056", 0.310, None, "天方药业《关于调整现金选择权价格、换股价格及换股比例的公告》2013-05-16"),
    "sz.000562": ("merger", "sz.000166", 2.049382716, None, "宏源证券《关于申万宏源发行股份吸收合并本公司的换股实施公告》2015-01-21"),
    "sh.601299": ("merger", "sh.601766", 1.10, None, "中国北车《关于A股股票终止上市的公告》2015-05-17"),
    "sh.600832": ("merger", "sh.600637", 1 / 3.05, None, "东方明珠《关于换股价格、换股比例及现金选择权价格调整的公告》2015-04-16（3.05 股换 1 股）"),
    "sz.300186": ("merger", "sz.300498", 0.8070, None, "大华农《关于温氏股份发行股份吸收合并公司的换股实施公告》2015-10-29"),
    "sz.000024": ("merger", "sz.001979", 1.6008, None, "招商地产《关于招商局蛇口控股换股吸收合并本公司的换股实施公告》2015-12-27（A 股 1:1.6008）"),
    "sz.000748": ("merger", "sz.000066", 0.5424, None, "长城信息《关于本次重大资产重组换股实施的提示性公告》2017-01-12"),
    "sh.600005": ("merger", "sh.600019", 0.56, None, "武钢股份《关于公司股票终止上市的公告》2017-02-07"),
    "sz.000916": ("merger", "sz.001965", 0.6956, None, "华北高速《关于招商公路换股吸收合并本公司换股实施的公告》2017-12-20"),
    "sh.600270": ("merger", "sh.601598", 3.8225, None, "外运发展《异议股东现金选择权实施公告》2018-12（换股比例 1:3.8225）"),
    "sz.000418": ("merger", "sz.000333", 1.15144821, None, "小天鹅《关于美的集团实施2018年度利润分配方案后调整换股比例事项的提示性公告》2019-05-22（A 股）"),
    "sh.600317": ("merger", "sh.601880", 1.5030, None, "营口港《关于公司股票终止上市的公告》2021-01-25"),
    "sh.600068": ("merger", "sh.601868", 4.4337, None, "葛洲坝《关于公司股票终止上市的公告》2021-09-09"),
    "sh.600723": ("merger", "sh.600859", 0.3058, None, "首商股份《关于公司股票终止上市的公告》2021-10-14"),
    "sz.000780": ("merger", "sz.001289", 0.3407, None, "ST平能《关于公司股票终止上市并摘牌的公告》2022-01-20"),
    "sz.002013": ("merger", "sh.600372", 0.66053, None, "中航机电《关于中航电子换股吸收合并本公司的换股实施公告》2023-03-16"),
    "sh.600837": ("merger", "sh.601211", 0.62, None, "海通证券换股实施公告 2025-02-28"),
    "sh.601989": ("merger", "sh.600150", 0.1339, None, "中国重工换股实施公告 2025-09-03"),
    # 主动终止上市（现金选择权）/ 全面要约收购后退市
    "sz.000602": ("voluntary", None, None, 13.46, "金马集团《要约收购结果公告》2013-07-23（全面要约 13.46 元）"),
    "sh.601268": ("voluntary", None, None, 2.59, "*ST二重《要约收购报告书》2015-02-16（全面要约 2.59 元）"),
    "sh.600680": ("voluntary", None, None, 6.74, "*ST上普《主动终止公司股票上市现金选择权申报提示性公告》2019-04-16（A 股 6.74 元）"),
    "sh.600677": ("voluntary", None, None, 4.18, "*ST航通《关于主动终止公司股票上市现金选择权申报的提示性公告》2021-02-03"),
    "sz.000666": ("voluntary", None, None, 9.24, "经纬纺机《主动终止上市现金选择权行权暨实施公告》2023-10-16"),
    "sh.600213": ("voluntary", None, None, 6.42, "*ST亚星 主动终止上市财务顾问意见 2024-09-13"),
    "sh.600705": ("voluntary", None, None, 3.54, "中航产融《关于向上交所提出终止上市申请的公告》2025-04-28"),
    "sh.601028": ("voluntary", None, None, 13.20, "玉龙股份《现金选择权申报结果的公告》2025-04-25"),
    "sz.000627": ("voluntary", None, None, 1.60, "*ST天茂《主动终止上市现金选择权行权暨实施公告》2025-09-16"),
    "sh.603056": ("voluntary", None, None, 19.00, "德邦股份《关于向上交所提出终止上市申请的公告》2026-02-26"),
}
# *ST宜生（600978）的 1.15 元为 6% 部分要约，并非全体股东可兑现；按强制退市处理


def main():
    dc = load_df(META / "delist_class.parquet")
    rows = []
    for r in dc.itertuples():
        if r.code in SOURCED:
            cls, tgt, ratio, cash, src = SOURCED[r.code]
        else:
            cls = "forced" if r.tui_trade_days > 0 else "forced_no_period"
            tgt = ratio = cash = None
            src = "整理期可交易 %d 天" % r.tui_trade_days if r.tui_trade_days > 0 else "无可交易的退市整理期，退市日按 0 计"
        rows.append(dict(code=r.code, name=r.name, delist_date=r.delist_date, last_trade=r.last_trade,
                         exit_class=cls, target_code=tgt, ratio=ratio, cash=cash, source=src))
    df = pd.DataFrame(rows)
    miss = dc[dc["class"].isin(["merger", "voluntary"]) & ~dc.code.isin(SOURCED)]
    save_df(df, META / "corporate_exits.parquet")
    L = ["# 退市处理依据", "", df.exit_class.value_counts().rename_axis("类别").reset_index(name="数量").to_markdown(index=False), "",
         "## 换股吸收合并与主动退市（逐条出处）", "",
         df[df.exit_class.isin(["merger", "voluntary"])][["code", "name", "delist_date", "last_trade", "exit_class", "target_code", "ratio", "cash", "source"]]
         .sort_values("delist_date").to_markdown(index=False), "",
         f"自动分类为换股/主动退市但未录入出处的：{', '.join(miss.code) if len(miss) else '无'}（600978 已说明按强制退市处理）"]
    (REPORTS / "corporate_exits.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
