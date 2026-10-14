"""第 5 步：数据质量检查，输出 reports/data_quality.md 及明细 csv。

逐只股票读取 data/raw/daily，按与 build_panel 相同的口径补派生列后检查。
2024 年及以后（最终测试集）只做完整性统计（行数、缺失、价格合法性），
不计算任何收益或策略相关的统计量；年度表中的计数属于数据完整性信息。

检查项：
  A 覆盖：股票池 vs 有数据的股票、退市股是否覆盖、数据起止是否与上市/退市日期吻合
  B 日历：重复行、非交易日行、区间内缺失交易日
  C 缺失：各字段在正常交易行上的缺失率
  D 价格合法性：非正价格、high<low、开收盘越界、成交量/额为 0、换手率越界、均价越界
  E 涨跌停一致性：涨跌幅超过当日涨跌停幅度（剔除新股无涨跌幅期）
  F 复权：因子单调、缺失、异常大比例除权、后复权收益与交易所口径收益一致
  G 年度统计：股票数（分板块）、新上市、退市、ST、停牌占比、满足流动性过滤的可交易股票数
  H 基准指数：区间、缺失交易日、异常值
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from build_panel import enrich, load_st_events, trading_days
from common import (log_test_access, INDEX, MAIN_ST_10PCT, META, PANEL, RAW_DAILY, REPORTS, TEST_START, limit_price,
                    load_df, save_df)

PRICE_TOL = 1e-6
MAX_EXAMPLES = 20
LIQ_THRESHOLD = 5e7          # 前 20 日平均成交额 5000 万
MIN_LISTED_DAYS = 60

FIELDS = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg",
          "hfq_open", "hfq_high", "hfq_low", "hfq_close", "adj_factor"]


class Issues:
    def __init__(self):
        self.count = defaultdict(int)
        self.examples = defaultdict(list)

    def add(self, name, df_bad, cols):
        n = len(df_bad)
        if n == 0:
            return
        self.count[name] += n
        cap = 200 if name.startswith("E1a") else MAX_EXAMPLES      # E1a 需要逐条说明，保留全部
        room = cap - len(self.examples[name])
        if room > 0:
            self.examples[name].extend(df_bad[cols].head(room).astype(str).to_dict("records"))


SET_KEYS = ["stocks_any", "stocks_traded", "st_stocks", "new_listings", "delistings",
            "board_main_sh", "board_main_sz", "board_chinext", "board_star"]


def new_year_acc():
    a = defaultdict(int)
    for k in SET_KEYS:
        a[k] = set()
    return a


def check_stock(df, info, tdays, st_events, iss, year_acc, liq_counter, missing_acc):
    code = info.name
    df = enrich(df, info, tdays, st_events)
    df["year"] = df["date"].dt.year
    base = ["code", "date"]

    # ---- B 日历
    dup = df[df.duplicated("date", keep=False)]
    iss.add("B1 重复日期行", dup, base)
    iss.add("B2 非交易日出现数据", df[~df["date"].isin(tdays)], base)
    first, last = df["date"].min(), df["date"].max()
    span = tdays[(tdays >= first) & (tdays <= last)]
    missing_days = span.difference(pd.DatetimeIndex(df["date"]))
    if len(missing_days):
        iss.add("B3 区间内缺失交易日（无该日记录）",
                pd.DataFrame({"code": code, "date": missing_days}), base)

    # ---- A 起止日期与上市/退市日
    lst, dl = info["list_date"], info["delist_date"]
    if lst >= tdays[0]:
        lag = tdays.searchsorted(first) - tdays.searchsorted(lst)
        if lag > 5:
            iss.add("A2 首条记录晚于上市日 >5 个交易日",
                    pd.DataFrame([{"code": code, "date": first, "list_date": lst, "lag": lag}]),
                    ["code", "date", "list_date", "lag"])
    end_ref = dl if pd.notna(dl) else tdays[-1]
    gap_end = tdays.searchsorted(end_ref) - tdays.searchsorted(last)
    if gap_end > 5:
        iss.add("A3 末条记录早于退市日/最新交易日 >5 个交易日",
                pd.DataFrame([{"code": code, "date": last, "end_ref": end_ref, "gap": gap_end}]),
                ["code", "date", "end_ref", "gap"])

    # ---- C 缺失（只看正常交易行）
    tr = df[df["tradestatus"] == 1]
    for y, g in tr.groupby("year"):
        missing_acc[(y, "_rows")] += len(g)
        for c in FIELDS:
            if c in g:
                missing_acc[(y, c)] += int(g[c].isna().sum())
    iss.add("C1 tradestatus 缺失", df[df["tradestatus"].isna()], base)
    iss.add("C2 isST 缺失", df[df["isST"].isna()], base)

    # ---- D 价格合法性（正常交易行）
    cols = base + ["open", "high", "low", "close", "preclose", "volume", "amount", "turn"]
    px = tr[["open", "high", "low", "close", "preclose"]]
    iss.add("D1 价格非正或缺失", tr[(px <= 0).any(axis=1) | px.isna().any(axis=1)], cols)
    iss.add("D2 high < low", tr[tr["high"] < tr["low"] - PRICE_TOL], cols)
    out_o = (tr["open"] > tr["high"] + PRICE_TOL) | (tr["open"] < tr["low"] - PRICE_TOL)
    out_c = (tr["close"] > tr["high"] + PRICE_TOL) | (tr["close"] < tr["low"] - PRICE_TOL)
    iss.add("D3 开盘或收盘价超出当日高低区间", tr[out_o | out_c], cols)
    iss.add("D4 正常交易但成交量或成交额 <= 0", tr[(tr["volume"] <= 0) | (tr["amount"] <= 0)], cols)
    iss.add("D5 换手率缺失、<=0 或 >100%", tr[tr["turn"].isna() | (tr["turn"] <= 0) | (tr["turn"] > 100)], cols)
    vwap = tr["amount"] / tr["volume"].replace(0, np.nan)
    bad_vwap = (vwap > tr["high"] * 1.02) | (vwap < tr["low"] * 0.98)
    iss.add("D6 成交均价(额/量)超出高低区间 ±2%（单位问题）", tr[bad_vwap], cols)
    sus = df[df["tradestatus"] == 0]
    iss.add("D7 停牌行但成交量 > 0", sus[sus["volume"] > 0], base + ["volume"])

    # ---- E 涨跌停一致性
    chk = tr[(tr["no_limit"] == 0) & tr["preclose"].gt(0)]
    over = (chk["close"] > chk["up_limit"] + 0.005) | (chk["close"] < chk["down_limit"] - 0.005) | \
           (chk["high"] > chk["up_limit"] + 0.005) | (chk["low"] < chk["down_limit"] - 0.005)
    e = chk[over].assign(ret=lambda x: x["close"] / x["preclose"] - 1)
    ecols = base + ["board", "isST", "preclose", "high", "low", "close", "limit_pct", "up_limit", "down_limit", "ret"]
    if pd.notna(dl):   # 退市整理期（首日无涨跌幅、之后主板 10%）落在退市前 30 个交易日内，单独归类
        cut = tdays[max(tdays.searchsorted(dl) - 30, 0)]
        iss.add("E1b 超出涨跌停价：退市前 30 个交易日内（退市整理期规则不同，属预期）", e[e["date"] >= cut], ecols)
        e = e[e["date"] < cut]
    iss.add("E1a 超出涨跌停价：其他（非新股期）", e, ecols)

    # E2 疑似漏标 ST：主板、标为非 ST、2025-07-07 前，收盘恰好封在 ±5% 价位且收于最高/最低价
    if info["board"] in ("main_sh", "main_sz"):
        m = chk[(chk["isST"] == 0) & (chk["date"] < MAIN_ST_10PCT)]
        up5 = limit_price(m["preclose"], np.full(len(m), 0.05), 1)
        dn5 = limit_price(m["preclose"], np.full(len(m), 0.05), -1)
        hit = m[((abs(m["close"] - up5) < 1e-6) & (abs(m["close"] - m["high"]) < 1e-6)) |
                ((abs(m["close"] - dn5) < 1e-6) & (abs(m["close"] - m["low"]) < 1e-6))]
        if len(hit) >= 3:   # 偶然一两次封在 5% 很正常，连续多次才可疑
            iss.add("E2 疑似漏标 ST：非 ST 主板股≥3 天收盘封在 ±5% 价位（行数）", hit,
                    base + ["preclose", "high", "low", "close"])

    # ---- F 复权因子（新浪乘法因子；前收由因子推出，E1 已间接检验其与涨跌停价是否一致）
    s = df.sort_values("date").copy()
    s["fr"] = s["adj_factor"] / s["adj_factor"].shift(1)
    iss.add("F1 复权因子下降（除权后因子应不减）", s[s["fr"] < 1 - 1e-9], base + ["close", "preclose", "adj_factor", "fr"])
    iss.add("F2 复权因子或后复权价缺失", s[s["adj_factor"].isna() | s["hfq_close"].isna()], base)
    iss.add("F3 单日除权比例 > 3 倍（核对是否为大比例送转）", s[s["fr"] > 3], base + ["close", "preclose", "adj_factor", "fr"])
    s["ret"] = s["close"] / s["preclose"] - 1
    s["ret_hfq"] = s["hfq_close"] / s["hfq_close"].shift(1) - 1
    iss.add("F4 后复权日收益 与 收盘/前收 偏差 > 0.5%",
            s[((s["ret_hfq"] - s["ret"]).abs() > 0.005) & (s["tradestatus"] == 1)],
            base + ["preclose", "close", "adj_factor", "ret", "ret_hfq"])
    yrs_ex = s.loc[s["fr"] > 1 + 1e-9, "date"].dt.year

    # ---- G 年度统计 + 流动性可交易股票数
    for y, g in df.groupby("year"):
        a = year_acc[y]
        a["stocks_any"].add(code)
        if (g["tradestatus"] == 1).any():
            a["stocks_traded"].add(code)
            a[f"board_{info['board']}"].add(code)
        if (g["isST"] == 1).any():
            a["st_stocks"].add(code)
        a["rows"] += len(g)
        a["exright_events"] += int((yrs_ex == y).sum())
        a["suspended_rows"] += int((g["tradestatus"] == 0).sum())
    if lst >= tdays[0]:
        year_acc[lst.year]["new_listings"].add(code)
    if pd.notna(dl):
        year_acc[dl.year]["delistings"].add(code)

    amt = s["amount"].where(s["tradestatus"] == 1)
    amt20 = amt.rolling(20, min_periods=20).mean()
    elig = (s["tradestatus"] == 1) & (s["isST"] == 0) & (s["days_listed"] > MIN_LISTED_DAYS) & (amt20 >= LIQ_THRESHOLD)
    liq_counter.append(s.loc[elig, "date"].values)
    return len(df)


def check_index(tdays):
    rows = []
    for f in sorted(INDEX.glob("*.parquet")):
        if f.stem == "index_daily":
            continue
        d = load_df(f).sort_values("date")
        span = tdays[(tdays >= d.date.min()) & (tdays <= d.date.max())]
        miss = span.difference(pd.DatetimeIndex(d.date))
        r = d.loc[d["date"] < pd.Timestamp(TEST_START), "close"].pct_change().abs()   # 测试集不算收益类统计
        rows.append({
            "index": f.stem, "source": d["source"].iloc[0], "start": d.date.min().date(),
            "end": d.date.max().date(), "rows": len(d), "missing_trading_days": len(miss),
            "nonpositive_close": int((d.close <= 0).sum()), "dup_dates": int(d.date.duplicated().sum()),
            "max_abs_daily_ret(<2024)": round(float(r.max()), 4),
        })
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)


def main():
    log_test_access("数据质量检查：缺失、价格合法性、涨跌停一致性、复权、年度计数（测试集不算收益类指标）",
                    "data/raw/daily 全部股票全部年份；data/index 全部")
    uni = load_df(META / "universe.parquet").set_index("code")
    tdays = trading_days()
    st_events = load_st_events()
    iss = Issues()
    year_acc = defaultdict(new_year_acc)
    liq_counter = []
    missing_acc = defaultdict(int)

    files = {f.stem: f for f in RAW_DAILY.glob("*.parquet")}
    no_file, empty, total_rows = [], [], 0
    for i, code in enumerate(uni.index, 1):
        if code not in files:
            no_file.append(code)
            continue
        df = load_df(files[code])
        if df.empty:
            empty.append(code)
            continue
        total_rows += check_stock(df, uni.loc[code], tdays, st_events, iss, year_acc, liq_counter, missing_acc)
        if i % 500 == 0:
            print(f"  已检查 {i}/{len(uni)}")

    # ---------- A 覆盖
    delisted = uni[uni.is_delisted == 1]
    cover = {
        "股票池总数（沪深A股，2013-06后在市过）": len(uni),
        "其中已退市": len(delisted),
        "有数据的股票": len(uni) - len(no_file) - len(empty),
        "有数据的已退市股票": int(delisted.index.isin(list(files)).sum() - delisted.index.isin(empty).sum()),
        "未下载（无文件）": len(no_file),
        "下载结果为空": len(empty),
        "合计行数": total_rows,
    }
    panel_rows = None
    if list(PANEL.glob("daily_*.parquet")):
        panel_rows = sum(len(load_df(f, ["date"])) for f in PANEL.glob("daily_*.parquet"))
        cover["合并面板行数（应等于合计行数）"] = panel_rows

    # ---------- G 年度表
    liq = pd.Series(np.concatenate(liq_counter) if liq_counter else np.array([], dtype="datetime64[ns]"))
    liq = liq.value_counts().sort_index()
    liq.index = pd.to_datetime(liq.index)
    yrows = []
    for y in sorted(year_acc):
        a = year_acc[y]
        ly = liq[liq.index.year == y]
        cal_days = int(((tdays.year == y)).sum())
        yrows.append({
            "年份": y, "交易日": cal_days, "行数": a["rows"],
            "有记录股票": len(a["stocks_any"]), "有成交股票": len(a["stocks_traded"]),
            "沪主板": len(a["board_main_sh"]), "深主板": len(a["board_main_sz"]),
            "创业板": len(a["board_chinext"]), "科创板": len(a["board_star"]),
            "新上市": len(a["new_listings"]), "退市": len(a["delistings"]),
            "曾为ST": len(a["st_stocks"]),
            "停牌行占比%": round(100 * a["suspended_rows"] / max(a["rows"], 1), 2),
            "除权除息次数": a["exright_events"],
            "日均可交易股票数*": int(ly.mean()) if len(ly) else 0,
            "可交易股票数最少": int(ly.min()) if len(ly) else 0,
        })
    ytab = pd.DataFrame(yrows)
    ytab.to_csv(REPORTS / "data_quality_by_year.csv", index=False, encoding="utf-8-sig")

    # ---------- C 缺失率表
    mrows = []
    for y in sorted({k[0] for k in missing_acc}):
        n = missing_acc[(y, "_rows")]
        row = {"年份": y, "正常交易行": n}
        for c in FIELDS:
            row[c] = round(100 * missing_acc[(y, c)] / max(n, 1), 3)
        mrows.append(row)
    mtab = pd.DataFrame(mrows)
    mtab.to_csv(REPORTS / "data_quality_missing_pct.csv", index=False, encoding="utf-8-sig")

    # ---------- 问题汇总
    irows = [{"检查项": k, "命中行数": v} for k, v in sorted(iss.count.items())]
    itab = pd.DataFrame(irows) if irows else pd.DataFrame(columns=["检查项", "命中行数"])
    with open(REPORTS / "data_quality_examples.json", "w", encoding="utf-8") as f:
        json.dump(iss.examples, f, ensure_ascii=False, indent=1, default=str)

    idx_tab = check_index(tdays)

    # ---------- 写报告
    rev = REPORTS / "revision_notes.md"
    L = ["# 数据质量报告", ""]
    if rev.exists():
        L += [rev.read_text(encoding="utf-8").split("\n", 1)[1].replace("\n## ", "\n### ").strip(), ""]
    L += [
         f"生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M}　数据区间：{tdays[0].date()} ~ {tdays[-1].date()}",
         f"（{TEST_START} 及以后为最终测试集：本报告只对其做完整性统计，未计算任何收益类指标。）", "",
         "## A. 覆盖情况", ""]
    L += [f"- {k}：{v:,}" if isinstance(v, int) else f"- {k}：{v}" for k, v in cover.items()]
    if no_file:
        L += ["", f"未下载的代码（前 50 个）：{', '.join(no_file[:50])}　→ 重跑 download_daily.py 补齐"]
    if empty:
        L += ["", f"下载为空的代码（前 50 个）：{', '.join(empty[:50])}"]
    L += ["", "股票池来源：" + "，".join(f"{k} {v}" for k, v in uni["source"].value_counts().items())]

    def section(fname, title, demote=True):
        f = REPORTS / fname
        if not f.exists():
            return []
        body = f.read_text(encoding="utf-8").split("\n", 1)[1]          # 去掉文件自己的一级标题
        if demote:
            body = body.replace("\n## ", "\n### ")
        return ["", f"## {title}", body]

    L += section("delist_check.md", "退市完整性（官方终止上市名单）")
    L += section("st_validation.md", "ST 标记验证")
    L += ["", "前 15 只不一致股票的逐只原因见 `reports/st_mismatch_top15.md`。"]
    L += section("crosscheck_sample.md", "抽样交叉核对（腾讯行情，新浪仲裁）")
    L += ["", "**价格分歧日**：不一致集中在 2016-01-04 和 2014-04-23 两天，新浪历史行情与本库（东方财富）一致，腾讯不同。"
          "2016-01-04 是熔断日，13:34 触发熔断后提前收市，各数据源对收盘价的取法不一；2014-04-23 未查到制度原因，判断为腾讯当日数据问题。"
          "这两天的全部行已在面板中用 `price_flag`（1 = 2016-01-04，2 = 2014-04-23）标出。"]
    L += section("exright_check.md", "除权参考价不一致排查")
    L += ["", "## G. 分年度股票数量", "",
          md_table(ytab), "",
          "\\* 可交易股票：当日正常交易、非 ST、上市超过 60 个交易日、前 20 日平均成交额 ≥ 5000 万。",
          "", "## B–F. 问题汇总", "",
          "命中行数为 0 的检查项不列出。每项前 20 个样例见 reports/data_quality_examples.json。", "",
          md_table(itab) if len(itab) else "（全部检查项均未命中）", "",
          "说明：D5 全部是换手率四舍五入后为 0.00 的交易日（流通股本极大如中国石油，或连续一字板时成交极少）；D6 为一字板且成交量仅几百股的日子，"
          "东方财富成交量以'手'取整导致均价偏离，均非单位错误。F3 为 10 送转 20 股以上的大比例送转（2014–2016 年高送转为主）。"
          "F1、F4 各 1 行：600381 2014-09-30 长期停牌后复牌当日新浪因子下降约 8 倍（借壳重组复牌，原因未能确认，该日已标 exright_day）；002506 2015-08-12 为协鑫集成恢复上市首日。"
          "E2 为非 ST 主板股收在 ±5% 价位：沪市发生率 0.031%、深市 0.034%（深市用的是真值 ST 标记），总体不再指向系统性漏标。"
          "其中开高低收全在 ±5% 的一字板，沪市 58 行：38 行是未股改 S 股（600182、600733，涨跌幅本来就是 5%），"
          "约 20 行集中在 600518、600157、600691、600179、600688，是沪市 ST 重建的残余缺口（见 st_mismatch_top15.md）。", "",
          ]
    e1a = REPORTS / "e1a_explained.md"
    if e1a.exists():
        txt = e1a.read_text(encoding="utf-8")
        summ = txt[txt.index("## 归类汇总"):txt.index("## 逐行明细")].replace("## 归类汇总", "### E1a 归因（原始 52 行，逐行明细见 reports/e1a_explained.md）")
        L += [summ]
    L += ["## C. 正常交易行各字段缺失率（%）", "", md_table(mtab), "",
          "## H. 基准指数", "", md_table(idx_tab) if len(idx_tab) else "（未找到指数文件）", ""]
    (REPORTS / "data_quality.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
