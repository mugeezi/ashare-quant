"""第 3 步：重建每日 ST（风险警示）标记。

东方财富日线没有 isST 字段，BaoStock 在云端连不上，所以用两个公开来源重建：
  深市：深交所"证券简称变更"全量记录（含生效日期）——精确，直接用。
  沪市：巨潮资讯检索标题含"风险警示"的公告（2010 年起），用两类信息：
        a) 公告时的证券简称（secName）是否带 ST/退 —— 当日状态的观测；
        b) 标题判定"实施…风险警示"（次一交易日起生效）、"撤销…风险警示"（第二个交易日起生效，
           中间一般停牌一天），排除"可能被实施""申请撤销""进展""律师意见"等非生效公告。
  巨潮方法同时在深市上跑一遍，与深交所精确记录对比，报告一致率，作为沪市 ST 标记的质量证据。

ST 口径：简称含 ST（ST、*ST、SST、S*ST）或处于退市整理期（简称含"退"）。
输出：
  data/meta/sz_name_changes.parquet   深交所原始记录
  data/meta/cninfo_risk_ann.parquet   巨潮原始公告列表
  data/meta/st_events.parquet         code, eff_date, state(0/1), source —— 每只股票的状态切换点
  reports/st_validation.md            巨潮方法 vs 深交所真值的一致率
"""
import re
import time
import unicodedata

import akshare as ak
import numpy as np
import pandas as pd

from build_panel import trading_days
from common import META, REPORTS, is_universe_code, limit_price, load_df, log_download, save_df, today_str
from sources import cninfo_search, retry

# 注意："撤销…的提示性公告"往往就是生效公告，所以不排除"提示性"；"可能被实施…"靠"可能"排除
NON_EFFECTIVE = ["申请", "不予", "争取", "措施", "消除", "进展", "意见", "说明", "律师", "核查", "专项", "问询", "回复",
                 "预告", "延期", "补充", "更正", "取消", "法律", "鉴证", "审核报告", "函", "问答"]


EV_COLS = ["code", "eff_date", "state", "source", "prio", "name"]


def _norm(name) -> str:
    return unicodedata.normalize("NFKC", str(name)).upper().replace(" ", "")


def is_s_name(name) -> bool:
    """未完成股权分置改革的股票（简称前缀 S、SST、S*ST），涨跌幅 5%。"""
    n = _norm(name)
    return n.startswith("S") and not n.startswith("ST")


def is_tui_name(name) -> bool:
    """退市整理期简称（"退市XX" 或 "XX退"）。"""
    n = _norm(name)
    return n.startswith("退") or n.endswith("退") or ("退市" in n)


def special_flags(ev_code, dates) -> tuple[np.ndarray, np.ndarray]:
    """由简称观测推出两个附加状态：
    isS  未股改（S 前缀）：从首次（或样本起点，若最早一条观测就是 S）到最后一次 S 简称观测；股改完成后不会再回到 S。
    tui  退市整理期：从"进入退市整理期"公告生效日或首次出现"退"字简称起，直到退市。"""
    d = pd.to_datetime(pd.Series(dates)).values
    s = np.zeros(len(d), dtype="int8")
    t = np.zeros(len(d), dtype="int8")
    if ev_code is None or not len(ev_code):
        return s, t
    obs = ev_code[ev_code["name"].fillna("").astype(str) != ""].sort_values("eff_date")
    sm = obs["name"].map(is_s_name)
    if sm.any():
        first = obs["eff_date"].values[0] if sm.iloc[0] else obs.loc[sm, "eff_date"].values[0]
        last = obs.loc[sm, "eff_date"].values[-1]
        s[(d >= (np.datetime64("1900-01-01") if sm.iloc[0] else first)) & (d <= last)] = 1
    starts = list(obs.loc[obs["name"].map(is_tui_name), "eff_date"].values)
    starts += list(ev_code.loc[ev_code["source"] == "cninfo_delist_start", "eff_date"].values)
    if starts:
        t[d >= min(starts)] = 1
    return s, t


def is_st_name(name: str) -> bool:
    n = unicodedata.normalize("NFKC", str(name)).upper().replace(" ", "")
    return ("ST" in n) or n.startswith("退") or n.endswith("退") or ("退市" in n)


def to_code(num: str):
    num = str(num).zfill(6)
    code = ("sh." if num.startswith("6") else "sz.") + num
    return code if is_universe_code(code) else None


def classify(title: str):
    """把公告标题判为 ST 状态的生效事件：start（实施风险警示 / 进入退市整理期）、end（撤销风险警示）或 None。"""
    t = re.sub(r"<[^>]+>", "", title).replace("撤消", "撤销")
    if any(w in t for w in ("债券", "公司债", "可转债")) and "股票" not in t:   # 只针对债券的"风险警示"与股票无关
        return None
    if "退市整理期" in t:
        return "start" if re.search(r"进入退市整理期", t) and not any(w in t for w in NON_EFFECTIVE) else None
    if "风险警示" not in t or any(w in t for w in NON_EFFECTIVE):
        return None
    # "可能被实施…""存在被实施…"是风险提示而非生效；但"被实施退市风险警示和公司债券可能被暂停交易"是生效公告
    if re.search(r"(可能|存在)(被)?(继续)?(实施|实行|撤销)", t):
        return None
    # "将被实施…的（第 N 次）提示性公告"是年报前的预警；"将被实施…暨股票停（复）牌"、
    # "法院受理重整…暨将被实施…的公告"是生效公告
    if "将被" in t and "提示" in t and not re.search(r"停(复)?牌", t):
        return None
    cont = re.search(r"继续(被)?(实施|实行)", t)                       # "继续被实施"：状态不变（"继续停牌"不算）
    if "即将" in t and not re.search(r"停(复)?牌", t):
        return None
    if "撤销" in t and "情形" in t:             # "撤销退市风险警示情形的公告"只说明条件已消除，不是生效
        return None
    if "撤销" in t:
        # "撤销退市风险警示并（继续）实施其他风险警示"：仍为 ST，不是摘帽
        if cont or (re.search(r"撤销.*(实施|实行)其他风险", t) and "及其他风险警示" not in t) \
                or re.search(r"撤销.*(实施|实行)退市风险", t):
            return None
        return "end"
    if ("实施" in t or "实行" in t) and not cont:
        return "start"
    return None


def fetch_cninfo(start_year=2010, keyword="风险警示"):
    rows = []
    end_year = int(today_str()[:4])
    for y in range(start_year, end_year + 1):
        s, e = f"{y}-01-01", f"{y}-12-31"
        first = cninfo_search(keyword, s, e, 1)
        pages = int(first.get("totalpages") or 0) + 1
        total = first.get("totalAnnouncement")
        got = 0
        for p in range(1, pages + 1):
            j = first if p == 1 else cninfo_search(keyword, s, e, p)
            anns = j.get("announcements") or []
            for a in anns:
                rows.append({"secCode": a.get("secCode"), "secName": a.get("secName"),
                             "title": re.sub(r"<[^>]+>", "", a.get("announcementTitle") or ""),
                             "date": pd.Timestamp(a["announcementTime"], unit="ms", tz="Asia/Shanghai").tz_localize(None).normalize(),
                             "announcementId": a.get("announcementId")})
            got += len(anns)
            if not anns:
                break
            time.sleep(0.3)
        print(f"  巨潮[{keyword}] {y}: 共 {total} 条，取得 {got}", flush=True)
    df = pd.DataFrame(rows).drop_duplicates("announcementId")
    df["code"] = df["secCode"].map(to_code)
    return df


def events_from_cninfo(ann: pd.DataFrame, tdays: pd.DatetimeIndex) -> pd.DataFrame:
    """把巨潮公告转成状态切换点。同一天先放观测，再放事件（事件优先）。"""
    def pos(d, k):
        """公告日之后第 k 个交易日。日历从 2013-06 开始，更早的公告用工作日近似。"""
        if d < tdays[0]:
            return d + pd.offsets.BDay(k)
        return tdays[min(tdays.searchsorted(d, side="right") + k - 1, len(tdays) - 1)]
    out = []
    for r in ann.dropna(subset=["code"]).itertuples():
        out.append((r.code, r.date, int(is_st_name(r.secName)), "cninfo_name", 0, r.secName))
        kind = classify(r.title)
        if kind == "start":
            out.append((r.code, pos(r.date, 1), 1,
                        "cninfo_delist_start" if "退市整理" in r.title else "cninfo_start", 1, ""))
        elif kind == "end":
            out.append((r.code, pos(r.date, 2), 0, "cninfo_end", 1, ""))
    ev = pd.DataFrame(out, columns=EV_COLS)
    return ev.sort_values(["code", "eff_date", "prio"]).reset_index(drop=True)


def events_from_szse(chg: pd.DataFrame) -> pd.DataFrame:
    c = chg.rename(columns={"变更日期": "d", "证券代码": "num", "变更前简称": "before", "变更后简称": "after"})
    c["d"] = pd.to_datetime(c["d"], errors="coerce")
    c["code"] = c["num"].astype(str).str.zfill(6).map(lambda n: "sz." + n)
    c = c[c.code.map(is_universe_code)].sort_values(["code", "d"])
    out = []
    for code, g in c.groupby("code"):
        out.append((code, pd.Timestamp("1990-01-01"), int(is_st_name(g["before"].iloc[0])), "szse_init", 0, g["before"].iloc[0]))
        for r in g.itertuples():
            out.append((code, r.d, int(is_st_name(r.after)), "szse_change", 1, r.after))
    return pd.DataFrame(out, columns=EV_COLS)


def initial_state(ev_code: pd.DataFrame) -> int:
    """第一条记录之前的状态：首条是实施→0，是撤销→1，是观测→观测值。"""
    first = ev_code.iloc[0]
    if first.source == "cninfo_start":
        return 0
    if first.source == "cninfo_end":
        return 1
    return int(first.state)


def flags_for(ev_code: pd.DataFrame | None, dates: pd.Series) -> np.ndarray:
    """给定某只股票的切换点和日期序列，返回每天的 ST 标记（0/1）。无记录的股票视为从未 ST。"""
    d = pd.to_datetime(dates).values
    if ev_code is None or len(ev_code) == 0:
        return np.zeros(len(d), dtype="int8")
    ev = ev_code.sort_values(["eff_date", "prio"], kind="stable")
    idx = np.searchsorted(ev["eff_date"].values, d, side="right") - 1
    st = np.where(idx >= 0, ev["state"].values[np.clip(idx, 0, None)], initial_state(ev))
    return st.astype("int8")


def refine_with_limits(flags: np.ndarray, df: pd.DataFrame, ev_code: pd.DataFrame | None) -> np.ndarray:
    """用涨跌停制度纠正巨潮方法（仅主板、仅 2026-07-06 前有效，之后主板 ST 也是 10%）。
    反证：标为 ST 的某天 收盘/最高/最低 偏离前收 > 5%×前收 + 0.006 → ST 必已结束：从这一天起置 0，
          直到下一条 ST 证据（实施公告生效日或带 ST 的简称观测）。
    正证：开高低收全部等于 ±5% 涨跌停价的"一字板"几乎只可能是 ST（深市真值里非 ST 股 13 年仅 1 次）→
          补标 ST，并向前/向后延伸到最近的反证日或撤销公告生效日，单向最多 250 个交易日，且不越过 2026-07-06。
    df 需含 date, close, high, low, preclose, tradestatus, board, no_limit。"""
    from common import MAIN_ST_10PCT
    f = flags.copy()
    if ev_code is None or not len(df) or df["board"].iloc[0] not in ("main_sh", "main_sz"):
        return f
    d = df["date"].values
    cutoff = np.datetime64(MAIN_ST_10PCT)
    pre_cut = d < cutoff
    ok = ((df["tradestatus"] == 1) & (df["no_limit"] == 0) & (df["preclose"] > 0)).values & pre_cut
    move = np.maximum.reduce([np.abs(df[c] - df["preclose"]).values for c in ("close", "high", "low")])
    contra = ok & (move > df["preclose"].values * 0.05 + 0.006)
    support = np.sort(ev_code.loc[ev_code["state"] == 1, "eff_date"].values)
    # 退市整理期（主板 10%）和未股改 S 股（5%，但不是 ST）不适用这两条制度证据；
    # 长期停牌（≥20 个交易日）后的复牌首日、股改完成后 10 个交易日内，常有不设涨跌幅或前收重定的特殊安排，不作反证
    s_flag, tui = special_flags(ev_code, df["date"])
    contra &= (tui == 0)
    ts = df["tradestatus"].values
    run = np.zeros(len(ts), dtype=int)                 # 截至前一日的连续停牌天数
    for k in range(1, len(ts)):
        run[k] = run[k - 1] + 1 if ts[k - 1] == 0 else 0
    contra &= ~((ts == 1) & (run >= 20))
    s_end = np.where((np.r_[0, s_flag[:-1]] == 1) & (s_flag == 0))[0]
    for k in s_end:
        contra[k:k + 10] = False
    ok_onep = ok & (s_flag == 0) & (tui == 0)

    pre = df["preclose"].values
    up5 = limit_price(pre, np.full(len(pre), 0.05), 1)
    dn5 = limit_price(pre, np.full(len(pre), 0.05), -1)
    hi, lo, cl = df["high"].values, df["low"].values, df["close"].values
    onep = ok_onep & (np.abs(hi - lo) < 1e-6) & ((np.abs(cl - up5) < 1e-6) | (np.abs(cl - dn5) < 1e-6))
    ends = np.sort(np.concatenate([
        ev_code.loc[ev_code["source"] == "cninfo_end", "eff_date"].values,
        ev_code.loc[(ev_code["state"] == 0) & ev_code["source"].str.contains("name"), "eff_date"].values]))  # 非 ST 简称观测也阻断延伸
    for i in np.where(onep & (f == 0))[0]:
        j = i
        while j >= 0 and i - j < 250 and not contra[j] and not (len(ends) and np.any((ends > d[j]) & (ends <= d[i]))):
            f[j] = 1
            j -= 1
        j = i + 1
        while j < len(f) and pre_cut[j] and j - i < 250 and not contra[j] \
                and not (len(ends) and np.any((ends > d[i]) & (ends <= d[j]))):
            f[j] = 1
            j += 1
    i = 0
    while i < len(f):
        if f[i] == 1 and contra[i]:
            k = np.searchsorted(support, d[i], side="right")
            nxt = support[k] if k < len(support) else np.datetime64("2100-01-01")
            j = i
            while j < len(f) and d[j] < nxt:
                f[j] = 0
                j += 1
            i = j
        else:
            i += 1
    return f


def cninfo_method_flags(ev_code: pd.DataFrame | None, df: pd.DataFrame) -> np.ndarray:
    """沪市 ST 标记（也用于在深市做验证）。
    2026-07-06 前：巨潮风险警示公告事件 + 公告简称观测 + 涨跌幅纠正。
    2026-07-06 起（简称法）：巨潮"全部公告"的发布日证券简称（逐只股票，约每周数条）+ 风险警示事件
      + 东方财富当日简称；不再使用涨跌幅纠正。2026 年没有任何 ST 迹象（无风险警示公告、当前简称不含 ST）
      的股票，2026-07-06 起直接判为非 ST。"""
    from common import MAIN_ST_10PCT
    f = flags_for(ev_code, df["date"])
    f = refine_with_limits(f, df, ev_code)
    has_dense = ev_code is not None and (ev_code["source"] == "cninfo_dense_name").any()
    if not has_dense:
        f[df["date"].values >= np.datetime64(MAIN_ST_10PCT)] = 0
    return f


def dense_name_events(which: str = "post_cutoff", since=None) -> pd.DataFrame:
    """巨潮逐只全量公告的简称观测 → 事件（state = 简称是否含 ST），以及东方财富当前简称观测。"""
    from common import MAIN_ST_10PCT
    since = pd.Timestamp(since or MAIN_ST_10PCT)
    out = []
    for f in sorted((META / "cninfo_names" / which).glob("*.parquet")):
        x = load_df(f)
        x = x[x["date"] >= since]
        out.append(pd.DataFrame({"code": f.stem, "eff_date": x["date"], "state": x["is_st_name"].astype(int),
                                 "source": "cninfo_dense_name", "prio": 0, "name": x["secName"]}))
    ev = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=EV_COLS)
    uni = load_df(META / "universe.parquet")
    uni = uni[uni.code.isin(ev.code.unique())]
    from build_panel import trading_days
    last = trading_days()[-1]
    cur = pd.DataFrame({"code": uni.code, "eff_date": last, "state": uni.code_name.map(is_st_name).astype(int),
                        "source": "cninfo_dense_name", "prio": 0, "name": uni.code_name})  # 东方财富当日简称
    return pd.concat([ev, cur], ignore_index=True)


def _px(code, info, tdays):
    """验证用：单只股票的日线 + 板块 + 新股无涨跌幅标记。"""
    from common import RAW_DAILY, no_limit_days
    px = load_df(RAW_DAILY / f"{code}.parquet", ["date", "close", "high", "low", "preclose", "tradestatus"])
    px["board"] = info.board
    lst = info.list_date if pd.notna(info.list_date) else px.date.min()
    px["no_limit"] = (px["date"] < tdays[min(tdays.searchsorted(lst) + no_limit_days(info.board, lst), len(tdays) - 1)]).astype(int)
    return px


def _score(rows):
    t = pd.DataFrame(rows, columns=["a", "b"])
    tp = int(((t.a == 1) & (t.b == 1)).sum()); fp = int(((t.a == 1) & (t.b == 0)).sum())
    fn = int(((t.a == 0) & (t.b == 1)).sum()); n = len(t)
    return dict(n=n, agree=(t.a == t.b).mean() if n else np.nan, rec=tp / max(tp + fn, 1), prec=tp / max(tp + fp, 1),
                fn=fn, fp=fp, pos=tp + fn)


def validate(ev_cn: pd.DataFrame, ev_sz: pd.DataFrame, tdays: pd.DatetimeIndex) -> str:
    from common import MAIN_ST_10PCT, log_test_access
    log_test_access("ST 标记验证：沪市方法 vs 深交所真值的逐日一致率（含 2024 年以后两个区间）",
                    "深市全部股票日线（判断涨跌幅纠正所需的价格）、深交所简称变更、巨潮公告")
    uni = load_df(META / "universe.parquet").set_index("code")
    g_cn = dict(tuple(ev_cn.groupby("code")))
    g_sz = dict(tuple(ev_sz.groupby("code")))
    periods = {"研究期 2014-01-01~2023-12-31": ("2014-01-01", "2023-12-31"),
               "2024-01-01~2026-07-03（主板 ST 5% 末期）": ("2024-01-01", "2026-07-03"),
               "2026-07-06 起（主板 ST 10%，简称法）": (str(MAIN_ST_10PCT.date()), "2100-01-01")}
    acc = {k: {"raw": [], "final": []} for k in periods}
    per_stock = []
    for code, info in uni[uni.index.str.startswith("sz.")].iterrows():
        px = _px(code, info, tdays)
        a_raw = flags_for(g_cn.get(code), px.date)
        a = cninfo_method_flags(g_cn.get(code), px)
        b = flags_for(g_sz.get(code), px.date)
        for k, (lo, hi) in periods.items():
            m = ((px.date >= lo) & (px.date <= hi)).values
            acc[k]["raw"] += list(zip(a_raw[m], b[m]))
            acc[k]["final"] += list(zip(a[m], b[m]))
        m = ((px.date >= "2014-01-01") & (px.date <= "2023-12-31")).values
        per_stock.append((code, info.code_name, int((a[m] != b[m]).sum())))

    L = ["# ST 标记验证", "",
         "ST 口径：证券简称含 ST/*ST/SST/S*ST，或处于退市整理期（简称含\"退\"）。", "",
         "## 1. 深市：沪市方法 vs 深交所简称变更（真值）", "",
         "沪市方法＝巨潮风险警示/退市整理公告 + 公告简称；2026-07-06 前再加涨跌幅纠正（反证 >5% 波动、正证 ±5% 一字板），"
         "2026-07-06 起改用简称法（巨潮逐只全量公告的发布日简称 + 东方财富当日简称）。股票-日粒度，仅在市期间。", "",
         "| 区间 | 方法 | 股票-日 | 一致率 | 召回率 | 精确率 | 漏标 | 误标 | 真值 ST 天数 |", "|---|---|---|---|---|---|---|---|---|"]
    for k in periods:
        for mk, label in (("raw", "仅公告"), ("final", "最终方法")):
            r = _score(acc[k][mk])
            L.append(f"| {k} | {label} | {r['n']:,} | {r['agree']:.3%} | {r['rec']:.2%} | {r['prec']:.2%} | "
                     f"{r['fn']:,} | {r['fp']:,} | {r['pos']:,} |")
    per_stock.sort(key=lambda x: -x[2])
    L += ["", "研究期内不一致天数最多的 15 只（原因见 st_mismatch_top15.md）：", "",
          "| 代码 | 名称 | 不一致天数 |", "|---|---|---|"] + [f"| {c} | {n} | {m} |" for c, n, m in per_stock[:15]]
    (REPORTS / "st_top15_sz.csv").write_text(
        pd.DataFrame(per_stock[:15], columns=["code", "name", "mismatch_days"]).to_csv(index=False), encoding="utf-8")

    # ---- 沪市验证：独立的巨潮逐只全量公告简称（不参与标记算法）
    vdir = META / "cninfo_names" / "validation_sh"
    if vdir.exists() and any(vdir.glob("*.parquet")):
        ann_ids = set(load_df(META / "cninfo_risk_ann.parquet")["announcementId"])
        rows, rows_tol, sh_stock = [], [], []
        for f in sorted(vdir.glob("*.parquet")):
            code = f.stem
            if code not in uni.index:
                continue
            obs = load_df(f)
            obs = obs[(obs.date >= "2014-01-01") & (obs.date <= "2023-12-31") & ~obs.announcementId.isin(ann_ids)]
            if obs.empty:
                continue
            px = _px(code, uni.loc[code], tdays)
            px["flag"] = cninfo_method_flags(g_cn.get(code), px)
            fl = px.set_index("date")["flag"]
            # 公告日可能是非交易日：取不晚于公告日的最近交易日；容差版允许相邻一个交易日
            idx = px.date.searchsorted(obs.date, side="right") - 1
            ok = idx >= 0
            a = px.flag.values[np.clip(idx, 0, None)][ok]
            a_next = px.flag.values[np.clip(idx + 1, 0, len(px) - 1)][ok]
            b = obs.is_st_name.values[ok]
            rows += list(zip(a, b))
            rows_tol += list(zip(np.where(a_next == b, b, a), b))
            sh_stock.append((code, uni.loc[code, "code_name"], int(ok.sum()), int((a != b).sum()),
                             int(((a != b) & (a_next != b)).sum())))
        r, rt = _score(rows), _score(rows_tol)
        n_st = sum(1 for x in sh_stock if x[0] in set(ev_cn.code))
        L += ["", "## 2. 沪市：最终 ST 标记 vs 巨潮逐只全量公告的发布日简称（独立来源）", "",
              f"样本：{len(sh_stock)} 只沪市股票（2010 年以来有风险警示公告或曾被标记 ST 的全部股票 + 随机 60 只对照），"
              "2014-2023 年每一条公告的发布日简称（剔除标记算法已用过的风险警示公告），与该日的 ST 标记比对。", "",
              "| 口径 | 比对点 | 一致率 | 召回率 | 精确率 | 漏标 | 误标 | 简称含 ST 的点 |", "|---|---|---|---|---|---|---|---|",
              f"| 严格（公告当日） | {r['n']:,} | {r['agree']:.3%} | {r['rec']:.2%} | {r['prec']:.2%} | {r['fn']:,} | {r['fp']:,} | {r['pos']:,} |",
              f"| 容差 1 个交易日 | {rt['n']:,} | {rt['agree']:.3%} | {rt['rec']:.2%} | {rt['prec']:.2%} | {rt['fn']:,} | {rt['fp']:,} | {rt['pos']:,} |"]
        sh_stock.sort(key=lambda x: -x[4])
        L += ["", "容差口径下不一致点最多的 15 只（原因见 st_mismatch_top15.md）：", "",
              "| 代码 | 名称 | 比对点 | 严格不一致 | 容差后不一致 |", "|---|---|---|---|---|"]
        L += [f"| {c} | {n} | {k} | {m} | {mt} |" for c, n, k, m, mt in sh_stock[:15]]
        (REPORTS / "st_top15_sh.csv").write_text(
            pd.DataFrame(sh_stock[:15], columns=["code", "name", "points", "mismatch", "mismatch_tol"]).to_csv(index=False),
            encoding="utf-8")
    return "\n".join(L)


def build_and_validate(chg, ann, tdays):
    ev_cn = events_from_cninfo(ann, tdays)
    ev_cn = pd.concat([ev_cn, dense_name_events("post_cutoff")], ignore_index=True) \
        .sort_values(["code", "eff_date", "prio"]).reset_index(drop=True)
    ev_sz = events_from_szse(chg)
    rep = validate(ev_cn, ev_sz, tdays)
    (REPORTS / "st_validation.md").write_text(rep, encoding="utf-8")
    print(rep)
    # 最终：深市用深交所真值；沪市用巨潮方法
    final = pd.concat([ev_sz, ev_cn[ev_cn.code.str.startswith("sh.")]], ignore_index=True)
    save_df(final, META / "st_events.parquet")
    print(f"ST 切换点 {len(final)} 条（深 {len(ev_sz)}，沪 {len(final)-len(ev_sz)}）")


def main():
    import sys
    tdays = trading_days()
    if "--rebuild" in sys.argv:   # 不重新下载，只用已保存的原始数据重建
        build_and_validate(load_df(META / "sz_name_changes.parquet"), load_df(META / "cninfo_risk_ann.parquet"), tdays)
        return
    t0 = time.time()
    chg = retry(lambda: ak.stock_info_sz_change_name(symbol="简称变更"))
    save_df(chg, META / "sz_name_changes.parquet")
    log_download("sz_name_changes", "akshare.stock_info_sz_change_name(简称变更) / szse.cn", "all", len(chg), t0)
    print(f"深交所简称变更 {len(chg)} 条")

    t0 = time.time()
    ann = pd.concat([fetch_cninfo(keyword="风险警示"), fetch_cninfo(keyword="退市整理")]).drop_duplicates("announcementId")
    save_df(ann, META / "cninfo_risk_ann.parquet")
    log_download("cninfo_risk_ann", "cninfo hisAnnouncement/query searchkey=风险警示 / 退市整理", "2010~今", len(ann), t0)
    build_and_validate(chg, ann, tdays)


if __name__ == "__main__":
    main()
