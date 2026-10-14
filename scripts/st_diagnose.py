"""ST 不一致排查：对验证中不一致最多的股票，列出不一致区间及附近的证据（深交所简称变更、巨潮公告、行情）。
输出 reports/st_diagnose_raw.txt（人工归因的原始材料）。"""
import numpy as np
import pandas as pd

from build_panel import trading_days
from common import log_test_access, META, REPORTS, load_df
from download_st import (_px, cninfo_method_flags, dense_name_events, events_from_cninfo, events_from_szse,
                         flags_for)


def intervals(dates, mask):
    out, start, prev = [], None, None
    for d, m in zip(dates, mask):
        if m and start is None:
            start = d
        if not m and start is not None:
            out.append((start, prev)); start = None
        prev = d
    if start is not None:
        out.append((start, prev))
    return out


def main():
    log_test_access("ST 标记不一致排查（只看研究期，读取日线以复现标记）", "不一致最多股票的日线")
    td = trading_days()
    uni = load_df(META / "universe.parquet").set_index("code")
    ann = load_df(META / "cninfo_risk_ann.parquet")
    chg = load_df(META / "sz_name_changes.parquet")
    ev_cn = pd.concat([events_from_cninfo(ann, td), dense_name_events("post_cutoff")])
    g_cn = dict(tuple(ev_cn.groupby("code")))
    g_sz = dict(tuple(events_from_szse(chg).groupby("code")))
    L = []
    sz = pd.read_csv(REPORTS / "st_top15_sz.csv")
    for code in sz.code:
        info = uni.loc[code]
        px = _px(code, info, td)
        px = px[(px.date >= "2014-01-01") & (px.date <= "2023-12-31")].reset_index(drop=True)
        a = cninfo_method_flags(g_cn.get(code), px)
        b = flags_for(g_sz.get(code), px.date)
        L.append(f"=== {code} {info.code_name}  深市真值对比，不一致 {(a != b).sum()} 天")
        for s, e in intervals(px.date, a != b):
            seg = px[(px.date >= s) & (px.date <= e)]
            L.append(f"  区间 {s.date()}~{e.date()} 共 {len(seg)} 天：方法={a[seg.index[0]]} 真值={b[seg.index[0]]}；"
                     f"其中停牌 {(seg.tradestatus == 0).sum()} 天")
        z = chg[chg["证券代码"].astype(str).str.zfill(6) == code[3:]]
        L.append("  深交所简称变更: " + "; ".join(f"{d} {x}->{y}" for d, x, y in zip(z["变更日期"], z["变更前简称"], z["变更后简称"])
                                         if str(d) >= "2012"))
        an = ann[ann.code == code].sort_values("date")
        L.append("  巨潮风险警示公告: " + " | ".join(f"{d.date()}[{n}]{t[:30]}" for d, n, t in zip(an.date, an.secName, an.title)
                                              if d.year >= 2012)[:1500])
    sh = pd.read_csv(REPORTS / "st_top15_sh.csv")
    ids = set(ann.announcementId)
    for code in sh.code:
        info = uni.loc[code]
        px = _px(code, info, td)
        px["flag"] = cninfo_method_flags(g_cn.get(code), px)
        obs = load_df(META / "cninfo_names" / "validation_sh" / f"{code}.parquet")
        obs = obs[(obs.date >= "2014-01-01") & (obs.date <= "2023-12-31") & ~obs.announcementId.isin(ids)]
        idx = px.date.searchsorted(obs.date, side="right") - 1
        obs["flag"] = px.flag.values[np.clip(idx, 0, None)]
        bad = obs[obs.flag != obs.is_st_name]
        L.append(f"=== {code} {info.code_name}  沪市独立简称对比，不一致 {len(bad)} 点")
        L.append("  不一致点: " + " | ".join(f"{d.date()}[{n}]标记={f}" for d, n, f in zip(bad.date, bad.secName, bad.flag)))
        an = ann[ann.code == code].sort_values("date")
        L.append("  巨潮风险警示公告: " + " | ".join(f"{d.date()}[{n}]{t[:30]}" for d, n, t in zip(an.date, an.secName, an.title)
                                              if d.year >= 2012)[:1500])
    (REPORTS / "st_diagnose_raw.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
