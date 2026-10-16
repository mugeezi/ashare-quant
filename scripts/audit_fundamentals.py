"""第 3 项：基本面数据核对（只做数据完整性检查，不计算收益）。
A) 随机抽 50 个公司-报告期（2013–2023 报告期、公告日不晚于 2023-12-31），从巨潮下载该期首次披露的定期报告原文（PDF），
   核对东方财富业绩报表（RPT_LICO_FN_CPD）中的 营业总收入、归母净利润、销售毛利率，以及公告日。
   判定：东财数值能在原文中找到（按 元 / 千元 / 万元 / 百万元 换算，容差为原文末位的半个单位）记为一致；
   毛利率用原文"营业成本"候选值和东财营收重算，相差 ≤ 0.05 个百分点记为一致。
   找不到时，再到下一年同期报告里找（"上年同期"列），判断东财存的是否为追溯调整后的值。
B) 业绩预告：随机抽 50 条预告记录，查巨潮该股该报告期全部"业绩预告"类公告（含修正）；
   另从巨潮直接检索 30 条"业绩预告修正"公告，看东财预告表如何记录：
   修正公告是否单独成行（日期 = 修正日），还是只有一行、日期为原公告日、内容却是修正后的数字（= 修正版沿用原公告日期）。
PDF 存 data/audit/pdf/（下载的外部文件，只用 pdftotext 抽文本）。结果写 reports/audit_fin.csv、audit_yjyg.csv、AUDIT.md。"""
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import sources as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data/audit/pdf"
PDF.mkdir(parents=True, exist_ok=True)
CUT = pd.Timestamp("2023-12-31")
CAT = {12: "category_ndbg_szsh", 6: "category_bndbg_szsh", 3: "category_yjdbg_szsh", 9: "category_sjdbg_szsh"}
_org = {}


def org_id(code):
    if not _org:
        m = pd.read_parquet(ROOT / "data/meta/cninfo_orgid.parquet")
        _org.update(dict(zip(m["code"].astype(str).str.zfill(6), m["orgId"])))
    if code not in _org:
        js = S.get_json("https://www.cninfo.com.cn/new/information/topSearch/query", data={"keyWord": code, "maxNum": 5})
        hit = [x for x in js if x.get("code") == code]
        _org[code] = hit[0]["orgId"] if hit else None
    return _org[code]


def announcements(code, cat, s, e, key=""):
    org = org_id(code)
    if org is None:
        return []
    col = "sse" if code.startswith("6") else "szse"
    out, page = [], 1
    while True:
        js = S.get_json(S.CNINFO_QUERY, data=dict(pageNum=page, pageSize=30, column=col, tabName="fulltext", plate="",
                                                   stock=f"{code},{org}", searchkey=key, secid="", category=cat, trade="",
                                                   seDate=f"{s.date()}~{e.date()}", sortName="", sortType="", isHLtitle="true"))
        ann = js.get("announcements") or []
        for a in ann:
            out.append(dict(title=re.sub(r"<.*?>", "", a["announcementTitle"]), url=a["adjunctUrl"],
                            date=pd.to_datetime(a["announcementTime"], unit="ms", utc=True).tz_convert("Asia/Shanghai")
                            .tz_localize(None).normalize()))
        if page >= (js.get("totalpages") or 0) or not ann:
            break
        page += 1
        time.sleep(0.3)
    return out


def pdf_text(url):
    f = PDF / url.replace("/", "_")
    if not f.exists():
        r = S.session().get("https://static.cninfo.com.cn/" + url, timeout=120)
        r.raise_for_status()
        f.write_bytes(r.content)
        time.sleep(0.3)
    t = subprocess.run(["pdftotext", "-layout", str(f), "-"], capture_output=True, text=True, timeout=180).stdout
    return t


NUM = re.compile(r"(?<![\d.])-?\(?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?|(?<![\d.,])-?\d+\.\d+")


def numbers(text):
    out = []
    for m in NUM.finditer(text):
        s = m.group(0)
        neg = s.startswith("-") or s.startswith("(")
        s2 = s.strip("-()").replace(",", "")
        try:
            v = float(s2)
        except ValueError:
            continue
        dec = len(s2.split(".")[1]) if "." in s2 else 0
        out.append((-v if neg else v, dec))
    return out


def present(v, nums):
    if v is None or pd.isna(v):
        return None
    for x, dec in nums:
        for sc in (1, 1e3, 1e4, 1e6):
            tol = 0.5 * sc * 10 ** (-dec) + 1.0
            if abs(x * sc - v) <= tol and abs(x) > 0:
                return True
    return False


def cost_candidates(text):
    vals = []
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"^(其中[:：])?\s*营业成本", s) and "税金" not in s:
            vals += [x for x, _ in numbers(s)]
    return vals


def pick_report(anns, notice):
    good = [a for a in anns if not re.search(r"摘要|英文|English|更正|修订|补充|更新|取消|问询|说明|审计|内部控制|社会责任", a["title"])]
    if not good:
        return None
    good.sort(key=lambda a: (abs((a["date"] - notice).days), a["date"]))
    return good[0]


def in_universe(df, date_col):
    """只保留回测股票池中的 A 股（沪深京主板 / 创业板 / 科创板），且公告时已上市、未退市；去掉新三板、B 股和上市前的报告期。"""
    u = pd.read_parquet(ROOT / "data/meta/universe.parquet")
    u["code6"] = u["code"].str[-6:]
    df = df.copy()
    df["code6"] = df["code6"].astype(str).str.zfill(6)
    df = df.merge(u[["code6", "list_date", "delist_date"]], on="code6", how="inner")
    d = pd.to_datetime(df[date_col])
    ok = (d >= pd.to_datetime(df["list_date"])) & (df["delist_date"].isna() | (d < pd.to_datetime(df["delist_date"])))
    return df[ok].drop(columns=["list_date", "delist_date"])


def audit_fin(n=50, seed=20261016):
    f = pd.read_parquet(ROOT / "data/events/fin.parquet")
    f = f[(f.period >= "2013-01-01") & (f.period <= CUT) & (f.notice_date <= CUT) & f.notice_date.notna()]
    f = in_universe(f, "notice_date")
    smp = f.sample(n=n, random_state=seed).reset_index(drop=True)
    rows = []
    for _, r in smp.iterrows():
        code, per, nd = r.code6, pd.Timestamp(r.period), pd.Timestamp(r.notice_date)
        cat = CAT[per.month]
        row = dict(code=code, period=per.date(), em_notice=nd.date(), em_revenue=r.total_operate_income,
                   em_netprofit=r.parent_netprofit, em_gm=r.xsmll)
        try:
            anns = announcements(code, cat, per, per + pd.DateOffset(months=5))
            rep = pick_report(anns, nd)
            if rep is None:
                row["status"] = "未找到原文"
                rows.append(row)
                continue
            row.update(cninfo_title=rep["title"], cninfo_date=rep["date"].date(), date_match=rep["date"] == nd,
                       revised_versions=sum(bool(re.search(r"更正|修订|更新", a["title"])) for a in anns))
            txt = pdf_text(rep["url"])
            nums = numbers(txt)
            row["revenue_ok"] = present(r.total_operate_income, nums)
            row["netprofit_ok"] = present(r.parent_netprofit, nums)
            if pd.notna(r.xsmll) and pd.notna(r.total_operate_income) and r.total_operate_income > 0:
                cands = cost_candidates(txt)
                gms = []
                for c in cands:
                    for sc in (1, 1e3, 1e4, 1e6):
                        gms.append((1 - c * sc / r.total_operate_income) * 100)
                row["gm_ok"] = any(abs(g - r.xsmll) <= 0.05 for g in gms) if gms else None
            # 不一致时到下一年同期报告找（追溯调整后的值会出现在"上年同期"列）
            bad = [k for k in ("revenue_ok", "netprofit_ok") if row.get(k) is False]
            if bad:
                nxt = per + pd.DateOffset(years=1)
                anns2 = announcements(code, cat, nxt, nxt + pd.DateOffset(months=5))
                rep2 = pick_report(anns2, nxt + pd.DateOffset(months=3))
                if rep2:
                    nums2 = numbers(pdf_text(rep2["url"]))
                    row["found_in_next_year"] = {k: present(r.total_operate_income if k == "revenue_ok" else r.parent_netprofit, nums2)
                                                 for k in bad}
            row["status"] = "已核对"
        except Exception as e:  # noqa: BLE001
            row["status"] = f"出错：{e}"[:120]
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
    return pd.DataFrame(rows)


def forecast_numbers(text):
    """预告文字里的金额 / 幅度数字（用于判断记录内容来自原公告还是修正公告）。"""
    return [x for x, _ in numbers(text or "")] + [float(x) for x in re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)\s*%", text or "")]


def match_pdf(nums, txt):
    if not nums:
        return None
    pn = numbers(txt) + [(float(x), 2) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", txt)]
    hit = sum(any(abs(abs(a) - abs(b)) <= max(0.006, 1e-6 * abs(a)) for b, _ in pn) for a in nums)
    return hit / len(nums)


def classify(y_rows, anns):
    """y_rows：东财该股该报告期的预告记录；anns：巨潮该股该报告期的业绩预告类公告。"""
    orig = sorted([a for a in anns if "修正" not in a["title"]], key=lambda a: a["date"])
    rev = sorted([a for a in anns if "修正" in a["title"]], key=lambda a: a["date"])
    em_dates = sorted(set(pd.to_datetime(y_rows.notice_date).dt.normalize()))
    out = dict(n_cninfo_orig=len(orig), n_cninfo_rev=len(rev), n_em_dates=len(em_dates),
               em_dates=",".join(str(d.date()) for d in em_dates),
               cninfo_dates=",".join(f"{a['date'].date()}{'(修正)' if '修正' in a['title'] else ''}" for a in sorted(anns, key=lambda a: a["date"])))
    if not rev:
        out["verdict"] = "无修正公告"
        return out
    rev_dates = {a["date"] for a in rev}
    if any(d in rev_dates for d in em_dates):
        out["verdict"] = "修正单独成行（日期为修正日）"
        return out
    # 修正公告没有对应日期的记录：看记录内容来自原公告还是修正公告
    yr = y_rows.iloc[-1]
    nums = forecast_numbers(str(yr.text))
    try:
        m_rev = match_pdf(nums, pdf_text(rev[-1]["url"]))
        m_org = match_pdf(nums, pdf_text(orig[-1]["url"])) if orig else None
    except Exception as e:  # noqa: BLE001
        out["verdict"] = f"下载失败：{e}"[:80]
        return out
    out.update(match_rev=m_rev, match_orig=m_org)
    if m_rev is not None and m_rev >= 0.99 and (m_org is None or m_org < 0.99):
        out["verdict"] = "修正版沿用原公告日期"
    elif m_org is not None and m_org >= 0.99:
        out["verdict"] = "只记录原公告，修正未入表"
    else:
        out["verdict"] = "无法判定"
    return out


def audit_yjyg(n=50, n_rev=30, seed=20261016):
    y = pd.read_parquet(ROOT / "data/events/yjyg.parquet")
    y = y[(y.notice_date <= CUT) & (y.report_period >= "2013-01-01") & (y.report_period <= CUT)]
    y = y[y.metric.astype(str).str.contains("归属于上市公司股东的净利润", na=False)]
    y = in_universe(y, "notice_date")
    rows = []
    smp = y.drop_duplicates(["code6", "report_period"]).sample(n=n, random_state=seed)
    targets = [("随机抽样", r.code6, pd.Timestamp(r.report_period)) for _, r in smp.iterrows()]
    # 定向：巨潮检索"业绩预告修正"公告，随机取 n_rev 条
    random.seed(seed)
    revs = []
    for yr in range(2014, 2024):
        for s, e in ((f"{yr}-01-01", f"{yr}-04-30"), (f"{yr}-07-01", f"{yr}-08-31"), (f"{yr}-10-01", f"{yr}-10-31")):
            js = S.cninfo_search("业绩预告修正", s, e, 1)
            for a in js.get("announcements") or []:
                t = re.sub(r"<.*?>", "", a["announcementTitle"])
                if "修正" in t and "预告" in t and a.get("secCode"):
                    revs.append((a["secCode"], pd.to_datetime(a["announcementTime"], unit="ms"), t))
            time.sleep(0.3)
    for code, d, t in random.sample(revs, min(n_rev, len(revs))):
        m = re.search(r"(20\d\d)\s*年(?:度|半年度|第一季度|前三季度|第三季度|1-3月|1-9月)?", t)
        yr = int(m.group(1)) if m else d.year - (1 if d.month <= 4 else 0)
        if "半年" in t or "1-6" in t:
            per = pd.Timestamp(f"{yr}-06-30")
        elif "第一季度" in t or "1-3" in t:
            per = pd.Timestamp(f"{yr}-03-31")
        elif "三季" in t or "1-9" in t:
            per = pd.Timestamp(f"{yr}-09-30")
        else:
            per = pd.Timestamp(f"{yr}-12-31")
        if per <= CUT and d <= CUT:
            targets.append(("定向：修正公告", code, per))
    for src, code, per in targets:
        row = dict(source=src, code=code, period=per.date())
        try:
            anns = announcements(code, "category_yjygjxz_szsh", per - pd.DateOffset(days=60), min(per + pd.DateOffset(months=5), CUT))
            anns = [a for a in anns if re.search(r"预告|预亏|预盈|预增|预减|扭亏", a["title"])
                    and not re.search(r"英文|取消|说明|问询|快报", a["title"])]
            yy = y[(y.code6 == code) & (pd.to_datetime(y.report_period) == per)].sort_values("notice_date")
            row["n_em_rows"] = len(yy)
            if len(yy) == 0:
                row["verdict"] = "东财无记录"
            else:
                row.update(classify(yy, anns))
        except Exception as e:  # noqa: BLE001
            row["verdict"] = f"出错：{e}"[:120]
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("fin", "all"):
        audit_fin().to_csv(ROOT / "reports/audit_fin.csv", index=False)
    if what in ("yjyg", "all"):
        audit_yjyg().to_csv(ROOT / "reports/audit_yjyg.csv", index=False)
