"""Tushare"调整前"财务数据与巨潮首次披露原文核对（2026-10-17，用户第 2 步）。
token 只从环境变量 TUSHARE_TOKEN 读取，不写入任何文件、不打印。
样本：第 3 项同一批 50 个公司-报告期中找到原文的 45 个（reports/audit_fin.csv），原文 PDF 已缓存在 data/audit/pdf/。

版本选取规则（运行前写定）：同一公司-报告期，若有 report_type=5（调整前合并报表）的记录，取其中 f_ann_date（缺失时用 ann_date）
最早的一条；否则取 report_type=1（合并报表）中 f_ann_date 最早的一条。
逐字段判定（方法同 audit_fundamentals.present：数值能在首次披露原文中找到）：
  营收 = total_revenue（营业总收入）；归母净利润 = n_income_attr_p；
  毛利率 = revenue（营业收入）与 oper_cost（营业成本）都能在原文中找到，且由它们算出的毛利率与原文一致。
公告日：f_ann_date 是否等于巨潮原文发布日。结果写 reports/audit_tushare.csv。"""
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
import audit_fundamentals as A  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.tushare.pro"
FIELDS = "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,update_flag,total_revenue,revenue,oper_cost,n_income_attr_p"


def ts(api, **params):
    tok = os.environ["TUSHARE_TOKEN"]
    for i in range(5):
        r = requests.post(API, json=dict(api_name=api, token=tok, params=params, fields=FIELDS), timeout=30)
        js = r.json()
        if js.get("code") == 0:
            d = js["data"]
            return pd.DataFrame(d["items"], columns=d["fields"])
        if "每分钟" in str(js.get("msg")) or "频率" in str(js.get("msg")):
            time.sleep(20 * (i + 1))
            continue
        raise RuntimeError(f"tushare {js.get('code')}: {js.get('msg')}")
    raise RuntimeError("tushare 频率限制重试失败")


def pick(df):
    if df.empty:
        return None, None
    df = df.copy()
    df["d"] = df["f_ann_date"].fillna(df["ann_date"])
    for rt in ("5", "1"):
        x = df[df["report_type"].astype(str) == rt].sort_values("d")
        if len(x):
            return x.iloc[0], rt
    return None, None


def main():
    a = pd.read_csv(ROOT / "reports/audit_fin.csv", dtype={"code": str})
    a = a[a.status == "已核对"]
    rows = []
    for _, r in a.iterrows():
        code = str(r.code).zfill(6)
        per = pd.Timestamp(r.period)
        tsc = code + (".SH" if code.startswith(("6", "9")) else ".BJ" if code.startswith(("8", "4")) else ".SZ")
        anns = A.announcements(code, A.CAT[per.month], per, per + pd.DateOffset(months=5))
        rep = A.pick_report(anns, pd.Timestamp(r.em_notice))
        txt = A.pdf_text(rep["url"])
        nums = A.numbers(txt)
        row = dict(code=code, period=per.date(), first_date=rep["date"].date())
        try:
            parts = [ts("income", ts_code=tsc, period=per.strftime("%Y%m%d"), report_type=rt) for rt in ("1", "5")]
            df = pd.concat(parts, ignore_index=True)
            row["n_rows"] = len(df)
            row["types"] = ",".join(sorted(df["report_type"].astype(str).unique())) if len(df) else ""
            x, rt = pick(df)
            if x is None:
                row["status"] = "Tushare 无记录"
            else:
                row["used_type"] = rt
                row["f_ann_date"] = x.f_ann_date
                row["ann_date"] = x.ann_date
                row["date_ok"] = str(x.f_ann_date or x.ann_date) == rep["date"].strftime("%Y%m%d")
                row["rev_ok"] = A.present(float(x.total_revenue), nums) if pd.notna(x.total_revenue) else None
                row["np_ok"] = A.present(float(x.n_income_attr_p), nums) if pd.notna(x.n_income_attr_p) else None
                if pd.notna(x.revenue) and pd.notna(x.oper_cost) and float(x.revenue) > 0:
                    row["gm_ok"] = bool(A.present(float(x.revenue), nums) and A.present(float(x.oper_cost), nums))
                    row["ts_gm"] = round((1 - float(x.oper_cost) / float(x.revenue)) * 100, 3)
                row["status"] = "已核对"
        except Exception as e:  # noqa: BLE001
            row["status"] = f"出错：{e}"[:120]
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
        time.sleep(0.4)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reports/audit_tushare.csv", index=False)
    c = df[df.status == "已核对"]
    for k in ("rev_ok", "np_ok", "gm_ok", "date_ok"):
        if k in c:
            v = c[k].dropna().astype(bool)
            print(k, f"{v.sum()}/{len(v)} = {v.mean():.1%}" if len(v) else "无")


if __name__ == "__main__":
    main()
