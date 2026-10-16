"""第 3 项（方案 1 可行性）：用第 3 项核对用过的同一批公司-报告期，检查其他可访问数据源的数值是否等于巨潮首次披露原文。
判定方法与 audit_fundamentals.py 相同（数值能否在首次披露原文中找到）。"""
import json, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import audit_fundamentals as A  # noqa: E402
import akshare as ak  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    a = pd.read_csv(ROOT / "reports/audit_fin.csv", dtype={"code": str})
    a = a[a.status == "已核对"]
    rows = []
    for _, r in a.iterrows():
        code = str(r.code).zfill(6)
        per = pd.Timestamp(r.period)
        sym = ("SH" if code.startswith("6") else "SZ") + code
        anns = A.announcements(code, A.CAT[per.month], per, per + pd.DateOffset(months=5))
        rep = A.pick_report(anns, pd.Timestamp(r.em_notice))
        nums = A.numbers(A.pdf_text(rep["url"]))
        row = dict(code=code, period=per.date(), first_date=rep["date"].date())
        try:
            d = ak.stock_financial_report_sina(stock=sym.lower(), symbol="利润表")
            x = d[d["报告日"].astype(str) == per.strftime("%Y%m%d")].iloc[0]
            rev = x.get("营业总收入") if pd.notna(x.get("营业总收入")) else x.get("营业收入")
            npf = x.get("归属于母公司所有者的净利润")
            row.update(sina_rev_ok=A.present(float(rev), nums) if pd.notna(rev) else None,
                       sina_np_ok=A.present(float(npf), nums) if pd.notna(npf) else None,
                       sina_date=str(x.get("公告日期")))
        except Exception as e:  # noqa: BLE001
            row["sina_err"] = str(e)[:80]
        try:
            d = ak.stock_profit_sheet_by_report_em(symbol=sym)
            x = d[d.REPORT_DATE.astype(str).str.startswith(str(per.date()))].iloc[0]
            row.update(emf10_rev_ok=A.present(float(x.TOTAL_OPERATE_INCOME), nums) if pd.notna(x.TOTAL_OPERATE_INCOME) else None,
                       emf10_np_ok=A.present(float(x.PARENT_NETPROFIT), nums) if pd.notna(x.PARENT_NETPROFIT) else None,
                       emf10_notice=str(x.NOTICE_DATE)[:10], emf10_update=str(x.UPDATE_DATE)[:10])
        except Exception as e:  # noqa: BLE001
            row["emf10_err"] = str(e)[:80]
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
        time.sleep(0.5)
    pd.DataFrame(rows).to_csv(ROOT / "reports/audit_sources.csv", index=False)


if __name__ == "__main__":
    main()
