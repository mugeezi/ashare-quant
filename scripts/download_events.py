"""阶段 3 外部事件数据（研究期 2013-06 ~ 2023-12；测试期在最终测试获批后再下载）。

输出 data/events/：
- lhb_jg.parquet   龙虎榜机构买卖每日统计（东方财富，AKShare stock_lhb_jgmmtj_em），avail_date = 上榜日
- dzjy.parquet     大宗交易每日明细（东方财富，AKShare stock_dzjy_mrmx），avail_date = 交易日
- buyback.parquet  巨潮"回购"标题公告（hisAnnouncement/query），avail_date = 公告日（另由信号按盘后处理顺延）

时点说明：三者都在收盘后发布；东方财富历史数据不做事后修订（龙虎榜、大宗交易为交易所当日公开信息）。
用法：python download_events.py [lhb] [dzjy] [buyback]
"""
import sys
import time

import pandas as pd

import sources  # noqa: F401  （代理设置）
from common import DATA, log_download, save_df
from sources import cninfo_search, get_json, retry

EM_DC = "https://datacenter-web.eastmoney.com/api/data/v1/get"

OUT = DATA / "events"
START, END = "2013-06-01", "2023-12-31"


def months():
    for s in pd.date_range(START, END, freq="MS"):
        e = min(s + pd.offsets.MonthEnd(0), pd.Timestamp(END))
        yield s, e


def lhb():
    import akshare as ak
    t0 = time.time()
    parts = []
    for s, e in months():
        df = retry(lambda: ak.stock_lhb_jgmmtj_em(start_date=s.strftime("%Y%m%d"), end_date=e.strftime("%Y%m%d")))
        if df is not None and len(df):
            parts.append(df)
        print("lhb", s.date(), 0 if df is None else len(df), flush=True)
    df = pd.concat(parts, ignore_index=True)
    df = df.rename(columns={"代码": "code6", "上榜日期": "date", "买方机构数": "n_buy_inst", "卖方机构数": "n_sell_inst",
                            "机构买入总额": "inst_buy", "机构卖出总额": "inst_sell", "机构买入净额": "inst_net",
                            "市场总成交额": "amount", "机构净买额占总成交额比": "net_ratio_pct", "上榜原因": "reason"})
    df["date"] = pd.to_datetime(df["date"])
    df["avail_date"] = df["date"]
    df = df[["date", "avail_date", "code6", "n_buy_inst", "n_sell_inst", "inst_buy", "inst_sell", "inst_net", "amount",
             "net_ratio_pct", "reason"]].drop_duplicates()
    save_df(df, OUT / "lhb_jg.parquet")
    log_download("events/lhb_jg", "东方财富 龙虎榜机构买卖每日统计（AKShare stock_lhb_jgmmtj_em）", f"{START}~{END}",
                 len(df), t0, note="只保留当日统计字段；不下载含'上榜后N日'未来收益的明细接口")


def dzjy():
    import akshare as ak
    t0 = time.time()
    parts = []
    for s, e in months():
        try:
            df = retry(lambda: ak.stock_dzjy_mrmx(symbol="A股", start_date=s.strftime("%Y%m%d"), end_date=e.strftime("%Y%m%d")))
        except Exception as err:  # noqa: BLE001
            print("dzjy fail", s.date(), err, flush=True)
            continue
        if df is not None and len(df):
            parts.append(df)
        print("dzjy", s.date(), 0 if df is None else len(df), flush=True)
    df = pd.concat(parts, ignore_index=True)
    df = df.rename(columns={"交易日期": "date", "证券代码": "code6", "收盘价": "close", "成交价": "price", "折溢率": "premium",
                            "成交量": "volume", "成交额": "amount", "成交额/流通市值": "amt_to_fmv_pct",
                            "买方营业部": "buyer", "卖方营业部": "seller"})
    df["date"] = pd.to_datetime(df["date"])
    df["avail_date"] = df["date"]
    df = df[["date", "avail_date", "code6", "close", "price", "premium", "volume", "amount", "amt_to_fmv_pct",
             "buyer", "seller"]]
    save_df(df, OUT / "dzjy.parquet")
    log_download("events/dzjy", "东方财富 大宗交易每日明细（AKShare stock_dzjy_mrmx）", f"{START}~{END}", len(df), t0)


def buyback(key="回购", name="buyback"):
    t0 = time.time()
    rows = []
    for s, e in months():
        page, total = 1, None
        while True:
            js = retry(lambda: cninfo_search(key, s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"), page))
            ann = js.get("announcements") or []
            total = js.get("totalpages", 0)
            for a in ann:
                rows.append(dict(code6=a.get("secCode"), name=a.get("secName"), title=a.get("announcementTitle"),
                                 ann_time=a.get("announcementTime"), ann_id=a.get("announcementId"),
                                 column=a.get("columnId"), org=a.get("orgId")))
            if page >= total or not ann:
                break
            page += 1
            time.sleep(0.3)
        print(name, s.date(), total, len(rows), flush=True)
    df = pd.DataFrame(rows).drop_duplicates("ann_id")
    df["title"] = df["title"].str.replace(r"<.*?>", "", regex=True)
    df["ann_date"] = pd.to_datetime(df["ann_time"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    df["avail_date"] = df["ann_date"].dt.normalize()
    save_df(df, OUT / f"{name}.parquet")
    log_download(f"events/{name}", f"巨潮资讯 hisAnnouncement/query searchkey={key}", f"{START}~{END}", len(df), t0,
                 note="avail_date = 公告日；公告时刻无法确认是否盘后，信号统一顺延到下一交易日")


def em_report(report, filt, sort="NOTICE_DATE", page_size=500, columns="ALL"):
    out, page = [], 1
    while True:
        js = get_json(EM_DC, params=dict(reportName=report, columns=columns, filter=filt, pageSize=page_size,
                                         pageNumber=page, sortColumns=sort, sortTypes="1", source="WEB", client="WEB"),
                      timeout=60)
        res = js.get("result") or {}
        data = res.get("data") or []
        out += data
        if page >= (res.get("pages") or 0) or not data:
            break
        page += 1
    return out


def jgdy_slow():
    """机构调研历史（东方财富 RPT_ORG_SURVEY，每行一家接待对象）。该表分页超过约 50 页、或加额外过滤条件时
    返回"服务器繁忙"。每次调研的第一行 NUMBERNEW="1"，且 SUM = 该次接待机构总数；按 NUMBERNEW 升序排序后，
    只读到 NUMBERNEW 不再为 "1" 为止。按 5 天窗口请求并限速。avail_date = 公告日期。"""
    import random
    t0 = time.time()
    part_dir = OUT / "jgdy_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    for old in part_dir.glob("*.parquet"):
        if not old.name.startswith("w5_"):
            old.unlink()
    for s in pd.date_range(START, END, freq="5D"):
        e = min(s + pd.Timedelta(days=4), pd.Timestamp(END))
        pf = part_dir / f"w5_{s.date()}.parquet"
        if pf.exists():
            continue
        rows, page, busy = [], 1, 0
        while page <= 50:
            f = f"(NOTICE_DATE>='{s.date()}')(NOTICE_DATE<='{e.date()}')"
            js = get_json(EM_DC, params=dict(reportName="RPT_ORG_SURVEY", columns="ALL", filter=f, pageSize=50,
                                             pageNumber=page, sortColumns="NUMBERNEW", sortTypes="1",
                                             source="WEB", client="WEB"), timeout=60, tries=8)
            res = js.get("result") or {}
            data = res.get("data") or []
            if not data and js.get("message") != "ok" and "为空" not in str(js.get("message")):
                busy += 1
                if busy > 6:
                    raise RuntimeError(f"{s.date()} 第 {page} 页持续返回：{js.get('message')}")
                time.sleep(20 * busy)
                continue
            busy = 0
            first = [d for d in data if str(d.get("NUMBERNEW")) == "1"]
            rows += [{k: d.get(k) for k in ("SECURITY_CODE", "NOTICE_DATE", "RECEIVE_START_DATE", "RECEIVE_END_DATE",
                                             "RECEIVE_WAY_EXPLAIN", "SUM")} for d in first]
            if len(first) < len(data) or page >= (res.get("pages") or 0) or not data:
                break
            page += 1
            time.sleep(0.3 + random.random() * 0.5)
        pd.DataFrame(rows).to_parquet(pf)
        print("jgdy", s.date(), len(rows), flush=True)
    df = pd.concat([pd.read_parquet(p) for p in sorted(part_dir.glob("w5_*.parquet"))], ignore_index=True)
    df = df.rename(columns={"SECURITY_CODE": "code6", "NOTICE_DATE": "notice_date", "RECEIVE_START_DATE": "visit_start",
                            "RECEIVE_END_DATE": "visit_end", "RECEIVE_WAY_EXPLAIN": "way", "SUM": "n_inst"})
    for c in ("notice_date", "visit_start", "visit_end"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    g = df.groupby(["code6", "notice_date", "visit_start"], as_index=False).agg(n_inst=("n_inst", "max"),
                                                                                way=("way", "first"))
    g["avail_date"] = g["notice_date"].dt.normalize()
    save_df(g, OUT / "jgdy.parquet")
    log_download("events/jgdy", "东方财富 机构调研历史 RPT_ORG_SURVEY（每次调研取 NUMBERNEW=1 的首行，SUM=机构数）",
                 f"{START}~{END}", len(g), t0, note="avail_date = 公告日期")


def jgdy():
    """机构调研（东方财富 RPT_ORG_SURVEY 历史表，NUMBERNEW=1 为每次调研一条汇总记录，SUM = 接待机构数）。
    RPT_ORG_SURVEYNEW（AKShare stock_jgdy_tj_em 所用）只保留最近一年，不能用于历史回测。avail_date = 公告日期。"""
    t0 = time.time()
    rows = []
    for s, e in months():
        f = f"(NUMBERNEW=\"1\")(IS_SOURCE=\"1\")(NOTICE_DATE>='{s.date()}')(NOTICE_DATE<='{e.date()}')"
        data = em_report("RPT_ORG_SURVEY", f, columns="SECURITY_CODE,SECURITY_NAME_ABBR,NOTICE_DATE,RECEIVE_START_DATE,"
                         "RECEIVE_END_DATE,SUM,RECEIVE_WAY_EXPLAIN,RECEIVE_OBJECT_TYPE")
        rows += data
        print("jgdy", s.date(), len(data), flush=True)
    df = pd.DataFrame(rows)
    keep = {"SECURITY_CODE": "code6", "SECURITY_NAME_ABBR": "name", "NOTICE_DATE": "notice_date",
            "RECEIVE_START_DATE": "visit_start", "RECEIVE_END_DATE": "visit_end", "SUM": "n_inst",
            "RECEIVE_WAY_EXPLAIN": "way", "RECEIVE_OBJECT_TYPE": "obj_type"}
    df = df[[c for c in keep if c in df.columns]].rename(columns=keep)
    for c in ("notice_date", "visit_start", "visit_end"):
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df["avail_date"] = df["notice_date"].dt.normalize()
    df = df.drop_duplicates()
    save_df(df, OUT / "jgdy.parquet")
    log_download("events/jgdy", "东方财富 机构调研 RPT_ORG_SURVEY（NUMBERNEW=1 汇总记录）", f"{START}~{END}",
                 len(df), t0, note="avail_date = 公告日期（调研记录披露日），不是接待日")


def yjyg():
    """业绩预告（AKShare stock_yjyg_em，按报告期）。avail_date = 公告日期。"""
    import akshare as ak
    t0 = time.time()
    parts = []
    for p in pd.date_range("2013-06-30", "2024-03-31", freq="QE"):
        df = retry(lambda: ak.stock_yjyg_em(date=p.strftime("%Y%m%d")))
        if df is not None and len(df):
            df["report_period"] = p
            parts.append(df)
        print("yjyg", p.date(), 0 if df is None else len(df), flush=True)
    df = pd.concat(parts, ignore_index=True)
    df = df.rename(columns={"股票代码": "code6", "股票简称": "name", "预测指标": "metric", "业绩变动": "text",
                            "预测数值": "forecast", "业绩变动幅度": "chg_pct", "业绩变动原因": "reason",
                            "预告类型": "type", "上年同期值": "last_year", "公告日期": "notice_date"})
    df["notice_date"] = pd.to_datetime(df["notice_date"], errors="coerce")
    df["avail_date"] = df["notice_date"]
    df = df[df["avail_date"] <= END]                    # 研究期以内（测试期公告不下载使用）
    df = df.drop(columns=[c for c in df.columns if c in ("序号",)])
    save_df(df, OUT / "yjyg.parquet")
    log_download("events/yjyg", "东方财富 业绩预告（AKShare stock_yjyg_em）", "报告期 2013Q2~2024Q1，公告日 ≤ 2023-12-31",
                 len(df), t0, note="avail_date = 公告日期；同一预告的修正公告为单独记录")


def gdhs():
    """股东户数（东方财富，AKShare stock_zh_a_gdhs，按统计截止日批量；只取季末截止日）。
    avail_date = 公告日期。若东方财富记录的公告日期晚于真实首次披露日，只会让信号更晚（保守），不会产生前视。
    截止日 2013-06-30 ~ 2023-12-31；保存前丢弃公告日在 2023-12-31 之后的记录（测试期）。"""
    import akshare as ak
    from common import log_test_access
    t0 = time.time()
    parts = []
    for p in pd.date_range("2013-06-30", "2023-12-31", freq="QE"):
        df = retry(lambda: ak.stock_zh_a_gdhs(symbol=p.strftime("%Y%m%d")))
        if df is None or not len(df):
            continue
        df = df.rename(columns={"代码": "code6", "股东户数-本次": "holders", "股东户数-上次": "holders_prev",
                                "股东户数-增减比例": "chg_pct", "股东户数统计截止日-本次": "period_end",
                                "股东户数统计截止日-上次": "period_prev", "户均持股数量": "shares_per_holder",
                                "总股本": "total_shares", "公告日期": "notice_date"})
        keep = ["code6", "holders", "holders_prev", "chg_pct", "period_end", "period_prev", "shares_per_holder",
                "total_shares", "notice_date"]
        df = df[[k for k in keep if k in df.columns]].copy()
        parts.append(df)
        print("gdhs", p.date(), len(df), flush=True)
    df = pd.concat(parts, ignore_index=True)
    df["notice_date"] = pd.to_datetime(df["notice_date"], errors="coerce")
    n_test = int((df["notice_date"] > END).sum())
    df = df[df["notice_date"] <= END].copy()
    df["code6"] = df["code6"].astype(str).str.zfill(6)
    df["avail_date"] = df["notice_date"].dt.normalize()
    save_df(df, OUT / "gdhs.parquet")
    log_download("events/gdhs", "东方财富 股东户数（AKShare stock_zh_a_gdhs，季末截止日）", "截止日 2013Q2~2023Q4",
                 len(df), t0, note=f"avail_date = 公告日期；丢弃公告日在测试期的 {n_test} 条")
    if n_test:
        log_test_access(f"下载股东户数时接口返回 {n_test} 条 2024 年以后公告的记录，保存前丢弃",
                        "股东户数元数据（未保存、未计算统计）")


def reports():
    """个股研报（东方财富 reportapi，逐只股票，发布日 2013-06-01 ~ 2023-12-31）。avail_date = publishDate。
    只保留元数据（机构、评级、评级变动、是否首次覆盖），不下载 PDF；接口按发布日过滤，不含测试期。"""
    import random
    from common import META, load_df
    t0 = time.time()
    part_dir = OUT / "report_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    codes = load_df(META / "universe.parquet")["code"].tolist()
    keep = ["stockCode", "publishDate", "orgSName", "emRatingName", "lastEmRatingName", "ratingChange", "indvIsNew",
            "sRatingName", "reportType", "infoCode"]
    for i, c in enumerate(codes):
        pf = part_dir / f"{c}.parquet"
        if pf.exists():
            continue
        rows, page = [], 1
        while True:
            p = dict(industryCode="*", pageSize=100, industry="*", rating="*", ratingChange="*",
                     beginTime=START, endTime="2023-12-31", pageNo=page, fields="", qType=0, orgCode="",
                     code=c.split(".")[1], rcode="", p=page, pageNum=page, pageNumber=page)
            js = get_json("https://reportapi.eastmoney.com/report/list", params=p, timeout=30, tries=6)
            data = js.get("data") or []
            rows += [{k: d.get(k) for k in keep} for d in data]
            if page >= (js.get("TotalPage") or 0) or not data:
                break
            page += 1
            time.sleep(0.2)
        pd.DataFrame(rows, columns=keep).astype(str).to_parquet(pf)
        if i % 200 == 0:
            print("reports", i, c, len(rows), round(time.time() - t0), flush=True)
        time.sleep(0.1 + random.random() * 0.2)
    df = pd.concat([pd.read_parquet(p) for p in sorted(part_dir.glob("*.parquet"))], ignore_index=True)
    df = df.rename(columns={"stockCode": "code6", "publishDate": "publish"})
    df["publish"] = pd.to_datetime(df["publish"], errors="coerce")
    df = df[df["publish"] <= END].drop_duplicates("infoCode")
    df["avail_date"] = df["publish"].dt.normalize()
    save_df(df, OUT / "reports.parquet")
    log_download("events/reports", "东方财富 个股研报元数据（reportapi.eastmoney.com/report/list）",
                 f"{START}~{END}", len(df), t0, note="avail_date = 发布日；不含 PDF 与盈利预测数值")


def unlock():
    """限售解禁（东方财富，AKShare stock_restricted_release_detail_em），解禁日 2013-06 ~ 2023-12。
    接口自带"解禁后 20 日涨跌幅"（未来数据）和"解禁前 20 日涨跌幅"，两列下载时直接丢弃。
    时点：解禁日程在发行时已公开，实际解禁数量在解禁前约 3 个交易日的提示性公告中披露；
    信号只在解禁日前 3 个交易日起使用。"""
    import akshare as ak
    t0 = time.time()
    parts = []
    for s, e in months():
        df = retry(lambda: ak.stock_restricted_release_detail_em(start_date=s.strftime("%Y%m%d"),
                                                                  end_date=e.strftime("%Y%m%d")))
        if df is not None and len(df):
            parts.append(df.drop(columns=[c for c in df.columns if "涨跌幅" in c]))
        print("unlock", s.date(), 0 if df is None else len(df), flush=True)
    df = pd.concat(parts, ignore_index=True)
    df = df.rename(columns={"股票代码": "code6", "解禁时间": "unlock_date", "限售股类型": "kind", "解禁数量": "shares",
                            "实际解禁数量": "shares_actual", "实际解禁市值": "value_actual",
                            "占解禁前流通市值比例": "ratio_fmv", "解禁前一交易日收盘价": "close_prev"})
    df = df.drop(columns=[c for c in ("序号", "股票简称") if c in df.columns])
    df["unlock_date"] = pd.to_datetime(df["unlock_date"])
    save_df(df, OUT / "unlock.parquet")
    log_download("events/unlock", "东方财富 限售解禁明细（AKShare stock_restricted_release_detail_em）", f"{START}~{END}",
                 len(df), t0, note="丢弃解禁前后 20 日涨跌幅两列；信号只在解禁日前 3 个交易日起使用")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for job in sys.argv[1:] or ["lhb", "dzjy", "buyback"]:
        {"lhb": lhb, "dzjy": dzjy, "buyback": buyback, "jgdy": jgdy, "yjyg": yjyg,
         "zengchi": lambda: buyback("增持", "zengchi"), "jgdy_slow": jgdy_slow,
         "gdhs": gdhs, "reports": reports, "unlock": unlock}[job]()
