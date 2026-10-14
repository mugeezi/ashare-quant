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


def buyback():
    t0 = time.time()
    rows = []
    for s, e in months():
        page, total = 1, None
        while True:
            js = retry(lambda: cninfo_search("回购", s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"), page))
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
        print("buyback", s.date(), total, len(rows), flush=True)
    df = pd.DataFrame(rows).drop_duplicates("ann_id")
    df["title"] = df["title"].str.replace(r"<.*?>", "", regex=True)
    df["ann_date"] = pd.to_datetime(df["ann_time"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    df["avail_date"] = df["ann_date"].dt.normalize()
    save_df(df, OUT / "buyback.parquet")
    log_download("events/buyback", "巨潮资讯 hisAnnouncement/query searchkey=回购", f"{START}~{END}", len(df), t0,
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


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for job in sys.argv[1:] or ["lhb", "dzjy", "buyback"]:
        {"lhb": lhb, "dzjy": dzjy, "buyback": buyback, "jgdy": jgdy, "yjyg": yjyg}[job]()
