"""HTTP 数据源封装：东方财富 K 线 / 股票列表、巨潮公告检索。统一重试与限速。

在 Cowork 云端运行时 HTTPS 走代理；pypi 等域名在 NO_PROXY 里会被直连而失败，
所以这里显式清空 NO_PROXY，让所有请求都走 HTTPS_PROXY。本地直连环境不受影响。
"""
from __future__ import annotations

import os
import random
import threading
import time

import pandas as pd
import requests

if os.environ.get("HTTPS_PROXY"):
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = ""
    ca = "/root/.ccr/ca-bundle.crt"
    if os.path.exists(ca):
        os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)

# AKShare 内部的 requests 调用不设超时，遇到半开连接会永久挂起；统一加默认超时
_orig_request = requests.Session.request


def _request_with_timeout(self, method, url, **kw):
    kw.setdefault("timeout", 30)
    return _orig_request(self, method, url, **kw)


requests.Session.request = _request_with_timeout

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_local = threading.local()


def session() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = UA
        _local.s = s
    return s


def get_json(url, params=None, data=None, tries=6, timeout=25):
    err = None
    for i in range(tries):
        try:
            if data is None:
                r = session().get(url, params=params, timeout=timeout)
            else:
                r = session().post(url, data=data, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # 限流 / 断连：指数退避
            err = e
            _local.s = None
            time.sleep(min(60, 1.5 * 2 ** i) + random.random())
    raise RuntimeError(f"请求失败 {url} {params or data}: {err}")


# ---------------------------------------------------------------- 东方财富
EM_KLINE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
EM_CLIST = "https://push2.eastmoney.com/api/qt/clist/get"
KLINE_COLS = ["date", "open", "close", "high", "low", "vol_lots", "amount",
              "amplitude", "pctChg", "change", "turn"]


def em_secid(code: str) -> str:
    """'sh.600000' -> '1.600000'；'sz.000001' -> '0.000001'。"""
    mkt, num = code.split(".")
    return ("1." if mkt == "sh" else "0.") + num


def em_kline(secid: str, fqt: int, beg="19900101", end="20500101") -> tuple[str, pd.DataFrame]:
    """fqt: 0 不复权，1 前复权，2 后复权。返回 (名称, DataFrame)。停牌日不返回记录。"""
    j = get_json(EM_KLINE, params=dict(
        secid=secid, fields1="f1,f2,f3", klt=101, fqt=fqt, beg=beg, end=end,
        fields2="f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"))
    d = j.get("data") or {}
    rows = [k.split(",") for k in d.get("klines") or []]
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
        for c in KLINE_COLS[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return d.get("name", ""), df


def em_a_share_list() -> pd.DataFrame:
    """当前在市的沪深 A 股（主板、创业板、科创板）：代码、名称、上市日期。"""
    out, pn = [], 1
    while True:
        j = get_json(EM_CLIST, params=dict(pn=pn, pz=100, po=1, np=1, fltt=2, invt=2, fid="f12",
                                           fs="m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
                                           fields="f12,f13,f14,f26"))
        diff = (j.get("data") or {}).get("diff") or []
        out += diff
        if len(diff) < 100:
            break
        pn += 1
    df = pd.DataFrame(out).rename(columns={"f12": "num", "f13": "mkt", "f14": "name", "f26": "list_date"})
    df["code"] = df.apply(lambda r: ("sh." if r["mkt"] == 1 else "sz.") + r["num"], axis=1)
    df["list_date"] = pd.to_datetime(df["list_date"].astype(str), format="%Y%m%d", errors="coerce")
    return df[["code", "name", "list_date"]].drop_duplicates("code")


# ---------------------------------------------------------------- 新浪复权因子
def sina_hfq_factor(code: str, tries: int = 8) -> pd.DataFrame:
    """新浪后复权因子（乘法）：d = 除权除息日，f = 当日起生效的累计因子。返回 date, factor 升序。"""
    sym = code.replace(".", "")
    url = f"https://finance.sina.com.cn/realstock/company/{sym}/hfq.js"
    err = None
    for i in range(tries):
        try:
            r = session().get(url, timeout=20)
            r.raise_for_status()
            body = r.text.split("=", 1)[1].split("\n")[0].strip().rstrip(";")
            import json
            data = json.loads(body)["data"]
            if not data:   # 例如 CDR（689009）没有复权记录
                return pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]"), "factor": pd.Series([], dtype=float)})
            df = pd.DataFrame(data).rename(columns={"d": "date", "f": "factor"})
            df["date"] = pd.to_datetime(df["date"])
            df["factor"] = pd.to_numeric(df["factor"])
            return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        except Exception as e:
            err = e
            _local.s = None
            time.sleep(min(30, 1.0 * 2 ** i) + random.random())
    raise RuntimeError(f"新浪复权因子获取失败 {code}: {err}")


# ---------------------------------------------------------------- 巨潮资讯
CNINFO_QUERY = "https://www.cninfo.com.cn/new/hisAnnouncement/query"


def cninfo_search(searchkey: str, start: str, end: str, page: int, page_size: int = 30) -> dict:
    return get_json(CNINFO_QUERY, data=dict(
        pageNum=page, pageSize=page_size, column="szse", tabName="fulltext", plate="", stock="",
        searchkey=searchkey, secid="", category="", trade="", seDate=f"{start}~{end}",
        sortName="", sortType="", isHLtitle="true"))


def retry(fn, tries=6, wait=3):
    """给 AKShare 之类不自带重试的调用包一层。"""
    err = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            err = e
            time.sleep(wait * (i + 1))
    raise err
