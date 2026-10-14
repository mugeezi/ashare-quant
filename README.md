# A 股日线短线波段研究

## 阶段 1：数据

全部数据在 Cowork 云端工作区下载和处理。BaoStock 使用专用 TCP 端口 10030，云端出口代理只放行 HTTPS，
连接建立后即被切断，因此按任务说明改用 AKShare 体系（东方财富、新浪、中证指数官网）及交易所、巨潮资讯公开接口。

### 字段与数据来源

面板文件 `data/panel/daily_YYYY.parquet`，每行一只股票一个交易日。

| 字段 | 含义 | 来源 / 计算 |
|---|---|---|
| date | 交易日 | 交易日历：东方财富上证指数 ∪ 深证成指 K 线日期 |
| code | 代码（sh./sz. 前缀） | 东方财富 A 股列表 |
| open, high, low, close | 不复权价（元） | 东方财富 K 线 `push2his.eastmoney.com/api/qt/stock/kline/get`，fqt=0（AKShare `stock_zh_a_hist` 同源）。抽样与腾讯行情比对不一致率 0.016%，分歧行以新浪历史行情仲裁 |
| preclose | 前收（元），用于涨跌停价 | 非除权日 = 前一交易日收盘；除权除息日 = 前一交易日收盘 × 新浪因子比，按交易所四舍五入到 0.01 元。东方财富不提供交易所前收（其"涨跌额"相对未除权的上一收盘）。与上市公司实施公告核对：送转派息含回购折算的情形一致；配股按实际配售比例、重整转增按调整参考价的少数情形有偏差（`exright_day` 标出，见 `reports/e1a_explained.md`） |
| pctChg | 涨跌幅（%） | 本库计算：close / preclose − 1 |
| volume | 成交量（股） | 东方财富（手）× 100 |
| amount | 成交额（元） | 东方财富 |
| turn | 换手率（%） | 东方财富口径（成交量 ÷ 流通股本） |
| amplitude | 振幅（%） | 东方财富 |
| tradestatus | 1 = 正常交易，0 = 停牌 | 东方财富当日有 K 线且成交量 > 0 记为 1；按交易日历补出的停牌行记为 0 |
| filled | 1 = 补出的停牌行 | 本库：上市日至退市前一交易日（或最新交易日）内东方财富缺失的交易日；价格 = 上一收盘，量额 = 0 |
| adj_factor | 后复权因子（乘法） | 新浪 `finance.sina.com.cn/realstock/company/<code>/hfq.js`（AKShare `stock_zh_a_daily` 同源）。东方财富和腾讯的"后复权价"是加法复权，不使用 |
| hfq_open/high/low/close | 后复权价 | 不复权价 × adj_factor；收益一律用后复权价计算 |
| board | 板块 | 代码前缀：main_sh / main_sz / chinext / star |
| list_date | 上市日 | 东方财富 A 股列表（f26），缺失时用 K 线首日 |
| delist_date | 终止上市日 | 上交所、深交所官方终止上市名单（`query.sse.com.cn`、`szse.cn`，经 AKShare）；2013-06 后 256 只，与本库逐一核对一致 |
| days_listed | 上市以来第几个交易日 | 本库按交易日历计算 |
| isST | 风险警示（ST/*ST/退市整理期） | 深市：深交所"证券简称变更"全量记录（精确）。沪市：巨潮资讯"风险警示""退市整理"公告（实施/撤销事件 + 公告时简称）；2026-07-06 前再用涨跌幅制度纠正，2026-07-06 起用简称法（巨潮逐只全量公告的发布日简称 + 东方财富当日简称）。验证见 `reports/st_validation.md` |
| isS | 未完成股改（简称 S 前缀），涨跌幅 5% | 深交所简称变更、巨潮公告简称 |
| delist_period | 退市整理期 | 巨潮"进入退市整理期"公告、"退"字简称 |
| no_limit | 当日不设涨跌幅 | 新股上市初期（注册制前 5 日 / 老规则首日）、退市整理期首日、恢复上市首日、股改复牌首日 |
| resume_no_limit | 恢复上市 / 股改复牌首日 | 巨潮"股票恢复上市""股权分置改革方案实施"公告（`download_special.py`）后的首个复牌日；S 股停牌期间完成股改 |
| limit_pct | 涨跌幅比例 | 主板 10%；主板 ST/*ST 2026-07-06 前 5%、之后 10%；S 股 5%；创业板 2020-08-24 前 10%（ST 5%）、之后 20%；科创板 20%；退市整理期按本板块普通股票 |
| up_limit, down_limit | 涨跌停价 | preclose × (1 ± limit_pct)，整数"分"运算，四舍五入到 0.01 元 |
| exright_day | 除权除息日 | 复权因子变化日（事先公告，不构成前视信息） |
| price_flag | 已知数据源分歧日 | 1 = 2016-01-04 熔断日，2 = 2014-04-23（见数据质量报告） |

其他数据：

| 文件 | 内容 | 来源 |
|---|---|---|
| `data/index/{hs300,zz500,zz1000,zzqz}.parquet` | 沪深300、中证500、中证1000、中证全指 | 中证指数官网（AKShare `stock_zh_index_hist_csindex`）为主，与东方财富交叉核对；发现东方财富 2014-12-09 中证全指收盘错误，已以官网为准 |
| `data/index/gz2000.parquet` | 国证2000（参考） | 东方财富 |
| `data/meta/universe.parquet` | 股票池（5,481 只，含退市 256 只） | 东方财富 A 股列表（包含退市股）∪ 交易所终止上市名单，另对代码区间穷举探测；剔除与 601607 重复的 600849 |
| `data/meta/st_events.parquet` | ST 状态切换点 | 见上 isST |
| `data/meta/cninfo_*` | 巨潮公告原始列表 | `www.cninfo.com.cn/new/hisAnnouncement/query` |
| `data/download_log.csv` | 每个数据集的来源、参数、行数、下载时间（含时区） | |

### 运行

```bash
pip install -r requirements.txt
bash scripts/run_all.sh            # 首次约 1 小时；download_meta.py 之后可加 --skip-probe
```

### 研究纪律

- 研究代码统一用 `common.load_panel()` 读数据：默认不读取 2024-01-01 及以后的文件，并截断日期。
  只有最终测试获批后才使用 `allow_test=True`。
- 数据质量检查和 ST 规则验证读取了 2024 年以后的数据，但只做完整性和规则一致性统计，不计算任何收益或策略指标。

## 新会话恢复步骤

代码和报告存在 GitHub 私有仓库，数据存在阿里云 OSS（香港，存储桶开启版本控制）。凭据只通过环境变量传入，
不写进任何文件。

```bash
# 1. 克隆仓库（token 通过环境变量传给 git，不写进 .git/config）
export GITHUB_TOKEN=...   # 由用户在会话中提供
git -c credential.helper= \
    -c credential.helper='!f() { echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f' \
    clone https://github.com/mugeezi/ashare-quant.git
cd ashare-quant

# 2. 设置 OSS 环境变量（由用户在会话中提供）
export OSS_ACCESS_KEY_ID=... OSS_ACCESS_KEY_SECRET=... OSS_ENDPOINT=oss-cn-hongkong.aliyuncs.com OSS_BUCKET=...

# 3. 安装依赖（Cowork 云端：pip 需走代理；oss2 依赖的 crcmod 需要 PEP 517 构建）
NO_PROXY= no_proxy= PIP_PROXY="$HTTPS_PROXY" pip install --use-pep517 -r requirements.txt --break-system-packages

# 4. 拉取数据：按 manifest 下载并逐个校验 SHA256
python scripts/sync_oss.py pull

# 5. 确认环境正常
cd engine && python unit_tests.py    # 应显示"失败 0 项"
```

说明：
- Cowork 云端的 GitHub 访问经过会话代理，仓库需要先加入该会话的授权仓库列表，否则代理会返回 403。
  GitHub 不可用时，代码以 git bundle 形式备份在 OSS 的 `code/ashare-quant.bundle`
  （`git clone ashare-quant.bundle ashare-quant` 即可恢复完整历史）。
- `python scripts/sync_oss.py verify` 比较本地与远端 manifest；`push` 只上传新增或变化的文件。
- 每次汇报进展时：`git push` 代码和报告，`sync_oss.py push` 同步新增数据。

## 外部数据时点规则（阶段 3 起强制执行）

1. 所有外部数据按"公开可得日期"对齐，不用报告期，也不用数据库收录日期。
   - 股东户数用公告日；公告用发布日（巨潮 `announcementTime`）。盘后发布的公告最早在下一交易日收盘形成信号；
     发布时间无法确定是否盘后的，一律按盘后处理。
   - 更正公告、补充公告单独处理：只在其自身发布日之后生效，不回溯修改原公告的信号。
2. 不得使用当前的概念板块或行业成分回测历史。需要板块或产业链关系时，必须使用有历史时点的成分数据，
   或只用当时已公开的信息构造关系（例如用 t 日以前的收益相关性构造"同伴股"）。
3. 每个外部数据集在 `research/data_registry.md` 记录：来源、时点字段、是否存在事后修订或回溯。
4. 外部数据也纳入前视检测：扰动 t 日之后发布的外部记录，t 日信号必须不变。
