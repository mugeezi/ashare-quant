# research2：第二轮研究（round2 分支）

与最终测试隔离：独立 git worktree，不改 main，不改 engine/、research/（只读引用）。最终测试开始补数据时，后台探测脚本写入
`PAUSED_FOR_FINAL_TEST` 并终止本目录下登记了 pid 的进程（logs/*.pid），`r2.check_pause()` 拒绝新运行。

| 文件 | 内容 |
|---|---|
| r2.py | 公共模块：试验登记（experiments.csv，从 938 累加）、进攻档 / 稳健档门槛、集中度诊断、暂停检查 |
| stock_strats.py | A01 集中版、W02 低波动高股息、W03 大盘增强 |
| a01_compare.py | A01 的 10 只对照重跑（不计试验） |
| dividends.py | 由复权因子跳变推算每股现金分红与 TTM 股息率 |
| etf_download.py | 腾讯 ETF 日线下载 |
| etf_rotation.py | W01 ETF 轮动 |
| sync_oss2.py | data2/ 同步到 OSS 的 round2/ 前缀 |

## 数据口径（data2/，与股票数据分开）
- `data2/dividend/events_2013_2023.parquet`：由研究期行情（2013-06 至 2023-12-29）推算的除权除息事件。字段 date、code、R（复权因子比）、
  P（除权前收盘）、q（流通股本前后比）、s（送转比例，0.1 网格）、D（每股现金分红）、d = D/P、flag。
  核对：招商银行、贵州茅台（含 2022-12 特别分红 21.91）、宁德时代（2023-04 10 转 8 派 25.2）、恒瑞医药、格力、工商银行、万科逐年与公告一致。
- `data2/etf/{代码}.parquet`：腾讯 web.ifzq.gtimg.cn fqkline 接口，2013-01 至 2023-12-31，不复权（raw_*）与后复权（hfq_*）两套；
  volume 单位为手，amount = volume × 100 × 不复权收盘价（估算）。接口附带的当前行情字段丢弃不存。
  腾讯 ETF 后复权价只有 3 位小数且为仿射调整，策略中日收益用不复权收盘价，分红 / 折算日（两者日收益差 > 0.3%）改用后复权日收益。
- 不请求东方财富 push2his；不使用 2024-01-01 以后的数据；新浪"分红送配"页面对本环境返回空内容，未使用。
