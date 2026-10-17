"""最终测试数据准备：下载测试期（2024-01-01 至截止日）的外部事件数据，写到 data/events_final/。
口径与研究期完全相同（同一套下载函数，只改日期参数）；研究期数据（data/events/）不动。

只有在仓库根目录存在 FINAL_TEST_APPROVED 时才会真正运行；否则只允许 --dry-run：
  python final_test_data.py --dry-run          # 用 2023-05 一个月写到临时目录，只检验流程，不碰测试期
  python final_test_data.py --cutoff 2026-10-16 # 获批后：下载 2024-01-01 ~ cutoff
需要的数据源（与入选策略的特征一一对应）：业绩预告、回购、增持、股东户数、龙虎榜机构、大宗交易、融资融券、业绩报表（只用公告日）。"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import download_events as DE  # noqa: E402
from common import DATA, log_test_access  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
JOBS = ["yjyg", "buyback", "zengchi", "gdhs", "lhb", "dzjy", "fin", "margin"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cutoff")
    ap.add_argument("--jobs", nargs="*", default=JOBS)
    a = ap.parse_args()
    if a.dry_run:
        out = Path(tempfile.mkdtemp(prefix="events_dry_"))
        DE.START, DE.END = "2023-05-01", "2023-05-31"         # 报告期最多请求到 2023-06-30，接口返回的公告全在研究期内
        DE.PERIOD_FIRST = DE.FIN_FIRST = "2023-03-31"
        DE.NOTICE_MIN = "2023-05-01"
        DE.log_download = lambda *x, **k: None                # 演练不写下载日志
        import common
        common.log_test_access = lambda *x, **k: None         # 演练的截止日在研究期内，不涉及测试期访问
    else:
        if not (ROOT / "FINAL_TEST_APPROVED").exists():
            raise SystemExit("未找到 FINAL_TEST_APPROVED：最终测试未获批，不能下载测试期数据")
        if not a.cutoff:
            raise SystemExit("必须给出 --cutoff（测试期截止日）")
        out = DATA / "events_final"
        DE.START, DE.END = "2024-01-01", a.cutoff
        DE.PERIOD_FIRST = DE.FIN_FIRST = "2023-09-30"     # 2023Q4 报告期的公告大多落在 2024 年
        DE.NOTICE_MIN = "2024-01-01"
        log_test_access("最终测试数据准备：下载测试期外部事件数据", f"{', '.join(a.jobs)}；2024-01-01 ~ {a.cutoff}", "final_test")
    out.mkdir(parents=True, exist_ok=True)
    DE.OUT = out
    fns = {"yjyg": DE.yjyg, "buyback": DE.buyback, "zengchi": lambda: DE.buyback("增持", "zengchi"), "gdhs": DE.gdhs,
           "lhb": DE.lhb, "dzjy": DE.dzjy, "fin": DE.fin, "margin": DE.margin}
    for j in a.jobs:
        fns[j]()
        print("完成", j, flush=True)
    print("输出目录", out)


if __name__ == "__main__":
    main()
