"""第二轮数据（data2/）同步到 OSS 的 round2/ 前缀下，与主线 data/ 完全分开。

用法：python research2/sync_oss2.py push | verify
凭据只从环境变量读取（同 scripts/sync_oss.py），不写入、不打印。
远端：round2/data2/<相对路径>，清单 round2/data2/manifest.json，历史清单 round2/manifests/。
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

R2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R2 / "scripts"))
import sync_oss as S  # noqa: E402

S.PREFIX = "round2/data2/"
S.MANIFEST_KEY = S.PREFIX + "manifest.json"
DATA = R2 / "data2"


def push():
    b = S.bucket()
    man = S.build_manifest(DATA)
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda e: S.upload_one(b, DATA, e), man["files"]))
    body = json.dumps(man, ensure_ascii=False, indent=1).encode()
    b.put_object(S.MANIFEST_KEY, body)
    b.put_object(f"round2/manifests/manifest_{time.strftime('%Y%m%d_%H%M%S')}.json", body)
    print(f"round2 数据：{man['n_files']} 个文件，{man['total_bytes'] / 1e6:.1f} MB；上传 {res.count('upload')}，跳过 {res.count('skip')}")


def verify():
    b = S.bucket()
    remote = json.loads(b.get_object(S.MANIFEST_KEY).read())
    local = S.build_manifest(DATA)
    same = remote["files"] == local["files"]
    print("一致" if same else "不一致")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit({"push": push, "verify": verify}[sys.argv[1]]())
