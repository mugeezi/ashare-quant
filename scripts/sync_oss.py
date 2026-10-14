"""把 data/ 同步到阿里云 OSS（持久化）。

用法：
    python scripts/sync_oss.py push                # 上传 data/ 全部文件 + manifest.json
    python scripts/sync_oss.py pull [--dest DIR]   # 按 manifest 下载并逐个校验 SHA256
    python scripts/sync_oss.py verify              # 只比较本地与远端 manifest，不传输

凭据只从环境变量读取：OSS_ACCESS_KEY_ID、OSS_ACCESS_KEY_SECRET、OSS_ENDPOINT、OSS_BUCKET。
本脚本不写入、不打印任何凭据。

规则：
- 远端对象路径 = "data/" + 相对路径；每个对象带自定义元数据 x-oss-meta-sha256。
- push：远端已存在且 sha256 元数据与本地一致 → 跳过（存储桶开了版本控制，避免产生重复版本）。
  大于 16MB 的文件用分片上传，断点信息存在 ~/.cache/ashare-quant-oss/，中断后重跑会续传。
- manifest.json 列出每个文件的 path、size、sha256，push 结束后上传到 data/manifest.json，
  同时按时间戳存一份 manifests/manifest_YYYYmmdd_HHMMSS.json，保留历史。
- pull：先下载 manifest；本地已存在且 SHA256 一致的文件跳过；下载到临时文件，校验通过再改名；
  任何一个文件校验失败都会报错并以非 0 退出。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import oss2

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PREFIX = "data/"
MANIFEST_KEY = PREFIX + "manifest.json"
MULTIPART_THRESHOLD = 16 * 1024 * 1024
PART_SIZE = 8 * 1024 * 1024
CKPT_DIR = Path.home() / ".cache" / "ashare-quant-oss"
EXCLUDE_SUFFIX = (".tmp", ".part", ".pyc")
EXCLUDE_NAMES = {".DS_Store"}


def bucket() -> oss2.Bucket:
    missing = [k for k in ("OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET", "OSS_ENDPOINT", "OSS_BUCKET") if not os.environ.get(k)]
    if missing:
        sys.exit(f"缺少环境变量：{', '.join(missing)}")
    auth = oss2.Auth(os.environ["OSS_ACCESS_KEY_ID"], os.environ["OSS_ACCESS_KEY_SECRET"])
    ep = os.environ["OSS_ENDPOINT"]
    if not ep.startswith("http"):
        ep = "https://" + ep
    return oss2.Bucket(auth, ep, os.environ["OSS_BUCKET"], connect_timeout=60)


def sha256(path: Path, buf=4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def local_files(base: Path) -> list[Path]:
    out = []
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.name not in EXCLUDE_NAMES and not p.name.endswith(EXCLUDE_SUFFIX) \
                and p.name != "manifest.json":
            out.append(p)
    return out


def build_manifest(base: Path, workers=8) -> dict:
    files = local_files(base)
    with ThreadPoolExecutor(workers) as ex:
        hashes = list(ex.map(sha256, files))
    entries = [{"path": p.relative_to(base).as_posix(), "size": p.stat().st_size, "sha256": h}
               for p, h in zip(files, hashes)]
    return {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_files": len(entries),
            "total_bytes": sum(e["size"] for e in entries), "files": entries}


def remote_sha(b: oss2.Bucket, key: str) -> str | None:
    try:
        meta = b.head_object(key)
    except oss2.exceptions.NotFound:
        return None
    return meta.headers.get("x-oss-meta-sha256")


def upload_one(b: oss2.Bucket, base: Path, e: dict) -> str:
    key = PREFIX + e["path"]
    if remote_sha(b, key) == e["sha256"]:
        return "skip"
    headers = {"x-oss-meta-sha256": e["sha256"]}
    src = str(base / e["path"])
    if e["size"] > MULTIPART_THRESHOLD:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        oss2.resumable_upload(b, key, src, headers=headers, multipart_threshold=MULTIPART_THRESHOLD,
                              part_size=PART_SIZE, num_threads=4, store=oss2.ResumableStore(root=str(CKPT_DIR)))
    else:
        b.put_object_from_file(key, src, headers=headers)
    # 上传后复核：远端元数据必须等于本地哈希
    if remote_sha(b, key) != e["sha256"]:
        raise RuntimeError(f"上传后校验失败：{e['path']}")
    return "upload"


def push(args):
    b = bucket()
    t0 = time.time()
    man = build_manifest(DATA)
    print(f"本地 {man['n_files']} 个文件，{man['total_bytes'] / 1e6:,.1f} MB；哈希耗时 {time.time() - t0:.0f}s", flush=True)
    stats = {"skip": 0, "upload": 0, "error": 0}
    done_bytes = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(upload_one, b, DATA, e): e for e in man["files"]}
        for i, f in enumerate(as_completed(futs), 1):
            e = futs[f]
            try:
                stats[f.result()] += 1
                done_bytes += e["size"]
            except Exception as err:  # noqa: BLE001
                stats["error"] += 1
                print(f"失败 {e['path']}: {type(err).__name__}: {err}", flush=True)
            if i % 200 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}  上传 {stats['upload']}  跳过 {stats['skip']}  失败 {stats['error']}  "
                      f"{done_bytes / 1e6:,.0f} MB  {time.time() - t0:.0f}s", flush=True)
    if stats["error"]:
        sys.exit(f"有 {stats['error']} 个文件上传失败，manifest 未更新。重跑 push 会跳过已完成的文件并续传。")
    body = json.dumps(man, ensure_ascii=False, indent=1).encode()
    try:
        old = b.get_object(MANIFEST_KEY).read()
    except oss2.exceptions.NoSuchKey:
        old = None
    same = old is not None and json.loads(old)["files"] == man["files"]
    if not same:
        b.put_object(MANIFEST_KEY, body)
        b.put_object(f"manifests/manifest_{time.strftime('%Y%m%d_%H%M%S')}.json", body)
    print(f"完成：上传 {stats['upload']}，跳过 {stats['skip']}；manifest {'未变化' if same else '已更新'}；"
          f"共 {time.time() - t0:.0f}s")


def download_one(b: oss2.Bucket, dest: Path, e: dict) -> str:
    p = dest / e["path"]
    if p.exists() and p.stat().st_size == e["size"] and sha256(p) == e["sha256"]:
        return "skip"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    key = PREFIX + e["path"]
    if e["size"] > MULTIPART_THRESHOLD:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        oss2.resumable_download(b, key, str(tmp), multiget_threshold=MULTIPART_THRESHOLD, part_size=PART_SIZE,
                                num_threads=4, store=oss2.ResumableDownloadStore(root=str(CKPT_DIR)))
    else:
        b.get_object_to_file(key, str(tmp))
    h = sha256(tmp)
    if h != e["sha256"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"SHA256 不一致：{e['path']}（期望 {e['sha256'][:12]}…，实际 {h[:12]}…）")
    tmp.replace(p)
    return "download"


def pull(args):
    b = bucket()
    dest = Path(args.dest).resolve() if args.dest else DATA
    t0 = time.time()
    man = json.loads(b.get_object(MANIFEST_KEY).read())
    print(f"远端 manifest：{man['n_files']} 个文件，{man['total_bytes'] / 1e6:,.1f} MB（生成于 {man['created']}）", flush=True)
    stats = {"skip": 0, "download": 0, "error": 0}
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(download_one, b, dest, e): e for e in man["files"]}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                stats[f.result()] += 1
            except Exception as err:  # noqa: BLE001
                stats["error"] += 1
                print(f"失败 {futs[f]['path']}: {err}", flush=True)
            if i % 500 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}  下载 {stats['download']}  跳过 {stats['skip']}  失败 {stats['error']}  "
                      f"{time.time() - t0:.0f}s", flush=True)
    (dest / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1))
    if stats["error"]:
        sys.exit(f"{stats['error']} 个文件下载或校验失败")
    print(f"完成：{man['n_files']} 个文件全部校验一致（下载 {stats['download']}，跳过 {stats['skip']}），"
          f"{man['total_bytes'] / 1e6:,.1f} MB，{time.time() - t0:.0f}s")


def verify(args):
    b = bucket()
    remote = json.loads(b.get_object(MANIFEST_KEY).read())
    local = build_manifest(DATA)
    r = {e["path"]: e["sha256"] for e in remote["files"]}
    l_ = {e["path"]: e["sha256"] for e in local["files"]}
    only_l = sorted(set(l_) - set(r))
    only_r = sorted(set(r) - set(l_))
    diff = sorted(k for k in set(l_) & set(r) if l_[k] != r[k])
    print(f"本地 {len(l_)}，远端 {len(r)}；仅本地 {len(only_l)}，仅远端 {len(only_r)}，内容不同 {len(diff)}")
    for k in (only_l + diff)[:20]:
        print("  需要 push:", k)
    return 0 if not (only_l or only_r or diff) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "pull", "verify"])
    ap.add_argument("--dest", help="pull 的目标目录（默认 data/）")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    sys.exit({"push": push, "pull": pull, "verify": verify}[args.cmd](args))


if __name__ == "__main__":
    main()
