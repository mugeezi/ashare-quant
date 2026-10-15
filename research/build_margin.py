"""把融资融券明细压缩成 T×N 的 float32 矩阵（按数据日期对齐，未做顺延），存到 data/derived/margin.npz。"""
import gc
import numpy as np
import pandas as pd
from framework import DERIVED, ROOT, load_md

md = load_md()
cm = {c.split(".")[1]: i for i, c in enumerate(md.codes)}
df = pd.read_parquet(ROOT / "data/events/margin.parquet", columns=["date", "code6", "rzye", "rzjme", "rqyl"])
j = df["code6"].astype(str).str.zfill(6).map(cm)
t = md.dates.get_indexer(pd.to_datetime(df["date"]))
ok = j.notna().values & (t >= 0)
t, j = t[ok], j[ok].astype(int).values
out = {}
for c in ("rzye", "rzjme", "rqyl"):
    m = np.full((md.T, md.N), np.nan, dtype=np.float32)
    m[t, j] = pd.to_numeric(df[c], errors="coerce").values[ok].astype(np.float32)
    out[c] = m
del df; gc.collect()
DERIVED.mkdir(parents=True, exist_ok=True)
np.savez_compressed(DERIVED / "margin.npz", dates=md.dates.values.astype("datetime64[D]").astype(str), codes=md.codes, **out)
print({k: float(np.isfinite(v).mean()) for k, v in out.items()})
