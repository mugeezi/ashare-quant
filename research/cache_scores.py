"""把训练集用的模型打分（只训练、预测到 2020 年）存到暂存目录，供筛选器反复组合使用。"""
import sys
import time

import numpy as np

import ml
from framework import CACHE, Lab
from run_batch3 import SPECS, load_ext

OUT = CACHE.parent


def main(hyp, var):
    sp = SPECS[(hyp, var)]
    lab = Lab()
    md = lab.md
    ext = load_ext(md, sp.get("plus", False)) if sp["use_events"] else None
    extra = {k: sp[k] for k in ("keep", "seeds", "model", "rank_inputs") if k in sp}
    t = time.time()
    sc = ml.walk_forward_scores(md, ext if sp["use_events"] else None, H=sp["H"], last_year=2020, **extra)
    np.save(OUT / f"score_{hyp}_{var}_train.npy", sc.astype(np.float32))
    print(hyp, var, "saved", round(time.time() - t), "s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
