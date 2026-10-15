"""把训练集用的模型打分（只训练、预测到 2020 年）存到暂存目录，供筛选器反复组合使用。"""
import sys
import time

import numpy as np

import ml
from framework import CACHE, Lab
from run_batch3 import SPECS, load_ext

OUT = CACHE.parent


def main(hyp, var, split="train"):
    sp = SPECS[(hyp, var)]
    lab = Lab()
    md = lab.md
    ext = load_ext(md, sp.get("plus", False)) if sp["use_events"] else None   # plus="fin" 时加载财报
    extra = {k: sp[k] for k in ("keep", "seeds", "model", "rank_inputs", "window_years", "label", "universe") if k in sp}
    t = time.time()
    last = 2023 if split == "valid" else 2020          # valid：用于验证集运行（每年用截至上年末的数据重训）
    sc = ml.walk_forward_scores(md, ext if sp["use_events"] else None, H=sp["H"], last_year=last, free_md=True, **extra)
    np.save(OUT / f"score_{hyp}_{var}_{split}.npy", sc.astype(np.float32))
    print(hyp, var, "saved", round(time.time() - t), "s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(sys.argv[3:4] or ["train"]))
