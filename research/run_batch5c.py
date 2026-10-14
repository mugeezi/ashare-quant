"""第五批（续）：模型打分的组合构建变化 + 新族。"""
import numpy as np

import families as F
from framework import CACHE, Lab, eligible_matrix
from run_batch3 import load_ext
from screen import Screener

lab = Lab()
md = lab.md
ext = load_ext(md, plus=True)
sc = Screener(lab)
el = eligible_matrix(md)
h22 = np.load(CACHE.parent / "score_H22_base_train.npy").astype(np.float64)
sc.run("H22", "calib", h22, H=10, start="2016-01-01", notes="筛选器校准（对照引擎 E0029：2016–2020 超额 +21.7%）")
sc.run("F18", "h10", F.f18_avoid_all(md, el), H=10, notes="多重避雷（事后）")
sc.run("F24", "h10", F.f24_peer_info_transfer(md, ext), H=10, notes="同伴信息溢出")
sc.run("F22", "h10", F.f22_large_only(md, h22), H=10, start="2016-01-01", notes="H22 打分，只在流通市值前 40% 中选")
sc.run("F23", "h10", F.f23_model_avoid(md, h22, el), H=10, start="2016-01-01", notes="H22 打分，剔除避雷分最差 30%")
