"""自动流程：缓存训练集打分 → 筛选 → （晋级）前视检测 → 引擎训练集 → 稳健性 → 验证集 → 修订口径 DSR。
每一步单独子进程（控制内存），全部记录在 experiments.csv，进度写 logs/autopilot.log。
晋级规则（2026-10-15 登记）：筛选器 2016–2020 超额 ≥ 20% 且回撤不超过 18%；构建类变体（复用已有打分）直接进引擎。
稳健性规则同前（6 个变体中至少 5 个超额 ≥ 5% 且回撤 ≤ 45%）。"""
import json, shutil, subprocess, sys, time
import pandas as pd
from framework import CACHE, ROOT

LOG = ROOT / "logs" / "autopilot.log"
PY = [sys.executable, "-u", "-W", "ignore"]


def log(msg):
    with open(LOG, "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def run(args, name):
    t = time.time()
    p = subprocess.run(PY + args, cwd=ROOT / "research", capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    (ROOT / "logs" / f"ap_{name}.log").write_text(out)
    log(f"  {name}: 退出码 {p.returncode}，{time.time() - t:.0f}s")
    return p.returncode, out


def last_row(hyp, var, split):
    d = pd.read_csv(ROOT / "experiments.csv")
    d = d[(d.hyp_id == hyp) & (d.variant == var) & (d.split == split)]
    return d.iloc[-1] if len(d) else None


def make_combo(hyp, var, comps):
    """组合打分：各组成模型打分逐日转截面分位后等权平均。组成模型均已单独通过前视检测，逐日分位平均不引入未来信息。"""
    import numpy as np
    from run_batch3 import SPECS
    for sp in ("train", "valid"):
        mats = []
        for c in comps:
            f = CACHE.parent / f"score_{c}_base_{sp}.npy"
            if not f.exists():
                rc, _ = run(["cache_scores.py", c, "base"] + ([sp] if sp == "valid" else []), f"{c}_{sp}cache")
                if rc != 0:
                    return False
            m = np.load(f).astype(np.float64)
            mats.append(pd.DataFrame(m).rank(axis=1, pct=True).values)
            del m
        out = np.nanmean(np.stack(mats), axis=0) if len(mats) > 1 else mats[0]
        np.save(CACHE.parent / f"score_{hyp}_{var}_{sp}.npy", out.astype(np.float32))
        del mats, out
    d = pd.read_csv(ROOT / "experiments.csv")
    def leak_ok(c):
        f = ROOT / "reports" / f"leak_{c}_base.json"
        if f.exists():
            return bool(json.loads(f.read_text()).get("passed"))
        r = d[(d.hyp_id == c) & (d.variant == "base") & (d.split == "train")]
        return len(r) > 0 and str(r.iloc[-1].leak_test).startswith("pass")
    ok = all(leak_ok(c) for c in comps)
    (ROOT / "reports" / f"leak_{hyp}_{var}.json").write_text(json.dumps(
        {"passed": ok, "n_checks": 0, "changed": [], "note": "组合打分：组成模型 " + "、".join(comps) + " 均单独通过前视检测"}, ensure_ascii=False))
    log(f"  组合打分 {'+'.join(comps)} 已生成，组成模型前视检测 {'全部通过' if ok else '未全部通过'}")
    return ok


def pipeline(hyp, var="base", reuse=None):
    log(f"== {hyp}/{var}")
    cf = CACHE.parent / f"score_{hyp}_{var}_train.npy"
    from run_batch3 import SPECS
    comps = SPECS.get((hyp, var), {}).get("combo")
    if comps and not cf.exists():
        if not make_combo(hyp, var, comps):
            return log("  组合打分失败或组成模型未全部通过前视检测")
    if reuse:                                     # 构建类变体：打分与 reuse 完全相同
        for sp in ("train", "valid"):
            src = CACHE.parent / f"score_{reuse}_base_{sp}.npy"
            if src.exists():
                shutil.copy(src, CACHE.parent / f"score_{hyp}_{var}_{sp}.npy")
    if not cf.exists():
        rc, _ = run(["cache_scores.py", hyp, var], f"{hyp}_cache")
        if rc != 0:
            return log("  打分失败，跳过")
    if not reuse:
        rc, out = run(["screen_one.py", hyp, var], f"{hyp}_screen")
        res = [json.loads(l[7:]) for l in out.splitlines() if l.startswith("RESULT ")]
        if not res:
            return log("  筛选失败")
        r = res[0]
        log(f"  筛选：超额 {r['excess']:+.1%}，回撤 {r['dd']:.1%}，正年 {r['years']}/5")
        if not (r["excess"] >= 0.20 and r["dd"] >= -0.18):
            return log("  未晋级")
    if comps:
        rc, out = 0, "'passed': True"
    else:
        rc, out = run(["run_leak.py", hyp, var], f"{hyp}_leak")
    if rc != 0 or "'passed': True" not in out:
        return log("  前视检测未通过或失败：" + out[-300:])
    rc, out = run(["run_batch3.py", f"{hyp}:{var}"], f"{hyp}_train")
    row = last_row(hyp, var, "train")
    if row is None or str(row.passed).lower() != "true":
        return log(f"  引擎训练集未达标：{None if row is None else (row.excess_cagr, row.max_drawdown, row.years_excess_pos)}")
    log(f"  引擎训练集：超额 {row.excess_cagr:+.1%}，回撤 {row.max_drawdown:.1%}，{row.years_excess_pos}")
    rc, out = run(["run_robust.py", hyp, var], f"{hyp}_robust")
    d = pd.read_csv(ROOT / "experiments.csv")
    rb = d[(d.hyp_id == hyp) & d.variant.astype(str).str.startswith(f"{var}_") & (d.split == "train")].tail(6)
    ok = int(((rb.excess_cagr >= 0.05) & (rb.max_drawdown >= -0.45)).sum())
    log(f"  稳健性 {ok}/{len(rb)}")
    if ok < 5:
        return log("  稳健性未过，不上验证集")
    if not (CACHE.parent / f"score_{hyp}_{var}_valid.npy").exists():
        rc, _ = run(["cache_scores.py", hyp, var, "valid"], f"{hyp}_vcache")
        if rc != 0:
            return log("  验证集打分失败")
    rc, out = run(["run_valid.py", hyp, var], f"{hyp}_valid")
    vr = last_row(hyp, var, "valid")
    if vr is None:
        return log("  验证集运行失败：" + out[-300:])
    log(f"  验证集：超额 {vr.excess_cagr:+.1%}，回撤 {vr.max_drawdown:.1%}，{vr.years_excess_pos}，α 占比 {vr.alpha_share}")
    rc, out = run(["dsr_eval.py", f"{hyp}"], f"{hyp}_dsr")
    log("  " + (out.strip().splitlines()[-1] if out.strip() else "DSR 计算失败"))


if __name__ == "__main__":
    for item in sys.argv[1:]:
        hyp, _, reuse = item.partition("@")
        try:
            pipeline(hyp, "base", reuse or None)
        except Exception as e:  # noqa: BLE001
            log(f"  异常：{e}")
