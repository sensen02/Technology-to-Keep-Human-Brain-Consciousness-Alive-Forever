"""Verify the receptor models against their exact steady states, and
cross-check the chemo-receptor against hormone.py's independent Langmuir solver."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from engine.receptors import (PhotoReceptor, MechanoReceptor, ChemoReceptor,
                              ThermoReceptor, HygroReceptor, ReceptorBank)
import hormone

res = []
def rec(name, kind, passed, detail, error=None, tol=None):
    res.append({"name": name, "kind": kind, "passed": bool(passed),
                "error": None if error is None else float(error),
                "tolerance": tol, "detail": detail})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}  {detail}")

# ---------------- 1. photoreceptor: steady state = Naka-Rushton -------------
print("=== 1. 光感受器：稳态必须收敛到解析 Naka-Rushton ===")
pr = PhotoReceptor(3, gain_mV=10.0, dt_ms=1.0, seed=0)
errs = []
for I in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0):
    p = PhotoReceptor(3, gain_mV=10.0, dt_ms=1.0, seed=0)
    for _ in range(8000):          # long enough for adaptation to settle
        p.step(np.full(3, I))
    ana = p.steady_state(np.full(3, I))
    errs.append(float(np.max(np.abs(p.state - ana))))
rec("photoreceptor_steady_state", "analytic", max(errs) < 1e-4,
    f"6 个光强下最大偏差 {max(errs):.2e}", max(errs), 1e-4)

# adaptation must be real: a step response overshoots then settles LOWER
p = PhotoReceptor(1, gain_mV=10.0, dt_ms=1.0, seed=0)
trace = [float(p.step(np.array([1.0]))[0]) for _ in range(3000)]
peak = max(trace); final = trace[-1]
rec("photoreceptor_adapts", "invariant", peak > final * 1.05,
    f"阶跃响应峰值 {peak:.4f} mV 高于稳态 {final:.4f} mV（存在适应）",
    peak - final, 0.0)

# ---------------- 2. mechanoreceptor: Boltzmann ----------------------------
print("\n=== 2. 机械感受器：稳态 = Boltzmann ===")
mr = MechanoReceptor(4, gain_mV=10.0, dt_ms=1.0, seed=0)
errs = []
for x in (0.0, 0.2, 0.4, 0.7, 1.0, 1.5):
    m = MechanoReceptor(4, gain_mV=10.0, dt_ms=1.0, seed=0)
    for _ in range(2000):
        m.step(np.full(4, x))
    errs.append(float(np.max(np.abs(m.state - m.steady_state(np.full(4, x))))))
rec("mechanoreceptor_boltzmann", "analytic", max(errs) < 1e-6,
    f"最大偏差 {max(errs):.2e}", max(errs), 1e-6)
half = 0.5
x_half = (mr.p["dG"]) / mr.p["k_gate"]
rec("mechano_half_point_exact", "analytic",
    abs(float(mr.steady_state(np.array([x_half]))[0]) - half) < 1e-12,
    f"x=dG/k={x_half:.4f} 时 P_open=0.5（解析恒等）")

# ---------------- 3. chemoreceptor vs hormone.py ---------------------------
print("\n=== 3. 化学感受器 vs hormone.py 独立实现 ===")
cr = ChemoReceptor(5, gain_mV=10.0, dt_ms=1.0, seed=0, kd=1.0)
mine = float(cr.steady_state(np.array([2.0]))[0])
theirs = float(hormone.equilibrium_occupancy(2.0, {"kd_nM": 1.0, "n_hill": 1.0}))
d = abs(mine - theirs)
rec("chemoreceptor_matches_hormone", "analytic", d < 1e-12,
    f"c=2, Kd=1：本模块 {mine:.12f} vs hormone.py {theirs:.12f}", d, 1e-12)

# ---------------- 4. thermo / hygro ---------------------------------------
print("\n=== 4. 温度 / 湿度感受器 ===")
tr = ThermoReceptor(1, t_pref=25.0, width=8.0)
rec("thermo_half_point", "analytic",
    abs(float(tr.steady_state(np.array([25.0]))[0]) - 0.5) < 1e-12,
    "在偏好温度处响应为 0.5")
hr = HygroReceptor(1, t_pref=0.5, width=0.15)
rec("hygro_half_point", "analytic",
    abs(float(hr.steady_state(np.array([0.5]))[0]) - 0.5) < 1e-12,
    "在 50% 湿度处响应为 0.5")

# ---------------- 5. bank: stimuli -> current array -------------------------
print("\n=== 5. ReceptorBank：刺激 -> 全脑电流数组 ===")
N = 1000
bank = ReceptorBank(N, dt_ms=1.0, seed=0)
bank.attach("vision", np.arange(0, 100), PhotoReceptor, gain_mV=10.0)
bank.attach("mechanosensory", np.arange(100, 200), MechanoReceptor, gain_mV=10.0)
bank.attach("olfactory", np.arange(200, 260), ChemoReceptor, gain_mV=10.0)
i = bank.current({"vision": 0.8, "mechanosensory": 0.9, "olfactory": 3.0})
rec("bank_shape_and_isolation", "invariant",
    i.shape == (N,) and np.all(i[260:] == 0.0) and i[:100].sum() > 0,
    f"电流数组 {i.shape}，未接受体的 740 个神经元电流为 0",
    float(np.abs(i[260:]).max()), 0.0)
rec("bank_stimulus_changes_output", "invariant",
    not np.allclose(bank.current({"vision": 0.0}), bank.current({"vision": 0.8})),
    "改变光强会改变注入电流")
rec("bank_unknown_modality_ignored", "invariant",
    float(np.abs(bank.current({"nonexistent": 5.0})).max()) == 0.0,
    "未知模态不注入任何电流")

npass = sum(1 for r in res if r["passed"])
summary = {"n_tests": len(res), "n_passed": npass, "n_failed": len(res)-npass,
           "all_passed": npass == len(res), "results": res,
           "what_this_proves": "每个受体模型的稳态都与其解析解一致；"
                               "化学感受器与 hormone.py 的独立实现逐位一致",
           "what_this_does_not_prove":
               "这些是最小转导模型；本项目没有任何果蝇受体动力学的实测值；"
               "转导产生电流，不产生知觉"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
with open(os.path.join(OUT, "metrics_receptor_selftest.json"), "w") as fh:
    json.dump(summary, fh, indent=2, ensure_ascii=False)
print(f"\n=== {npass}/{len(res)} 通过 ===")
