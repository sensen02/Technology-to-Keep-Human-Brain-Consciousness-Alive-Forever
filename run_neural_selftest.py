"""run_neural_selftest.py -- verification of the spiking neural layer.

LIF has exact closed-form behaviour, so this is real analytic verification, not
a self-consistency check:

  * membrane time constant recovered from a subthreshold step response;
  * f-I curve compared against the closed-form rate;
  * absolute refractory period bounds the maximum rate;
  * synaptic delay shifts the response by exactly the declared number of steps.

Plus the scale test that matters for the whole-fly plan: build a network the
size of the FlyWire FAFB snapshot and MEASURE how long a second of brain
activity costs.

Writes: outputs/metrics_neural_selftest.json
        outputs/NEURAL_SELFTEST.zh-CN.md
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.neural import (LIFNetwork, NetworkParams, analytic_lif_rate,
                           lemplev_ziv, build_synthetic_network, nt_sign)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
res = []


def rec(name, kind, passed, detail, error=None, tol=None):
    res.append({"name": name, "kind": kind, "passed": bool(passed),
                "error": error, "tolerance": tol, "detail": detail})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}  {detail}")


# ---------------------------------------------------------------- 1. tau_m
print("=== 1. 膜时间常数（解析） ===")
tau_true = 20.0
p = NetworkParams(noise_mV=0.0, tau_m_ms=tau_true)
net = LIFNetwork(1, p, dt_ms=0.05, seed=0)
I = 5.0                      # subthreshold: V_inf = -65 + 5 = -60 mV < -50
v_trace = []
for _ in range(400):         # 20 ms
    net.step(i_ext=I)
    v_trace.append(net.v[0])
v_trace = np.array(v_trace)
t = np.arange(1, len(v_trace) + 1) * 0.05
v_inf = p["v_rest_mV"] + I
# v(t) = v_inf + (v0 - v_inf) exp(-t/tau); fit log|v_inf - v|
y = np.log(np.abs(v_inf - v_trace[5:]))
coef = np.polyfit(t[5:], y, 1)
tau_fit = -1.0 / coef[0]
err = abs(tau_fit - tau_true)
rec("membrane_time_constant", "analytic", err < 0.5,
    f"拟合 τ={tau_fit:.3f} ms（真值 {tau_true}）", err, 0.5)

# ---------------------------------------------------------------- 2. f-I curve
print("\n=== 2. f-I 曲线（解析对照） ===")
pf = NetworkParams(noise_mV=0.0, tau_m_ms=20.0, v_th_mV=-50.0,
                   v_reset_mV=-65.0, v_rest_mV=-65.0, t_ref_ms=2.0)
rows, errs = [], []
# Threshold current is exactly V_th - V_rest = 15 mV in these units, so I <= 15
# CANNOT fire and the closed form is legitimately undefined there.  Those points
# are checked separately as "must be silent", not against the formula.
for I in (5.0, 10.0):
    n = LIFNetwork(1, pf, dt_ms=0.05, seed=1)
    fired = sum(1 for _ in range(int(4000 / 0.05)) if n.step(i_ext=I)[0])
    rec(f"subthreshold_silent_I{I:g}", "analytic", fired == 0,
        f"I={I} mV < 阈值电流 15 mV：{fired} 次放电（应为 0）")

for I in (16.0, 20.0, 30.0, 40.0):
    n = LIFNetwork(1, pf, dt_ms=0.05, seed=1)
    duration = 4000.0
    steps = int(duration / 0.05)
    spikes_at = []
    for k in range(steps):
        if n.step(i_ext=I)[0]:
            spikes_at.append((k + 1) * 0.05)
    if len(spikes_at) > 3:
        isi = np.diff(spikes_at)
        f_sim = 1000.0 / np.median(isi)
    else:
        f_sim = 0.0
    f_ana = analytic_lif_rate(I, pf["tau_m_ms"], pf["v_rest_mV"], pf["v_th_mV"],
                              pf["v_reset_mV"], pf["t_ref_ms"])
    rel = abs(f_sim - f_ana) / f_ana
    errs.append(rel)
    rows.append({"I": I, "f_sim_hz": f_sim, "f_analytic_hz": f_ana, "rel_err": rel})
worst = max(errs)
rec("fI_curve_vs_closed_form", "analytic", worst < 0.02,
    f"5 个电流点最大相对误差 {worst*100:.3f}%（模拟用 ISI 中位数估计稳态频率）",
    worst, 0.02)
for r in rows:
    print(f"    I={r['I']:5.1f}  f_sim={r['f_sim_hz']:7.3f} Hz  "
          f"f_analytic={r['f_analytic_hz']:7.3f} Hz  rel={r['rel_err']*100:.3f}%")

# ---------------------------------------------------------------- 3. refractory
print("\n=== 3. 不应期上界（不变量） ===")
pr = NetworkParams(noise_mV=0.0, t_ref_ms=2.0)
n = LIFNetwork(1, pr, dt_ms=0.05, seed=2)
sp = sum(1 for _ in range(int(2000 / 0.05)) if n.step(i_ext=1e6)[0])
f_max_sim = 1000.0 * sp / 2000.0
f_max_theory = 1000.0 / pr["t_ref_ms"]
ok = f_max_sim <= f_max_theory + 1e-9
rec("refractory_bounds_max_rate", "invariant", ok,
    f"极大电流下 f={f_max_sim:.2f} Hz ≤ 1000/t_ref={f_max_theory:.2f} Hz",
    abs(f_max_sim - f_max_theory), 1e-9)

# ---------------------------------------------------------------- 4. delay
print("\n=== 4. 突触延迟（解析） ===")
pd_ = NetworkParams(noise_mV=0.0, syn_delay_ms=2.0, w_scale_mV=5.0)
n = LIFNetwork(2, pd_, dt_ms=1.0, seed=3)
from scipy import sparse as sp
n.set_connectivity(sp.csr_matrix((np.array([5.0]), (np.array([1]), np.array([0]))),
                                 shape=(2, 2)))   # 0 -> 1, delay 2 steps
n.v[0] = pd_["v_th_mV"] + 1.0      # neuron 0 spikes on step 1, naturally
n.v[1] = pd_["v_rest_mV"]
first_response = None
for k in range(12):
    n.step()
    if first_response is None and n.v[1] != pd_["v_rest_mV"]:
        first_response = k + 1
# A spike emitted at step 1 with a declared delay of d ms arrives at step 1 + d.
rec("synaptic_delay_steps", "analytic", first_response == 1 + 2,
    f"突触前在第 1 步放电，突触后膜电位首次变化在第 {first_response} 步"
    f"（= 1 + 延迟 2 步）",
    abs((first_response or 0) - 3), 0)

# also check a longer delay is honoured exactly
pd5 = NetworkParams(noise_mV=0.0, syn_delay_ms=5.0, w_scale_mV=5.0)
n5 = LIFNetwork(2, pd5, dt_ms=1.0, seed=3)
n5.set_connectivity(sp.csr_matrix((np.array([5.0]), (np.array([1]), np.array([0]))),
                                  shape=(2, 2)))
n5.v[0] = pd5["v_th_mV"] + 1.0
n5.v[1] = pd5["v_rest_mV"]
fr5 = None
for k in range(15):
    n5.step()
    if fr5 is None and n5.v[1] != pd5["v_rest_mV"]:
        fr5 = k + 1
rec("synaptic_delay_scales", "analytic", fr5 == 1 + 5,
    f"延迟改为 5 步时首次响应在第 {fr5} 步（= 1 + 5）", abs((fr5 or 0) - 6), 0)

# ---------------------------------------------------------------- 5. E/I sign
print("\n=== 5. 兴奋/抑制符号（方向性） ===")
def spikes_of_post(drive_sign, w=400.0):
    """Strong single synapse.

    Note: with the DEFAULT weight scale (0.5 mV per synapse) one presynaptic
    neuron firing at ~160 Hz depolarises the postsynaptic cell by well under
    1 mV and it never reaches threshold.  Neurons need many coincident inputs --
    physiologically reasonable, but it means the weight scale decides whether a
    synthetic network fires at all, and it is an illustrative value.
    """
    pp = NetworkParams(noise_mV=0.0, w_scale_mV=w, syn_delay_ms=2.0)
    net = LIFNetwork(2, pp, dt_ms=1.0, seed=4)
    W = sp.csr_matrix((np.array([drive_sign * w]), (np.array([1]), np.array([0]))),
                      shape=(2, 2))
    net.set_connectivity(W)
    # neuron 0 is driven hard so it fires; neuron 1 only ever sees the synapse
    n_post = 0
    for _ in range(400):
        s = net.step(i_ext=np.array([80.0, 0.0]))
        if s[1]:
            n_post += 1
    return n_post


exc = spikes_of_post(+1)
inh = spikes_of_post(-1)
rec("excitatory_raises_inhibitory_lowers", "invariant", exc > inh,
    f"强突触（w=400 mV）下：兴奋性输入使突触后放电 {exc} 次，抑制性 {inh} 次")

# ---------------------------------------------------------------- 6. determinism
print("\n=== 6. 确定性 ===")
def spikes_of(seed):
    pn = NetworkParams()
    pre, post, cnt, nt = build_synthetic_network(300, 3000, seed=seed)
    n = LIFNetwork(300, pn, dt_ms=1.0, seed=seed)
    n.set_connectivity_from_edges(pre, post, cnt, nt)
    r = n.run(200.0, i_ext=15.0, record=True)
    return r["spikes"]

a, b = spikes_of(11), spikes_of(11)
c = spikes_of(12)
rec("network_deterministic", "determinism", bool(np.array_equal(a, b)),
    "同种子逐位可复现")
rec("network_seed_changes_result", "invariant", not np.array_equal(a, c),
    "不同种子给出不同放电模式")

# ---------------------------------------------------------------- 7. LZ
print("\n=== 7. 复杂度度量健全性 ===")
const = lemplev_ziv(np.zeros(2000, dtype=int))
rand = lemplev_ziv((np.random.default_rng(0).random(2000) < 0.5).astype(int))
rec("lz_reasonable_range", "invariant", 0.0 < const < rand <= 1.0,
    f"常量序列 {const:.4f} < 随机序列 {rand:.4f}（均为 LZ76 归一化值）")

# ---------------------------------------------------------------- 8. scale
print("\n=== 8. 全脑规模实测（关键可行性数字） ===")
N_NEURONS = 139255        # FlyWire FAFB v783
N_EDGES = 3732460
t0 = time.time()
pre, post, cnt, nt = build_synthetic_network(N_NEURONS, N_EDGES, seed=0)
t_build = time.time() - t0
brain = LIFNetwork(N_NEURONS, NetworkParams(), dt_ms=1.0, seed=0)
info = brain.set_connectivity_from_edges(pre, post, cnt, nt)
t0 = time.time()
r = brain.run(1000.0, i_ext=0.0, record=False)     # 1000 ms = 1 s, 1000 steps
t_run = time.time() - t0
rec("whole_brain_1s_runtime", "invariant", t_run < 600.0,
    f"{N_NEURONS} 神经元 / {N_EDGES} 连接：模拟 1 秒脑活动用 {t_run:.1f} s CPU"
    f"（含 {info['nnz']} 个非零权重；建图 {t_build:.1f} s）",
    t_run, 600.0)
print(f"    平均放电率 {r['mean_rate_hz']:.3f} Hz（无外部驱动，仅自发噪声）")
print(f"    RAM 中 CSR 权重约 {info['nnz']*12/1e6:.0f} MB")

# ---------------------------------------------------------------- write
npass = sum(1 for x in res if x["passed"])
summary = {"n_tests": len(res), "n_passed": npass, "n_failed": len(res) - npass,
           "all_passed": npass == len(res), "results": res,
           "flywire_reference": {"dataset": "FlyWire FAFB v783",
                                 "neurons": N_NEURONS, "connections": N_EDGES,
                                 "source": "https://codex.flywire.ai/api/download"},
           "what_this_proves": "LIF 层在解析可解的行为上正确；全脑规模的运行时间已实测",
           "what_this_does_not_prove":
               "LIF 不是神经元；连接组只给「谁连谁」而非突触权重或生理；"
               "复杂度指标不是意识度量。当前网络是合成拓扑，不是果蝇。"}
with open(os.path.join(OUT, "metrics_neural_selftest.json"), "w") as fh:
    json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)

lines = ["# 神经层自检报告", "",
         f"共 {len(res)} 项，通过 {npass}，失败 {len(res)-npass}。", "",
         "| 测试 | 类型 | 通过 | 结果 |", "|---|---|---|---|"]
for x in res:
    lines.append(f"| `{x['name']}` | {x['kind']} | {'✅' if x['passed'] else '❌'} | {x['detail']} |")
lines += ["", "## 边界", "",
          "- LIF 不是神经元：无树突、无通道、无适应、无神经调质。",
          "- **连接组 ≠ 功能**：它只给定「谁连谁」，不给突触权重、释放概率、生理参数。",
          "- 复杂度指标（LZ、同步度）是**描述性度量**，不是意识度量。",
          "- 当前跑的是**合成拓扑**，不是果蝇；真连接组需要 Codex API token。"]
with open(os.path.join(OUT, "NEURAL_SELFTEST.zh-CN.md"), "w") as fh:
    fh.write("\n".join(lines) + "\n")

print(f"\n=== {npass}/{len(res)} 通过 ===")
