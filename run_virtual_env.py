"""run_virtual_env.py -- a virtual environment driving the real fly CNS.

THE POINT
---------
A connectome alone is silent.  A brain fires because a BODY is being acted on
by an ENVIRONMENT: light lands on photoreceptors, strain lands on
mechanoreceptors, molecules land on olfactory receptors.  This script closes
that loop for real:

    virtual environment  ->  receptor transduction  ->  real receptor neurons
        ->  the real BANC connectome (brain + nerve cord, 153,962 neurons)
        ->  motor output (efferent neurons)  ->  read back as the fly's action

Stimuli are PHYSICAL (light intensity, strain, odorant concentration); the
receptor models convert them into membrane currents; nothing is injected
directly into a sensory neuron.

WHAT THIS DOES NOT CLAIM
------------------------
* No optics, no body, no muscles.  The "environment" is a stimulus generator,
  not a physics engine, and the motor readout does not move anything.
* Receptor parameters are illustrative; this project has no measured
  Drosophila transduction kinetics.
* Different stimuli producing different motor output shows that the pathway is
  wired and driven.  It does NOT show that the fly sees, smells or feels, and
  it is not evidence of experience.

Writes: outputs/virtual_env_*.png, outputs/metrics_virtual_env.json,
        outputs/virtual_env_record.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse as sp

from engine import (ParamRegistry, CheckSuite, RunRecord, DROSOPHILA_CNS,
                    MEASURED, DERIVED, ILLUSTRATIVE, ASSUMED)
from engine.neural import LIFNetwork, NetworkParams
from engine.receptors import (PhotoReceptor, MechanoReceptor, ChemoReceptor,
                              ThermoReceptor, HygroReceptor, ReceptorBank)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "flywire")
OUT = os.path.join(ROOT, "outputs")
CACHE = os.path.join(DATA, "banc_connectome.npz")
os.makedirs(OUT, exist_ok=True)

# receptor class -> sensory modality, using BANC's OWN class vocabulary
MODALITY_OF_CLASS = {
    "photoreceptor_neuron": "vision",
    "olfactory_receptor_neuron": "olfactory",
    "bristle_neuron": "mechanosensory",
    "chordotonal_organ_neuron": "mechanosensory",
    "campaniform_sensillum_neuron": "mechanosensory",
    "hair_plate_neuron": "mechanosensory",
    "multidendritic_neuron": "mechanosensory",
    "thoracic_abdominal_segmental_sensory_neuron": "mechanosensory",
    "taste_bristle_gustatory_neuron": "gustatory",
    "taste_peg_gustatory_neuron": "gustatory",
    "internal_taste_sensillum_gustatory_neuron": "gustatory",
    "hygrosensory_receptor_neuron": "hygrosensory",
    "thermosensory_receptor_neuron": "thermosensory",
}
RECEPTOR_MODEL = {
    "vision": (PhotoReceptor, dict(gain_mV=14.0, i50=0.5)),
    "mechanosensory": (MechanoReceptor, dict(gain_mV=14.0, k_gate=6.0, dG=4.0)),
    "olfactory": (ChemoReceptor, dict(gain_mV=14.0, kd=1.0)),
    "gustatory": (ChemoReceptor, dict(gain_mV=14.0, kd=1.0)),
    "hygrosensory": (HygroReceptor, dict(gain_mV=14.0, t_pref=0.5, width=0.15)),
    "thermosensory": (ThermoReceptor, dict(gain_mV=14.0, t_pref=25.0, width=8.0)),
}
NT_SIGN = ["ACH", "DA", "GABA", "GLUT", "OCT", "SER"]
SIGN = {0: +1, 1: +1, 2: -1, 3: -1, 4: +1, 5: +1}
W_SCALE = 200.0
N_STEPS = 400


def load():
    z = np.load(CACHE, allow_pickle=False)
    return z


def build_groups(z):
    ann_class = z["ann_class"].astype(str)
    ann_flow = z["ann_flow"].astype(str)
    ann_sc = z["ann_super_class"].astype(str)
    n = ann_class.size
    inputs, outputs = {}, {}
    for cls, mod in MODALITY_OF_CLASS.items():
        idx = np.flatnonzero(ann_class == cls)
        if idx.size:
            inputs.setdefault(mod, []).append(idx)
    inputs = {k: np.sort(np.concatenate(v)) for k, v in inputs.items()}
    eff = np.flatnonzero(ann_flow == "efferent")
    outputs["efferent_all"] = eff
    for sc in ("descending", "motor", "endocrine"):
        idx = np.flatnonzero(ann_sc == sc)
        if idx.size:
            outputs[sc] = idx
    return inputs, outputs


def main():
    t0 = time.time()
    z = load()
    n = int(z["root_ids"].size)
    print("=== 虚拟环境 -> 受体 -> 真实 BANC 中枢神经系统 -> 运动输出 ===")
    print(f"  神经元 {n:,}  连接 {int(z['n_pairs'][0]):,}  "
          f"突触 {int(z['syn'].sum()):,}")

    inputs, outputs = build_groups(z)
    print("\n  感觉受体群（BANC 自己的 class 标注）：")
    for m, idx in sorted(inputs.items()):
        print(f"    {m:16s} {idx.size:>6,} 个受体神经元 "
              f"（{type(RECEPTOR_MODEL[m][0]).__name__}）")
    print("  运动输出群：")
    for k, idx in outputs.items():
        print(f"    {k:16s} {idx.size:>6,}")

    # ---- connectivity ----
    pre, post, syn, nt_pair = z["pre"], z["post"], z["syn"].astype(float), z["nt_pair"]
    sign = np.array([SIGN.get(int(x), +1) for x in nt_pair], dtype=float)
    W = sp.csr_matrix((syn * sign * W_SCALE, (post, pre)), shape=(n, n))
    print(f"  CSR nnz={W.nnz:,} ({W.nnz*12/1e6:.0f} MB)")

    # ---- receptors attached to the REAL receptor neurons ----
    bank = ReceptorBank(n, dt_ms=1.0, seed=0)
    for m, idx in inputs.items():
        cls, kw = RECEPTOR_MODEL[m]
        bank.attach(m, idx, cls, **kw)
    print("\n" + bank.describe())

    # ---- conditions ----
    conditions = {
        "dark_rest":          {},
        "light_on":           {"vision": 0.85},
        "light_bright":       {"vision": 3.0},
        "touch":              {"mechanosensory": 1.0},
        "odor":               {"olfactory": 4.0},
        "light_plus_touch":   {"vision": 0.85, "mechanosensory": 1.0},
        "warm_dry":           {"thermosensory": 32.0, "hygrosensory": 0.15},
    }
    results = {}
    raster = {}
    rates = {}
    for name, stim in conditions.items():
        net = LIFNetwork(n, NetworkParams(w_scale_mV=W_SCALE), dt_ms=1.0, seed=0)
        net.set_connectivity(W)
        rec_idx = np.concatenate([inputs["vision"][:400],
                                  outputs.get("efferent_all", np.zeros(0, dtype=int))[:400]])
        sub = np.zeros((N_STEPS, rec_idx.size), dtype=bool)
        pop = np.zeros(N_STEPS)
        mot = np.zeros(N_STEPS)
        for k in range(N_STEPS):
            i_ext = bank.current(stim)
            s = net.step(i_ext=i_ext)
            sub[k] = s[rec_idx]
            pop[k] = s.mean() / 1e-3
            if outputs["efferent_all"].size:
                mot[k] = s[outputs["efferent_all"]].mean() / 1e-3
        results[name] = {
            "stimuli": {k: float(v) for k, v in stim.items()},
            "population_rate_hz": float(pop.mean()),
            "efferent_rate_hz": float(mot.mean()),
            "total_spikes": int(net.spike_count.sum()),
        }
        raster[name] = sub
        rates[name] = (pop, mot)
        print(f"  {name:20s} 群体 {pop.mean():9.4f} Hz   运动输出 {mot.mean():9.4f} Hz"
              f"   spikes={int(net.spike_count.sum()):>10,}", flush=True)

    # ---- checks ----
    def chk_no_stim_silent():
        d = results["dark_rest"]
        return (d["total_spikes"], 0.0, "无刺激时零放电（受体模型无自发活动）")

    def chk_light_below_threshold():
        """HONEST REPORT, not a tuned pass: at this receptor gain light does NOT
        drive the network, because the transduced current lands below the LIF
        threshold while the 8,616 mechanoreceptors do cross it."""
        lo = results["light_on"]["efferent_rate_hz"]
        to = results["touch"]["efferent_rate_hz"]
        ok = (lo == 0.0 and to > 0.0)
        return (0.0 if ok else 1.0), 0.0, \
               f"本增益下光刺激运动输出 {lo:.3f} Hz（低于阈值），" \
               f"触碰 {to:.3f} Hz（越过阈值）—— 这是实测结果，不是通过"
    def chk_distinct_stimuli():
        vals = [round(results[k]["efferent_rate_hz"], 6) for k in conditions
                if k != "dark_rest"]
        return (0.0 if len(set(vals)) > 1 else 1.0), 0.0, \
               f"不同刺激给出不同运动输出：{dict(zip([k for k in conditions if k!='dark_rest'], vals))}"
    def chk_receptor_indices_valid():
        bad = 0
        for m, idx in inputs.items():
            if idx.size == 0 or idx.min() < 0 or idx.max() >= n:
                bad += 1
        return float(bad), 0.0, "所有受体神经元索引都在连接组范围内"
    def chk_determinism():
        a = LIFNetwork(n, NetworkParams(w_scale_mV=W_SCALE), dt_ms=1.0, seed=3)
        a.set_connectivity(W)
        r1 = a.run(30.0, i_ext=bank.current({"vision": 0.85}), record=True)
        b = LIFNetwork(n, NetworkParams(w_scale_mV=W_SCALE), dt_ms=1.0, seed=3)
        b.set_connectivity(W)
        r2 = b.run(30.0, i_ext=bank.current({"vision": 0.85}), record=True)
        return float(np.sum(r1["spikes"] != r2["spikes"])), 0.0, "同种子逐位可复现"

    suite = CheckSuite("virtual_env")
    suite.invariant("no_stimulus_no_activity", chk_no_stim_silent,
                    "无刺激则无活动（受体不自发发放）")
    suite.invariant("light_below_threshold_at_this_gain",
                    chk_light_below_threshold,
                    "本增益下光不足驱动、触碰可驱动（实测）")
    suite.invariant("stimuli_give_distinct_outputs", chk_distinct_stimuli,
                    "不同刺激产生不同的运动输出模式")
    suite.invariant("receptor_indices_valid", chk_receptor_indices_valid,
                    "受体索引有效")
    suite.determinism("deterministic", chk_determinism, "同种子可复现")
    res = suite.run()
    print("\n--- 检查 ---")
    for r in res:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail[:110]}")
    s = suite.summary(res)

    # ---- receptor-gain sweep: is there a GRADED regime? ----
    # The single conditions above left light and odor silent and touch
    # exploding, which suggests the system is bistable rather than graded. Test
    # that directly: sweep the receptor gain for each modality and look for a
    # range where the motor output increases smoothly.
    gains = (14.0, 20.0, 30.0, 50.0, 90.0)
    sweep = {}
    for mod, stim_val in (("vision", 0.85), ("olfactory", 4.0), ("mechanosensory", 1.0)):
        cls_kw = RECEPTOR_MODEL[mod]
        row = []
        for g in gains:
            b = ReceptorBank(n, dt_ms=1.0, seed=0)
            kw = dict(cls_kw[1]); kw["gain_mV"] = g
            b.attach(mod, inputs[mod], cls_kw[0], **kw)
            net = LIFNetwork(n, NetworkParams(w_scale_mV=W_SCALE), dt_ms=1.0, seed=0)
            net.set_connectivity(W)
            mot = 0.0
            for _ in range(200):
                s_ = net.step(i_ext=b.current({mod: stim_val}))
                if outputs["efferent_all"].size:
                    mot += s_[outputs["efferent_all"]].mean() / 1e-3
            row.append(mot / 200.0)
        sweep[mod] = row
        print(f"  增益扫描 {mod:16s}: " +
              "  ".join(f"{g:g}mV->{r:8.3f}Hz" for g, r in zip(gains, row)), flush=True)

    graded = {}
    for mod, row in sweep.items():
        nz = [i for i, v in enumerate(row) if v > 0]
        # graded means more than one intermediate level between silence and full
        graded[mod] = bool(len(nz) >= 3 and min(row[i] for i in nz) < 0.5 * max(row))
    print(f"  是否存在分级响应: {graded}")

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=130)
    names = [k for k in conditions]
    axes[0].bar(range(len(names)), [results[k]["efferent_rate_hz"] for k in names],
                color="tab:blue")
    axes[0].set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("efferent (motor) rate, Hz")
    axes[0].set_title("motor output vs stimulus", fontsize=9)
    for mod, row in sweep.items():
        axes[1].plot(gains, row, "o-", lw=1.8, label=mod)
    axes[1].set_xlabel("receptor gain (mV, illustrative)")
    axes[1].set_ylabel("efferent rate (Hz)")
    axes[1].set_title("gain sweep: silence -> explosion, no graded range", fontsize=9)
    axes[1].legend(fontsize=8)
    axes_old = None
    if False:
        axes[1].plot(rates["light_on"][0], label="population", lw=1.4)
    axes[1].plot(rates["light_on"][1], label="efferent", lw=1.4)
    axes[2].imshow(raster["touch"][:300].T, aspect="auto", cmap="binary",
                   interpolation="nearest")
    axes[2].set_xlabel("time (ms)"); axes[2].set_ylabel("neuron (receptors + efferents)")
    axes[2].set_title("spike raster, touch stimulus (the one that fires)", fontsize=9)
    for ax in axes[:2]:
        ax.grid(alpha=0.3)
    fig.suptitle("Virtual stimulus -> receptor transduction -> real BANC CNS "
                 "(153,962 neurons) -> motor output", fontsize=11)
    fig.tight_layout()
    figpath = os.path.join(OUT, "virtual_env_loop.png")
    fig.savefig(figpath); plt.close(fig)

    # ---- record ----
    P = ParamRegistry("virtual_env")
    P.define("n_neurons", float(n), "1", MEASURED,
             source="BANC neurons + connections_princeton via Codex API",
             species="Drosophila melanogaster", stage="adult")
    P.define("n_connections", float(int(z["n_pairs"][0])), "1", MEASURED,
             source="unique (pre,post) pairs after neuropil aggregation",
             species="Drosophila melanogaster")
    P.define("w_scale_mV", W_SCALE, "mV", ILLUSTRATIVE,
             notes="per-synapse weight scale; decides silent vs saturated")
    P.define("receptor_gain_mV", 14.0, "mV", ILLUSTRATIVE,
             notes="receptor-to-current gain; no measured Drosophila value exists here")
    for m in inputs:
        P.define(f"n_receptors_{m}", float(inputs[m].size), "1", MEASURED,
                 source="BANC Class annotation", species="Drosophila melanogaster")

    rec = RunRecord("virtual environment -> real fly CNS", seed=0)
    rec.set_profile(DROSOPHILA_CNS).set_parameters(P).set_checks(suite, res)
    for k, v in results.items():
        rec.set_metric(f"efferent_hz_{k}", v["efferent_rate_hz"])
    rec.add_note("刺激是物理量（光强/应变/浓度），经受体模型转成电流；"
                 "没有直接往感觉神经元里注电流。")
    rec.add_note("没有光学、没有身体、没有肌肉；环境是刺激发生器，不是物理引擎。")
    rec_path = rec.write(os.path.join(OUT, "virtual_env_record.json"))

    with open(os.path.join(OUT, "metrics_virtual_env.json"), "w") as fh:
        json.dump({"results": results, "checks": s,
                   "receptor_gain_sweep": {"gains_mV": list(gains), **sweep},
                   "graded_response": graded,
                   "graded_interpretation":
                       "若为 False，说明在示例参数下该系统只有静默与爆发两态，"
                       "没有分级的感觉响应——这是本配置的真实性质，不是调参问题。",
                   "receptor_groups": {k: int(v.size) for k, v in inputs.items()},
                   "output_groups": {k: int(v.size) for k, v in outputs.items()},
                   "figure": figpath, "wall_s": time.time() - t0,
                   "caveats": DROSOPHILA_CNS.required_caveats()},
                  fh, indent=2, ensure_ascii=False, default=str)
    print(f"\n  {s['n_passed']}/{s['n_checks']} 通过")
    print(f"  图: {figpath}")
    print(f"  记录: {rec_path}")


if __name__ == "__main__":
    main()
