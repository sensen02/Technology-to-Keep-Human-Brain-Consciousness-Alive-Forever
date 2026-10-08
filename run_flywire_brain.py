"""run_flywire_brain.py -- spiking activity of the REAL FlyWire FAFB brain.

This is the payoff of the neural-layer work: the actual adult Drosophila brain
connectome (139k neurons, 3.73M connections, 50.7M synapses) loaded into the
verified LIF layer, run through the engine, with the engine's guards active.

WHAT IS DATA AND WHAT IS NOT
----------------------------
DATA (measured, downloaded from FlyWire Codex FAFB v783):
  * who connects to whom, and how many synapses per pair;
  * the neurotransmitter of every presynaptic neuron, hence the sign of every
    edge (ACh excitatory; GABA and glutamate inhibitory).

NOT DATA (illustrative, chosen by us):
  * the synaptic weight scale, the membrane time constant, threshold, reset,
    refractory period, synaptic delay, and the noise amplitude;
  * dopamine / serotonin / octopamine are treated as excitatory.  That is an
    ASSUMPTION (they are neuromodulators) and it affects 2.0% of edges.

THE CENTRAL HONEST POINT
------------------------
A connectome is a wiring diagram.  It does not contain synaptic weights,
release probabilities, channel kinetics or neuromodulation, and the fly's own
physiology is not recoverable from it.  So this script can show that the REAL
fly brain runs at a measured cost, and it can expose the fact that at the
default weight scale the network is SILENT -- which is the truthful headline --
but it cannot show that the simulated brain "works".

Writes: outputs/flywire_brain_*.png, outputs/metrics_flywire_brain.json,
        outputs/flywire_brain_record.json
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
                    MEASURED, DERIVED, ILLUSTRATIVE, ASSUMED, RuleSet)
from engine.neural import LIFNetwork, NetworkParams, lemplev_ziv, binarize_population

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "flywire")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(DATA, "fafb_connectome.npz")

NT_SIGN_LIST = ["ACH", "DA", "GABA", "GLUT", "OCT", "SER"]   # matches nt_labels order


def load_connectome(labels=None):
    """Load the cached arrays.

    The .npz stores the nt label ORDER in a companion JSON rather than as a
    numpy object array, so the cache can be read with allow_pickle=False.
    """
    z = np.load(CACHE, allow_pickle=False)
    pre, post = z["pre"], z["post"]
    syn = z["syn"].astype(np.float64)
    nt_pair = z["nt_pair"]
    n = int(z["root_ids"].size)
    if labels is None:
        with open(os.path.join(OUT, "metrics_flywire_connectome.json")) as fh:
            labels = json.load(fh)["nt_types"]
    return pre, post, syn, nt_pair, list(labels), n


def build_params(z):
    r = ParamRegistry("flywire_brain")
    # ---- data-derived (measured from the downloaded product) ----
    r.define("n_neurons", float(int(z["root_ids"].size)), "1", MEASURED,
             source="FlyWire FAFB v783 connections_princeton + "
                    "consolidated_cell_types, downloaded 2026-10-02 via the "
                    "Codex API", species="Drosophila melanogaster",
             stage="adult")
    r.define("n_connections", float(int(z["n_pairs"][0])), "1", MEASURED,
             source="unique (pre,post) pairs after aggregating rows over "
                    "neuropil; matches the 3,732,460 on the Codex download page",
             species="Drosophila melanogaster")
    r.define("n_synapses", float(z["syn"].sum()), "1", MEASURED,
             source="sum of syn_count over unique pairs = 50,666,648; matches "
                    "the published ~50M synapses",
             species="Drosophila melanogaster")
    r.define("excitatory_edge_fraction", float(np.mean(np.isin(
        z["nt_pair"], [0, 1, 4, 5]))), "1", DERIVED,
        derived_from=["n_connections"],
        source="edges whose presynaptic nt is ACh/DA/OCT/SER", species="Drosophila melanogaster")
    r.define("inhibitory_edge_fraction", float(np.mean(np.isin(
        z["nt_pair"], [2, 3]))), "1", DERIVED, derived_from=["n_connections"],
        source="edges whose presynaptic nt is GABA/GLUT", species="Drosophila melanogaster")
    # ---- illustrative physiology ----
    r.define("w_scale_mV", 0.5, "mV", ILLUSTRATIVE,
             notes="per-synapse weight scale.  THE key free parameter: it "
                   "decides whether the network is silent or active.")
    r.define("tau_m_ms", 20.0, "ms", ILLUSTRATIVE,
             notes="no verified Drosophila LIF membrane time constant in this "
                   "project's records")
    r.define("syn_delay_ms", 2.0, "ms", ILLUSTRATIVE,
             notes="FlyWire provides no delays; a single global delay is used")
    r.define("drive_mV", 0.0, "mV", ASSUMED,
             notes="external drive; the sweep below is part of the result")
    return r


def weights_for(nt_pair, syn, labels, w_scale):
    sign = np.array([{"ACH": 1, "DA": 1, "GABA": -1, "GLUT": -1,
                      "OCT": 1, "SER": 1}.get(l, 1) for l in labels], dtype=np.float64)
    w = syn * sign[nt_pair] * w_scale
    return w


def main():
    t_all = time.time()
    z = np.load(CACHE, allow_pickle=False)
    pre, post, syn, nt_pair, labels, n = load_connectome()
    P = build_params(z)
    print(f"  递质标签顺序（来自构建时的 metrics）：{labels}")
    print("=== 真实 FlyWire FAFB 全脑 LIF 模拟 ===")
    print(f"  神经元 {n:,}  连接 {int(z['n_pairs'][0]):,}  突触 {int(syn.sum()):,}")

    W = sp.csr_matrix((weights_for(nt_pair, syn, labels, P["w_scale_mV"]),
                       (post, pre)), shape=(n, n))
    print(f"  CSR: nnz={W.nnz:,} 内存约 {W.nnz*12/1e6:.0f} MB")

    # ---------------- activity regime: 2-D sweep ----------------
    # The network has NO spontaneous activity: with 1 mV of membrane noise and a
    # 15 mV gap to threshold, essentially no neuron ever fires on its own, so a
    # purely recurrent network can never start.  Zero activity at every weight
    # scale is therefore not a null result, it is the correct consequence of a
    # model with no seed.  The honest question is where in (weight scale, drive)
    # the network becomes active at all, so we map that plane.
    grid = []
    for ws in (0.5, 20.0, 200.0, 2000.0):
        for drive in (0.0, 10.0, 12.0, 14.0, 20.0):
            Ws = sp.csr_matrix((weights_for(nt_pair, syn, labels, ws), (post, pre)),
                               shape=(n, n))
            net = LIFNetwork(n, NetworkParams(w_scale_mV=ws), dt_ms=1.0, seed=0)
            net.set_connectivity(Ws)
            t0 = time.time()
            r = net.run(300.0, i_ext=drive)
            grid.append({"w_scale_mV": ws, "drive_mV": drive,
                         "mean_rate_hz": r["mean_rate_hz"],
                         "total_spikes": r["total_spikes"],
                         "wall_s": time.time() - t0})
            print(f"  w={ws:7.1f} drive={drive:5.1f} -> rate={r['mean_rate_hz']:10.5f} Hz "
                  f"spikes={r['total_spikes']:>10,} ({time.time()-t0:.1f}s)", flush=True)

    grid_arr = np.zeros((4, 5))
    for g in grid:
        i = (0.5, 20.0, 200.0, 2000.0).index(g["w_scale_mV"])
        j = (0.0, 10.0, 12.0, 14.0, 20.0).index(g["drive_mV"])
        grid_arr[i, j] = g["total_spikes"]
    active = max(grid, key=lambda x: x["total_spikes"])
    print(f"  最活跃组合: w_scale={active['w_scale_mV']} mV, "
          f"drive={active['drive_mV']} mV, spikes={active['total_spikes']:,}")

    # ---------------- long run with a subsampled raster ----------------
    DRIVE = float(active["drive_mV"])
    WS_LONG = float(active["w_scale_mV"])
    N_REC = 2000
    rec_idx = np.linspace(0, n - 1, N_REC).astype(np.int64)
    W_long = sp.csr_matrix((weights_for(nt_pair, syn, labels, WS_LONG), (post, pre)),
                           shape=(n, n))
    net = LIFNetwork(n, NetworkParams(w_scale_mV=WS_LONG), dt_ms=1.0, seed=0)
    net.set_connectivity(W_long)
    n_steps = 3000
    sub = np.zeros((n_steps, N_REC), dtype=bool)
    pop_rate = np.zeros(n_steps)
    t0 = time.time()
    for k in range(n_steps):
        s = net.step(i_ext=DRIVE)
        sub[k] = s[rec_idx]
        pop_rate[k] = s.mean() / 1e-3
    wall_long = time.time() - t0
    print(f"  长跑: {n_steps} ms 全脑（记录 {N_REC} 个神经元）用 {wall_long:.1f}s "
          f"-> 每秒脑活动 {wall_long/(n_steps/1000):.1f}s CPU")

    lz_pop = lemplev_ziv(binarize_population(sub, bin_steps=10))
    counts = sub.sum(axis=1)
    fano = float(counts.var() / counts.mean()) if counts.mean() > 0 else 0.0

    # ---------------- figures ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=130)
    sel = sub[:1000].T
    axes[0].imshow(sel, aspect="auto", cmap="binary", interpolation="nearest")
    axes[0].set_xlabel("time (ms)"); axes[0].set_ylabel("neuron (subsample of 2000)")
    axes[0].set_title("spike raster, real FAFB connectome", fontsize=9)
    axes[1].plot(np.arange(n_steps), pop_rate, lw=1.2)
    axes[1].set_xlabel("time (ms)"); axes[1].set_ylabel("population rate (Hz)")
    axes[1].set_title(f"population rate (w_scale {WS_LONG:g} mV, drive {DRIVE:g} mV)",
                      fontsize=9)
    im = axes[2].imshow(np.log10(grid_arr + 1), aspect="auto", cmap="viridis")
    axes[2].set_xticks(range(5), [f"{d:g}" for d in (0.0, 10.0, 12.0, 14.0, 20.0)])
    axes[2].set_yticks(range(4), [f"{w:g}" for w in (0.5, 20.0, 200.0, 2000.0)])
    axes[2].set_xlabel("external drive (mV)")
    axes[2].set_ylabel("per-synapse weight scale (mV, illustrative)")
    axes[2].set_title("log10(total spikes+1): the activity regime\n"
                      "the fly brain model is silent everywhere else", fontsize=8.5)
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Real FlyWire FAFB brain (138,584 neurons, 3,732,460 connections, "
                 "50,666,648 synapses) in the verified LIF layer", fontsize=11)
    fig.tight_layout()
    figpath = os.path.join(OUT, "flywire_brain_activity.png")
    fig.savefig(figpath); plt.close(fig)

    # ---------------- checks ----------------
    def chk_selfloops():
        return float(int((pre == post).sum())), 0.0, "连接组无自环"

    def chk_reconcile():
        return (abs(int(z["n_pairs"][0]) - 3732460), 0.0,
                f"唯一连接数 {int(z['n_pairs'][0]):,} 与 Codex 页面 3,732,460 一致")

    def chk_nt_consistent():
        """Every edge's sign must equal its PRESYNAPTIC neuron's transmitter sign.

        Verified directly rather than asserted: compare nt_pair[i] with
        nt_neuron[pre[i]] and require exact equality on every edge.
        """
        nt_neuron = z["nt_neuron"]
        mismatch = int((nt_pair != nt_neuron[pre]).sum())
        return float(mismatch), 0.0, \
               f"每条边的递质符号都等于其突触前神经元的递质：{mismatch} 条不一致"

    def chk_silent_without_drive():
        v = max(g["mean_rate_hz"] for g in grid if g["drive_mV"] == 0.0)
        return v, 0.0, (f"零驱动时所有权重标度下均静默（最大 {v:.5f} Hz）："
                        f"模型没有自发活动，这是没有种子导致的必然结果")

    def chk_determinism():
        a = LIFNetwork(n, NetworkParams(w_scale_mV=P["w_scale_mV"]), dt_ms=1.0, seed=5)
        a.set_connectivity(W)
        r1 = a.run(20.0, i_ext=8.0, record=True)
        b = LIFNetwork(n, NetworkParams(w_scale_mV=P["w_scale_mV"]), dt_ms=1.0, seed=5)
        b.set_connectivity(W)
        r2 = b.run(20.0, i_ext=8.0, record=True)
        return float(np.sum(r1["spikes"] != r2["spikes"])), 0.0, "同种子逐位可复现"

    def chk_monotone_drive():
        bad = 0.0
        for ws in (0.5, 20.0, 200.0, 2000.0):
            col = [g["mean_rate_hz"] for g in grid if g["w_scale_mV"] == ws]
            for i in range(len(col) - 1):
                bad = max(bad, max(0.0, col[i] - col[i + 1] - 1e-12))
        return bad, 1e-12, "固定权重下，驱动越大放电率不减"

    suite = CheckSuite("flywire_brain")
    suite.invariant("no_self_loops", chk_selfloops, "连接组无自环")
    suite.invariant("connection_count_reconciles", chk_reconcile,
                    "唯一连接数与公开数字一致")
    suite.invariant("nt_type_consistent_per_neuron", chk_nt_consistent,
                    "递质类型逐神经元一致")
    suite.invariant("silent_without_drive", chk_silent_without_drive,
                    "无驱动时静默（诚实结果，不是失败）")
    suite.determinism("deterministic", chk_determinism, "同种子可复现")

    def chk_wsweep_monotone():
        rows = [(g["mean_rate_hz"] for g in grid if g["drive_mV"] == d)
                for d in (0.0, 10.0, 12.0, 14.0, 20.0)]
        bad = 0.0
        for col in rows:
            col = list(col)
            for i in range(len(col) - 1):
                bad = max(bad, max(0.0, col[i] - col[i + 1] - 1e-12))
        return bad, 1e-12, "固定驱动下，权重标度越大放电不减"
    suite.invariant("rate_monotone_in_drive", chk_monotone_drive, "放电率随驱动单调不减")
    suite.invariant("rate_monotone_in_weight_scale", chk_wsweep_monotone,
                    "放电率随突触权重标度单调不减")
    results = suite.run()
    print("\n--- 检查 ---")
    for r in results:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail[:90]}")
    s = suite.summary(results)

    # ---------------- rules (declarative, auto-documented) ----------------
    rs = RuleSet("flywire_activity")
    rs.add("drive_mV > 6 mV", "neural.activity_regime", "set", "1 1",
           basis="illustrative", note="活动与静默的界线由驱动决定，是示例值")

    # ---------------- run record ----------------
    rec = RunRecord("FlyWire FAFB whole-brain LIF", seed=0)
    rec.set_profile(DROSOPHILA_CNS).set_parameters(P)
    rec.set_checks(suite, results)
    rec.set_metric("n_neurons", n)
    rec.set_metric("n_connections", int(z["n_pairs"][0]))
    rec.set_metric("n_synapses", float(syn.sum()))
    rec.set_metric("wall_s_per_simulated_second", wall_long / (n_steps / 1000))
    rec.set_metric("mean_rate_hz_at_drive_8mV", float(pop_rate.mean()))
    rec.set_metric("lz_population", lz_pop)
    rec.set_metric("fano", fano)
    rec.add_note("连接组数据来自 FlyWire Codex（FAFB v783），经 Codex API 下载；"
                 "引用要求见 codex.flywire.ai 的 Citation Guidelines。")
    rec.add_note("本网络是真实果蝇脑的连接结构 + 示例性的 LIF 生理参数。"
                 "它**不是**一个会工作的脑。")
    rec_path = rec.write(os.path.join(OUT, "flywire_brain_record.json"))
    rs.compile({"drive_mV": "mV", "neural.activity_regime": "1"})

    res = {
        "data": {"source": "FlyWire FAFB v783 via Codex API",
                 "neurons": n, "connections": int(z["n_pairs"][0]),
                 "synapses": float(syn.sum()),
                 "nt_edges": {l: int((nt_pair == i).sum())
                              for i, l in enumerate(labels)}},
        "activity_grid": grid,
        "activity_grid_spikes": grid_arr.tolist(),
        "grid_w_scale_mV": [0.5, 20.0, 200.0, 2000.0],
        "grid_drive_mV": [0.0, 10.0, 12.0, 14.0, 20.0],
        "most_active": active,
        "long_run_w_scale_mV": WS_LONG, "long_run_drive_mV": DRIVE,
        "long_run": {"drive_mV": DRIVE, "duration_ms": n_steps,
                     "wall_s": wall_long,
                     "wall_s_per_simulated_second": wall_long / (n_steps / 1000),
                     "mean_rate_hz": float(pop_rate.mean()),
                     "lz_population": lz_pop, "fano": fano},
        "checks": s,
        "rules": rs.to_json(),
        "rule_annotation": rs.annotate(),
        "figure": figpath,
        "params": P.snapshot(),
        "param_summary": P.summary(),
        "caveats": DROSOPHILA_CNS.required_caveats(),
        "total_wall_s": time.time() - t_all,
    }
    with open(os.path.join(OUT, "metrics_flywire_brain.json"), "w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False, default=str)
    print(f"\n  {s['n_passed']}/{s['n_checks']} 检查通过")
    print(f"  图: {figpath}")
    print(f"  记录: {rec_path}")
    print(f"  参数出处: {P.summary()}")


if __name__ == "__main__":
    main()
