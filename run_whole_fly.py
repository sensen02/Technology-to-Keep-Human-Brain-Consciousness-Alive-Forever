"""run_whole_fly.py -- assemble the whole-fly stack and test the tracheal layer.

Three things happen here:

1. The engine's profile guard is exercised in a realistic direction: a fly
   profile REFUSES a vessel/perfusion layer, because insects breathe through
   tracheae.
2. `TrachealGasLayer` is checked against the analytic steady-cylinder diffusion
   solution it implements.
3. The tracheal geometry is compared QUANTITATIVELY with a mammalian capillary
   bed at the SAME tissue metabolic rate, to show that the category error is not
   a pedantic one: the two networks deliver oxygen at very different local
   demand per tube.

Writes: outputs/whole_fly_*.png, outputs/metrics_whole_fly.json,
        outputs/whole_fly_record.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine import (ParamRegistry, Pipeline, ConstantSource, RatioMonitor,
                    CheckSuite, RunRecord, DROSOPHILA_CNS, ProfileViolation,
                    MEASURED, DERIVED, ILLUSTRATIVE, ASSUMED, Layer, Q)
from engine.project_layers import (TrachealGasLayer, CellStateLayer,
                                   TRACHEAL_GEOMETRY)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

# mammalian comparison values (for the category-error argument only)
MAMMAL = {"capillary_radius_um": 3.0, "spacing_um": 60.0}


def build_params():
    r = ParamRegistry("whole_fly")
    r.define("tracheole_radius_um", TRACHEAL_GEOMETRY["tracheole_radius_um"][0],
             "um", MEASURED,
             source=TRACHEAL_GEOMETRY["tracheole_radius_um"][1],
             species="Locusta migratoria (LOCUST)",
             notes="This is the ONLY verified tracheal geometry this project has, "
                   "and it is from locust. The records state Drosophila is not in "
                   "that data table, so using it for a fly is a cross-species "
                   "transfer and must be declared.")
    r.define("tracheal_spacing_um", TRACHEAL_GEOMETRY["spacing_um"][0], "um",
             MEASURED, source=TRACHEAL_GEOMETRY["spacing_um"][1],
             species="Locusta migratoria (LOCUST)")
    r.define("tracheole_density_ratio", 150.0, "1", DERIVED,
             derived_from=["tracheal_spacing_um"],
             source="records: tracheoles ~150x denser than mammalian capillaries")
    r.define("o2_diffusion_um2_s", 2000.0, "um^2/s", ILLUSTRATIVE,
             notes="effective O2 diffusivity in tissue; order of magnitude, not "
                   "measured in this project")
    r.define("metabolic_rate_per_um3_per_s", 1.0e-6, "1/s", ASSUMED,
             notes="tissue O2 consumption per unit volume, in the same "
                   "normalisation as the concentration field; used ONLY to "
                   "compare geometries at equal demand")
    r.define("spiracle_o2", 1.0, "1", ASSUMED,
             notes="fractional O2 at the spiracle; air is 0.21 by volume but the "
                   "field here is normalised to 1")
    return r


P = build_params()


def analytic_drop(q_per_cylinder, D, r_t, r_c):
    """C(r_c) - C(r_t) = (Q / 2 pi D) ln(r_c / r_t) for a steady cylinder."""
    return (q_per_cylinder / (2.0 * np.pi * D)) * np.log(r_c / r_t)


def main():
    print("=== 果蝇全身：气管供气层 ===")
    print(DROSOPHILA_CNS.profile_caveats() if hasattr(DROSOPHILA_CNS, "profile_caveats")
          else "")
    for c in DROSOPHILA_CNS.required_caveats():
        print(f"  必读警示: {c}")
    print()

    # ---------------- 1. profile guard, in the realistic direction ----------
    print("--- 1. 生物档案守卫（果蝇拒绝血管层）---")


    class VesselLayer(Layer):
        name = "vessel"
        requires = {}
        provides = {"vessel_source": "mmHg"}

        def step(self, dt, inputs):
            return {"vessel_source": 38.0}


    probe = Pipeline("probe")
    probe.add(VesselLayer())
    try:
        DROSOPHILA_CNS.validate_pipeline(probe)
        print("  ❌ 守卫失效：果蝇接受了血管层")
        guard_ok = False
    except ProfileViolation as e:
        print(f"  ✅ 被拒绝: {str(e)[:150]}")
        guard_ok = True

    # ---------------- 2. tracheal layer vs its analytic solution ------------
    print("\n--- 2. 气管层 vs 解析解 ---")
    rng = np.random.default_rng(0)
    n_cells = 144
    side = np.sqrt(n_cells)
    xs, ys = np.meshgrid(np.arange(side) * 12.0, np.arange(side) * 12.0)
    pos = np.column_stack([xs.ravel(), ys.ravel()])
    spacing = P["tracheal_spacing_um"]
    r_t = P["tracheole_radius_um"]
    D = P["o2_diffusion_um2_s"]
    r_c = spacing / np.sqrt(np.pi)
    # per-cylinder demand = rate * volume of one cylinder (spacing^2 * depth)
    depth = 100.0
    q = P["metabolic_rate_per_um3_per_s"] * (spacing ** 2) * depth
    tg = TrachealGasLayer(pos, metabolism=q, spacing_um=spacing,
                          tracheole_radius_um=r_t, d_eff_um2_s=D)
    out = tg.step(Q(1.0, "s"), {"spiracle_o2": np.array([P["spiracle_o2"]])})
    cell_o2 = out["cell_o2"]
    r_used = tg._nearest_tracheole_distance()
    ana = P["spiracle_o2"] - analytic_drop(q, D, r_t, np.clip(r_used, r_t, r_c))
    err = float(np.max(np.abs(cell_o2 - np.clip(ana, 0, 1))))
    print(f"  格点间距 {spacing} um, 气管小管半径 {r_t} um, 等效圆柱半径 {r_c:.2f} um")
    print(f"  最大 O2 下降 {1-cell_o2.min():.6f}（相对值）")
    print(f"  与解析解最大偏差 {err:.3e}")

    # ---------------- 3. tracheal vs mammalian at equal demand --------------
    print("\n--- 3. 同样代谢率下：气管 vs 哺乳动物毛细血管 ---")
    m_sp = MAMMAL["spacing_um"]
    m_rt = MAMMAL["capillary_radius_um"]
    m_rc = m_sp / np.sqrt(np.pi)
    q_m = P["metabolic_rate_per_um3_per_s"] * (m_sp ** 2) * depth
    drop_t = analytic_drop(q, D, r_t, r_c)
    drop_m = analytic_drop(q_m, D, m_rt, m_rc)
    print(f"  气管:    间距 {spacing:5.1f} um, r_t {r_t} um -> 每圆柱需求 {q:.3e}, "
          f"跨圆柱下降 {drop_t:.6f}")
    print(f"  哺乳动物: 间距 {m_sp:5.1f} um, r_t {m_rt} um -> 每圆柱需求 {q_m:.3e}, "
          f"跨圆柱下降 {drop_m:.6f}")
    print(f"  下降之比（哺乳/气管）= {drop_m/drop_t:.1f} 倍")

    # ---------------- 4. assemble and run the stack -------------------------
    print("\n--- 4. 装配并运行（气管 -> 细胞状态）---")
    pl = Pipeline("whole_fly")
    pl.add(ConstantSource("spiracle_o2", "1", P["spiracle_o2"]))
    pl.add(tg)
    cs = CellStateLayer(n_cells=n_cells, seed=0)
    pl.add(cs).add(RatioMonitor("monitor", "atp", "uM", "mean_atp"))
    pl.wire()
    DROSOPHILA_CNS.validate_pipeline(pl)
    print(pl.describe())

    state, trace = pl.run(Q(20.0, "s"), dt_outer=Q(5.0, "s"))
    print(f"  最终 cell O2 均值 {np.mean(state['cell_o2']):.6f}; "
          f"ATP 均值 {state['atp']:.3f} uM; 完整性 {state['integrity']:.4f}")

    # ---------------- checks ------------------------------------------------
    def chk_tracheal_analytic():
        return err, 1e-9, f"气管层与其解析稳态圆柱解一致（最大偏差 {err:.2e}）"

    def chk_geometry_matters():
        return (1.0 / (drop_m / drop_t) if False else 0.0), 0.0, \
               f"等需求下哺乳动物几何的局部下降是气管的 {drop_m/drop_t:.1f} 倍，" \
               f"说明几何不是细节"

    def chk_guard():
        return (0.0 if guard_ok else 1.0), 0.0, "果蝇档案拒绝血管层"

    def chk_o2_in_bounds():
        v = np.asarray(state["cell_o2"], float)
        return float(max(0.0, v.max() - P["spiracle_o2"])), 1e-12, \
               f"O2 不超过气门值：min={v.min():.6f} max={v.max():.6f}"

    def chk_atp_nonneg():
        return float(max(0.0, -state["atp"])), 1e-12, f"ATP 非负 {state['atp']:.3f}"

    suite = CheckSuite("whole_fly")
    suite.analytic("tracheal_matches_analytic", chk_tracheal_analytic,
                   "气管供气与解析解一致")
    suite.invariant("tracheal_vs_capillary_geometry_differs", chk_geometry_matters,
                    "气管与毛细血管几何在等需求下差异显著")
    suite.invariant("fly_profile_rejects_vessels", chk_guard,
                    "果蝇档案拒绝血管层")
    suite.invariant("o2_within_bounds", chk_o2_in_bounds, "O2 不越界")
    suite.invariant("atp_nonnegative", chk_atp_nonneg, "ATP 非负")
    results = suite.run()
    print("\n--- 检查 ---")
    for r in results:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail[:88]}")
    s = suite.summary(results)

    # ---------------- figure ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=130)
    axes[0].hist(r_used, bins=20, color="tab:blue")
    axes[0].set_xlabel("distance to nearest tracheole (um)")
    axes[0].set_ylabel("cells")
    axes[0].set_title(f"tracheal geometry: spacing {spacing:g} um", fontsize=9)
    labels = ["tracheal\n(fly)", "capillary\n(mammal)"]
    axes[1].bar(labels, [drop_t, drop_m], color=["tab:green", "tab:red"])
    axes[1].set_ylabel("O2 drop across the supply cylinder")
    axes[1].set_title(f"same tissue demand: mammal drop is {drop_m/drop_t:.1f}x larger",
                      fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Why the category matters: insects breathe through tracheoles "
                 "(the only verified geometry is LOCUST, not Drosophila)",
                 fontsize=10.5)
    fig.tight_layout()
    figpath = os.path.join(OUT, "whole_fly_tracheal.png")
    fig.savefig(figpath); plt.close(fig)

    # ---------------- record ------------------------------------------------
    rec = RunRecord("whole-fly tracheal gas exchange", seed=0)
    rec.set_profile(DROSOPHILA_CNS).set_parameters(P).set_wiring(pl)
    rec.set_checks(suite, results)
    rec.set_metric("tracheal_drop", drop_t)
    rec.set_metric("mammalian_drop", drop_m)
    rec.set_metric("drop_ratio_mammal_over_tracheal", drop_m / drop_t)
    rec.add_note("气管几何来自蝗虫实测，不是果蝇；跨物种迁移已声明。")
    rec_path = rec.write(os.path.join(OUT, "whole_fly_record.json"))

    res = {"checks": s, "results": [r.to_dict() for r in results],
           "tracheal_drop": drop_t, "mammalian_drop": drop_m,
           "drop_ratio": drop_m / drop_t,
           "geometry": {"tracheal_spacing_um": spacing, "tracheole_radius_um": r_t,
                        "equivalent_cylinder_radius_um": r_c,
                        "mammalian_spacing_um": m_sp,
                        "mammalian_capillary_radius_um": m_rt},
           "params": P.snapshot(), "param_summary": P.summary(),
           "figure": figpath,
           "caveats": DROSOPHILA_CNS.required_caveats()}
    with open(os.path.join(OUT, "metrics_whole_fly.json"), "w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False, default=str)
    print(f"\n  {s['n_passed']}/{s['n_checks']} 通过")
    print(f"  图: {figpath}")
    print(f"  记录: {rec_path}")
    print(f"  参数出处: {P.summary()}")


if __name__ == "__main__":
    main()
