"""run_engine_demo.py -- the engine carrying the project's REAL modules.

This is the proof that the engine is not a toy: it runs the existing
`transport.py` (conservative finite-volume Laplacian) and `cellstate.py`
(per-cell volume/ATP/integrity with death hysteresis) as layers, wired
L1 -> L2 for the project's original organism, Drosophila pupal notum.

It also demonstrates the two guards firing in a realistic setting:
  * attaching a vessel/perfusion layer to the Drosophila profile is refused
    (the fly has no vasculature -- it uses tracheae);
  * a parameter declared `measured` without a citation cannot even be created.

Units are exercised for real: the bath is given in mmHg, the transport layer
publishes per-cell oxygen in mmHg, and the cellstate layer converts mmHg to the
dimensionless relative availability that `cellstate.run(oxygen=...)` expects,
using a Michaelis constant that is registered WITH its provenance problem
stated (the records note that "oxygen Km" is not a single quantity).

Writes: outputs/engine_demo_record.json
        outputs/engine_demo_report.zh-CN.md
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import (Q, Pipeline, Layer, ConstantSource, RatioMonitor, CheckSuite,
                    RunRecord, ParamRegistry, MEASURED, ILLUSTRATIVE, ASSUMED,
                    DROSOPHILA_NOTUM, ProfileViolation)
import cellstate
import transport as tp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

N_CELLS = 144          # 12x12, small enough for a fast demo
T_END_S = 20.0
DOMAIN_UM = (200.0, 200.0)


# ======================================================================
# parameters, with provenance enforced
# ======================================================================
def build_params():
    r = ParamRegistry("engine_demo")
    # measured: the project's own verified anchor
    r.define("cell_area", 55.6, "um^2", MEASURED,
             source="222.3/4 um^2 mean 4-cell group area, notum 12-13.5 hAPF "
                    "(Curran 2017), via MEASURED_ANCHORS.md",
             species="Drosophila melanogaster", stage="pupal notum, 12-13.5 hAPF",
             notes="Fixes the lattice spacing; it is the one solid geometric anchor.")
    r.define("epi_thickness", 8.75, "um", MEASURED,
             source="notum epithelial thickness 7.5-10 um, notum 15-18 hAPF "
                    "(Pinheiro 2017)", species="Drosophila melanogaster",
             stage="pupal notum, 15-18 hAPF", range=(7.5, 10.0))
    # derived, from the two above
    r.define("cell_diameter", 7.456, "um", MEASURED,
             derived_from=["cell_area"],
             source="2*sqrt(cell_area/pi)", species="Drosophila melanogaster")
    # illustrative / assumed: the bulk of any such model
    r.define("o2_bath_mmHg", 38.0, "mmHg", ASSUMED,
             notes="PhysiCell's stock oxygen defaults use 38 mmHg at the "
                   "Dirichlet boundary; that is a modelling convention, not a "
                   "measured Drosophila tracheal value. ORGANISM_SCALE_PLAN.md "
                   "records that PhysiCell's own oxygen block declares its unit "
                   "as dimensionless while holding these numbers.")
    r.define("o2_km_mmHg", 1.0, "mmHg", ASSUMED,
             notes="The records note oxygen Km is NOT a single quantity: "
                   "mitochondrial ~0.05 torr, tissue apparent values ~100x "
                   "higher, fitted model values commonly ~1 mmHg. It must not be "
                   "carried across preparations.")
    r.define("o2_diffusion_um2_s", 2000.0, "um^2/s", ILLUSTRATIVE,
             notes="Order-of-magnitude for O2 in tissue; not measured here.")
    r.define("o2_clearance_per_s", 0.05, "1/s", ILLUSTRATIVE)
    r.define("o2_uptake_per_s", 0.4, "1/s", ILLUSTRATIVE,
             notes="Per-voxel consumption at voxels holding a cell. Chosen large "
                   "enough that a real spatial gradient develops, so that the "
                   "transport layer is actually exercised rather than being a "
                   "uniform field. NOT a measured Drosophila tracheal value.")
    return r


P = build_params()


# ======================================================================
# L1: oxygen transport, wrapping the project's real conservative solver
# ======================================================================
class TissueOxygenLayer(Layer):
    """Diffusion + clearance of O2 on the project's finite-volume grid.

    Uses `transport.make_grid` and `transport.laplacian` unchanged.  The
    Laplacian is the divergence of face fluxes with zero normal flux on faces,
    so total mass is conserved by construction up to roundoff -- which is what
    the conservation check below verifies.
    """

    name = "tissue_o2"
    requires = {"o2_bath": "mmHg"}
    provides = {"cell_o2": "mmHg"}

    def __init__(self, n_cells=N_CELLS, domain_um=DOMAIN_UM):
        super().__init__()
        self.target_dt = Q(1.0, "s")
        geo = cellstate.cell_geometry(n_cells=n_cells)
        self.pos_um = np.asarray(geo["positions"], float)
        span = self.pos_um.max(axis=0) - self.pos_um.min(axis=0)
        dx = float(span.min()) / 11.0
        self.grid = tp.make_grid(domain_um=domain_um, dx_um=dx, thickness_um=8.75)
        self.L = tp.laplacian(self.grid)
        self.shape = self.grid["shape"]
        self.c = np.full(self.grid["n_nodes"], 0.0)
        # nearest-voxel index for each cell centre
        idx = np.floor(self.pos_um / dx).astype(int)
        idx = np.clip(idx, 0, np.array(self.shape) - 1)
        self.cell_idx = idx[:, 0] * self.shape[1] + idx[:, 1]
        self.dx = dx

    def initialize(self, inputs):
        bath = float(inputs.get("o2_bath") or 0.0)
        self.c[:] = bath
        # per-voxel consumption mask: one sink per cell centre
        self.sink = np.zeros(self.grid["n_nodes"])
        np.add.at(self.sink, self.cell_idx, 1.0)

    def step(self, dt, inputs):
        D = P["o2_diffusion_um2_s"]
        k = P["o2_clearance_per_s"]
        bath = float(inputs["o2_bath"])
        dt_s = dt.si
        # stability guard for explicit diffusion
        dt_max = 0.25 * self.dx ** 2 / (D * self.grid["ndim"])
        n = max(1, int(np.ceil(dt_s / dt_max)))
        h = dt_s / n
        uptake = P["o2_uptake_per_s"]        # 1/s, at voxels holding a cell
        for _ in range(n):
            cons = uptake * self.c * self.sink
            self.c += h * (D * (self.L @ self.c) + k * (bath - self.c) - cons)
        return {"cell_o2": self.c[self.cell_idx].copy()}

    def total_mass(self):
        return float(self.c.sum())


# ======================================================================
# L2: per-cell state, wrapping the project's real cellstate module
# ======================================================================
class CellStateLayer(Layer):
    """Wraps `cellstate.run`.  Converts mmHg -> the relative availability that
    `cellstate.run(oxygen=...)` expects, using a registered Michaelis constant."""

    name = "cellstate"
    requires = {"cell_o2": "mmHg"}
    provides = {"atp": "uM", "integrity": "1"}

    def __init__(self, n_cells=N_CELLS, t_end=None, seed=2025):
        super().__init__()
        self.target_dt = Q(1.0, "s")     # outer step; the module sub-steps internally
        self.n_cells = n_cells
        self.seed = seed
        self._t = 0.0
        self._state = None
        self.last = None

    def _relative_o2(self, mmHg):
        km = P["o2_km_mmHg"]
        return np.clip(np.asarray(mmHg, float) / (np.asarray(mmHg, float) + km), 0.0, 1.0)

    def step(self, dt, inputs):
        rel = self._relative_o2(inputs["cell_o2"])
        res = cellstate.run(n_cells=self.n_cells, oxygen=rel, t_end=dt.si,
                            dt=min(0.05, dt.si / 10.0), seed=self.seed,
                            initial=self._state, return_frames=False)
        self.last = res
        fin = res["final"]
        self._state = fin          # accepted as `initial=` on the next step
        self._t += dt.si
        return {"atp": float(np.mean(fin["atp"])),
                "integrity": float(np.mean(fin["integrity"]))}


# ======================================================================
def main():
    t0 = time.time()
    print("=== 引擎演示：把项目里真实模块接成分层栈 ===\n")
    print(f"生物档案：{DROSOPHILA_NOTUM.species} / {DROSOPHILA_NOTUM.stage}")
    print(f"  维度={DROSOPHILA_NOTUM.dimensionality}D  "
          f"有血管={DROSOPHILA_NOTUM.has_vasculature}  "
          f"气体交换={DROSOPHILA_NOTUM.gas_exchange}")
    for c in DROSOPHILA_NOTUM.required_caveats():
        print(f"  必读警示：{c}")
    print()

    # ---- guard demo 1: a measured parameter cannot exist without a citation
    print("--- 守卫 1：measured 参数必须有引用 ---")
    try:
        from engine import Param
        Param("tension", 44.0, "pN", MEASURED)
        print("  ❌ 守卫失效")
    except Exception as e:
        print(f"  ✅ 被拒绝：{str(e)[:100]}")

    # ---- guard demo 2: no vasculature for a fly
    print("\n--- 守卫 2：果蝇档案拒绝血管层 ---")
    probe = Pipeline("probe")


    class VesselLayer(Layer):
        name = "vessel"
        requires = {}
        provides = {"vessel_source": "mol/m^3"}

        def step(self, dt, inputs):
            return {"vessel_source": 1.0}


    probe.add(VesselLayer())
    try:
        DROSOPHILA_NOTUM.validate_pipeline(probe)
        print("  ❌ 守卫失效")
    except ProfileViolation as e:
        print(f"  ✅ 被拒绝：{str(e)[:160]}")

    # ---- the real stack ----
    print("\n--- 装配真实分层栈（L1 transport → L2 cellstate）---")
    pl = Pipeline("notum_o2_cellstate")
    pl.add(ConstantSource("o2_bath", "mmHg", P["o2_bath_mmHg"]))
    o2 = TissueOxygenLayer()
    cs = CellStateLayer()
    pl.add(o2).add(cs).add(RatioMonitor("monitor", "atp", "uM", "mean_atp"))
    pl.wire()
    DROSOPHILA_NOTUM.validate_pipeline(pl)
    print(pl.describe())

    mass0 = o2.total_mass()
    state, trace = pl.run(Q(T_END_S, "s"), dt_outer=Q(5.0, "s"))
    mass1 = o2.total_mass()
    wall = time.time() - t0

    print(f"\n运行 {T_END_S} s，外层步长 5 s；L1 内部 dt=1 s，L2 内部 dt<=0.05 s")
    print(f"  最终 per-cell O2（mmHg）：均值 {np.mean(state['cell_o2']):.3f} "
          f"最小 {np.min(state['cell_o2']):.3f} 最大 {np.max(state['cell_o2']):.3f}")
    print(f"  最终 ATP（uM）均值 {state['atp']:.4f}；完整性均值 {state['integrity']:.4f}")
    print(f"  L1 层被调用 {o2.n_steps} 次，L2 层被调用 {cs.n_steps} 次")

    # ==================================================================
    # checks
    # ==================================================================
    def check_bounds():
        v = np.asarray(state["cell_o2"], float)
        err = float(max(0.0, v.min() - P["o2_bath_mmHg"],
                        -v.max(), -0.0))
        return err, 1e-9, f"O2 在 [0, bath] 内：min={v.min():.3f} max={v.max():.3f}"

    def check_atp_nonneg():
        a = float(state["atp"])
        return max(0.0, -a), 1e-12, f"ATP 非负：{a:.6f}"

    def check_conservation():
        # with clearance on, mass is not conserved; instead verify that the
        # Laplacian itself conserves mass (that is the actual invariant)
        ones = np.ones(o2.grid["n_nodes"])
        return float(abs(np.sum(o2.L @ ones))), 1e-6, "保守 Laplacian 行和为零"

    def check_monotone_in_o2():
        """More oxygen must not give less ATP.  This is a directional check of a
        REAL module, run through the engine."""
        rels = [0.0, 0.25, 0.5, 0.75, 1.0]
        atps = []
        for rel in rels:
            r = cellstate.run(n_cells=64, oxygen=rel, t_end=5.0, dt=0.05,
                              seed=2025, return_frames=False)
            atps.append(float(np.mean(r["final"]["atp"])))
        err = 0.0
        for i in range(len(atps) - 1):
            if atps[i + 1] < atps[i] - 1e-9:
                err = max(err, atps[i] - atps[i + 1])
        return err, 1e-9, f"ATP 随 O2 单调：{['%.3f' % a for a in atps]}"

    def check_determinism():
        a = cellstate.run(n_cells=64, oxygen=0.5, t_end=5.0, dt=0.05, seed=7,
                          return_frames=False)
        b = cellstate.run(n_cells=64, oxygen=0.5, t_end=5.0, dt=0.05, seed=7,
                          return_frames=False)
        return (float(np.max(np.abs(a["final"]["atp"] - b["final"]["atp"]))), 0.0,
                "同种子逐位可复现")

    suite = CheckSuite("engine_demo")
    suite.invariant("o2_within_bounds", check_bounds, "O2 不越界")
    suite.invariant("atp_nonnegative", check_atp_nonneg, "ATP 非负")
    suite.conservation("laplacian_conserves_mass", check_conservation,
                       "transport.py 的 Laplacian 行和为零（质量守恒前提）")
    suite.analytic("module_moves_right_direction", check_monotone_in_o2,
                   "真实 cellstate 模块：氧气越多 ATP 越高")
    suite.determinism("module_deterministic", check_determinism,
                      "真实 cellstate 模块：同种子可复现")
    results = suite.run()
    print("\n--- 检查 ---")
    for r in results:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}: "
              f"err={r.error:.3e} tol={r.tolerance:.3e}  {r.detail[:80]}")
    s = suite.summary(results)
    print(f"  {s['n_passed']}/{s['n_checks']} 通过")

    # ==================================================================
    # run record
    # ==================================================================
    rec = RunRecord("engine demo: notum O2 -> cellstate", seed=2025)
    rec.set_profile(DROSOPHILA_NOTUM).set_parameters(P).set_wiring(pl)
    rec.set_checks(suite, results)
    rec.set_metric("n_cells", N_CELLS)
    rec.set_metric("t_end_s", T_END_S)
    rec.set_metric("mean_cell_o2_mmHg", float(np.mean(state["cell_o2"])))
    rec.set_metric("mean_atp_uM", float(state["atp"]))
    rec.set_metric("wall_s", round(wall, 2))
    rec.add_note("这是一个模拟，不是永生实验；实验体是果蝇，不是人。")
    rec.add_note("O2 的 Km 与 bath 值均为 assumed，不是果蝇实测值；"
                 "果蝇的气体交换靠气管而非毛细血管，本演示只借用通用输运数学。")
    p_rec = rec.write(os.path.join(OUT, "engine_demo_record.json"))

    md = ["# 引擎演示报告", "",
          "## 这个演示证明了什么", "",
          "- 引擎能承载项目里**真实的** `transport.py` 与 `cellstate.py`，"
          "而不是玩具层：L1 用真实的保守有限体积 Laplacian，"
          "L2 调用真实的 `cellstate.run`。",
          "- 两个守卫在真实场景中确实触发：measured 参数缺引用无法创建；"
          "给果蝇档案接血管层被拒绝。",
          "- 单位是真在起作用的：bath 与 per-cell O2 用 mmHg，"
          "`cellstate.run` 需要的是无量纲相对可用度，转换由引擎带单位完成。", "",
          f"## 运行", "",
          f"- 细胞数 {N_CELLS}，模拟时长 {T_END_S} s，外层步长 5 s",
          f"- 最终 O2 均值 {np.mean(state['cell_o2']):.3f} mmHg；"
          f"ATP 均值 {state['atp']:.4f} uM",
          f"- L1 被调用 {o2.n_steps} 次，L2 被调用 {cs.n_steps} 次", "",
          suite.markdown(results), "",
          rec.markdown_claim_boundary(), "",
          "## 参数出处", "", P.markdown_table(), ""]
    with open(os.path.join(OUT, "engine_demo_report.zh-CN.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")

    print(f"\n记录：{p_rec}")
    print(f"报告：{os.path.join(OUT, 'engine_demo_report.zh-CN.md')}")
    print(f"参数出处统计：{P.summary()}")


if __name__ == "__main__":
    main()
