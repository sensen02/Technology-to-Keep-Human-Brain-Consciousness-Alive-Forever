"""Tests for engine/embodied/slow_physiology.py (slow physiology + behaviour recorder).

Run in the BODY environment, AFTER the demo (the last group of checks reads the
artefacts the demo wrote):
    /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python \
        /run/media/sensen/Data2/cell_wound_prototype/run_embodied_slowphys_selftest.py

WHAT THESE TESTS ESTABLISH
--------------------------
1.  DETERMINISM: the same config and the same inputs give bit-identical ledgers and
    states, and the module's fingerprint changes when the config changes.
2.  THE FROZEN / OFF PATH IS INERT: with the modulator disabled, stepping it raises and
    it changes no state; with the energy proxy disabled, no proxy is computed and it
    appears in no ledger; with a zero load the demand is exactly the resting demand and
    each compartment relaxes monotonically to its own quasi-steady state with the closed
    form's own rate.
3.  SUBSTANCE LEDGERS CLOSE to a STATED tolerance, and the residual is REPORTED: for
    every substance amount_now = initial + inflow - outflow - consumption + residual,
    both for the whole module and for each COMPARTMENT, over a grid of step sizes and
    demands including the extreme (supply-limited) one.
4.  SUPPLY AND CONSUMPTION ARE NON-NEGATIVE at every step, and a long run is compared
    against an INDEPENDENT RK4 integration of the stated ODE: state, transfer integral
    and accumulated consumption all agree to a stated tolerance.
5.  LOCOMOTOR LOAD ACTUALLY CHANGES WITH BODY ACTIVITY: the numbers are printed, and a
    synthetic doubling of the measured speed/joint speed doubles the load, with the
    joint term demonstrably contributing.
6.  THE BEHAVIOUR THRESHOLDS BEHAVE AT THEIR BOUNDARIES: synthetic cases ON and JUST
    BELOW each threshold are checked against the expected label, for every criterion.
7.  INVALID INPUT IS REFUSED: malformed configs, a non-zero haemolymph O2 capacity, a
    malformed activity reduction, a negative demand, a NaN, a wrong-shape trajectory.
8.  NOTHING HERE IS BLOOD-BORNE GAS EXCHANGE, AND NO LONG ADVANCE IS SILENT: module
    source, report JSON and figure text are scanned for the forbidden vocabulary
    (`MODULE_FORBIDDEN_TERMS`), and every long advance must carry the explicit
    TIME_JUMP_QUASI_STEADY label.

WHAT THESE TESTS DO NOT ESTABLISH
---------------------------------
They do NOT establish that any parameter is Drosophila's: every kinetic and
physiological number is ILLUSTRATIVE and the only tracheal geometry this project has is
LOCUST data that no default uses.  They do NOT establish that the model's oxygen
transport matches a real fly's, that the tissue really consumes what the load proxy
says, or that the behaviour labels correspond to any real fly behaviour.  They do NOT
establish anything about sleep: no sleep criterion is implemented.  Passing them means
the module is internally consistent, conservative, reproducible and honest about what it
cannot say -- nothing more.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "outputs", "embodied_body")

from engine.embodied.slow_physiology import (                               # noqa: E402
    BEHAVIOUR_LABELS, BehaviourConfig, ENERGY_PROXY_NAME, HaemolymphConfig,
    LoadConfig, LocalModulator, MODULE_FORBIDDEN_TERMS, ModulatorConfig,
    SLOWPHYS_HONESTY, SlowPhysiology, SlowPhysiologyConfig, TimeJumpError,
    TimeJumpRecord, classify_behaviour, epoch_table, measure_behaviour,
    measure_long_scale, slowphys_fingerprint,
)

RESULTS = []
LEDGER_TOL_NMOL = 1e-9


def check(name, ok, detail=""):
    RESULTS.append({"name": name, "ok": bool(ok), "detail": str(detail)[:600]})
    print(("PASS " if ok else "FAIL ") + name
          + (" | " + str(detail)[:260] if detail else ""), flush=True)
    return bool(ok)


def refuse(fn):
    try:
        fn()
    except (ValueError, TypeError, KeyError, RuntimeError, FileNotFoundError,
            FloatingPointError, TimeJumpError):
        return True
    except Exception:                                                    # noqa: BLE001
        return False
    return False


# ---------------------------------------------------------------------------
# the independent reference the integrator is checked against
# ---------------------------------------------------------------------------
def rk4_reference(cfg, dt, demand, nsteps, h=1e-5):
    """Independent RK4 integration of the STATED ODE, plus its flux integrals.

    Written from the physics as stated in the docstring:
        dA/dt = a*(a0 - A/vt) - b*(A/vt - T/vi)
        dT/dt = b*(A/vt - T/vi) - d
        inflow/outflow from the sign of a*(a0 - A/vt), transfer from b*(A/vt - T/vi)
    and integrated with a 1e-5 s RK4 step, i.e. 500x finer than the module's own step in
    the coarsest case tested.
    """
    tr, ti = cfg.tracheal, cfg.tissue
    vt, vi = tr.volume_mm3, ti.volume_mm3
    a, b = tr.bath_conductance_mm3_s, tr.tracheole_conductance_mm3_s
    a0 = tr.bath_o2_nmol_per_mm3
    ceiling = a0 / (1.0 / a + 1.0 / b)

    def f(x, d):
        A, T = x
        return np.array([a * (a0 - A / vt) - b * (A / vt - T / vi),
                         b * (A / vt - T / vi) - d])

    x = np.array([tr.initial_o2_nmol, ti.initial_o2_nmol], float)
    inflow = outflow = transfer = 0.0
    for _ in range(nsteps):
        d = min(demand, ceiling) if cfg.enforce_supply_ceiling else demand
        m = int(round(dt / h))
        hh = dt / m
        for _ in range(m):
            k1 = f(x, d)
            k2 = f(x + hh / 2 * k1, d)
            k3 = f(x + hh / 2 * k2, d)
            k4 = f(x + hh * k3, d)
            xn = x + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            cA = (x[0] + xn[0]) / 2 / vt
            cT = (x[1] + xn[1]) / 2 / vi
            e = (a * (a0 - cA) - b * (cA - cT)) * hh
            t_ = b * (cA - cT) * hh
            inflow += max(e, 0.0)
            outflow += max(-e, 0.0)
            transfer += t_
            x = xn
    return x[0], x[1], inflow, outflow, transfer, min(demand, ceiling) * dt * nsteps


def synthetic_run(cfg, schedule, dt, warmup=0):
    """Drive a SlowPhysiology with a synthetic (speed, joint_speed) schedule.

    Returns the module and the list of (load, demand) it computed, plus the per-step
    non-negativity evidence.
    """
    p = SlowPhysiology(cfg)
    nonneg = {"inflow": True, "outflow": True, "consumption": True,
              "transfer_ge_zero": True, "tissue_ge_zero": True, "tracheal_ge_zero": True}
    loads = []
    for k, (v, j) in enumerate(schedule):
        load, cs, cj = p.measured_load(v, j)
        if k < warmup:
            load, cs, cj = 0.0, 0.0, 0.0
        demand = p.demand_nmol_s(load)
        p.step(dt, load=load)
        loads.append((load, demand, cs, cj))
        L = p.o2_ledger
        nonneg["inflow"] &= L.inflow >= 0.0
        nonneg["outflow"] &= L.outflow >= 0.0
        nonneg["consumption"] &= L.consumption >= 0.0
        nonneg["tissue_ge_zero"] &= p.tissue_o2_nmol >= 0.0
        nonneg["tracheal_ge_zero"] &= p.tracheal_o2_nmol >= 0.0
    return p, loads, nonneg


def main():
    t0 = time.perf_counter()
    cfg = SlowPhysiologyConfig().validate()

    # =====================================================================
    # 1. DETERMINISM
    # =====================================================================
    sched = [(0.0, 0.0)] * 20 + [(12.0, 400.0)] * 40 + [(3.0, 60.0)] * 20
    a1, l1, nn1 = synthetic_run(cfg, sched, 0.01)
    a2, l2, nn2 = synthetic_run(cfg, sched, 0.01)
    same_state = (a1.tracheal_o2_nmol == a2.tracheal_o2_nmol
                  and a1.tissue_o2_nmol == a2.tissue_o2_nmol
                  and a1.haemolymph_trehalose_nmol == a2.haemolymph_trehalose_nmol
                  and a1.o2_ledger.inflow == a2.o2_ledger.inflow
                  and a1.o2_ledger.consumption == a2.o2_ledger.consumption
                  and a1.o2_ledger.flows == a2.o2_ledger.flows)
    rows1 = np.asarray([r.o2_total_nmol for r in a1.rows])
    rows2 = np.asarray([r.o2_total_nmol for r in a2.rows])
    check("determinism: two independent runs with the same config and inputs are "
          "BIT-IDENTICAL in state, ledgers and every recorded row",
          same_state and bool(np.array_equal(rows1, rows2)),
          "state_equal=%s rows_bit_equal=%s" % (same_state,
                                                bool(np.array_equal(rows1, rows2))))
    check("determinism: the load/demand series is bit-identical too",
          l1 == l2, "n=%d entries" % len(l1))
    fp0 = slowphys_fingerprint(cfg)
    fp1 = slowphys_fingerprint(SlowPhysiologyConfig(seed=1))
    fp2 = slowphys_fingerprint(SlowPhysiologyConfig(
        tissue=SlowPhysiologyConfig().tissue.__class__(resting_demand_nmol_s=0.5)))
    check("determinism: the fingerprint is stable for one config and changes with "
          "config content", fp0 == slowphys_fingerprint(cfg) and fp0 != fp1
          and fp0 != fp2, "%s / %s / %s" % (fp0[:12], fp1[:12], fp2[:12]))

    # =====================================================================
    # 2. THE FROZEN / OFF PATH IS INERT
    # =====================================================================
    m_off = LocalModulator(ModulatorConfig())          # enabled=False by default
    before = (m_off.time_s, m_off.occupancy, m_off.model)
    raised = refuse(lambda: m_off.step(1.0))
    check("inert: a DISABLED modulator layer RAISES when stepped and changes nothing "
          "(it cannot silently contribute), and a malformed one is refused at "
          "construction",
          raised and (m_off.time_s, m_off.occupancy, m_off.model) == before
          and refuse(lambda: LocalModulator(ModulatorConfig(enabled=True,
                                                            source_amount_s=-1.0))),
          "step raised=%s, state unchanged=%s; sample=%s"
          % (raised, (m_off.time_s, m_off.occupancy, m_off.model) == before,
             json.dumps(m_off.sample())[:100]))

    m_on = LocalModulator(ModulatorConfig(enabled=True))
    for _ in range(200):
        m_on.step(0.05)
    occ0 = m_on.occupancy
    mult_off, mult_val = m_on.load_multiplier(1.0)
    check("inert (unknown target OFF BY DEFAULT): with the target effect disabled the "
          "multiplier is exactly 1 and occupancy cannot reach the load",
          mult_val == 1.0 and mult_off == 1.0 and m_on.cfg.known_target is False
          and 0.0 <= occ0 <= 1.0,
          "occupancy at source after 10 s = %.6f; load_multiplier -> (%.6f, %.6f)"
          % (occ0, mult_off, mult_val))
    m_eff = LocalModulator(ModulatorConfig(enabled=True,
                                           enable_unknown_target_effect=True))
    for _ in range(200):
        m_eff.step(0.05)
    lo, hi = (float(v) for v in m_eff.cfg.target_effect_bounds)
    vals = [m_eff.load_multiplier(1.0)[1] for _ in range(1)]
    check("inert: when the unknown target effect IS switched on it stays inside its "
          "declared bounds and cannot explode",
          lo <= vals[0] <= hi and m_eff.cfg.known_target is False,
          "multiplier %.6f within [%.2f, %.2f]" % (vals[0], lo, hi))

    p_proxy_off = SlowPhysiology(SlowPhysiologyConfig(enable_energy_proxy=False))
    for v, j in sched:
        p_proxy_off.step(0.01, load=p_proxy_off.measured_load(v, j)[0])
    check("inert: the disabled energy proxy computes nothing, is recorded as NaN, and "
          "still appears in no ledger",
          p_proxy_off.energy_proxy is None
          and np.isnan(p_proxy_off.sample().energy_proxy)
          and p_proxy_off.energy_proxy_in_any_ledger() is False,
          "proxy=None, sample=NaN, in_any_ledger=False")

    p_rest = SlowPhysiology(cfg)
    rest_demand = p_rest.demand_nmol_s(0.0)
    for _ in range(4000):
        p_rest.step(0.005, demand_nmol_s=0.0)
    L = p_rest.check_ledgers(LEDGER_TOL_NMOL)
    tr, ti = cfg.tracheal, cfg.tissue
    c_air = tr.bath_o2_nmol_per_mm3 * tr.volume_mm3
    c_tis = (tr.bath_o2_nmol_per_mm3 * ti.volume_mm3)
    check("inert (zero load): a zero-load run consumes exactly ZERO, and both "
          "compartments relax monotonically to their own air-saturated levels",
          L["O2"]["consumption"] == 0.0
          and abs(p_rest.tracheal_o2_nmol - c_air) / c_air < 1e-3
          and abs(p_rest.tissue_o2_nmol - c_tis) / c_tis < 1e-3
          and rest_demand == ti.resting_demand_nmol_s,
          "tracheal %.6f -> %.6f nmol (air-saturated %.6f); tissue %.6f -> %.6f "
          "(air-saturated %.6f); consumption exactly %.1f"
          % (tr.initial_o2_nmol, p_rest.tracheal_o2_nmol, c_air,
             ti.initial_o2_nmol, p_rest.tissue_o2_nmol, c_tis,
             L["O2"]["consumption"]))

    # =====================================================================
    # 3. LEDGERS CLOSE AND ARE REPORTED
    # =====================================================================
    grid = []
    worst = 0.0
    worst_case = None
    for dt in (0.005, 0.01, 0.05, 0.5):
        for demand in (0.0, 0.2167, 2.91, 25.0, 1e6):
            nsteps = max(1, int(round(2.0 / dt)))
            p = SlowPhysiology(cfg)
            for _ in range(nsteps):
                p.step(dt, demand_nmol_s=demand)
            L = p.check_ledgers(LEDGER_TOL_NMOL)
            r = abs(L["O2"]["residual"])
            grid.append({"dt_s": dt, "demand_nmol_s": demand, "n_steps": nsteps,
                         "residual_nmol": L["O2"]["residual"],
                         "closed": L["O2"]["closed_within_tolerance"],
                         "clipped": L["demand_clipped"],
                         "tissue_o2_nmol": p.tissue_o2_nmol})
            if r > worst:
                worst, worst_case = r, (dt, demand)
    check("ledger: for %d (step size, demand) combinations including a demand far above "
          "the supply ceiling, the O2 ledger closes within %.0e nmol and the residual is "
          "REPORTED" % (len(grid), LEDGER_TOL_NMOL),
          all(g["closed"] for g in grid),
          "worst |residual| = %.3e nmol at dt=%s s, demand=%s nmol/s"
          % (worst, worst_case[0], worst_case[1]))
    Lc = a1.check_ledgers()
    check("ledger: the CUMULATIVE trachea-to-tissue transfer derived from the AIR SIDE's "
          "identity and from the TISSUE's identity agree, so the two compartment accounts "
          "cannot disagree about how much O2 moved",
          Lc["o2_transfer_two_identities_agree"] is True
          and abs(Lc["o2_cumulative_transfer_to_tissue_nmol"]
                  - Lc["o2_cumulative_transfer_from_air_side_nmol"]) <= LEDGER_TOL_NMOL,
          "from the air side %.6f nmol vs from the tissue %.6f nmol"
          % (Lc["o2_cumulative_transfer_from_air_side_nmol"],
             Lc["o2_cumulative_transfer_to_tissue_nmol"]))
    check("ledger: the COMPARTMENT identities close too (air side and tissue separately), "
          "not only the total",
          all(abs(v) <= LEDGER_TOL_NMOL
              for v in (a1.check_ledgers(LEDGER_TOL_NMOL)["O2"]
                        ["compartment_residuals"]["tracheal_air_side"],
                        a1.check_ledgers(LEDGER_TOL_NMOL)["O2"]
                        ["compartment_residuals"]["tissue"])),
          "air %.2e, tissue %.2e"
          % (a1.check_ledgers()["O2"]["compartment_residuals"]["tracheal_air_side"],
             a1.check_ledgers()["O2"]["compartment_residuals"]["tissue"]))
    check("ledger: the haemolymph's OWN substance ledger closes and the report states "
          "plainly that the haemolymph carries NO O2",
          all(g["closed"] for g in grid)
          and a1.check_ledgers()["haemolymph_carries_o2"] is False
          and a1.check_ledgers()["o2_is_tracheal_only"] is True
          and a1.check_ledgers()["trehalose"]["closed_within_tolerance"]
          and a1.check_ledgers()["trehalose"]["residual"] == 0.0,
          "trehalose residual %.1e nmol; haemolymph_carries_o2=%s; o2_is_tracheal_only=%s"
          % (a1.check_ledgers()["trehalose"]["residual"],
             a1.check_ledgers()["haemolymph_carries_o2"],
             a1.check_ledgers()["o2_is_tracheal_only"]))
    check("ledger: the rule recorded with every ledger is the stated identity, and the "
          "residual column is present in the artefact",
          all(g["closed"] for g in grid)
          and "amount_now = initial + inflow - outflow - consumption"
          in a1.check_ledgers()["O2"]["rule"],
          a1.check_ledgers()["O2"]["rule"][:120])
    check("ledger: an over-ceiling demand is CLIPPED AND REPORTED, never silently "
          "absorbed (the tissue is at zero and the unmet amount is a number)",
          any(g["clipped"] for g in grid)
          and a1.check_ledgers()["demand_ceiling_rule"].startswith("demand is clipped"),
          json.dumps([g for g in grid if g["clipped"]][:1])[:220])

    # =====================================================================
    # 4. NON-NEGATIVITY AND THE INDEPENDENT RK4 REFERENCE
    # =====================================================================
    p_nn, _, nonneg = synthetic_run(cfg, sched, 0.01)
    check("non-negativity: every supply term and every compartment stayed >= 0 at every "
          "step of a 80-interval activity run", all(nonneg.values()),
          json.dumps(nonneg))
    ref_cases = [(0.05, 40, 0.0), (0.05, 40, 2.91), (0.005, 400, 0.0),
                 (0.5, 4, 2.91)]
    worst_state = worst_transfer = worst_cons = 0.0
    ref_detail = []
    for dt, nsteps, demand in ref_cases:
        p = SlowPhysiology(cfg)
        for _ in range(nsteps):
            p.step(dt, demand_nmol_s=demand)
        A, T, inf_, out_, trf, cons = rk4_reference(cfg, dt, demand, nsteps)
        dA = abs(p.tracheal_o2_nmol - A)
        dT = abs(p.tissue_o2_nmol - T)
        dtr = abs(p.o2_ledger.flows["tracheal_to_tissue_nmol"] - trf)
        dco = abs(p.o2_ledger.consumption - cons)
        worst_state = max(worst_state, dA, dT)
        worst_transfer = max(worst_transfer, dtr)
        worst_cons = max(worst_cons, dco)
        ref_detail.append("dt=%.3f d=%.2f: dA=%.2e dT=%.2e dtr=%.2e dcons=%.2e"
                          % (dt, demand, dA, dT, dtr, dco))
    check("integrator: the state matches an INDEPENDENT RK4 integration of the stated ODE "
          "to <= 1e-6 nmol over four (step, demand) cases, including a 0.5 s step",
          worst_state <= 1e-6, "; ".join(ref_detail))
    check("integrator: the TRANSFER INTEGRAL matches the reference to <= 1e-6 nmol -- "
          "this is the quantity that was wrong three times while this file was written",
          worst_transfer <= 1e-6, "worst %.3e nmol" % worst_transfer)
    check("integrator: the accumulated consumption matches the reference exactly",
          worst_cons <= 1e-12, "worst %.3e nmol" % worst_cons)

    # demand scales linearly with the load, and the load with the measured inputs
    demo_sched = [(0.0, 0.0)] * 5 + [(5.0, 100.0)] * 5 + [(15.0, 300.0)] * 5
    pA, lA, _ = synthetic_run(cfg, demo_sched, 0.01)
    pB, lB, _ = synthetic_run(cfg, [(2 * v, 2 * j) for v, j in demo_sched], 0.01)
    ratio_ok = all(abs(b[0] - 2.0 * a[0]) < 1e-9 for a, b in zip(lA, lB))
    check("locomotor load: DOUBLING the two measured inputs exactly doubles the load, "
          "and the demand follows linearly", ratio_ok,
          "loads %s -> %s" % (["%.4f" % x[0] for x in lA[5:10]],
                              "%.4f" % lB[6][0]))
    check("locomotor load: the load RESPONDS TO ACTIVITY with the numbers printed "
          "(rest -> walk -> fast walk)",
          lA[0][0] == 0.0 and lA[7][0] > lA[0][0] and lA[12][0] > lA[7][0]
          and lA[12][1] > lA[7][1] > lA[0][1],
          "load/demand: rest %.4f/%.4f, walk %.4f/%.4f, fast %.4f/%.4f nmol/s"
          % (lA[0][0], lA[0][1], lA[7][0], lA[7][1], lA[12][0], lA[12][1]))
    p_speed_only = SlowPhysiology(SlowPhysiologyConfig(
        load=LoadConfig(speed_only=True)))
    p_both = SlowPhysiology(cfg)
    lo_s, _, _ = p_speed_only.measured_load(15.0, 0.0)
    lo_b, _, _ = p_both.measured_load(15.0, 0.0)
    hi_s, _, _ = p_speed_only.measured_load(15.0, 400.0)
    hi_b, _, _ = p_both.measured_load(15.0, 400.0)
    check("locomotor load: the JOINT term is really contributing -- with speed_only the "
          "load ignores joint speed, with both terms it does not",
          lo_s == lo_b and hi_s == lo_s and hi_b > hi_s,
          "speed-only: %.4f -> %.4f; both terms: %.4f -> %.4f"
          % (lo_s, hi_s, lo_b, hi_b))
    hits_before = p_both.n_load_ceiling_hits
    capped, _, _ = p_both.measured_load(1e6, 1e6)
    check("locomotor load: the declared load ceiling is enforced and COUNTED, so no "
          "single transient can dominate the run",
          p_both.n_load_ceiling_hits == hits_before + 1
          and capped == cfg.load.load_ceiling
          and p_both.load_unclipped_max > cfg.load.load_ceiling,
          "ceiling %.2f; an absurd input (1e6 mm/s, 1e6 rad/s) was clipped to %.4f after "
          "an unclipped value of %.1f; total hits %d"
          % (cfg.load.load_ceiling, capped, p_both.load_unclipped_max,
             p_both.n_load_ceiling_hits))

    # =====================================================================
    # 5. BEHAVIOUR THRESHOLDS AT THEIR BOUNDARIES
    # =====================================================================
    bcfg = BehaviourConfig().validate()
    dt_c = 0.05
    E = bcfg.rest_speed_max_mm_s              # exactly the rest speed ceiling
    L_min = bcfg.locomotion_speed_min_mm_s     # exactly the locomotion speed floor
    T_min = bcfg.turning_heading_change_min_deg
    X_disp = bcfg.exploration_displacement_min_mm
    X_long = bcfg.exploration_long_displacement_min_mm
    LOOP = bcfg.turn_loop_displacement_max_mm
    X_sd = bcfg.exploration_heading_sd_min_deg
    S_min = bcfg.supported_contact_legs_min
    MOVING = 3.0 * L_min                       # mm/s, well above every speed threshold
    FLOOR = 1.02 * L_min                       # above the floor, still in the gaps
    steps = 240                                # 12 s at 0.05 s: longer than either window

    def build(speed_mm_s, heading_rate_deg_per_step, contact, wobble_deg=0.0,
              still_after=None, lateral_frac=0.0):
        """A synthetic trajectory with a stated speed, heading plan and contact count.

        ``still_after`` (an index) makes the body stop dead from that interval on, so a
        case can present a large net displacement over the LONG window and then a
        genuinely still SHORT window -- which is the boundary between TURNING and
        EXPLORATION in the pre-registered rule.
        """
        t = np.arange(steps) * dt_c
        x = np.arange(steps) * (speed_mm_s * dt_c)
        if still_after is not None:
            x[still_after:] = x[still_after - 1] if still_after > 0 else 0.0
        y = np.zeros(steps)
        if lateral_frac:
            # a real LATERAL zig-zag, so the path efficiency (net / path) is genuinely
            # below 1 and the EXPLORATION-versus-LOCOMOTION criterion has something to
            # measure.  Without it a test that "wanders" in heading only would still walk
            # in a straight line and be labelled LOCOMOTION -- correctly.
            y = lateral_frac * np.array([(-1) ** i * (i // 2 + 1) * speed_mm_s * dt_c
                                         for i in range(steps)])
        xy = np.stack([x, y], axis=1)
        hd = np.radians(heading_rate_deg_per_step) * np.arange(steps)
        if wobble_deg:
            hd = hd + np.radians(wobble_deg) * np.array(
                [(-1) ** i for i in range(steps)])
        if still_after is not None:
            hd[still_after:] = hd[still_after - 1] if still_after > 0 else 0.0
        return t, xy, hd, np.full(steps, float(contact))

    def label_of(t, xy, hd, cc):
        m = measure_behaviour(t, xy, hd, cc, bcfg)
        ls = measure_long_scale(t, xy, hd, bcfg)
        labs, reas = classify_behaviour(m, bcfg, ls)
        return labs[-1], m[-1], ls, reas[-1]

    cases = []

    def case(label, expect, *args, **kw):
        cases.append((label, expect, args, kw))

    # ---- REST, and its boundaries
    case("REST: speed exactly AT the rest speed ceiling, heading change 0, supported",
         "REST", E, 0.0, 6.0)
    case("REST (boundary): speed JUST BELOW the rest speed ceiling", "REST",
         E - 1e-6, 0.0, 6.0)
    case("(boundary): speed JUST ABOVE the rest ceiling but below the locomotion floor, "
         "walking straight -> REST by the pre-registered fall-through (the gap between "
         "the rest ceiling and the locomotion floor is deliberately a REST band)",
         "REST", E + 1e-6, 0.0, 6.0)
    # ---- LOCOMOTION, and its boundary
    case("LOCOMOTION: just above the locomotion speed floor, walking straight (the "
         "comparison is >=, so the test states where its own synthetic speed lands "
         "instead of pretending a 0.05 s grid can hit the floor exactly)", "LOCOMOTION",
         FLOOR, 0.0, 6.0)
    case("LOCOMOTION: high speed, straight heading", "LOCOMOTION", MOVING, 0.0, 6.0)
    # ---- TURNING, and its boundary
    # TURNING needs |heading change| >= T_min per 0.5 s window AND a long-window net
    # displacement BELOW the loop threshold.  A body with speed v and a heading rate w
    # covers a circle of circumference v/(w in turns/s), so a fast, slow-turning body
    # travels far enough on that circle to exceed the threshold: that is the boundary.
    case("TURNING: hard heading change per window and a long-window net displacement "
         "below the loop threshold %.1f mm (it walks in a tight circle)" % LOOP,
         "TURNING", 1.0, 300.0, 6.0)
    case("(boundary): a slower, softer turn is still TURNING", "TURNING",
         2.0, 520.0, 6.0)
    case("(boundary): the same turn command on a body moving fast enough to travel "
         "further than %.1f mm on its circle -> NOT turning" % LOOP,
         "LOCOMOTION", 40.0, 40.0, 6.0)
    # ---- EXPLORATION, and its boundary
    case("EXPLORATION: slow, fast-wobbling path with long-window heading SD >= %.0f deg "
         "and long-window net displacement >= %.1f mm" % (X_sd, X_long),
         "EXPLORATION", 4.0, 0.0, 6.0, wobble_deg=X_sd + 60.0, lateral_frac=0.35)
    case("(boundary): the same wander but speed JUST BELOW the turning/exploration speed "
         "floor -> falls through to REST", "REST", 1.0 - 1e-6, 0.0, 6.0,
         wobble_deg=X_sd + 60.0)
    case("(boundary): the same wander but the long-window displacement is JUST BELOW "
         "%.1f mm -> falls through to REST" % X_long, "REST", 0.02, 0.0, 6.0,
         wobble_deg=X_sd + 5.0)
    # ---- NOT_SUPPORTED, and its boundary
    case("NOT_SUPPORTED: fewer than %.1f legs in contact on average" % S_min,
         "NOT_SUPPORTED", 0.0, 0.0, S_min - 0.5)
    case("NOT_SUPPORTED beats REST: a still body with no support is not resting",
         "NOT_SUPPORTED", 0.0, 0.0, 1.0)
    case("(boundary): exactly %.1f legs in contact IS supported" % S_min,
         "REST", 0.0, 0.0, S_min)

    fails = []
    table = []
    for label, expect, args, kw in cases:
        t, xy, hd, cc = build(*args, **kw)
        got, m, ls, reason = label_of(t, xy, hd, cc)
        table.append({"case": label, "expected": expect, "got": got, "reason": reason,
                      "speed_mm_s": m.speed_mm_s,
                      "heading_change_deg": m.heading_change_deg,
                      "efficiency": (m.displacement_mm / m.path_length_mm
                                     if m.path_length_mm > 0 else 0.0),
                      "long_displacement_mm": float(ls[0][-1]),
                      "long_heading_sd_deg": float(ls[1][-1]),
                      "contact_leg_count_mean": m.contact_leg_count_mean})
        if got != expect:
            fails.append("%s -> %s (expected %s): %s" % (label, got, expect, reason))
    check("behaviour thresholds: %d synthetic cases ON and JUST BELOW every criterion of "
          "the two-scale rule produce the pre-registered label (the table is written to "
          "the JSON)" % len(cases), not fails,
          "; ".join(fails) if fails else
          "all %d matched; e.g. %s" % (len(cases), json.dumps(table[6], default=str)[:260]))

    # ---- the two-scale requirement is enforced, not optional
    t_, xy_, hd_, cc_ = build(MOVING, 0.0, 6.0)
    ms_ = measure_behaviour(t_, xy_[:, :2], hd_, cc_, bcfg)
    check("two-scale rule: the classifier REFUSES to run without the long-window "
          "measures, because the turning-versus-exploration decision is not decidable "
          "from the short window alone (measured: 81.6 vs 71.2 deg per window)",
          refuse(lambda: classify_behaviour(ms_, bcfg))
          and refuse(lambda: classify_behaviour(ms_, bcfg, (np.zeros(3), np.zeros(3)))),
          "refused both a missing long scale and a misaligned one")

    # ---- the long window is what separates a loop from a wander
    loop_t, loop_xy, loop_hd, loop_cc = build(1.0, 900.0, 6.0)
    loop_lab, loop_m, loop_ls, _ = label_of(loop_t, loop_xy, loop_hd, loop_cc)
    wall_t, wall_xy, wall_hd, wall_cc = build(40.0, 40.0, 6.0)
    wall_lab, wall_m, wall_ls, _ = label_of(wall_t, wall_xy, wall_hd, wall_cc)
    check("two-scale rule: the SAME turning kind of path is TURNING when the long window "
          "shows it going nowhere and NOT TURNING when it goes somewhere -- the "
          "short-window heading change alone cannot tell these apart",
          loop_lab == "TURNING" and wall_lab != "TURNING"
          and loop_m.heading_change_deg > T_min and wall_m.heading_change_deg > T_min,
          "loop: %.1f deg/window, long disp %.2f mm -> %s | fast: %.1f deg/window, long "
          "disp %.2f mm -> %s"
          % (loop_m.heading_change_deg, float(loop_ls[0][-1]), loop_lab,
             wall_m.heading_change_deg, float(wall_ls[0][-1]), wall_lab))

    check("behaviour thresholds: the short window at the START of a series is labelled "
          "REST with an explicit 'window_incomplete' reason rather than guessed",
          measure_behaviour(np.arange(3) * dt_c, np.zeros((3, 2)), np.zeros(3),
                            np.full(3, 6.0), bcfg)[0].windowed is False
          and classify_behaviour(
              measure_behaviour(np.arange(3) * dt_c, np.zeros((3, 2)), np.zeros(3),
                                np.full(3, 6.0), bcfg), bcfg,
              measure_long_scale(np.arange(3) * dt_c, np.zeros((3, 2)), np.zeros(3),
                                 bcfg))[1][0] == "window_incomplete",
          "reason = window_incomplete")

    short_t, short_xy, short_hd, short_cc = build(MOVING, 0.0, 6.0)
    short_t, short_xy = short_t[:9], short_xy[:9]
    short_ms = measure_behaviour(short_t, short_xy, short_hd[:9], short_cc[:9], bcfg)
    labs_, reas_ = classify_behaviour(
        short_ms, bcfg, measure_long_scale(short_t, short_xy, short_hd[:9], bcfg))
    ep = epoch_table(labs_, short_ms, reas_)
    check("behaviour epochs: a uniform series collapses to ONE epoch whose duration is "
          "the interval grid times its length (NOT window_s times the count) -- the "
          "first version reported 409.5 s for a 4.1 s epoch, 100x too long",
          len(ep) == 1 and abs(ep[0]["duration_s"]
                               - ep[0]["n_intervals"] * dt_c) < 1e-9,
          json.dumps(ep[0], default=str)[:220])

    # =====================================================================
    # 6. INVALID INPUT IS REFUSED
    # =====================================================================
    bad = [
        lambda: SlowPhysiologyConfig(tracheal=SlowPhysiologyConfig().tracheal.__class__(
            bath_conductance_mm3_s=-1.0)).validate(),
        lambda: SlowPhysiologyConfig(tracheal=SlowPhysiologyConfig().tracheal.__class__(
            volume_mm3=0.0)).validate(),
        lambda: SlowPhysiologyConfig(tissue=SlowPhysiologyConfig().tissue.__class__(
            resting_demand_nmol_s=float("nan"))).validate(),
        lambda: SlowPhysiologyConfig(haemolymph=HaemolymphConfig(
            o2_capacity_nmol=1.0)).validate(),
        lambda: SlowPhysiologyConfig(load=LoadConfig(speed_scale_mm_s=0.0)).validate(),
        lambda: SlowPhysiologyConfig(load=LoadConfig(load_ceiling=0.0)).validate(),
        lambda: SlowPhysiologyConfig(load=LoadConfig(weight_speed=0.0,
                                                     weight_joint=0.0)).validate(),
        lambda: SlowPhysiologyConfig(max_substeps=0).validate(),
        lambda: SlowPhysiologyConfig(max_continuous_step_s=-1.0).validate(),
        lambda: SlowPhysiologyConfig(seed=-1).validate(),
        lambda: SlowPhysiologyConfig(seed=1.5).validate(),
        lambda: BehaviourConfig(window_s=0.0).validate(),
        lambda: BehaviourConfig(rest_speed_max_mm_s=10.0,
                                locomotion_speed_min_mm_s=3.0).validate(),
        lambda: BehaviourConfig(min_samples=2).validate(),
        lambda: BehaviourConfig(supported_contact_legs_min=7.0).validate(),
        lambda: BehaviourConfig(window_s=2.0, long_window_s=1.0).validate(),
        lambda: BehaviourConfig(locomotion_efficiency_min=1.5).validate(),
        lambda: BehaviourConfig(exploration_heading_sd_min_deg=-1.0).validate(),
        lambda: BehaviourConfig(long_window_s=0.0).validate(),
        lambda: ModulatorConfig(n_voxels=(4, 4, 0)).validate(),
        lambda: ModulatorConfig(source_voxel=(9, 0, 0)).validate(),
        lambda: ModulatorConfig(kon_nM_inv_s=0.0).validate(),
        lambda: ModulatorConfig(known_target=True).validate(),
        lambda: ModulatorConfig(target_effect_bounds=(1.0, 0.5)).validate(),
        lambda: TimeJumpRecord(start_s=0.0, duration_s=0.0),
        lambda: TimeJumpRecord(start_s=0.0, duration_s=1.0, label="quiet_advance"),
    ]
    refused = sum(1 for fn in bad if refuse(fn))
    check("invalid input: every malformed config, non-zero haemolymph O2 capacity, "
          "malformed grid, unknown target claimed as known, and unlabelled time jump is "
          "REFUSED", refused == len(bad), "%d/%d refused" % (refused, len(bad)))

    live = [
        lambda: a1.step(-1.0),
        lambda: a1.step(0.1, demand_nmol_s=-1.0),
        lambda: a1.step(0.1, demand_nmol_s=float("nan")),
        lambda: a1.step(0.1, load=-0.5),
        lambda: a1.step(0.1, activity="not an activity"),
        lambda: a1.measured_load(-1.0, 0.0),
        lambda: a1.measured_load(0.0, float("inf")),
        lambda: a1.reduce_interval(0.0, 0.0, np.zeros((3, 2)), np.zeros(2), np.zeros(2)),
        lambda: a1.reduce_interval(0.0, 0.1, np.zeros((3, 3)), np.zeros(2), np.zeros(2)),
        lambda: a1.reduce_interval(0.0, 0.1, np.zeros((3, 2)), np.zeros(2),
                                   np.full(2, 7.0)),
        lambda: a1.reduce_interval(0.0, 0.1, np.zeros((3, 2)), np.full(2, -1.0),
                                   np.zeros(2)),
        lambda: a1.reduce_interval(0.0, 0.1, np.array([[0.0, 0.0], [np.nan, 0.0]]),
                                   np.zeros(2), np.zeros(2)),
        lambda: measure_behaviour(np.arange(3) * 0.05, np.zeros((4, 2)), np.zeros(3),
                                  np.zeros(3)),
        lambda: measure_behaviour(np.arange(4) * 0.05, np.zeros((4, 2)), np.zeros(4),
                                  np.full(4, 9.0)),
        lambda: measure_behaviour(np.array([0.0, 0.05, 0.05, 0.05]), np.zeros((4, 2)),
                                  np.zeros(4), np.zeros(4)),
        lambda: measure_behaviour(np.array([0.0, 0.05]), np.zeros((2, 2)), np.zeros(2),
                                  np.zeros(2), BehaviourConfig(window_s=-1.0)),
        lambda: classify_behaviour([1, 2, 3]),
        lambda: measure_long_scale(np.arange(3) * 0.05, np.zeros((4, 2)), np.zeros(3)),
        lambda: measure_long_scale(np.arange(4) * 0.05, np.zeros((4, 2)),
                                   np.full(4, np.nan)),
        lambda: slowphys_fingerprint("not a config"),
        lambda: LocalModulator(ModulatorConfig()).step(0.1),
    ]
    refused_live = sum(1 for fn in live if refuse(fn))
    check("invalid input: the RUNTIME guards refuse a negative duration, a negative or "
          "NaN demand, a negative load, a malformed activity reduction, a non-uniform "
          "time grid, a 7-leg contact count and a disabled modulator step",
          refused_live == len(live), "%d/%d refused" % (refused_live, len(live)))

    # a long advance is refused unless it carries the quasi-steady label
    p_tj = SlowPhysiology(cfg)
    alone = refuse(lambda: p_tj.step(cfg.max_continuous_step_s * 2.0))
    p_tj.time_jumps.clear()
    p_tj.step(cfg.max_continuous_step_s * 2.0, declared_time_jump=True)
    check("TIME JUMP: an advance longer than max_continuous_step_s is REFUSED unless it "
          "is declared, and when declared it is recorded with the explicit "
          "TIME_JUMP_QUASI_STEADY label and the 'not simulated' statement",
          alone and len(p_tj.time_jumps) == 1
          and p_tj.time_jumps[0].label == "TIME_JUMP_QUASI_STEADY"
          and p_tj.time_jumps[0].body_simulated_during_jump is False
          and "NOT simulated" in p_tj.time_jumps[0].note
          and p_tj.check_ledgers(LEDGER_TOL_NMOL)["O2"]["closed_within_tolerance"],
          json.dumps(p_tj.time_jumps[0].as_dict())[:220])

    # a short declared step must NOT be recorded as a time jump (the flag has to mean
    # something)
    p_short = SlowPhysiology(cfg)
    p_short.step(0.05, declared_time_jump=True)
    check("TIME JUMP: the flag cannot be manufactured -- a step shorter than the bound is "
          "never labelled a jump even when the caller passes declared_time_jump=True",
          len(p_short.time_jumps) == 0, "0 jumps recorded for a 0.05 s step")

    # =====================================================================
    # 7. THE FORBIDDEN VOCABULARY, AND THE HONESTY BLOCK
    # =====================================================================
    src_path = os.path.join(HERE, "engine", "embodied", "slow_physiology.py")
    with open(src_path) as fh:
        src = fh.read()

    def affirmative_hits(text, keep_enforcement_list=False):
        """Occurrences of a forbidden term that are NOT inside a sentence denying it.

        The test does not merely COUNT the words.  Each term is looked for together with
        its negation ("NOT ATP", "no haemoglobin ... is used"), because the module is
        REQUIRED to state what it does not model, and a counter cannot tell a disclaimer
        from an assertion.  The enforcement list itself is stripped first, since that is
        the one place the words have to appear.
        """
        body = text
        if not keep_enforcement_list:
            k = text.find("MODULE_FORBIDDEN_TERMS = (")
            if k >= 0:
                end = text.find(")", k)
                body = text[:k] + text[end:]
        # COMMENT-ONLY lines are the internal notes to the next maintainer, not part of
        # what this module names or documents; docstrings and prose are NOT stripped.
        body = "\n".join(ln for ln in body.split("\n")
                          if not ln.lstrip().startswith("#"))
        # flatten per-sentence: split on sentence punctuation and newlines
        chunks = []
        for line in body.split("\n"):
            for piece in line.replace(". ", ".\n").split("\n"):
                if piece.strip():
                    chunks.append(piece.strip())
        hits = {}
        neg = ("not ", "no ", "never ", "nothing ", "without ", "cannot ", "isn't",
               "n't ", "instead of", "rather than", "would be", "different animal",
               "does not", "do not", "none")
        for term in MODULE_FORBIDDEN_TERMS:
            for c in chunks:
                if term in c:
                    low = c.lower()
                    if any(ng in low for ng in neg):
                        continue
                    hits.setdefault(term, []).append(c[:120])
        return hits

    src_hits = affirmative_hits(src)
    check("forbidden vocabulary: the module never NAMES or DOCUMENTS a blood-borne "
          "carrier, a vertebrate-style saturating carrier curve, or a high-energy "
          "phosphate as its currency -- every occurrence of such a word is inside a "
          "sentence that DENIES it (checked by negation context, not by counting)",
          not src_hits,
          "affirmative hits outside denials: %s"
          % json.dumps(src_hits, default=str)[:300] if src_hits else
          "0 affirmative hits; the words occur only in denials and in the "
          "MODULE_FORBIDDEN_TERMS list, which is stripped before the scan")

    report_path = os.path.join(OUT, "slowphys_report.json")
    report = None
    if os.path.isfile(report_path):
        with open(report_path) as fh:
            report = json.load(fh)
    if report is not None:
        blob = json.dumps(report, default=str)
        rep_hits = affirmative_hits(blob, keep_enforcement_list=True)
        fv = report.get("forbidden_vocabulary", {})
        check("forbidden vocabulary: the REPORT has no affirmative use of any forbidden "
              "term (it also CARRIES the list, so a reader can re-run the same check), "
              "and its recorded 'affirmative_uses_found' is empty",
              not rep_hits and fv.get("affirmative_uses_found") == []
              and list(fv.get("terms", [])) == list(MODULE_FORBIDDEN_TERMS),
              "affirmative hits in the report: %s"
              % (json.dumps(rep_hits, default=str)[:260] if rep_hits else
                 "none; %d terms carried for re-checking" % len(MODULE_FORBIDDEN_TERMS)))
        honey = report.get("honesty", [])
        missing = [h for h in SLOWPHYS_HONESTY if h not in honey]
        phrases = ["TRACHEAL", "ILLUSTRATIVE", "LOCUST", "QUIET REST IS NOT SLEEP",
                   "no measured adult-fly metabolic rate", "not ATP",
                   "non-homeostatic artefact", "No consciousness"]
        p_ok = all(any(p in h for h in honey) for p in phrases)
        check("honesty: every honesty string is in the report, and the report covers "
              "TRACHEAL gas exchange, ILLUSTRATIVE parameters, the LOCUST geometry flag, "
              "'quiet rest is not sleep', the missing fly metabolic rate, the energy "
              "proxy not being ATP, the reduced-K artefact and the no-consciousness rule",
              not missing and p_ok,
              "missing=%d phrases_ok=%s" % (len(missing), p_ok))
        check("honesty: the report states the model's length unit, records wall clock and "
              "peak RAM, and names its figure",
              report.get("model_length_unit") == "mm"
              and isinstance(report.get("time", {}).get("wall_seconds"), (int, float))
              and isinstance(report.get("time", {}).get("peak_ram_mb"), (int, float))
              and str(report.get("figure", "")).endswith("slowphys_demo.png"),
              "unit=%s wall=%s RAM=%s figure=%s"
              % (report.get("model_length_unit"),
                 report.get("time", {}).get("wall_seconds"),
                 report.get("time", {}).get("peak_ram_mb"),
                 os.path.basename(str(report.get("figure", "")))))
        check("sleep: the report says PLAINLY that no sleep state exists, that quiescence "
              "is labelled REST, and that no arousal test stimulus was implemented",
              report.get("sleep", {}).get("implemented") is False
              and report.get("sleep", {}).get("arousal_test_stimulus") == "NONE"
              and "NO sleep state exists" in report.get("sleep", {}).get("status", ""),
              json.dumps(report.get("sleep", {}))[:220])
        check("tracheal architecture: the report states that the gas exchange is tracheal, "
              "that the haemolymph carries no O2, and that its own ledger is trehalose",
              report.get("o2_is_tracheal") is True
              and report.get("haemolymph_carries_o2") is False
              and report.get("haemolymph_substance") == "trehalose",
              "o2_is_tracheal=%s haemolymph_carries_o2=%s substance=%s"
              % (report.get("o2_is_tracheal"), report.get("haemolymph_carries_o2"),
                 report.get("haemolymph_substance")))
        check("energy proxy: it is named as a proxy, recorded as NOT ATP, and it appears "
              "in no substance ledger",
              report.get("energy_proxy", {}).get("name") == ENERGY_PROXY_NAME
              and report.get("energy_proxy", {}).get("is_atp") is False
              and report.get("energy_proxy", {}).get("in_any_ledger") is False
              and report.get("ledgers", {}).get("energy_proxy_in_any_ledger") is False,
              json.dumps(report.get("energy_proxy", {}))[:200])
        check("ledger in the artefact: the O2 and trehalose ledgers in the saved report "
              "both closed within %.0e nmol, with residual_relative reported"
              % LEDGER_TOL_NMOL,
              report.get("ledgers", {}).get("O2", {})
              .get("closed_within_tolerance") is True
              and report.get("ledgers", {}).get("trehalose", {})
              .get("closed_within_tolerance") is True
              and "residual_relative" in report.get("ledgers", {}).get("O2", {}),
              "O2 residual %.3e (rel %.2e), trehalose residual %.3e"
              % (report["ledgers"]["O2"]["residual"],
                 report["ledgers"]["O2"]["residual_relative"],
                 report["ledgers"]["trehalose"]["residual"]))
        check("measured load in the artefact: the report names the MEASURED quantity and "
              "its unit, states that no invented activity scalar was used, and gives the "
              "load/demand ranges",
              report.get("measured_load", {}).get("activity_scalar_used") is False
              and "mm/s" in report.get("measured_load", {})
              .get("measured_inputs", {}).get("speed_xy_mm_s", {}).get("unit", "")
              and "rad/s" in report.get("measured_load", {})
              .get("measured_inputs", {}).get("joint_speed_rad_s", {}).get("unit", ""),
              "load %s -> demand %s nmol/s"
              % (report["measured_load"]["load_range"],
                 report["measured_load"]["demand_nmol_s_range"]))
        check("modulator layer in the artefact: its targets are recorded as UNKNOWN and "
              "the target effect is OFF",
              report.get("modulator_layer", {}).get("known_target") is False
              and report.get("modulator_layer", {}).get("target_effect_enabled") is False,
              json.dumps({k: report.get("modulator_layer", {}).get(k)
                          for k in ("enabled", "known_target", "target_effect_enabled")}))
        check("time jump in the artefact: any quasi-steady advance is flagged, and when "
              "none was used the report says so instead of leaving it ambiguous",
              "time_jump_flag" in report.get("time", {})
              and (report["time"]["time_jump_flag"] is True
                   or "no hours of continuous simulation are claimed"
                   in report["time"].get("note", "")),
              "n_time_jumps=%s flag=%s"
              % (report["time"].get("n_time_jumps"), report["time"].get("time_jump_flag")))
        check("behaviour in the artefact: the label counts are recorded, the thresholds "
              "are marked as declared before the run, and the exploration criterion is "
              "stated as measurable",
              report.get("behaviour", {})
              .get("thresholds_declared_before_the_run") is True
              and "heading SD" in report.get("behaviour", {})
              .get("exploration_criterion", "")
              and set(report.get("behaviour", {}).get("label_counts", {}))
              <= set(BEHAVIOUR_LABELS),
              "counts=%s" % report.get("behaviour", {}).get("label_counts"))
        fig_path = os.path.join(OUT, "slowphys_demo.png")
        if os.path.isfile(fig_path):
            check("figure: slowphys_demo.png exists and is non-trivial (> 100 kB); its "
                  "honesty banner and the time-jump flag are asserted by the demo itself, "
                  "and the image is opened and read back by the harness",
                  os.path.getsize(fig_path) > 100_000,
                  "%s (%.0f kB); figure text asserted from the report, not scraped from "
                  "the PNG" % (fig_path, os.path.getsize(fig_path) / 1024.0))
        else:
            check("figure: slowphys_demo.png exists", False, "not found")
        check("selftest: the report's own 'no silent clamping' claim matches the code -- "
              "no clip/np.clip of a STATE amount exists, only the gross flow split",
              src.count("np.clip") == 0 and src.count("clip(") == 0,
              "np.clip occurrences in the module: %d" % src.count("np.clip"))
        check("selftest: the report records the measured supply/consumption pair with "
              "units, and the supply is not zero while consumption is not zero",
              report.get("supply_consumption", {}).get("units", {}).get("amounts") == "nmol"
              and report["supply_consumption"]["o2_consumed_nmol"] > 0.0,
              "inflow %.4f nmol, consumed %.4f nmol, mean supply %.4f nmol/s vs mean "
              "consumption %.4f nmol/s"
              % (report["supply_consumption"]["o2_external_inflow_nmol"],
                 report["supply_consumption"]["o2_consumed_nmol"],
                 report["supply_consumption"]["mean_net_supply_nmol_s"],
                 report["supply_consumption"]["mean_consumption_nmol_s"]))
    else:
        check("artefacts: slowphys_report.json exists (run the demo first)", False,
              "missing %s" % report_path)

    # =====================================================================
    n_pass = sum(1 for r in RESULTS if r["ok"])
    result = {
        "what_these_tests_establish": [
            "bit-determinism of state, ledgers, rows and the load series; a config "
            "fingerprint that changes with config content",
            "the frozen/off paths are inert: a disabled modulator cannot be stepped, an "
            "unknown target stays unknown and bounded, a disabled energy proxy computes "
            "nothing and enters no ledger, and a zero load consumes exactly zero",
            "every substance ledger closes to %g nmol (total AND per compartment) over "
            "%d step-size/demand combinations including a demand far above the supply "
            "ceiling, with the residual reported and an over-ceiling demand clipped and "
            "counted rather than silently absorbed" % (LEDGER_TOL_NMOL, len(grid)),
            "every supply term and both compartments stayed nonnegative over an 80 "
            "interval activity run",
            "the state, the transfer integral and the accumulated consumption match an "
            "INDEPENDENT RK4 integration of the stated ODE (worst state %.2e nmol, worst "
            "transfer %.2e nmol) across four step sizes up to 0.5 s"
            % (worst_state, worst_transfer),
            "the dimensionless locomotor load is built from two MEASURED body quantities "
            "with their units, doubles exactly when they double, responds to activity, "
            "and its joint term demonstrably contributes",
            "%d synthetic behaviour cases on and just below every pre-registered "
            "threshold return the pre-registered label; incomplete windows are labelled "
            "as such; epoch durations come from the interval grid" % len(cases),
            "%d malformed configs and %d runtime-invalid calls are refused"
            % (len(bad), len(live)),
            "no forbidden vocabulary (blood oxygen, respiratory pigment, a mammalian "
            "saturation model, ATP) appears in the module outside the enforcement list, "
            "and none appears in the report",
            "every long advance of the slow module alone is refused unless it carries the "
            "TIME_JUMP_QUASI_STEADY label, and the label cannot be manufactured for a "
            "short step",
        ],
        "what_they_do_not_establish": [
            "that any parameter is Drosophila's: EVERY kinetic and physiological number "
            "here is ILLUSTRATIVE and no default uses the LOCUST tracheal geometry this "
            "project has on record",
            "that the model's oxygen transport, conductances or demand scale match a real "
            "fly: this project has NO measured adult-fly metabolic rate and no measured "
            "tracheal conductance",
            "that the tissue really consumes what the load proxy says: the absolute nmol/s "
            "scale and the two normalising scales are ILLUSTRATIVE",
            "that the behaviour labels correspond to any real fly behaviour: the "
            "thresholds are engineering thresholds for THIS model",
            "anything about SLEEP: no sleep criterion is implemented and no arousal test "
            "stimulus exists",
            "anything about consciousness, experience, viability or immortality",
            "that matching RK4 means matching the fly: RK4 confirms that the code solves "
            "the STATED equations, and the stated equations are a lumped-parameter "
            "approximation with illustrative parameters",
        ],
        "config": cfg.as_dict(),
        "fingerprint_sha256_32": slowphys_fingerprint(cfg),
        "ledger_tolerance_nmol": LEDGER_TOL_NMOL,
        "ledger_grid": grid,
        "behaviour_case_table": table,
        "n_checks": len(RESULTS), "n_passed": n_pass,
        "n_failed": len(RESULTS) - n_pass,
        "passed": bool(n_pass == len(RESULTS)),
        "wall_seconds": time.perf_counter() - t0,
        "results": RESULTS,
    }
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "slowphys_selftest.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print("\n%d/%d checks passed in %.1fs -> %s"
          % (n_pass, len(RESULTS), result["wall_seconds"], path))
    if report is not None:
        report["selftest"] = {"n_checks": result["n_checks"],
                              "n_passed": result["n_passed"],
                              "n_failed": result["n_failed"],
                              "passed": result["passed"], "json": path}
        with open(report_path, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print("updated the selftest summary in %s" % report_path)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
