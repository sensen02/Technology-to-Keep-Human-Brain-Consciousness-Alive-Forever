"""Tests for the first closed sensorimotor loop (scheduler + adapters + recording).

Run in the BODY environment, after the demo (the report check reads loop_report.json):
    ./venv_body/bin/python run_embodied_loop_selftest.py

WHAT THESE TESTS ESTABLISH, AND WHAT THEY DO NOT
-----------------------------------------------
They establish: the clocks and their integer alignment; that the bounded BANC tier is
a SELECTION with reported coverage and UNSIGNED edges; that the body's joint angles
map onto the CPG's DOF order as claimed; that the neural arm is bit-reproducible; that
the cpg_baseline arm is bit-identical to a plain CPG walk; that a neutral command
reproduces the baseline trajectory exactly; that the decoded command comes from the
PREVIOUS completed interval (checked structurally AND by injecting a distinctive spike
pattern); that no observed channel is a truth-only quantity and that every observed
channel is exactly reproducible from truth through its declared transduction; that the
body is stepped by exactly one writer and that the neural tier has no way to write a
joint target; that a forced descending rate changes speed and heading measurably; and
that malformed input is refused.

They do NOT establish: that the connectome generates the gait (it does not -- FlyGym's
demo CPG does, in both arms); that the assumed map from firing rate to CPG knobs is
correct; that the receptor kinetics are Drosophila's; that any of this is a fly
behaving; or anything about experience, perception or consciousness.
"""
from __future__ import annotations

import json
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "outputs", "embodied_body")

from engine.embodied.adapters import (                                          # noqa: E402
    HONESTY, LEGS, UNITS, CommandDecoder, CommandedTripodCPG, DecodeConfig,
    MechanoReceptorBank, NeuralTier, OwnershipError, OwnershipLedger,
    ReceptorConfig, TierConfig, build_neural_tier, dof_names, joint_name_index_map,
    leg_rows_from_dof_order, sensor_matrix_row_map,
)
from engine.embodied.loop import (                                              # noqa: E402
    COMMAND_FIELDS, LoopConfig, MultirateScheduler, OBSERVED_FIELDS,
    OBSERVED_PROVENANCE, TRUTH_FIELDS, heading_yaw_rad,
)
from engine.embodied.body_backend import BodyBackend, BodyConfig                  # noqa: E402

RESULTS = []
BUILT = None
SEED = 0
EQUIV_SECONDS = 0.4
CAUSAL_SECONDS = 1.5


def check(name, ok, detail=""):
    RESULTS.append({"name": name, "ok": bool(ok), "detail": str(detail)[:400]})
    print(("PASS " if ok else "FAIL ") + name +
          (" | " + str(detail)[:200] if detail else ""), flush=True)
    return bool(ok)


def refuse(fn):
    try:
        fn()
    except (ValueError, TypeError, KeyError, RuntimeError, FileNotFoundError,
            OwnershipError):
        return True
    except Exception:                                                    # noqa: BLE001
        return False
    return False


def run_episode(**kw):
    cfg = LoopConfig(gl_backend=None, **kw)
    sch = MultirateScheduler(cfg, built_tier=BUILT if kw.get("arm") != "cpg_baseline"
                             else None)
    ep = sch.run()
    return sch, ep


def plain_cpg_walk(seed, seconds, every_body_steps):
    """A plain CPG walk with no scheduler at all: the reference the baseline must match."""
    be = BodyBackend(BodyConfig(seed=seed, timestep_s=1e-4), gl_backend=None)
    be.attach_cpg_baseline()
    n = int(round(seconds / 1e-4))
    xs = []
    for k in range(n):
        if k % every_body_steps == 0:
            xs.append(be.observe().thorax_position_mm.copy())
        be.step()
    be.close()
    return np.asarray(xs)


class SpikeInjector:
    """Stand-in for ``NeuralTier.advance`` that emits a chosen pattern at ONE interval.

    Everything else is silenced, so the ONLY way the command can move is by reading the
    injected interval -- which makes the timing of the move a direct test of causality.
    """

    def __init__(self, real_advance, n_neurons, n_substeps, inject_at, readout,
                 skip_calls):
        self.real = real_advance
        self.n = int(n_neurons)
        self.n_sub = int(n_substeps)
        self.inject_at = int(inject_at)
        self.readout = np.asarray(readout)
        # the first ``skip_calls`` calls are the calibration run: they pass the tier's own
        # spikes through, so the reference rate is real.  From the first EPISODE interval
        # on, the tier's spikes are replaced by a scripted pattern.
        self.skip = int(skip_calls)
        self.calls = 0
        self.episode_calls = 0

    def __call__(self, n_substeps, current):
        sp = self.real(n_substeps, current)
        if self.calls >= self.skip:
            scripted = np.zeros_like(sp)
            if self.episode_calls == self.inject_at:
                scripted[:, self.readout] = True
            sp = scripted
            self.episode_calls += 1
        self.calls += 1
        return sp


def main():
    global BUILT
    t0 = time.perf_counter()
    print("building the bounded BANC tier once for all tests...", flush=True)
    BUILT = build_neural_tier(TierConfig())
    cov = BUILT["coverage"]

    # ---------------------------------------------------------------- clocks
    cfg = LoopConfig(gl_backend=None)
    check("clocks: three explicit tiers with integer alignment (body 1e-4 s, neural "
          "5e-4 s, command 5e-3 s)",
          (cfg.dt_body_s, cfg.dt_neural_s, cfg.dt_command_s) == (1e-4, 5e-4, 5e-3)
          and cfg.neural_substeps == 10 and cfg.body_substeps == 50,
          "neural substeps=%d body substeps=%d" % (cfg.neural_substeps,
                                                   cfg.body_substeps))
    bad_clocks = [dict(dt_neural_s=4e-4), dict(dt_body_s=3e-4), dict(dt_command_s=7e-3),
                  dict(duration_s=0.0037), dict(dt_body_s=0.0), dict(duration_s=-1.0),
                  dict(dt_neural_s=float("nan"))]
    n_refused = sum(1 for kw in bad_clocks if refuse(lambda kw=kw: LoopConfig(**kw).validate()))
    check("clocks: every misaligned or non-positive clock is refused",
          n_refused == len(bad_clocks),
          "%d/%d refused" % (n_refused, len(bad_clocks)))

    check("units: the unit table says mm and the honesty block says the same",
          UNITS["body_length"] == "mm"
          and any("MILLIMETRES" in h for h in HONESTY)
          and any("ENGINEERING BASELINE" in h for h in HONESTY),
          "body_length=%s" % UNITS["body_length"])

    # ------------------------------------------------------------------ tier
    t = cov["tier"]
    check("tier: the selection is bounded and its coverage is reported",
          t["neurons"] <= TierConfig().max_neurons and t["neurons"] == 12000
          and "select_touch_tier" not in t and cov["is_a_selection_not_a_brain"] is True,
          "neurons=%d cap=%d cap_binding=%s"
          % (t["neurons"], t["max_neurons_config"], t["cap_binding_at_final_iteration"]))
    check("tier: the selection rule recorded is the module's own rule, applied twice",
          "partner_inflow_i" in cov["selection_rule"] and cov["growth_iterations"] == 2
          and len(cov["growth_history"]) == 2,
          json.dumps(cov["growth_history"]))
    check("tier: every cached edge is UNSIGNED (nt_pair == -1 on every row)",
          cov["nt_pair_all_minus_one"] is True
          and cov["edges"]["excluded_edge_accounting"]["nt_pair_used_for_sign"] is False,
          "nt_pair_all_minus_one=%s" % cov["nt_pair_all_minus_one"])
    check("tier: the index map is a bijection and out-of-tier ids can never be wrapped "
          "onto the last local row",
          BUILT["tier"]["local_of_global_index_map"]["is_a_bijection"] is True
          and BUILT["tier"]["local_of_global_index_map"]["outside_tier_sentinel"] == -1
          and int(np.count_nonzero(BUILT["local_of_global"]
                                   [BUILT["global_index"]] >= 0)) == BUILT["n"],
          "n=%d" % BUILT["n"])
    mods = [(k, v) for k, v in cov["modality_boundaries"].items() if v["in_dataset"]]
    ok_cov = all(v["in_tier"] > 0 and v["in_tier"] <= v["in_dataset"] for _, v in mods)
    check("tier: every modality boundary inside the tier is counted and is a fraction of "
          "the dataset's own group (never 100%% of it)",
          ok_cov and min(v["fraction"] for _, v in mods) < 1.0,
          "; ".join("%s %d/%d" % (k, v["in_tier"], v["in_dataset"]) for k, v in mods))
    dr = cov["descending_readout"]
    # These numbers were 7/9 and 177, i.e. the assertion had HARD-CODED the defect:
    # readout_local used to be selected by OUT-going degree (column count) while the
    # variable name, the comment and the coverage key all claimed IN-coming degree.
    # Corrected numbers: 195 of 196 descending neurons receive at least one modelled
    # in-tier edge; exactly 1 does not.
    check("tier: the descending readout's composition is reported, including the "
          "descending cells the sign rule cannot drive",
          dr["descending_in_tier"] == 196
          and dr["in_tier_with_at_least_one_modelled_incoming_edge"] == 195
          and dr["in_tier_with_zero_modelled_incoming_edges"] == 1
          and dr["in_tier_with_at_least_one_modelled_incoming_edge"]
          == dr["readout_left_biased"] + dr["readout_right_biased"]
          + dr["readout_no_bias_tie"],
          "in_tier=%d readout=%d (L%d/R%d/tie%d) no_incoming=%d"
          % (dr["descending_in_tier"],
             dr["in_tier_with_at_least_one_modelled_incoming_edge"],
             dr["readout_left_biased"], dr["readout_right_biased"],
             dr["readout_no_bias_tie"],
             dr["in_tier_with_zero_modelled_incoming_edges"]))

    # CONVENTION GUARD.  In this project the edge matrices are built as
    # csr_matrix((w, (loc[post], loc[pre]))) and consumed as `We @ arrivals`, so
    # ROW = postsynaptic target (incoming) and COLUMN = presynaptic source (outgoing).
    # That is the opposite of the usual CSR intuition and it has already caused one
    # wrong published claim (a "1-hop path from the visual group to the readout" that
    # was really a backwards traversal).  This check pins it: readout_local must equal
    # the ROW-based (incoming) selection and must NOT equal the column-based one.
    _Wc = (BUILT["edges"]["We"] + BUILT["edges"]["Wi"]).tocsr()
    _dsc = np.asarray(BUILT["descending_local"])
    _rowdeg = np.asarray(_Wc[_dsc, :].getnnz(axis=1)).ravel()      # incoming
    _coldeg = np.asarray(_Wc[:, _dsc].getnnz(axis=0)).ravel()      # outgoing
    _ro = np.asarray(BUILT["readout_local"])
    check("tier: edge-convention guard -- readout_local is the INCOMING(row) selection "
          "and NOT the outgoing(column) one",
          set(_ro.tolist()) == set(_dsc[_rowdeg >= 1].tolist())
          and set(_ro.tolist()) != set(_dsc[_coldeg >= 1].tolist()),
          "row>=1: %d  col>=1: %d  readout: %d"
          % (int((_rowdeg >= 1).sum()), int((_coldeg >= 1).sum()), _ro.size))
    # This check used to assert the literal string "NO VISION IS IMPLEMENTED" in the
    # note.  That wording was DELIBERATELY removed when TierConfig.include_visual_route
    # was added: the note is now MODE-DEPENDENT, because a tier built with the visual
    # route really does contain thousands of visual neurons.  Asserting a stale literal
    # is the wrong contract, so this now checks the SUBSTANCE instead: no
    # phototransduction is claimed by this module, the counts are recorded as numbers,
    # the note explains the selection rule, and the mode flag is actually reported.
    _va = cov["visual_absence"]
    _vr = cov["visual_route"]
    _note_ok = (isinstance(_va["note"], str) and len(_va["note"]) > 80
                and "phototransduction" in _va["note"].lower()
                and "SELECTION RULE" in _va["note"].upper())
    check("tier: the visual limitation is recorded as NUMBERS plus a mode-consistent note "
          "(this module models no phototransduction, and says so)",
          _va["photo_receptor_used"] is False
          and _va["visual_machinery_neurons_in_dataset"] == 79538
          and isinstance(_va["visual_projection_neurons_in_tier"], int)
          and _note_ok
          and _vr["include_visual_route"] is False,
          "visual machinery in tier = %d of %d in dataset; mode=%s; note_ok=%s"
          % (_va["visual_projection_neurons_in_tier"],
             _va["visual_machinery_neurons_in_dataset"],
             "leg-only" if not _vr["include_visual_route"] else "leg+visual",
             _note_ok))

    # Coverage for the NEW opt-in visual route.  The full "11 -> thousands" measurement
    # needs a second tier build and lives in the round-18 report; what is checked here
    # is that the switch is OFF by default and that a malformed config is REFUSED.
    _vr_ok, _vr_detail = True, []
    for _bad in (dict(include_visual_route="yes"),
                 dict(visual_route_seed_limit=0),
                 dict(visual_route_seed_limit=99999),
                 dict(visual_classes=()),
                 dict(visual_classes=("",))):
        try:
            # TierConfig is a plain frozen dataclass: constructing it does NOT run the
            # checks, so validate() must be called explicitly (which is how every other
            # call site in this project uses it).  An earlier version of this test
            # omitted validate() and therefore "accepted" every malformed config.
            TierConfig(**_bad).validate()
            _vr_ok = False
            _vr_detail.append("accepted %r" % (_bad,))
        except ValueError:
            pass
    _d = TierConfig().validate()
    check("visual route: OFF by default, and every malformed visual-route config is "
          "refused rather than coerced",
          _vr_ok and _d.include_visual_route is False
          and _d.visual_route_seed_limit == 4000 and len(_d.visual_classes) == 3,
          "default include_visual_route=%s limit=%d classes=%d; refusals=%s"
          % (_d.include_visual_route, _d.visual_route_seed_limit,
             len(_d.visual_classes), "5/5" if _vr_ok else _vr_detail))

    # the tier is silent without sensation: measured, not assumed
    tier_probe = NeuralTier(BUILT, TierConfig(), OwnershipLedger())
    zero_cur = np.zeros(tier_probe.n)
    sp_silent = tier_probe.advance(400, zero_cur)
    tier_probe.reset()
    loud_cur = np.zeros(tier_probe.n)
    for leg in LEGS:
        loud_cur[BUILT["sensory"][leg]] = 0.4
    sp_loud = tier_probe.advance(400, loud_cur)
    r_silent = tier_probe.group_rates_hz(sp_silent)
    r_loud = tier_probe.group_rates_hz(sp_loud)
    check("tier dynamics: with NO sensory drive the tier is silent, and with leg drive it "
          "is not (the loop's signal is the body's, not spontaneous noise)",
          r_silent["total_hz"] == 0.0 and int(sp_silent.sum()) <= 200
          and r_loud["total_hz"] > 5.0,
          "silent %.3f Hz (%d spikes in 200 ms) vs driven %.2f Hz"
          % (r_silent["total_hz"], int(sp_silent.sum()), r_loud["total_hz"]))

    # ------------------------------------------------------- body mapping
    sch_tmp = MultirateScheduler(LoopConfig(arm="cpg_baseline", seed=SEED,
                                            duration_s=2 * cfg.dt_command_s,
                                            gl_backend=None))
    dof_order = sch_tmp.dof_order
    model = sch_tmp.body.model
    jmap = joint_name_index_map(model)
    mismatches = [want for _, want in dof_names(dof_order) if want not in jmap]
    check("body mapping: every one of the 42 actuated DOFs names a real joint in the "
          "compiled model (matched by name, never by position)",
          not mismatches,
          "; ".join(mismatches[:3]) if mismatches else "42/42 DOF names resolved")
    maps = leg_rows_from_dof_order(dof_order, sch_tmp.body.model)
    rows, angle_rows = maps["dof_rows"], maps["angle_rows"]
    check("body mapping: the 42 DOFs split into exactly 7 per leg, disjointly",
          sum(len(v) for v in rows.values()) == 42
          and all(len(v) == 7 for v in rows.values()),
          json.dumps({k: len(v) for k, v in rows.items()}))
    check("body mapping: the actuated DOFs are NOT the first 42 joint angles, and the "
          "angle map is resolved by joint NAME (this check failed before the fix)",
          angle_rows is not None
          and int(maps["angle_of_dof"].min()) >= 0
          and not np.array_equal(maps["angle_of_dof"], np.arange(42))
          and int(maps["angle_of_dof"].max()) >= 42,
          "angle_of_dof[0:9] = %r, max = %d (the 66 hinge angles include passive tarsal "
          "joints)" % (maps["angle_of_dof"][:9].tolist(),
                       int(maps["angle_of_dof"].max())))
    audit = sensor_matrix_row_map(BUILT["sensory"], angle_rows)
    check("body mapping: no tier neuron and no joint-angle row is claimed by two legs",
          audit["neurons_claimed"] == sum(len(v) for v in BUILT["sensory"].values())
          and audit["dof_rows_claimed"] == 42, json.dumps(audit))
    sch_tmp.close()

    # -------------------------------------------------------- determinism
    s1, ep1 = run_episode(arm="neural_modulated", seed=SEED, duration_s=EQUIV_SECONDS)
    s2, ep2 = run_episode(arm="neural_modulated", seed=SEED, duration_s=EQUIV_SECONDS)
    same = all(np.array_equal(ep1.truth[k], ep2.truth[k]) for k in ep1.truth)
    same_obs = all(np.array_equal(ep1.observed[k], ep2.observed[k])
                   for k in ep1.observed)
    same_cmd = all(np.array_equal(ep1.commands[k], ep2.commands[k])
                   for k in ep1.commands)
    check("determinism: two independent neural-arm runs with the same seed are "
          "bit-identical in truth, observed AND commands",
          bool(same and same_obs and same_cmd),
          "truth %s observed %s commands %s" % (same, same_obs, same_cmd))
    worst = 0.0
    for k in ep1.truth:
        a_, b_ = np.asarray(ep1.truth[k]), np.asarray(ep2.truth[k])
        if a_.dtype == bool or b_.dtype == bool:
            worst = max(worst, float((a_ != b_).sum()))
        else:
            worst = max(worst, float(np.abs(a_ - b_).max()))
    check("determinism: the largest truth difference is exactly zero",
          worst == 0.0, "max |diff| = %r" % worst)

    _, ep_baseline = run_episode(arm="cpg_baseline", seed=SEED,
                                 duration_s=EQUIV_SECONDS)
    manual = plain_cpg_walk(SEED, EQUIV_SECONDS, cfg.body_substeps)
    check("baseline equivalence: the cpg_baseline arm is bit-identical to a plain CPG walk "
          "of the same length (the scheduler does not perturb the baseline)",
          np.array_equal(ep_baseline.truth["thorax_mm"], manual)
          and float(np.abs(ep_baseline.truth["thorax_mm"] - manual).max()) == 0.0,
          "n=%d max |diff| = %r"
          % (manual.shape[0],
             float(np.abs(ep_baseline.truth["thorax_mm"] - manual).max())))
    check("baseline equivalence: the baseline arm observes nothing and decodes no command, "
          "with the reason recorded rather than an empty file",
          all(np.asarray(ep_baseline.observed[k]).size == 0 for k in ep_baseline.observed)
          and all(np.asarray(ep_baseline.commands[k]).size == 0
                  for k in ep_baseline.commands)
          and "NO neural tier" in ep_baseline.meta.get("observed_note", ""),
          ep_baseline.meta.get("observed_note", "")[:80])

    # ------------------------------------------- zero-modulation equivalence
    _, ep_zero = run_episode(arm="neural_modulated", seed=SEED,
                             duration_s=EQUIV_SECONDS, forced_rate_mode="reference")
    check("zero-modulation: with the descending rate forced to its own reference the "
          "command is EXACTLY neutral (speed 1.0, turn 0.0), not merely close",
          np.all(ep_zero.commands["speed_scale"] == 1.0)
          and np.all(ep_zero.commands["turn"] == 0.0),
          "unique speeds %r unique turns %r"
          % (np.unique(ep_zero.commands["speed_scale"]).tolist(),
             np.unique(ep_zero.commands["turn"]).tolist()))
    check("zero-modulation: the neural arm then reproduces the baseline body trajectory "
          "bit for bit (truth arrays identical)",
          np.array_equal(ep_zero.truth["thorax_mm"], ep_baseline.truth["thorax_mm"])
          and np.array_equal(ep_zero.truth["joint_angles_rad"],
                             ep_baseline.truth["joint_angles_rad"])
          and np.array_equal(ep_zero.truth["contact_forces"],
                             ep_baseline.truth["contact_forces"]),
          "max |thorax diff| = %r"
          % float(np.abs(ep_zero.truth["thorax_mm"]
                         - ep_baseline.truth["thorax_mm"]).max()))
    s1.close(); s2.close()

    # ------------------------------------------------------- no lookahead
    src = np.asarray(ep1.commands["source_interval"])
    expect = np.arange(-1, len(src) - 1)
    check("no-lookahead: the command used in interval k was decoded from interval k-1, "
          "for every interval (source_interval == k-1)",
          np.array_equal(src, expect),
          "first five source intervals %r" % src[:5].tolist())
    log = np.asarray(s1.command_log, float)
    check("no-lookahead: what was actually written into the CPG during interval k equals "
          "the recorded command for interval k (one write, before the substeps)",
          log.shape[0] == len(src)
          and np.array_equal(log[:, 0], np.arange(len(src)))
          and np.array_equal(log[:, 1], ep1.commands["speed_scale"])
          and np.array_equal(log[:, 2], ep1.commands["turn"])
          and s1.ledger.total("cpg_intrinsic_freqs") == len(src),
          "%d command writes for %d intervals" % (log.shape[0], len(src)))
    # injected distinctive pattern: the command may move ONLY at the interval AFTER it
    inj_at = 3
    sch_inj = MultirateScheduler(LoopConfig(arm="neural_modulated", seed=SEED,
                                            duration_s=EQUIV_SECONDS, gl_backend=None),
                                 built_tier=BUILT)
    injector = SpikeInjector(sch_inj.tier.advance, sch_inj.tier.n,
                             sch_inj.cfg.neural_substeps, inj_at,
                             sch_inj.tier.readout_local,
                             skip_calls=sch_inj.cfg.calibration_intervals)
    sch_inj.tier.advance = injector
    ep_inj = sch_inj.run()
    r = np.asarray(ep_inj.commands["rate_total_hz"])
    r0 = sch_inj.calibration["reference_rates_hz"]["total_hz"]
    pre = r[:inj_at + 1]
    quiet_before = bool(np.all(np.diff(pre) <= 1e-9))
    jump = float(r[inj_at + 1] - pre[-1])
    check("no-lookahead (injection): with every episode spike silenced except interval %d, "
          "the command stays quiet through interval %d and moves FIRST at interval %d - "
          "i.e. it is decoded from the PREVIOUS completed interval"
          % (inj_at, inj_at, inj_at + 1),
          quiet_before and jump > 0.5 * r0 and r[inj_at + 1] > float(pre.max()) * 1.5
          and injector.episode_calls == len(r),
          "reference %.2f Hz; rate decays %.2f -> %.2f Hz over intervals 0..%d, then "
          "jumps to %.2f Hz at interval %d; episode calls %d == %d intervals"
          % (r0, pre[0], pre[-1], inj_at, r[inj_at + 1], inj_at + 1,
             injector.episode_calls, len(r)))
    sch_inj.close()

    # --------------------------------------------------- truth/observed split
    obs_keys = [k for k in ep1.observed if np.asarray(ep1.observed[k]).size]
    truth_keys = [k for k in ep1.truth if np.asarray(ep1.truth[k]).size]
    shared = set(obs_keys) & set(TRUTH_FIELDS)
    obs_t = np.asarray(ep1.observed["time_s"], float)
    grid = np.arange(obs_t.size) * cfg.dt_command_s
    truth_t = np.asarray(ep1.truth["time_s"], float)
    offset = truth_t - obs_t
    clock_ok = (shared <= {"time_s"} and np.array_equal(obs_t, grid)
                and float(np.ptp(offset)) < 1e-9)
    check("leak: no observed channel shares a name with a truth-only quantity except the "
          "shared clock; that clock is exactly the scheduler's interval grid and differs "
          "from the BODY clock by a CONSTANT (no drift), so it carries no body state",
          clock_ok,
          "shared names=%r; exact grid=%s; body-clock offset %.12f s with peak-to-peak "
          "%.3e s over %d intervals"
          % (sorted(shared), np.array_equal(obs_t, grid), float(offset[0]),
             float(np.ptp(offset)), obs_t.size))
    copies = []
    for ok_ in obs_keys:
        a_ = np.asarray(ep1.observed[ok_], float)
        for tk in truth_keys:
            b_ = np.asarray(ep1.truth[tk], float)
            if b_.shape == a_.shape and b_.size and np.ptp(b_) > 0:
                if np.array_equal(a_, b_):
                    copies.append((ok_, tk))
    check("leak: no observed array is a verbatim copy of a non-constant truth array of "
          "the same shape",
          not copies, "copies=%r" % (copies,))

    # replay: the observed sensory channels must be EXACTLY the declared transduction
    bank = MechanoReceptorBank(BUILT["sensory"], angle_rows, BUILT["n"],
                               LoopConfig().receptor, dt_ms=5.0,
                               ledger=OwnershipLedger(), seed=SEED)
    bank.set_reference_pose(np.asarray(ep1.truth["joint_angles_rad"])[0])
    replayed = {"force_strain_per_leg": [], "angle_strain_per_leg": [],
                "receptor_drive_contact": [], "receptor_drive_proprio": [],
                "current_nA_per_leg": []}
    for k in range(len(ep1.observed["time_s"])):
        obs_like = types.SimpleNamespace(
            contact_forces=np.asarray(ep1.truth["contact_forces"])[k],
            joint_angles_rad=np.asarray(ep1.truth["joint_angles_rad"])[k])
        _cur, drive = bank.transduce(obs_like)
        replayed["force_strain_per_leg"].append(drive["force_strain"])
        replayed["angle_strain_per_leg"].append(drive["angle_strain"])
        replayed["receptor_drive_contact"].append(drive["drive_contact"])
        replayed["receptor_drive_proprio"].append(drive["drive_proprio"])
        replayed["current_nA_per_leg"].append(drive["current_nA_per_leg"])
    replay_ok = all(np.array_equal(np.asarray(replayed[k]), np.asarray(ep1.observed[k]))
                    for k in replayed)
    check("leak: every sensory observed channel is EXACTLY reproducible from truth by the "
          "declared transduction, so no hidden channel exists",
          bool(replay_ok),
          "%d channels replayed" % len(replayed))
    check("leak: every observed channel has a declared provenance entry naming its truth "
          "sources and its transform",
          all(k in OBSERVED_PROVENANCE for k in OBSERVED_FIELDS)
          and all(isinstance(OBSERVED_PROVENANCE[k][1], str)
                  for k in OBSERVED_FIELDS),
          "%d declared channels" % len(OBSERVED_FIELDS))
    check("leak: the body's warmup clock offset is recorded in the metadata rather than "
          "silently absorbed",
          abs(float(ep1.meta.get("body_clock_offset_s", -1.0)) - 0.05) < 1e-6,
          "body_clock_offset_s=%r" % ep1.meta.get("body_clock_offset_s"))
    check("leak: the truth/observed split is documented in the episode metadata",
          "truth_observed_split" in ep1.meta
          and "training step" in ep1.meta["truth_observed_split"],
          ep1.meta.get("truth_observed_split", "")[:80])

    # ---------------------------------------------------------- ownership
    own = s1.check_ownership()
    check("ownership: exactly one joint-target writer per interval (the command adapter) "
          "and exactly one body stepper per substep (the scheduler)",
          own["ok"] is True
          and own["cpg_intrinsic_freqs_writers"] == ("command_adapter",)
          and own["cpg_intrinsic_freqs_writes"] == own["expected_cpg_writes"]
          and own["body_step_writers"] == ("scheduler",)
          and own["body_steps"] == own["expected_body_steps"],
          json.dumps(own))
    check("ownership: the neural tier exposes no way to write a joint target",
          own["neural_tier_has_no_body_reference"] is True
          and all(not hasattr(s1.tier, a) for a in
                  ("set_command", "write_joints", "body", "backend", "cpg", "sim")),
          "attributes checked")
    refused_writer = refuse(lambda: s1.ledger.write("cpg_intrinsic_freqs", "neural"))
    refused_resource = refuse(lambda: s1.ledger.write("joint_angles", "neural"))
    check("ownership: the ledger REFUSES a foreign writer, and refuses an unknown resource "
          "(the guard is live, not decorative)",
          refused_writer and refused_resource,
          "foreign writer refused=%s unknown resource refused=%s"
          % (refused_writer, refused_resource))
    check("ownership: the body has exactly one attached action source, and in the baseline "
          "arm it is FlyGym's own CPG object",
          s1.body._controller is s1.cpg
          and getattr(s1.body, "using_cpg_baseline", None) is False
          and ep_baseline.meta["arm_description"].startswith("FlyGym tripod CPG, unchanged"),
          "modulated source=%s baseline=%r"
          % (type(s1.body._controller).__name__,
             ep_baseline.meta["arm_description"][:40]))

    # ------------------------------------------------------- rate window
    probe_tier = NeuralTier(BUILT, TierConfig(), OwnershipLedger())
    fake = np.zeros((probe_tier.cfg.dt_ms and 10, probe_tier.n), dtype=bool)
    fake[0, probe_tier.readout_local[0]] = True
    dec = CommandDecoder(LoopConfig().decode, dt_ms_command=5.0, dt_ms_neural=0.5,
                         ledger=OwnershipLedger(), tier=probe_tier)
    want = 1.0 / (probe_tier.readout_local.size * 0.005)
    got_tier = probe_tier.group_rates_hz(fake)["total_hz"]
    got_dec = dec.rate_hz(fake)["total_hz"]
    check("rate window: one spike in a 5 ms interval gives exactly 1/(N*0.005) Hz in BOTH "
          "the tier and the decoder (guards the clock-conflation factor-of-ten bug)",
          abs(got_tier - want) < 1e-9 and abs(got_dec - want) < 1e-9,
          "want %.6f Hz, tier %.6f, decoder %.6f" % (want, got_tier, got_dec))
    check("rate window: decoding before calibration is refused",
          refuse(lambda: dec.decode(fake, source_interval=0)),
          "refused")

    # ---------------------------------------------------------- causality
    modes = ["reference", "high", "low", "turn_left", "turn_right"]
    caus = {}
    for mode in modes:
        _s, e = run_episode(arm="neural_modulated", seed=SEED,
                            duration_s=CAUSAL_SECONDS, forced_rate_mode=mode)
        caus[mode] = e
        _s.close()
    cmd_ok = (caus["high"].metrics["command_speed_scale_mean"] > 1.0
              > caus["low"].metrics["command_speed_scale_mean"]
              and caus["turn_left"].metrics["command_turn_mean"] > 0.2
              and caus["turn_right"].metrics["command_turn_mean"] < -0.2
              and abs(caus["reference"].metrics["command_speed_scale_mean"] - 1.0) < 1e-12)
    check("causality: the forced descending rates produce the intended commands "
          "(arithmetic: high > 1 > low, mirrored turn commands)", cmd_ok,
          "s(high)=%.4f s(ref)=%.4f s(low)=%.4f turn(L)=%.4f turn(R)=%.4f"
          % (caus["high"].metrics["command_speed_scale_mean"],
             caus["reference"].metrics["command_speed_scale_mean"],
             caus["low"].metrics["command_speed_scale_mean"],
             caus["turn_left"].metrics["command_turn_mean"],
             caus["turn_right"].metrics["command_turn_mean"]))
    v_ref = caus["reference"].metrics["mean_speed_mm_s"]
    v_hi = caus["high"].metrics["mean_speed_mm_s"]
    v_lo = caus["low"].metrics["mean_speed_mm_s"]
    check("causality (prediction P1, pre-registered): RAISING the descending rate raises "
          "the walking speed, and lowering it lowers the speed",
          (v_hi > v_ref + 0.5) and (v_lo < v_ref - 0.5),
          "speed: low %.2f < ref %.2f < high %.2f mm/s (differences %+.2f / %+.2f)"
          % (v_lo, v_ref, v_hi, v_hi - v_ref, v_lo - v_ref))
    y_l = caus["turn_left"].metrics["heading_change_deg"]
    y_r = caus["turn_right"].metrics["heading_change_deg"]
    y_0 = caus["reference"].metrics["heading_change_deg"]
    check("causality (prediction P2, pre-registered): the mirrored turn commands steer the "
          "body in measurably different directions, with the left-faster command yawing "
          "less (to the right) than the right-faster one",
          abs(y_l - y_r) > 10.0 and (y_l - y_r) < 0.0,
          "heading change: turn_left %+.2f deg, neutral %+.2f deg, turn_right %+.2f deg; "
          "difference %+.2f deg" % (y_l, y_0, y_r, y_l - y_r))
    check("causality: the forced runs differ from each other in the BODY, so the command "
          "channel is not inert",
          float(np.abs(caus["high"].truth["thorax_mm"]
                       - caus["low"].truth["thorax_mm"]).max()) > 0.1,
          "max thorax difference high vs low = %.3f mm"
          % float(np.abs(caus["high"].truth["thorax_mm"]
                         - caus["low"].truth["thorax_mm"]).max()))

    # ------------------------------------------------------ invalid input
    bad_cfgs = [
        lambda: LoopConfig(arm="bogus").validate(),
        lambda: LoopConfig(seed=-1).validate(),
        lambda: LoopConfig(seed=1.5).validate(),
        lambda: LoopConfig(arm="cpg_baseline", forced_rate_mode="high").validate(),
        lambda: LoopConfig(forced_rate_mode="sideways").validate(),
        lambda: LoopConfig(duration_s=0.0).validate(),
        lambda: LoopConfig(dt_neural_s=4e-4).validate(),
        lambda: LoopConfig(calibration_ms=7.0).validate(),
        lambda: TierConfig(max_neurons=0).validate(),
        lambda: TierConfig(max_neurons=20001).validate(),
        lambda: TierConfig(max_neurons="many").validate(),
        lambda: TierConfig(growth_iterations=9).validate(),
        lambda: TierConfig(dt_ms=0.3).validate(),
        lambda: TierConfig(partner_min_synapses=-1.0).validate(),
        lambda: ReceptorConfig(reference_force=-1.0).validate(),
        lambda: ReceptorConfig(receptor_tau_ms=0.0).validate(),
        lambda: ReceptorConfig(current_nA_per_drive=-0.1).validate(),
        lambda: DecodeConfig(rate_scale_hz=0.0).validate(),
        lambda: DecodeConfig(s_min=1.5, s_max=1.0).validate(),
        lambda: DecodeConfig(k_turn=-1.0).validate(),
        lambda: heading_yaw_rad([0.0, 0.0, 0.0]),
        lambda: heading_yaw_rad([0.0, 0.0, 0.0, 0.0]),
        lambda: build_neural_tier(TierConfig(max_neurons=10)),
    ]
    refused_n = sum(1 for fn in bad_cfgs if refuse(fn))
    check("invalid input: every malformed config, clock, quaternion and over-tight tier "
          "bound is refused rather than coerced",
          refused_n == len(bad_cfgs), "%d/%d refused" % (refused_n, len(bad_cfgs)))

    # runtime guards that need live objects
    live = [lambda: s1.cpg.set_command(0.0, 0.0),
            lambda: s1.cpg.set_command(1.0, 1.5),
            lambda: s1.cpg.set_command(float("nan"), 0.0),
            lambda: s1.tier.advance(10, np.zeros(s1.tier.n + 1)),
            lambda: s1.tier.advance(0, np.zeros(s1.tier.n)),
            lambda: s1.decoder.decode(np.zeros((10, s1.tier.n), bool), source_interval=1,
                                      forced_rate={"total_hz": -1.0, "left_hz": 0.0,
                                                   "right_hz": 0.0}),
            lambda: s1.decoder.decode(np.zeros((10, s1.tier.n), bool),
                                      source_interval="three"),
            lambda: s1.decoder.decode(np.zeros((10, s1.tier.n), bool), source_interval=1,
                                      forced_rate={"total_hz": 1.0}),
            lambda: MechanoReceptorBank({"lf": np.array([], dtype=int)},
                                        angle_rows, s1.tier.n, ReceptorConfig(),
                                        dt_ms=5.0, ledger=OwnershipLedger()),
            lambda: MechanoReceptorBank(BUILT["sensory"], angle_rows, 5, ReceptorConfig(),
                                        dt_ms=5.0, ledger=OwnershipLedger()),
            lambda: leg_rows_from_dof_order(["not-a-dof"] * 42),
            lambda: leg_rows_from_dof_order(dof_order, model=None)["angle_rows"][
                "lf"].max(),
            lambda: MechanoReceptorBank(BUILT["sensory"],
                                        {k: v for k, v in angle_rows.items()},
                                        BUILT["n"], ReceptorConfig(), dt_ms=5.0,
                                        ledger=OwnershipLedger()).set_reference_pose(
                np.zeros(41)),
            lambda: sensor_matrix_row_map({"lf": np.array([1, 2]), "lm": np.array([2, 3]),
                                           "lh": np.array([4]), "rf": np.array([5]),
                                           "rm": np.array([6]), "rh": np.array([7])},
                                          angle_rows),
            lambda: MechanoReceptorBank(BUILT["sensory"], rows, BUILT["n"],
                                        ReceptorConfig(), dt_ms=5.0,
                                        ledger=OwnershipLedger()).transduce(
                types.SimpleNamespace(contact_forces=np.zeros((6, 3)),
                                      joint_angles_rad=np.zeros(66))),
            ]
    refused_live = sum(1 for fn in live if refuse(fn))
    check("invalid input: the runtime guards refuse a bad command, a wrong-length current, "
          "an uncalibrated decode and a malformed receptor mapping",
          refused_live == len(live), "%d/%d refused" % (refused_live, len(live)))

    # -------------------------------------------------------------- honesty
    ep_honesty = list(ep1.meta.get("honesty", []))
    report_path = os.path.join(OUT, "loop_report.json")
    report = None
    if os.path.isfile(report_path):
        with open(report_path) as fh:
            report = json.load(fh)
    in_meta = all(h in ep_honesty for h in HONESTY)
    in_report = bool(report is not None
                     and all(h in report.get("honesty", []) for h in HONESTY))
    key_phrases = ["ENGINEERING BASELINE", "ASSUMED", "ILLUSTRATIVE", "SELECTION",
                   "UNSIGNED", "MILLIMETRES", "NOT evidence about experience",
                   "NO VISION"]
    phrases_ok = all(any(p in h for h in HONESTY) for p in key_phrases)
    check("honesty: every honesty string is present in the episode metadata, covers every "
          "required phrase, and is present in the human-readable report when it exists",
          in_meta and phrases_ok and (in_report or report is None),
          "metadata=%s report=%s (%s) phrases=%s"
          % (in_meta, in_report,
             "present" if report is not None else
             "loop_report.json not present yet: run run_embodied_loop.py first",
             phrases_ok))
    if report is not None:
        fig = report.get("figures", {}).get("figure")
        check("honesty: the report names its figure and video artefacts, and the figure "
              "exists on disk",
              bool(fig) and os.path.isfile(fig),
              "figure=%s" % fig)
        check("honesty: the report records wall clock and peak RAM",
              isinstance(report.get("wall_seconds_total"), (int, float))
              and report.get("peak_ram_mb", 0) > 0,
              "wall %.1f s, peak RAM %.0f MB" % (report.get("wall_seconds_total", -1),
                                                 report.get("peak_ram_mb", -1)))
        check("honesty: the report leaves its limitations visible (non-empty list with the "
              "CPG, the assumed map and the selection all named)",
              len(report.get("limitations_left_in_place", [])) >= 5
              and any("CPG still generates the gait" in s
                      for s in report["limitations_left_in_place"])
              and any("ASSUMED" in s for s in report["limitations_left_in_place"]),
              "%d limitations" % len(report.get("limitations_left_in_place", [])))
        check("honesty: the report's ground-visibility check found the ground in the "
              "rendered video frames",
              (report.get("ground_visibility", {})
               .get("fraction_with_textured_bottom_band") or 0) >= 0.99,
              json.dumps(report.get("ground_visibility", {})))

    s1.close()

    n_pass = sum(1 for r in RESULTS if r["ok"])
    result = {
        "what_these_tests_establish": [
            "the three clocks are explicit and refused when misaligned",
            "the bounded BANC tier is a SELECTION with reported coverage, UNSIGNED edges, "
            "and a silent-without-input dynamical property measured here",
            "the neural arm is bit-reproducible; the baseline arm is bit-identical to a "
            "plain CPG walk; a neutral command reproduces the baseline exactly",
            "the command used in an interval comes from the spike train of the PREVIOUS "
            "completed interval (structural check plus an injected spike pattern)",
            "no observed channel is a truth-only quantity and every observed channel is "
            "exactly reproducible from truth through its declared transduction",
            "exactly one writer per resource; the neural tier cannot write joints; the "
            "guard refuses a foreign writer",
            "a forced descending rate changes walking speed and heading measurably, in "
            "the pre-registered directions",
            "malformed input is refused",
        ],
        "what_they_do_not_establish": [
            "that the connectome generates the gait: FlyGym's engineering CPG does, in "
            "BOTH arms",
            "that the assumed map from firing rate to CPG knobs, or its gains, are "
            "correct: they are ASSUMED",
            "that the receptor kinetics or the force/angle scales are Drosophila's: they "
            "are ILLUSTRATIVE",
            "that the bounded selection stands in for a brain, or that the readout's "
            "left/right proxy is a real motor laterality",
            "anything about perception, experience or consciousness",
        ],
        "config": {"seed": SEED, "equivalence_seconds": EQUIV_SECONDS,
                   "causal_seconds": CAUSAL_SECONDS,
                   "tier": json.loads(json.dumps(BUILT["coverage"], default=str))},
        "n_checks": len(RESULTS), "n_passed": n_pass,
        "n_failed": len(RESULTS) - n_pass,
        "passed": bool(n_pass == len(RESULTS)),
        "wall_seconds": time.perf_counter() - t0,
        "results": RESULTS,
    }
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "loop_selftest.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print("\n%d/%d checks passed in %.1fs -> %s"
          % (n_pass, len(RESULTS), result["wall_seconds"], path))

    # keep the report and the tests consistent: only the 'selftest' summary field is
    # touched, so the two artefacts cannot disagree about whether the tests passed.
    if report is not None:
        report["selftest"] = {"n_checks": result["n_checks"],
                              "n_passed": result["n_passed"],
                              "n_failed": result["n_failed"],
                              "passed": result["passed"],
                              "json": path}
        with open(report_path, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print("updated the selftest summary in %s" % report_path)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
