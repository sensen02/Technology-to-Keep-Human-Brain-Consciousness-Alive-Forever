"""Does the neural loop's advantage over its cut controls GROW with disturbance?

WHY THIS EXPERIMENT EXISTS
--------------------------
The previous step measured, without any perturbation: normal 24.0 ms to 90% height vs
spike-output silence 14.9 ms.  A 9.1 ms difference is consistent with two very different
stories:

  (a) FEEDBACK: body-dependent neural commands might change the disturbance response;
      whether they increase this particular advantage is the hypothesis being tested;
  (b) FIXED OFFSET: the emitted command has some average effect that merely delays the fall
      (e.g. it holds a joint in an already-doomed configuration), so the difference is roughly
      CONSTANT and unrelated to the disturbance.

The experiment measures the SAME comparison at several external perturbation strengths.
The stories above are hypotheses, not exhaustive alternatives or necessary predictions:
feedback need not produce a growing advantage, and open-loop effects can depend on disturbance.
The response shape alone cannot distinguish these mechanisms or establish standing.

PRE-REGISTERED
--------------
  H19  The normal-minus-silence escape-time advantage INCREASES with perturbation strength.
       Supported if the advantage is monotone non-decreasing across the tested fractions and the
       strongest perturbation's advantage exceeds the unperturbed advantage by more than 20%.
  FALSIFIER: an interpretable flat or decreasing advantage does not support H19; it does
       NOT exclude feedback or establish a fixed offset. Escapes at/before push onset and
       unobserved escapes make this endpoint comparison non-informative.

WHAT THE PERTURBATION IS
------------------------
An experimenter-applied world-frame force on the thorax, expressed as a fraction of THE MODEL'S
OWN total weight (mass_sum * |gravity|, measured at runtime - no SI claim).  It is applied
identically in every arm, never targets a motor pool, and is cleared on every body substep.

Read-only on the canonical asset; guarded single-flight memory budget; no CPG/PD/baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

import engine.canonical_physics as cp  # noqa: E402
import engine.memory_guard as mg  # noqa: E402

OUT = HERE / "outputs/neural_muscle_closure"
ARMS = ("normal", "spike_output_silence", "sensory_cut")


def _run_arm(clo, model, seated, joints, geom_leg, built, manifest, groups, arm, duration,
             push, stride=100):
    """Run one arm and additionally record whether the PERTURBATION actually moved the body.

    MEASURED CONCERN (this is why the extra channels exist): the first version of this
    experiment reported the passive arms at an identical 14.9 ms for every perturbation
    strength, including 0.5x body weight.  A flat response is exactly what an intervention that
    was never applied would also look like, so the run must carry its own evidence that the
    force reached the body - otherwise a null result cannot be told apart from a broken probe.
    """
    import mujoco
    summary, traj, ctrl, spikes = clo.episode(
        model, seated, joints, geom_leg, built, manifest, groups, arm,
        duration, push=push, stride=stride)
    out = {
        "escape_time_s": summary["escape_time_s"],
        "final_height_model": summary["final_height_model"],
        "initial_height_model": summary["initial_height_model"],
        "commands_applied_nonzero_intervals": int(np.count_nonzero(ctrl.sum(axis=1))),
        "max_control_sum": float(ctrl.sum(axis=1).max()) if ctrl.size else 0.0,
        "final_height_fraction": (summary["final_height_model"]
                                  / summary["initial_height_model"]),
        "push_intervals_applied": int(sum(1 for r in summary["records"]
                                          if r.get("external_push_applied"))),
    }
    # lateral pose channels: the push is world +x, so x displacement is the probe's own witness
    nq = model.nq
    qpos = traj[:, :nq]
    out["final_x_model"] = float(qpos[-1, 0])
    out["final_y_model"] = float(qpos[-1, 1])
    out["final_quat_wxyz"] = [float(v) for v in qpos[-1, 3:7]]
    out["max_abs_x_model"] = float(np.max(np.abs(qpos[:, 0])))
    out["x_span_model"] = float(qpos[:, 0].max() - qpos[:, 0].min())
    return out


def endpoint_eligibility(escape_time_s, push_start_s):
    """Conservative temporal eligibility, not evidence that feedback was effective.

    An unobserved event is right-censored. An event at/before onset is already
    determined before the force can affect it (also checked in the no-push arm).
    """
    if not np.isfinite(push_start_s) or push_start_s < 0:
        raise ValueError("push onset must be finite and nonnegative")
    if escape_time_s is None:
        reason = "escape_not_observed_right_censored"
    elif not np.isfinite(escape_time_s) or escape_time_s < 0:
        reason = "invalid_escape_time"
    elif escape_time_s <= push_start_s + 1e-12:
        reason = "escape_at_or_before_push_onset"
    else:
        reason = "observed_after_push_onset"
    return {"eligible": reason == "observed_after_push_onset", "reason": reason}


def infer_h19(rows, push_start_s, *, push_was_applied, lateral_changed):
    """Evaluate only the registered dose-response shape; never exclude feedback."""
    fractions = [r["weight_fraction"] for r in rows]
    design_valid = (len(rows) >= 2 and fractions[0] == 0
                    and all(np.isfinite(f) for f in fractions)
                    and all(b > a for a, b in zip(fractions, fractions[1:])))
    endpoints = [{"weight_fraction": r["weight_fraction"], "arms": {
        a: endpoint_eligibility(r["arms"][a]["escape_time_s"], push_start_s)
        for a in ("normal", "spike_output_silence")}} for r in rows]
    eligible = bool(endpoints) and all(
        e["eligible"] for row in endpoints for e in row["arms"].values())
    advantages = [(None if any(r["arms"][a]["escape_time_s"] is None
                              for a in ("normal", "spike_output_silence")) else
                   1000 * (r["arms"]["normal"]["escape_time_s"]
                           - r["arms"]["spike_output_silence"]["escape_time_s"]))
                  for r in rows]
    finite = bool(advantages) and all(a is not None and np.isfinite(a) for a in advantages)
    monotone = finite and all(b >= a - 1e-12 for a, b in zip(advantages, advantages[1:]))
    ratio = (advantages[-1] / advantages[0]
             if finite and advantages[0] > 0 else None)
    interpretable = bool(design_valid and eligible and finite and ratio is not None
                         and push_was_applied and lateral_changed)
    supported = bool(monotone and ratio is not None and ratio > 1.2) if interpretable else None
    return {"interpretation_scope": "dose-response shape only; neither proves nor excludes feedback",
            "design_valid": bool(design_valid), "endpoint_eligibility": endpoints,
            "endpoint_comparison_eligible": eligible, "interpretable": interpretable,
            "monotone_non_decreasing": bool(monotone), "advantage_gain_ratio": ratio,
            "supported": supported,
            "verdict": "not_interpretable" if not interpretable else
                       "H19_supported" if supported else "H19_not_supported"}


def main(guard_report=None) -> None:
    import mujoco
    import run_neural_muscle_closure as clo
    from engine.embodied.adapters import build_neural_tier, TierConfig
    from engine.embodied.motor_output import build_motor_manifest
    from engine.embodied.sensory_channels import channel_index_map

    parser = argparse.ArgumentParser()
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.5])
    parser.add_argument("--duration", type=float, default=0.10)
    parser.add_argument("--push-at-ms", type=float, default=15.0)
    parser.add_argument("--push-duration-ms", type=float, default=5.0)
    args = parser.parse_args()
    if any(abs(f) > 5.0 for f in args.fractions) or not args.fractions:
        raise ValueError("declared perturbation range is 0..5x body weight")
    OUT.mkdir(parents=True, exist_ok=True)

    digest = cp.sha256_file(cp.CANONICAL_XML)
    assert digest == cp.CANONICAL_XML_SHA256, digest

    model, seated, gate, joints, geom_leg = clo.prepare_body()
    built = build_neural_tier(TierConfig(background_mean_nA=0, background_sd_nA=0))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    pools = HERE / "outputs/neural_rebuild/neural_rebuild_11_mn_pools.json"
    manifest = build_motor_manifest(json.loads(pools.read_text()), built["tier"]["root_ids"], names)
    groups, _ = channel_index_map(int(built["n"]), built["tier"]["root_ids"])
    total_weight = float(model.body_mass.sum() * abs(model.opt.gravity[2]))

    t0 = time.time()
    rows = []
    for frac in args.fractions:
        push = None
        if frac:
            push = {"start_s": args.push_at_ms/1000.0,
                    "end_s": (args.push_at_ms+args.push_duration_ms)/1000.0,
                    "weight_fraction": float(frac), "total_weight_model": total_weight,
                    "force_model": [float(frac*total_weight), 0.0, 0.0]}
        per_arm = {}
        for arm in ARMS:
            per_arm[arm] = _run_arm(clo, model, seated, joints, geom_leg, built, manifest,
                                    groups, arm, args.duration, push)
        esc = {a: per_arm[a]["escape_time_s"] for a in ARMS}
        advantage_ms = (None if esc["normal"] is None or esc["spike_output_silence"] is None
                        else 1000.0*(esc["normal"]-esc["spike_output_silence"]))
        rows.append({"weight_fraction": frac,
                     "push_force_model": None if push is None else push["force_model"],
                     "arms": per_arm, "advantage_normal_minus_silence_ms": advantage_ms})
        print(f"fraction {frac:>4}: normal {esc['normal']} silence {esc['spike_output_silence']} "
              f"sensory_cut {esc['sensory_cut']} -> advantage {advantage_ms}", flush=True)

    base = rows[0]["advantage_normal_minus_silence_ms"]
    strongest = rows[-1]["advantage_normal_minus_silence_ms"]

    # PROBE EFFICACY: did the declared force actually reach the body?
    #
    # The first version of this check demanded that the lateral span grow MONOTONICALLY with the
    # declared fraction, and it reported "force reached the body = False" - which was WRONG and
    # misleading, because the run's own records show the force was applied in exactly the
    # intended intervals for every non-zero fraction.  A strict monotonicity demand is the wrong
    # test: a brief lateral impulse can reduce the lateral span at low strength (it changes the
    # contact order) while still being applied.  The check is therefore split into what can
    # actually be evidenced: (1) the force window fired; (2) some measured channel, lateral span
    # OR escape time, differs from the no-push case.  It also records, separately, that the
    # PASSIVE arms' escape time may precede the push. Later lateral change cannot rescue
    # a pre-push endpoint: infer_h19 separately checks temporal eligibility.
    probe = {}
    for arm in ARMS:
        xs = [r["arms"][arm]["x_span_model"] for r in rows]
        esc = [r["arms"][arm]["escape_time_s"] for r in rows]
        applied = [r["arms"][arm]["push_intervals_applied"] for r in rows]
        base_x, base_esc = xs[0], esc[0]
        probe[arm] = {
            "x_span_model_by_fraction": xs,
            "escape_time_s_by_fraction": esc,
            "push_intervals_applied_by_fraction": applied,
            "push_window_fired_for_every_nonzero_fraction": bool(
                applied[0] == 0 and all(a > 0 for a in applied[1:])),
            "x_span_changed_from_no_push": bool(any(abs(x-base_x) > 1e-12 for x in xs[1:])),
            "x_span_max_relative_change": (None if base_x == 0 else
                                           max(abs(x-base_x) for x in xs[1:])/abs(base_x)),
            "escape_time_changed_from_no_push": bool(
                base_esc is not None and any(e is not None and abs(e-base_esc) > 1e-12
                                             for e in esc[1:])),
        }
    push_was_applied = all(p["push_window_fired_for_every_nonzero_fraction"] for p in probe.values())
    lateral_changed = any(p["x_span_changed_from_no_push"] for p in probe.values())
    escape_changed = any(p["escape_time_changed_from_no_push"] for p in probe.values())
    # is the ESCAPE-TIME metric itself sensitive to the probe, in the arms where the loop is cut?
    passive_metric_sensitive = any(
        probe[a]["escape_time_changed_from_no_push"] for a in ("spike_output_silence",
                                                               "sensory_cut"))
    probe_summary = {
        "push_force_was_applied": push_was_applied,
        "lateral_motion_changed_from_no_push": lateral_changed,
        "escape_time_changed_from_no_push_in_any_arm": escape_changed,
        "escape_time_metric_sensitive_in_passive_arms": passive_metric_sensitive,
        "meaning": "the push is world +x on the thorax; a null H19 is only interpretable if the "
                   "force demonstrably reached the body, and the H19 verdict must be read against "
                   "whether the escape-time metric responds in the passive arms at all",
    }

    inference = infer_h19(rows, args.push_at_ms/1000.0,
                          push_was_applied=push_was_applied, lateral_changed=lateral_changed)
    guard = guard_report("completed") if guard_report else None
    cp.write_result(
        OUT / "neural_muscle_perturbation.json",
        question=("Does the neural loop's escape-time advantage over its cut controls grow with "
                  "external perturbation strength, when endpoints follow push onset?"),
        status="PASS",
        measured={
            "status_meaning": "PASS = the measurement and its internal checks ran; it is a "
                              "statement about the SHAPE of the response, not about standing",
            "fractions_of_body_weight": list(args.fractions),
            "total_weight_model_units": total_weight,
            "duration_s": args.duration,
            "push_window_ms": [args.push_at_ms, args.push_at_ms+args.push_duration_ms],
            "arms": list(ARMS),
            "rows": rows,
            "baseline_advantage_ms": base,
            "strongest_advantage_ms": strongest,
            "advantage_gain_ratio": inference["advantage_gain_ratio"],
            "monotone_non_decreasing": inference["monotone_non_decreasing"],
            "probe_efficacy": {"by_arm": probe, "summary": probe_summary},
            "h19_interpretable": inference["interpretable"],
            "h19_inference": inference,
            "predictions": {"H19": {
                "supported": inference["supported"],
                "statement": "the normal-minus-silence advantage grows with perturbation strength",
                "falsifier": "an interpretable flat or decreasing advantage does not support H19; "
                             "it neither excludes feedback nor establishes a fixed offset"}}, 
            "gate": gate,
            "memory_guard": guard,
            "wall_s": time.time()-t0,
        },
        provenance={
            "code": str(Path(__file__).resolve()),
            "interpreter": sys.executable,
            "canonical_xml_sha256": digest,
            "canonical_xml_untouched": True,
            "pool_result_sha256": cp.sha256_file(pools),
            "harness": "reuses run_neural_muscle_closure.episode() so the clock, the sensory "
                       "channels and the motor translation are the SAME code as the closure run",
            "perturbation": "world-frame force on Thorax = fraction * (mass_sum * |gravity|), "
                            "measured at runtime, applied identically in every arm",
            "initialization": "gated round-37 pose, zero velocity, zero activation, no LP bias",
        },
        units={"force": "model force units (mass_unit*length_unit/s^2), SI uncalibrated",
               "time": "s", "fraction": "dimensionless ratio of this model's own weight"},
        illustrative_vs_measured="Escape times and heights are simulator measurements; the "
                                 "fractions, gains and translation parameters are declared",
        what_this_does_not_support=[
            "Pre-push escapes and unobserved escapes make the endpoint comparison non-informative; "
            "later lateral motion does not repair this. Flat dose response does not exclude feedback.",
            "A growing advantage would show the difference is disturbance-dependent; it would "
            "NOT show the loop stabilises the animal, and both arms still fall.",
            "Spike-output silence cuts emitted spikes, not network state.",
            "Sensory cut removes body feedback and the artificial touch together.",
            "One pose, one push axis, one push window; no generalisation to perturbations.",
            "No SI mass/force calibration, so the force magnitude cannot be compared to a "
            "measured fly."
        ],
    )
    print("wrote", OUT / "neural_muscle_perturbation.json")


if __name__ == "__main__":
    with mg.guarded_run("neural_muscle_perturbation",
                        need_mb=mg.BUDGETS["neural_muscle_closure"]["need_mb"],
                        cap_mb=mg.BUDGETS["neural_muscle_closure"]["cap_mb"]) as _rep:
        main(guard_report=_rep)
