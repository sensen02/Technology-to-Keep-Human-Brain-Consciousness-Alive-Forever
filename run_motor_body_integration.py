"""Bounded first frozen MN translator/body integration: manual research, not a product.
No training, body-state decoding, controller, pose repair, or closure.main invocation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
DURATION = .55
SEED = 0
# Diagnostic sensitivity is separate from receipt-validated trained integration.
LIMITATION = ("Scalar/contact preflight is not full geometry or static support qualification. "
              "No pose repair, seating or clipping for explicit qpos. Legacy prepare_body uses "
              "historical round37 pose and seating; invalid initial states reject before episodes. "
              "Units NOT_ESTABLISHED. INTERFACE_TEST is synthetic-label fitting, "
              "NOT behavioral training or biological calibration.")


def save_json(path, obj):
    def encode(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj, indent=2, default=encode, allow_nan=False) + "\n")


class BodyObserver:
    """Copies observational scalars only, after each body-substep mj_forward."""
    def __init__(self, model, geom_leg):
        self.geom_leg = geom_leg
        self.thorax = model.body("Thorax").id
        self.rows = []

    def __call__(self, model, data):
        import mujoco
        foot_normal = 0.
        nonfoot = 0
        for ci in range(data.ncon):
            c = data.contact[ci]
            a, b = int(c.geom1), int(c.geom2)
            planes = [g for g in (a, b) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
            if len(planes) != 1:
                continue
            other = b if a == planes[0] else a
            if other in self.geom_leg:
                f = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, f)
                foot_normal += float(f[0])
            elif c.efc_address >= 0:
                nonfoot += 1
        rotation = data.xmat[self.thorax].reshape(3, 3)
        tilt = float(np.arccos(np.clip(rotation[2, 2], -1., 1.)))
        self.rows.append((float(data.time), foot_normal, nonfoot,
                          float(data.xpos[self.thorax, 2]), tilt))

    def array(self):
        return np.asarray(self.rows, dtype=float).reshape(-1, 5)


def support_summary(samples, weight, h0, dt):
    # Tick selection robust to simulator accumulation; recorded time remains actual d.time.
    ticks = np.rint(samples[:, 0] / dt).astype(int)
    chosen = samples[(ticks > round(.05 / dt)) & (ticks <= round(.55 / dt))]
    if len(chosen) != round(.5 / dt):
        raise ValueError(f"incomplete 50–550ms endpoint window: {len(chosen)} samples")
    ratio = float(chosen[:, 1].mean() / weight)
    nonfoot_duration = float(np.count_nonzero(chosen[:, 2] > 0) * dt)
    hmin = float(chosen[:, 3].min())
    tilt = float(chosen[:, 4].max())
    gates = dict(mean_foot_normal_over_model_weight=ratio >= .8,
                 zero_effective_nonfoot_duration=nonfoot_duration == 0.,
                 minimum_height_at_least_085_initial=hmin >= .85*h0,
                 maximum_tilt_at_most_035_rad=tilt <= .35)
    return dict(window_s=[.05, .55], endpoint_count=len(chosen),
                interval_duration_s=len(chosen)*dt,
                convention="Right endpoint observations for DT_BODY intervals (50,550] ms; discrete integration evidence, NOT continuous physics proof.",
                mean_foot_normal_over_model_weight=ratio,
                effective_nonfoot_duration_s=nonfoot_duration,
                root_thorax_z_min_model=hmin, root_thorax_z_initial_model=h0,
                root_thorax_z_min_over_initial=hmin/h0, tilt_max_rad=tilt,
                gates=gates, pass_all=all(gates.values()))


def plot_results(output, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for arm in summaries.get("episodes", {}):
        path = output / arm / "arrays.npz"
        if not path.exists():
            continue
        with np.load(path) as z:
            s = z["body_substep_observations"]
            axes[0, 0].plot(s[:, 0], s[:, 1]/summaries["model_weight"], label=arm)
            axes[0, 1].plot(s[:, 0], s[:, 2], label=arm)
            axes[1, 0].plot(s[:, 0], s[:, 3], label=arm)
            axes[1, 1].plot(s[:, 0], s[:, 4], label=arm)
    labels = ("Tarsus-plane normal / model weight", "Effective nonfoot-plane contacts",
              "Thorax root z [model units]", "Thorax vertical tilt [rad]")
    for ax, label in zip(axes.flat, labels):
        ax.set_ylabel(label)
        ax.axvspan(.05, .55, alpha=.08, color="gray")
        ax.grid(alpha=.2)
        ax.set_xlabel("Actual simulator time [s]")
    if summaries.get("episodes"):
        axes[0, 0].legend(fontsize=8)
    else:
        axes[0, 0].text(.05, .5, "No episodes completed: see runtime failure", transform=axes[0, 0].transAxes)
    mode = summaries.get('mode', 'trained')
    initial = 'explicit saved qpos' if summaries.get('initial_state', {}).get('path') else 'legacy round37 entry'
    fig.suptitle(f"Bounded {mode} body integration — discrete endpoint evidence\nManual research; units NOT_ESTABLISHED; {initial}; not geometry-qualified")
    fig.tight_layout()
    fig.savefig(output / "body_integration.png", dpi=150)
    plt.close(fig)


def require_legal_initial_pose(gate):
    """No repair, clipping or integration is allowed after a failed initial gate."""
    limits = gate.get('initial_scalar_joint_ranges', {})
    if limits.get('within_declared_scalar_ranges') is not True:
        raise ValueError('INITIAL_POSE_REJECTED: scalar joint ranges invalid or missing')
    if gate.get('tarsus_only') is not True:
        raise ValueError('INITIAL_POSE_REJECTED: contact gate invalid or missing')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true', help='Validate initial state without neural/body integration')
    parser.add_argument('--initial-state', type=Path, help='Explicit saved NPZ qpos without seating; omission preserves legacy round37 entry')
    parser.add_argument('--sensitivity-diagnostic', action='store_true',
                        help='Exactly four fixed arms, no training or receipt: default, half weights, output silence, default repeat')
    parser.add_argument('--motor-translator', type=Path,
                        help='Actually trained frozen artifact; required with receipt outside diagnostic mode.')
    args = parser.parse_args()
    if not args.sensitivity_diagnostic and args.motor_translator is None:
        parser.error('--motor-translator is required outside --sensitivity-diagnostic')
    if args.sensitivity_diagnostic and args.motor_translator is not None:
        parser.error('--motor-translator cannot select diagnostic weights; diagnostic arms are fixed')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError("Refusing to rerun or overwrite an existing experiment")
    if not (output / "PREREGISTRATION.md").exists():
        raise FileNotFoundError("Write PREREGISTRATION.md in output before running")
    report = dict(status="RUNNING", duration_per_episode_s=DURATION,
                  maximum_episodes=4 if args.sensitivity_diagnostic else 3,
                  mode='sensitivity_diagnostic' if args.sensitivity_diagnostic else 'trained',
                  seed=SEED, limitation=LIMITATION, episodes={},
                  classification="P1/S1/S4/S5 bounded manual engineering integration, not training")
    save_json(output / "summary.json", report)
    observer = None
    stage = "imports"
    started = time.monotonic()
    try:
        import mujoco
        import run_neural_muscle_closure as closure
        from engine.embodied.motor_translation import FiniteMotorTranslator
        import hashlib
        if not args.sensitivity_diagnostic:
            stage = 'validate_training_artifact'
            receipt_path = Path(str(args.motor_translator)+'.training_receipt.json')
            receipt = json.loads(receipt_path.read_text())
            json.dumps(receipt, allow_nan=False)
            if receipt.get('target_kind') != 'INTERFACE_TEST':
                raise ValueError('BEHAVIOR provenance not established; only labeled INTERFACE_TEST artifact allowed')
            if receipt.get('artifact_sha256') != hashlib.sha256(args.motor_translator.read_bytes()).hexdigest():
                raise ValueError('training artifact/receipt hash mismatch')
            if receipt.get('changed_parameters', 0) <= 0:
                raise ValueError('receipt must record actual parameter updates')
            report['training_artifact'] = dict(path=str(args.motor_translator.resolve()),
                receipt=str(receipt_path.resolve()), target_kind='INTERFACE_TEST',
                behavior_training=False)
        stage = "prepare_body"
        initial_qpos = None
        if args.initial_state is not None:
            with np.load(args.initial_state, allow_pickle=False) as saved:
                initial_qpos = saved['qpos'].copy()
        model, seated, gate, joints, geom_leg = closure.prepare_body(initial_qpos=initial_qpos)
        report['initial_state'] = dict(
            path=str(args.initial_state.resolve()) if args.initial_state else None,
            sha256=hashlib.sha256(args.initial_state.read_bytes()).hexdigest() if args.initial_state else None,
            key='qpos', exact_copy=args.initial_state is not None,
            seating=args.initial_state is None, clipping=False,
            qualification='scalar/contact preflight only; not geometry or support acceptance')
        np.savez_compressed(output/'initial_runtime_state.npz', qpos=seated,
                            qvel=np.zeros(model.nv), activation=np.zeros(model.na), ctrl=np.zeros(model.nu))
        report['runtime_initialization'] = dict(qpos='exact prepared qpos', qvel='zero', activation='zero',
            ctrl='zero', semantics='Existing closure.episode fresh MjData initialization; no imported qvel/act/ctrl or static-check state.')
        save_json(output / "initial_gate.json", gate)
        report["initial_gate"] = gate
        stage = 'initial_pose_hard_gate'
        require_legal_initial_pose(gate)
        # Keep this integration entry consistent with new closure.main: a tarsus-only
        # contact is insufficient for a six-leg behavior start. No pose repair or clipping.
        stage = 'initial_six_foot_contact_gate'
        initial_contact_legs = closure.require_six_foot_initial_contacts(model, seated, geom_leg)
        report['initial_state']['initial_contact_legs'] = initial_contact_legs
        report['initial_state']['qualification'] = (
            'scalar/contact/six-foot geometric preflight only; not geometry or support acceptance')
        if args.preflight_only:
            report.update(status='INITIAL_PREFLIGHT_PASS_NO_INTEGRATION', body_steps=0)
            save_json(output/'summary.json', report)
            return 0
        report["model_weight"] = float(model.body_mass.sum()*np.linalg.norm(model.opt.gravity))
        stage = "build_neural_tier"
        np.random.seed(SEED)
        built = closure.build_neural_tier(closure.TierConfig(background_mean_nA=0, background_sd_nA=0, background_seed=SEED))
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        manifest = closure.build_motor_manifest(json.loads(closure.POOLS.read_text()), built["tier"]["root_ids"], names)
        save_json(output / "fresh_motor_manifest.json", manifest)
        groups, sensory_stats = closure.channel_index_map(int(built["n"]), built["tier"]["root_ids"])
        save_json(output / "neural_identity.json", dict(n=int(built["n"]), root_ids=built["tier"]["root_ids"], groups=groups, sensory_stats=sensory_stats))
        default = FiniteMotorTranslator(manifest)
        default.save(output/'default.json')
        if args.sensitivity_diagnostic:
            FiniteMotorTranslator(manifest, default.weights*.5).save(output/'fixed_half_weights.json')
            active_arms = (('default', 'normal', 'default.json'),
                           ('fixed_half_weights', 'normal', 'fixed_half_weights.json'),
                           ('spike_output_silence', 'spike_output_silence', 'default.json'),
                           ('default_repeat', 'normal', 'default.json'))
        else:
            trained = FiniteMotorTranslator.load(manifest, args.motor_translator)
            trained.save(output/'trained.json')
            active_arms = (('default', 'normal', 'default.json'), ('trained', 'normal', 'trained.json'),
                           ('trained_output_silence', 'spike_output_silence', 'trained.json'))
        for label, arm, translator in active_arms:
            stage = f"episode:{label}"
            directory = output / label
            directory.mkdir()
            observer = BodyObserver(model, geom_leg)
            np.random.seed(SEED)
            evidence = {}
            arm_started = time.monotonic()
            result, trajectory, commands, spikes = closure.episode(
                model, seated, joints, geom_leg, built, manifest, groups, arm, DURATION,
                stride=1, full_detail_intervals=0, push=None, external_touch=True,
                evidence=evidence, motor_translator=output/translator, body_observer=observer)
            samples = observer.array()
            # Save actual arrays and records before any derived statistics or plotting.
            np.savez_compressed(directory/"arrays.npz", body_substep_observations=samples,
                                trajectory=trajectory, applied_commands=commands, emitted_spike_counts=spikes,
                                **{k:v for k,v in evidence.items() if isinstance(v, np.ndarray)})
            save_json(directory/"episode_records.json", result)
            motor = {i for row in manifest["rows"] for i in row["local_indices"]}
            probe = result["probe_local_indices"]
            raw_motor = sum(sum(probe[col] in motor for step, col in r["probe_raw_spike_events_step_column"]) for r in result["records"])
            stats = support_summary(samples, report["model_weight"], result["initial_height_model"], closure.DT_BODY)
            report["episodes"][label] = dict(status="COMPLETED", runtime_s=time.monotonic()-arm_started,
                support=stats, raw_motor_events=raw_motor,
                emitted_motor_events=sum(r["motor_spikes"] for r in result["records"]),
                applied_command_nonzero_scalars=int(np.count_nonzero(commands)),
                applied_command_max=float(commands.max()), applied_command_mean=float(commands.mean()),
                body_substep_count=len(samples), final_time_s=float(samples[-1, 0]),
                native_positive_ctrl_floor_preserved=True)
            save_json(directory/"summary.json", report["episodes"][label])
            save_json(output/"summary.json", report)
            print(label, json.dumps(report["episodes"][label]), flush=True)
            observer = None
        stage = "compare"
        baseline_label = 'default' if args.sensitivity_diagnostic else 'trained'
        with np.load(output/baseline_label/"arrays.npz") as baseline:
            comparisons = {}
            for label, _, _ in active_arms:
                if label == baseline_label:
                    continue
                with np.load(output/label/"arrays.npz") as other:
                    comparisons[label] = {key:dict(array_equal=bool(np.array_equal(baseline[key], other[key])),
                        max_abs_difference=float(np.max(np.abs(baseline[key]-other[key]))))
                        for key in ("applied_commands", "trajectory", "emitted_spike_counts", "body_substep_observations")}
        report['comparison_baseline'] = baseline_label
        report['comparisons_to_'+baseline_label] = comparisons
        report["status"] = "COMPLETED_SENSITIVITY_DIAGNOSTIC" if args.sensitivity_diagnostic else "COMPLETED_TRAINED_PAIR"
    except Exception as exc:
        report["status"] = "FAIL_STOP"
        report["failure"] = dict(stage=stage, type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        if observer is not None:
            np.savez_compressed(output/"partial_failed_body_observations.npz", body_substep_observations=observer.array())
        print(traceback.format_exc(), file=sys.stderr, flush=True)
    report["runtime_s"] = time.monotonic()-started
    save_json(output/"summary.json", report)
    plot_results(output, report)
    return 0 if report["status"] in ("COMPLETED_TRAINED_PAIR", "COMPLETED_SENSITIVITY_DIAGNOSTIC") else 1


if __name__ == "__main__":
    raise SystemExit(main())
