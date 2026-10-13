"""Minimal canonical-body sensor -> connectome -> motor-pool closure.

Hand-built experimental translation, no CPG/PD/baseline command. Native XML
control floor remains intact and is reported, NOT credited to neural drive.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import uuid
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault('MUJOCO_GL', 'egl')
import engine.canonical_physics as cp
import run_fix_round39 as r39
import engine.memory_guard as mg
from engine.memory_guard import MemoryBudgetError
from engine.embodied.adapters import build_neural_tier, TierConfig
from engine.embodied.sensory_channels import ChannelSensoryEncoder, ChannelQuantities, ChannelConfig, channel_index_map, LEGS, LOOKUP
from engine.embodied.motor_output import build_motor_manifest, MotorOutput, MotorOutputConfig
from engine.neural_cond import ConductanceNetwork, ConductanceParams
from engine.embodied.closure_clock import grid_steps, push_steps

OUT = HERE / 'outputs/neural_muscle_closure'
POOLS = HERE / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json'
DT_BODY = .0001
DT_NEURAL = .0005
DT_COMMAND = .005


def prepare_body():
    import mujoco
    assert cp.sha256_file(cp.CANONICAL_XML) == cp.CANONICAL_XML_SHA256
    model = cp.load_model()
    if not np.isclose(model.opt.timestep, DT_BODY):
        raise ValueError('canonical body timestep differs from declared clock')
    data = mujoco.MjData(model)
    r39.apply_round37_pose(model, data, json.loads(r39.ROUND37_POSE.read_text()))
    base = data.qpos.copy()
    base[2] = cp.seat_root(model, data, base)
    data.qpos[:] = base
    data.qvel[:] = 0
    data.act[:] = 0  # declared initial state, never overwritten during integration
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)
    gate = cp.stance_facts(model, data).as_dict()
    from engine.embodied.initial_pose_contract import model_limit_report
    gate['initial_scalar_joint_ranges'] = model_limit_report(model, base)
    gate['gate_scope'] = 'Contact/burial gate enforced; joint range diagnostics reported separately, no silent pose clipping'
    if not gate['initial_scalar_joint_ranges']['within_declared_scalar_ranges']:
        print('WARNING: historical round37 initial pose has scalar joint range violations; contact PASS is not full pose validity.', flush=True)
    joints = {}
    for leg in LEGS:
        ids = [j for j in range(model.njnt) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or '').lower().startswith('joint_'+leg)]
        tib = [j for j in ids if 'tibia' in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,j).lower()]
        if len(ids) != 7 or len(tib) != 1:
            raise ValueError('ambiguous canonical leg joint identities')
        joints[leg] = dict(q=[int(model.jnt_qposadr[j]) for j in ids],
                           tib_q=int(model.jnt_qposadr[tib[0]]), tib_v=int(model.jnt_dofadr[tib[0]]))
    geom_leg = {}
    for g in range(model.ngeom):
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,g) or '').lower()
        body = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or '').lower()
        match = [leg for leg in LEGS if ('tars' in name+body and (name.startswith(leg) or body.startswith(leg)))]
        if len(match) == 1:
            geom_leg[g] = match[0]
    if set(geom_leg.values()) != set(LEGS):
        raise ValueError('foot geometry mapping incomplete')
    return model, base, gate, joints, geom_leg


def observe_quantities(model, data, reference, joints, geom_leg):
    import mujoco
    force = {leg: 0.0 for leg in LEGS}
    touches = {leg: 0.0 for leg in LEGS}
    for ci in range(data.ncon):
        c = data.contact[ci]
        for g, other in ((int(c.geom1), int(c.geom2)), (int(c.geom2), int(c.geom1))):
            if g in geom_leg and model.geom_type[other] == mujoco.mjtGeom.mjGEOM_PLANE:
                f = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, f)
                force[geom_leg[g]] += float(np.linalg.norm(f[:3]))
                touches[geom_leg[g]] = 1.0
    result = {}
    for leg, j in joints.items():
        result[leg] = ChannelQuantities(
            tibia_joint_dev_rad=float(data.qpos[j['tib_q']]-reference[j['tib_q']]),
            tibia_joint_vel_rad_s=float(data.qvel[j['tib_v']]),
            multi_joint_dev_rad=float(np.mean(data.qpos[j['q']]-reference[j['q']])),
            contact_force_norm=force[leg], foot_contact=touches[leg])
    return result


def episode(model, seated, joints, geom_leg, built, manifest, groups, arm, duration, replay=None,
            stride=10, full_detail_intervals=20, push=None):
    import mujoco
    intervals = grid_steps(duration, DT_COMMAND)
    body_steps = grid_steps(DT_COMMAND, DT_BODY)
    if not np.isclose(model.opt.timestep, DT_BODY, rtol=0, atol=1e-15):
        raise ValueError('episode body timestep differs from declared clock')
    neural_steps = grid_steps(DT_COMMAND, DT_NEURAL)
    window = push_steps(push, duration, DT_BODY)
    if stride < 1:
        raise ValueError('record stride must be >= 1')
    d = mujoco.MjData(model)
    d.qpos[:] = seated
    d.qvel[:] = 0
    d.act[:] = 0
    d.ctrl[:] = 0
    mujoco.mj_forward(model, d)
    thorax = model.body('Thorax').id
    h0 = float(d.xpos[thorax, 2])
    enc = ChannelSensoryEncoder(groups, int(built['n']), ChannelConfig())
    decoder = MotorOutput(manifest, MotorOutputConfig())
    net = ConductanceNetwork(int(built['n']), ConductanceParams(), dt_ms=DT_NEURAL*1000)
    net.set_connectivity(built['edges']['We']*.2, built['edges']['Wi']*.2)
    pending = np.zeros(model.nu)
    source = -1
    records, trajectories, commands, spike_counts = [], [], [], []
    # Compact diagnostic raster: only mapped motor and explicitly stimulated sensory cells.
    # Preserves 0.5ms spike timing without storing a full-tier raster in every arm.
    probe_indices = sorted({i for row in manifest['rows'] for i in row['local_indices']} |
                           {int(i) for i in groups['lf']['bristle_touch']})
    escape = None
    motor_indices = sorted({i for row in manifest['rows'] for i in row['local_indices']})
    cut_s = .015
    for k in range(intervals):
        t = k*DT_COMMAND
        quantities = observe_quantities(model, d, seated, joints, geom_leg)
        current, channels = enc.encode(quantities)
        # Declared experimenter's local tactile stimulus, never motor-targeted.
        # Compare it to a matched sensory-cut arm: zero all input currents there.
        stimulus_on = arm != 'no_external_touch' and .005 <= t < .015
        if stimulus_on:
            for i in groups['lf']['bristle_touch']:
                current[int(i)] += .3
        if arm == 'sensory_cut':
            current[:] = 0
        sp = np.zeros((neural_steps, int(built['n'])), dtype=bool)
        for j in range(neural_steps):
            sp[j] = net.step(current_nA=current)
        probe_raw_events = np.argwhere(sp[:, probe_indices]).tolist()
        if arm == 'spike_output_silence':
            sp[:] = False  # output silence intervention, not a claim of resetting net
        applied = pending.copy()
        if arm == 'motor_cut' and t >= cut_s:
            applied[:] = 0
        if arm == 'replay':
            applied = np.asarray(replay[k], float).copy()
        d.ctrl[:] = applied
        # EXTERNAL PERTURBATION: an experimenter-applied world-frame force on the thorax, applied
        # IDENTICALLY in every arm.  It is not a neural command and never targets a motor neuron
        # pool; it exists so "does the loop change recovery" can be asked at all.  Cleared on
        # every interval so an earlier push cannot leak into a later one.
        push_applied_steps = 0
        for substep in range(body_steps):
            absolute_step = k*body_steps + substep
            push_on = window is not None and window[0] <= absolute_step < window[1]
            d.xfrc_applied[:] = 0.0
            if push_on:
                d.xfrc_applied[thorax, :3] = push['force_model']
                push_applied_steps += 1
            mujoco.mj_step(model, d)
            mujoco.mj_forward(model, d)
            if escape is None and d.xpos[thorax, 2] < .9*h0:
                escape = float(d.time)
        if not all(np.all(np.isfinite(x)) for x in (d.qpos,d.qvel,d.act)):
            raise FloatingPointError(f'nonfinite body in {arm} at {d.time}')
        if abs(d.time-(t+DT_COMMAND)) > 1e-10 or abs(net.t_ms/1000-(t+DT_COMMAND)) > 1e-10:
            raise AssertionError('body/neural clock drift or simulator reset')
        warnings = [int(w.number) for w in d.warning]
        if any(warnings):
            raise RuntimeError(f'MuJoCo warnings in {arm}: {warnings}')
        pending, trace = decoder.consume(sp, spike_dt_s=DT_NEURAL, start_s=t,
                                         end_s=t+DT_COMMAND,
                                         connected=not(arm == 'motor_cut' and t+DT_COMMAND >= cut_s))
        if source >= k:
            raise AssertionError('future spikes applied to body')
        record = dict(time_start_s=t, time_end_s=float(d.time), applied_source_interval=source,
                      source_completed_by_s=t, decoder=trace,
                      probe_raw_spike_events_step_column=probe_raw_events,
                      probe_emitted_spike_events_step_column=np.argwhere(sp[:, probe_indices]).tolist(),
                      command_origin='offline_normal_replay' if arm == 'replay' else 'previous_completed_neural_interval',
                      mujoco_warning_counts=warnings,
                      sensory_current_min_nA=float(current.min()), sensory_current_max_nA=float(current.max()),
                      external_lf_touch_stimulus=stimulus_on,
                      external_push_applied=bool(push_applied_steps),
                      external_push_body_steps=push_applied_steps,
                      external_push_duration_s=push_applied_steps*DT_BODY,
                      external_push_impulse_model=(np.asarray(push['force_model'])*push_applied_steps*DT_BODY).tolist() if push else [0.0,0.0,0.0],
                      motor_spikes=int(sp[:,motor_indices].sum()),
                      active_control_count=int(np.count_nonzero(applied)),
                      native_control_floor_clamp_count=int(np.sum((applied < model.actuator_ctrlrange[:,0]) & model.actuator_ctrllimited)),
                      height_model=float(d.xpos[thorax,2]), height_mm_declared=float(cp.model_to_mm(d.xpos[thorax,2])),
                      muscle_force_max_abs_model=float(np.max(np.abs(d.actuator_force))),
                      activation_max=float(d.act.max()))
        # Per-interval channel detail is kept for the first intervals and every `stride` after,
        # so artifact size stays bounded as the episode lengthens instead of growing per interval.
        detailed = k < full_detail_intervals or k % stride == 0
        record['detail'] = 'full' if detailed else 'trimmed'
        if detailed:
            record['channel_record'] = channels
        records.append(record)
        source = k
        trajectories.append(np.r_[d.qpos.copy(), d.qvel.copy(), d.act.copy()])
        commands.append(applied)
        spike_counts.append(sp.sum(axis=0))
    return dict(arm=arm, probe_local_indices=probe_indices,
                probe_event_encoding='[step_index,column_in_probe_local_indices]; spike time=interval_start+(step_index+1)*DT_NEURAL; raw before output silence, emitted after',
                probe_spike_dt_s=DT_NEURAL,
                escape_time_s=escape, initial_height_model=h0,
                final_height_model=float(d.xpos[thorax,2]), records=records), np.array(trajectories), np.array(commands), np.array(spike_counts)


def main(guard_report=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=.05)
    parser.add_argument('--record-stride', type=int, default=10)
    parser.add_argument('--push-weight-fraction', type=float, default=0.0,
                        help='external world-frame force on the thorax as a FRACTION of the '
                             "model's own total weight (mass_sum * |gravity|), measured at "
                             'runtime; 0 disables the perturbation')
    parser.add_argument('--push-at-ms', type=float, default=15.0)
    parser.add_argument('--push-duration-ms', type=float, default=5.0)
    args = parser.parse_args()
    grid_steps(args.duration, DT_COMMAND)
    if args.record_stride < 1:
        raise ValueError('record stride must be >= 1')
    push_spec = None
    if args.push_weight_fraction:
        grid_steps(args.push_at_ms/1000, DT_BODY, name='push start', allow_zero=True)
        grid_steps(args.push_duration_ms/1000, DT_BODY, name='push duration')
        if not np.isfinite(args.push_weight_fraction):
            raise ValueError('push magnitude must be finite')
        if args.push_at_ms < 0 or args.push_duration_ms <= 0:
            raise ValueError('push window must start at >= 0 and last > 0 ms')
        if args.push_at_ms + args.push_duration_ms > args.duration*1000:
            raise ValueError('push window must lie inside the episode')
        if abs(args.push_weight_fraction) > 5.0:
            raise ValueError('push fraction above 5x body weight is outside the declared range')
        push_spec = {'start_s': args.push_at_ms/1000.0,
                     'end_s': (args.push_at_ms+args.push_duration_ms)/1000.0,
                     'weight_fraction': float(args.push_weight_fraction)}
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model, seated, gate, joints, geom_leg = prepare_body()
    # MEASURED BUG (fixed): this block first read `model.body_mass` BEFORE `prepare_body()` ran,
    # so the run died with UnboundLocalError - and only when the script was actually executed,
    # because a syntax check cannot see it.  The perturbation magnitude needs the compiled model,
    # so it is assembled here instead.
    push = None
    if push_spec is not None:
        total_weight_model = float(model.body_mass.sum() * abs(model.opt.gravity[2]))
        push = dict(push_spec, total_weight_model=total_weight_model,
                    force_model=[float(push_spec['weight_fraction']*total_weight_model), 0.0, 0.0],
                    note='world-frame force on Thorax as a fraction of this model\'s own total '
                         'weight, applied identically in every arm')
    built = build_neural_tier(TierConfig(background_mean_nA=0, background_sd_nA=0))
    names = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(model.nu)]
    manifest = build_motor_manifest(json.loads(POOLS.read_text()), built['tier']['root_ids'], names)
    groups, sensory_stats = channel_index_map(int(built['n']), built['tier']['root_ids'])
    from engine.embodied.channel_connectivity import channel_connectivity
    connectivity = channel_connectivity(built['edges']['We'], built['edges']['Wi'], groups,
        {i for row in manifest['rows'] for i in row['local_indices']})
    if connectivity['channels']['lf']['bristle_touch']['status'] != 'STRUCTURAL_PATH_ONLY':
        print('WARNING: LF tactile probe has no modeled path to motor output; not a feedback capability test.', flush=True)
    units = dict(time='s', spike_rate='Hz', current='nA', angle='rad', force='model force units (SI uncalibrated)',
                 length_model='mm (1:1 with model units; unit evidence in engine/canonical_physics.py D1)',
                 mass='model mass unit; SI conversion INFERRED (gram) but NOT established')
    provenance = dict(code=str(Path(__file__).resolve()), interpreter=sys.executable,
                      canonical_xml_sha256=cp.sha256_file(cp.CANONICAL_XML),
                      pool_result_sha256=cp.sha256_file(POOLS), sensory_lookup_sha256=cp.sha256_file(LOOKUP),
                      code_hashes={str(p):cp.sha256_file(p) for p in (Path(__file__).resolve(), HERE/'engine/embodied/motor_output.py', HERE/'engine/embodied/sensory_channels.py', HERE/'engine/neural_cond.py')},
                      connectome_sha256=cp.sha256_file(HERE/'data/flywire/banc_connectome.npz'),
                      source_feather_sha256=cp.sha256_file(HERE/'data/banc888/banc_888_meta.feather'),
                      network_weights_scale=.2, background_current_nA=0,
                      motor_translation=asdict(MotorOutputConfig()), sensory_translation=asdict(ChannelConfig()),
                      initialization='gated round37 qpos, zero qvel and activation; no LP bias',
                      sensory_geometry='explicit canonical joint names; sum magnitudes of foot-plane contact forces per leg',
                      activation='MuJoCo only; native positive ctrl lower bound preserved and reported')
    # Stage companions together; publish a verified complete bundle only after success.
    staged = OUT / '.staging' / uuid.uuid4().hex
    staged.mkdir(parents=True, exist_ok=False)
    provenance['staging_id'] = staged.name
    provenance['artifact_contract'] = 'Use verified current_run.json bundle; legacy flat files are historical, not current companions'
    provenance['code_hashes'].update({str(p):cp.sha256_file(p) for p in (
        HERE/'engine/embodied/closure_clock.py', HERE/'engine/embodied/closure_bundle.py',
        HERE/'engine/embodied/channel_connectivity.py', HERE/'engine/embodied/initial_pose_contract.py',
        HERE/'engine/embodied/source_snapshot.py',
        HERE/'engine/embodied/adapters.py', HERE/'engine/canonical_physics.py',
        Path(r39.__file__), r39.ROUND37_POSE)})
    from engine.embodied.source_snapshot import build_source_snapshot, verify_source_snapshot
    snapshot_path = staged/'source_snapshot.zip'
    build_source_snapshot(provenance['code_hashes'], snapshot_path)
    provenance['source_snapshot'] = dict(file='source_snapshot.zip', sha256=cp.sha256_file(snapshot_path),
        scope='Explicit recorded code/pose inputs only; not a full dependency or data archive',
        authenticity='Content integrity only, not signed authorship or biological validation')
    identity_path = staged/'neural_identity.json'
    identity_path.write_text(json.dumps(dict(
        tier_root_ids=[int(r) for r in built['tier']['root_ids']],
        sensory_groups={leg:{ch:[int(i) for i in ids] for ch,ids in row.items()} for leg,row in groups.items()},
        interpretation='Dataset tier identities and declared channel assignments, not validated per-cell tuning',
        spike_probe='Episode probe_local_indices index tier_root_ids; raw/emitted events preserve output-silence distinction'
    ),indent=2)+'\n')
    provenance['neural_identity_sha256'] = cp.sha256_file(identity_path)
    cp.write_result(staged/'motor_manifest.json', question='Which annotated muscle pools exist in this tier?',
                    status='PASS', measured=manifest, provenance=provenance, units=units,
                    illustrative_vs_measured='Computed annotation/tier intersections; aliases, shared-pool recruitment and present-cell normalization remain hypotheses',
                    what_this_does_not_support=['Aliases are declared hypotheses, not independently confirmed anatomy.',
                                               'Present-cell mean on partial pools is uncalibrated.'])
    summaries, all_arrays = {}, {}
    baseline_commands = None
    for arm in ('normal','no_external_touch','sensory_cut','motor_cut','spike_output_silence','replay','normal_repeat'):
        summary, traj, ctrl, spikes = episode(model,seated,joints,geom_leg,built,manifest,groups,
                                              arm,args.duration,replay=baseline_commands,
                                              stride=args.record_stride, push=push)
        summaries[arm] = summary
        all_arrays[arm+'_trajectory'] = traj
        all_arrays[arm+'_controls'] = ctrl
        all_arrays[arm+'_spike_counts'] = spikes
        if arm == 'normal':
            baseline_commands = ctrl
        print(arm, 'escape',summary['escape_time_s'],'active',int(np.count_nonzero(ctrl)), flush=True)
    checks = dict(repeat_trajectory_exact=np.array_equal(all_arrays['normal_trajectory'],all_arrays['normal_repeat_trajectory']),
                  replay_trajectory_exact=np.array_equal(all_arrays['normal_trajectory'],all_arrays['replay_trajectory']),
                  silenced_output_commands_zero=not bool(np.any(all_arrays['spike_output_silence_controls'])),
                  motor_cut_commands_zero_after_cut=not bool(np.any(all_arrays['motor_cut_controls'][3:])),
                  neural_commands_present=bool(np.any(all_arrays['normal_controls'])),
                  external_touch_changes_commands=not np.array_equal(all_arrays['normal_controls'],all_arrays['no_external_touch_controls']),
                  normal_vs_silence_body_max_difference=float(np.max(np.abs(all_arrays['normal_trajectory']-all_arrays['spike_output_silence_trajectory']))),
                  sensory_cut_changes_motor_spikes=not np.array_equal(all_arrays['normal_spike_counts'][:, sorted({i for r in manifest['rows'] for i in r['local_indices']})], all_arrays['sensory_cut_spike_counts'][:, sorted({i for r in manifest['rows'] for i in r['local_indices']})]),
                  native_floor_prevents_strict_zero_neural_only_actuation=bool(np.any((model.actuator_ctrlrange[:,0] > 0) & model.actuator_ctrllimited)))
    np.savez_compressed(staged/'closure_traces.npz',**all_arrays,seated_qpos=seated)
    assert cp.sha256_file(cp.CANONICAL_XML) == cp.CANONICAL_XML_SHA256
    guard = guard_report('completed') if guard_report else None
    mandatory = ('repeat_trajectory_exact','replay_trajectory_exact','silenced_output_commands_zero','motor_cut_commands_zero_after_cut')
    cp.write_result(staged/'closure_results.json', question='Does the actual motor-pool path produce traceable commands and body responses?',
        status='PASS' if all(checks[k] for k in mandatory) else 'FAIL',
        measured=dict(execution_status='completed',
                      status_meaning='PASS means interface/replay invariants only, not scientific closure or standing success',
                      strict_autonomy_acceptance=False, strict_autonomy_reason='Native nonzero control floor preserved; biological translation unvalidated',
                      interface_acceptance={k:checks[k] for k in mandatory},
                      scientific_checks=checks,gate=gate, sensory_stats=sensory_stats,
                       sensory_motor_structural_coverage=connectivity,
                      duration_s=args.duration, record_stride=args.record_stride,
                      external_perturbation=push,
                      memory_guard=guard, episodes=summaries, wall_s=time.time()-started),
        provenance=provenance, units=units,
        illustrative_vs_measured='Traces measured; transduction/gain/filter/weight scale are declared unvalidated assumptions',
        what_this_does_not_support=['No autonomous locomotion or standing success inferred from interface PASS.',
            'Native XML minimum control creates residual activation even for zero neural command; strict neural-only actuation is NOT established.',
            'External LF tactile injection is an experimenter stimulus, not an autonomous choice.',
            'Channel polarity and within-subtype tuning remain unvalidated; no artificial positive/negative neuron partition.',
            'Replaying a normal episode is an integration check, not proof of feedback benefit.',
            'Sensory-cut removes both body feedback and artificial touch; it establishes input dependence, not their separate effects.',
            'An external perturbation is an experimenter-applied force, not a neural command and '
            'not a behaviour; a difference between arms under it is still not proof of control.',
            'Spike-output silence does not silence network state; it cuts all emitted spikes before translation.',
            'Mass/force SI calibration unresolved; 0.05-second tests cannot establish robust behavior.'])
    verify_source_snapshot(snapshot_path, provenance['code_hashes'])
    if any(cp.sha256_file(Path(path)) != expected for path, expected in provenance['code_hashes'].items()):
        raise ValueError('Source changed during closure execution; refusing publication')
    from engine.embodied.closure_bundle import publish_bundle, validate_bundle
    bundle = publish_bundle(OUT, {name: staged/name for name in (
        'motor_manifest.json', 'closure_traces.npz', 'closure_results.json', 'neural_identity.json', 'source_snapshot.zip')})
    validate_bundle(OUT)
    print('Completed immutable run bundle:', bundle['path'], flush=True)
    print('Legacy flat files are historical; current_run.json identifies this completed run.', flush=True)
    print(json.dumps(checks,indent=2))


if __name__ == '__main__':
    # Guarded entry: refuses to start without declared free memory, caps this process's address
    # space so a runaway raises MemoryError here instead of a global OOM kill, and takes a
    # single-flight lock so two heavy runs of this project cannot stack.
    with mg.guarded_run('neural_muscle_closure', **mg.BUDGETS['neural_muscle_closure']) as _rep:
        main(guard_report=_rep)
