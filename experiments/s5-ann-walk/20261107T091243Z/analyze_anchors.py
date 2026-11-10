"""Read-only saved-trace diagnosis; scratch FK only, NEVER integrates physics.

Anchor-origin drift is not contact-material slip. Frozen independent replay is
responsible for material contact velocity and walking acceptance.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import mujoco as mj
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T091243Z')
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def stats(value):
    a = np.asarray(value, float).ravel()
    return dict(samples=int(a.size), mean=float(a.mean()), max=float(a.max()),
                p95=float(np.percentile(a, 95))) if a.size else dict(samples=0)


def readonly_audit(source):
    tree = ast.parse(source.read_text())
    physical_writes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            base = node.value
            if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
                if base.value.id == 'd' and base.attr in ('qpos', 'qvel'):
                    physical_writes.append(dict(line=node.lineno, target=ast.unparse(node)))
    return dict(actual_qpos_qvel_subscript_writes=physical_writes,
                snapshot_assertion_present='np.array_equal(before, p.snapshot(d))' in source.read_text(),
                limitation='Static source audit; actual handover arrays checked separately. Scratch FK writes are intentional.')


def analyze(folder, runtime, plant):
    S, T, B = [arrays(folder/(s+'.npz')) for s in ('SENSORY', 'TEACHER', 'BODY')]
    C, N = arrays(folder/'COMMANDS.npz'), arrays(folder/'NATIVE.npz')['rows']
    t = S['time']; release = t >= 1.-1e-9
    active = T['active'].astype(bool)
    start = int(np.flatnonzero(release)[0])
    ixs = [np.asarray([i for i,j in enumerate(plant.jids)
                      if plant.m.joint(int(j)).name.startswith('joint_'+leg)]) for leg in LEGS]
    scratch = mj.MjData(plant.m)
    feet = np.empty((len(t), 6, 3)); footrot = np.empty((len(t), 6, 3, 3))
    residual = np.empty((len(t), 6, 2)); recomputed = []
    for i in range(len(t)):
        scratch.qpos[:] = S['qpos'][i]; scratch.qvel[:] = S['qvel'][i]
        mj.mj_forward(plant.m, scratch)
        feet[i] = scratch.xpos[plant.bids]
        footrot[i] = scratch.xmat[plant.bids].reshape(6, 3, 3)
        recomputed.append(runtime.features(plant, scratch, T['target'][i], S['loads'][i]*runtime.n.dsm.WEIGHT))
        scratch.qpos[:7] = T['virtual_root'][i]
        scratch.qpos[plant.qa] = T['joint_desired'][i]
        mj.mj_forward(plant.m, scratch)
        residual[i, :, 0] = np.linalg.norm(T['foot_goal_position'][i]-scratch.xpos[plant.bids], axis=1)
        residual[i, :, 1] = np.linalg.norm(Rotation.from_matrix(
            T['foot_goal_rotation'][i]@scratch.xmat[plant.bids].reshape(6, 3, 3).transpose(0, 2, 1)).as_rotvec(), axis=1)
    # BODY is at END of each5ms; sensory/teacher are at START. BODY[i-1]
    # is the exact physical state for SENSORY[i], not BODY[i].
    alignment = dict(time_max=float(abs(t[1:]-B['time'][:-1]).max()),
        qpos_bitwise=bool(np.array_equal(S['qpos'][1:], B['qpos'][:-1])),
        qvel_bitwise=bool(np.array_equal(S['qvel'][1:], B['qvel'][:-1])),
        feet_fk_max=float(abs(feet[1:]-B['feet'][:-1]).max()))
    stance = T['stance'].astype(bool)
    continuing = stance.copy(); continuing[0] = False; continuing[1:] &= stance[:-1]
    mask = continuing & active[:, None]
    drift = np.linalg.norm(feet[:, :, :2]-T['anchor_position'][:, :, :2], axis=2)
    per_leg = {}
    expected = np.zeros_like(stance)
    expected[1:] = stance[1:] & ~stance[:-1]
    expected[:start+1] = False
    anchor_delta = np.zeros_like(stance, float)
    anchor_delta[1:] = np.max(abs(T['anchor_position'][1:]-T['anchor_position'][:-1]), axis=2)
    rotation_delta = np.zeros_like(anchor_delta)
    rotation_delta[1:] = np.max(abs(T['anchor_rotation'][1:]-T['anchor_rotation'][:-1]), axis=(2, 3))
    loads = S['loads']
    target_error, desire_error = T['target']-S['q'], T['joint_desired']-S['q']
    commanded = abs(T['rate']) >= runtime.CAP-1e-8
    torque_sat = abs(B['ctrl']) >= C['limits'][None, :]-1e-8
    raw = C['kp'][None, :]*target_error-C['kd'][None, :]*S['joint_velocity']
    for li, leg in enumerate(LEGS):
        ix = ixs[li]; entry = expected[:, li] & active
        per_leg[leg] = dict(ik_position_stance=stats(T['ik_residual'][active & stance[:, li], li, 0]),
            ik_position_swing=stats(T['ik_residual'][active & ~stance[:, li], li, 0]),
            ik_rotation_stance_rad=stats(T['ik_residual'][active & stance[:, li], li, 1]),
            ik_rotation_swing_rad=stats(T['ik_residual'][active & ~stance[:, li], li, 1]),
            continuing_stance_origin_anchor_xy_drift=stats(drift[mask[:, li], li]),
            loaded_continuing_stance_origin_anchor_xy_drift=stats(drift[mask[:, li] & (loads[:, li]>1e-9/runtime.n.dsm.WEIGHT), li]),
            nominal_stance_entries=int(entry.sum()), logged_entries=int(T['stance_entry'][active, li].sum()),
            anchor_position_changes=int(((anchor_delta[:, li]>1e-12) & active).sum()),
            unpermitted_anchor_position_change_max=float(anchor_delta[active & ~expected[:, li] & (np.arange(len(t))!=start), li].max()) if active.any() else 0.,
            unpermitted_anchor_rotation_change_max=float(rotation_delta[active & ~expected[:, li] & (np.arange(len(t))!=start), li].max()) if active.any() else 0.,
            entry_anchor_measurement_error_max=float(np.max(abs(T['anchor_position'][entry, li]-feet[entry, li]))) if entry.any() else None,
            entries_while_unloaded=int((entry & (loads[:, li]<=1e-9/runtime.n.dsm.WEIGHT)).sum()),
            unloaded_entry_times=t[entry & (loads[:, li]<=1e-9/runtime.n.dsm.WEIGHT)].tolist(),
            switch_load_W=float(loads[start, li]),
            switch_old_target_minus_actual_rad=target_error[start, ix].tolist(),
            switch_ik_desired_minus_actual_rad=desire_error[start, ix].tolist(),
            switch_old_target_error_norm=float(np.linalg.norm(target_error[start, ix])),
            switch_ik_desired_error_norm=float(np.linalg.norm(desire_error[start, ix])),
            switch_joint_desired_change_norm=float(np.linalg.norm(T['joint_desired'][start, ix]-T['target'][start, ix])),
            rate_cap_fraction=float(commanded[release][:, ix].mean()),
            sampled_actual_torque_cap_fraction=float(torque_sat[B['time']>=1][:, ix].mean()))
    actualrot = Rotation.from_quat(S['root'][:, [4, 5, 6, 3]])
    virtualrot = Rotation.from_quat(T['virtual_root'][:, [4, 5, 6, 3]])
    tilt = np.arccos(np.clip(actualrot.as_matrix()[:, 2, 2], -1, 1))
    virt_step = T['virtual_root'][:, :3]-S['root'][:, :3]
    virt_angle = np.linalg.norm((virtualrot*actualrot.inv()).as_rotvec(), axis=1)
    referr = np.linalg.norm(T['reference_body_position'][:, :2]-S['root'][:, :2], axis=1)
    hbefore, hafter = arrays(folder/'HANDOVER.npz'), arrays(folder/'HANDOVER_AFTER.npz')
    snapshot_ok = np.array_equal(hbefore['physics_integration'], hafter['physics_integration'])
    feature_error = np.max(abs(np.asarray(recomputed)-S['features']))
    first = lambda condition: float(t[np.flatnonzero(condition & release)[0]]) if (condition & release).any() else None
    events = dict(root_height_drop_0p01native=first(S['root'][:, 2]<S['root'][start, 2]-.01),
        root_height_drop_0p1native=first(S['root'][:, 2]<S['root'][start, 2]-.1),
        nominal_stance_ik_position_gt_01=first(np.max(np.where(stance, T['ik_residual'][:, :, 0], 0), axis=1)>.01),
        any_ik_position_gt_01=first(T['ik_residual'][:, :, 0].max(1)>.01),
        footload_below_08W=first(loads.sum(1)<.8), command_rate_cap=first(commanded.any(1)))
    switches = []
    for seconds in (1., 1.005, 1.01, 1.02, 1.05, 1.1, 1.2, 1.4):
        i = int(np.argmin(abs(t-seconds)))
        switches.append(dict(time=float(t[i]), root_height=float(S['root'][i, 2]),
            total_foot_load_W=float(loads[i].sum()), per_leg_load_W=loads[i].tolist(),
            nominal_stance=stance[i].tolist(), target_tracking_rms=float(np.sqrt(np.mean(target_error[i]**2))),
            ik_desired_minus_actual_rms=float(np.sqrt(np.mean(desire_error[i]**2))),
            max_ik_position=float(T['ik_residual'][i, :, 0].max()),
            per_leg_ik_position=T['ik_residual'][i, :, 0].tolist(),
            rate_capped_joints=int(commanded[i].sum())))
    report = dict(folder=str(folder), result=json.loads((folder/'RESULT.json').read_text()),
        state_alignment=alignment, sensory_feature_shape=list(S['features'].shape),
        sensory_features_finite=bool(np.isfinite(S['features']).all()), feature_recompute_max=float(feature_error),
        handover_snapshot_bitwise=bool(snapshot_ok), handover_target_matches_sensory=bool(np.array_equal(hbefore['target'], T['target'][start])),
        entry_flags_match_nominal=bool(np.array_equal(T['stance_entry'][active], expected[active])),
        initial_capture_anchor_position_error_max=float(abs(T['anchor_position'][start]-feet[start]).max()),
        recomputed_ik_residual_max_error=float(abs(residual[active]-T['ik_residual'][active]).max()) if active.any() else None,
        per_leg=per_leg, events=events, switch_trace=switches,
        virtual_xy_step=stats(np.linalg.norm(virt_step[release, :2], axis=1)),
        virtual_z_step=stats(abs(virt_step[release, 2])), virtual_angle_step_rad=stats(virt_angle[release]),
        root_height_min=float(S['root'][release, 2].min()), root_height_at_switch=float(S['root'][start, 2]),
        root_max_tilt_deg=float(np.rad2deg(tilt[release]).max()),
        reference_xy_error=stats(referr[active]), body_feedback_velocity=stats(np.linalg.norm(T['body_feedback_velocity'][release], axis=1)),
        sample_rate_cap_fraction=float(commanded[release].mean()),
        sample_actual_torque_cap_fraction=float(torque_sat[B['time']>=1].mean()),
        raw_PD_at_sensory_tick_over_cap_fraction=float((abs(raw[release])>C['limits'][None, :]).mean()),
        hashes={str(folder/(f+'.npz')):sha(folder/(f+'.npz')) for f in ('SENSORY', 'TEACHER', 'BODY', 'COMMANDS', 'NATIVE', 'HANDOVER', 'HANDOVER_AFTER')})
    return report, dict(t=t, S=S, T=T, B=B, C=C, N=N, drift=drift, tilt=tilt,
                        target_error=target_error, commanded=commanded, torque_sat=torque_sat, referr=referr)


def plot(traces, path):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for label, x in traces.items():
        t=x['t']; s=x['S']; T=x['T']; b=x['B']
        axes[0, 0].plot(t, s['root'][:, 2], label=label+' actual root z')
        axes[0, 0].plot(b['time'], b['com'][:, 2], '--', label=label+' actual COM z')
        axes[0, 1].plot(t, s['loads'].sum(1), label=label)
        axes[1, 1].plot(t, np.sqrt(np.mean(x['target_error']**2, axis=1)), label=label+' target-q RMS')
        axes[2, 1].plot(t, x['commanded'].mean(1), label=label+' rate cap fraction')
        axes[2, 1].plot(b['time'], x['torque_sat'].mean(1), '--', label=label+' actual torque cap fraction')
        if T['active'].any():
            active=T['active'].astype(bool)
            for li,leg in enumerate(LEGS):
                axes[1, 0].plot(t[active], T['ik_residual'][active, li, 0], label=leg)
                y=np.where(T['stance'][:, li] & active, x['drift'][:, li], np.nan)
                axes[2, 0].plot(t, y, label=leg)
            axes[1, 1].plot(t[active], x['referr'][active], ':', label='teacher reference xy error')
    titles=('Actual root/COM height', 'Actual foot support (W)',
            'Feedback teacher IK position residual (native)', 'Tracking and privileged body-reference error',
            'Nominal stance Tarsus5 origin-anchor XY drift (NOT material slip)', '5ms rate / sampled applied torque saturation')
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title, fontsize=10); ax.set_xlabel('physical time (s)'); ax.axvline(1., color='k', alpha=.3)
        ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.suptitle('Training teacher feedback failed; matched reference stays supported. No neural claim.\nSaved traces + scratch FK only; 0 physics steps.', fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--feedback', type=Path, default=OUT/'anchor_screen')
    ap.add_argument('--off', type=Path, default=OUT/'anchor_feedback_off')
    args=ap.parse_args()
    for folder in (args.feedback,args.off):
        for filename in ('RESULT.json','SENSORY.npz','TEACHER.npz','BODY.npz','COMMANDS.npz','NATIVE.npz','HANDOVER.npz','HANDOVER_AFTER.npz'):
            if not (folder/filename).exists():
                raise SystemExit('Not ready; no waiting or physics: '+str(folder/filename))
    def forbidden(*a, **kw):
        raise AssertionError('Analyzer forbids mj_step')
    mj.mj_step=forbidden
    source=HERE/'anchor_teacher.py'
    spec=importlib.util.spec_from_file_location('anchor_trace_runtime',source)
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    plant=runtime.n.Plant()
    reports={}; traces={}
    for label,folder in [('feedback',args.feedback),('feedback_off',args.off)]:
        reports[label],traces[label]=analyze(folder,runtime,plant)
    A,B=traces['feedback'],traces['feedback_off']
    warm=dict(commands_first1000_rows_bitwise=bool(np.array_equal(A['C']['rows'][:1000],B['C']['rows'][:1000])),
              native_first80000_rows_bitwise=bool(np.array_equal(A['N'][:80000],B['N'][:80000])),
              sensory_first200_states_bitwise=bool(np.array_equal(A['S']['qpos'][:200],B['S']['qpos'][:200])))
    output=dict(stage='S5 physical teacher failure diagnosis only', physics_steps=0,
        source=str(source),source_sha256=sha(source),analyzer_sha256=sha(__file__),
        actual_state_write_source_audit=readonly_audit(source), warmup=warm, runs=reports,
        interpretation='World-anchor IK feedback loses physical foot support after the unchanged1s warmup; compare switch joint errors, load loss, IK feasibility and saturation chronology. Causation beyond measured traces needs a separately declared teacher experiment.',
        limitations='Tarsus5 origin-anchor XY drift is NOT material contact-point slip. Source swing clearance is a geometric target only. No replayed physics, no ANN training, no neural or walking success claim.')
    path=OUT/'ANCHOR_DIAGNOSIS.json'; image=OUT/'ANCHOR_DIAGNOSIS.png'
    for existing in (path,image):
        if existing.exists():
            raise FileExistsError('Preserve prior diagnosis: '+str(existing))
    path.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    plot(traces,image)
    print(json.dumps(dict(report=str(path),image=str(image),warmup=warm,
        events=reports['feedback']['events'],features_recompute_max={k:v['feature_recompute_max'] for k,v in reports.items()}),indent=2))


if __name__=='__main__':
    main()
