"""Independent re-verification of the claimed 5 s ANN standing, from raw arrays.

Deliberately does NOT import verify_stand.py / verify_extended.py logic. Loads the
frozen joint-torque XML, replays the saved command stream from the keyframe with its
own PD formula, and recomputes every support number from the 12.5 us native rows.
Also quantifies how far the actuator target actually moved (walking would need a
large, sustained excursion; static holding should be near zero), and re-derives the
ANN action from the saved float16 rate frames.
"""
import json
import sys
from pathlib import Path

import numpy as np
import mujoco as mj

sys.path.insert(0, '/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-incremental/20261107T010000Z')
from incremental_actuator import load_full_model

DESKTOP = Path('/home/sensen/Desktop/engine修复验收/s5-ann-incremental/20261107T010000Z/learning')
OUT = Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T024636Z')
WEIGHT = 24.446354755878005
MASS = WEIGHT / 9801.0
H0 = 1.2072528224791905
DT = 0.0000125
NSUB = 80


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')


def load_model():
    model = load_full_model()
    assert (model.nq, model.nv, model.nu, model.nmesh, model.neq) == (49, 48, 42, 71, 0), 'model identity'
    roots = np.flatnonzero(model.jnt_type == mj.mjtJoint.mjJNT_FREE)
    assert len(roots) == 1 and model.jnt_qposadr[roots[0]] == 0, 'free root'
    assert np.array_equal(model.opt.gravity, [0., 0., -9801.]), 'gravity'
    assert int(model.opt.timestep * 1e12) == int(round(DT * 1e12)) or abs(model.opt.timestep - DT) < 1e-12
    return model


def support_gates(native_rows):
    """My own reimplementation of the D gates from raw native rows.

    Uses the same column conventions as the sealed native log:
    t | foot6 | nf_pos | nf_signed | ncontact | com_z | root7 | mean(ctrl^2) | max|qfrc_applied|.
    """
    r = np.asarray(native_rows, float)
    t = r[:, 0]
    foot = r[:, 1:7]
    nf_pos = r[:, 7]
    nf_signed = r[:, 8]
    comz = r[:, 10]
    assist_mask = (t >= 0.8) & (t < 1.0)
    rel_mask = t >= 1.0
    a_foot = foot[assist_mask].sum(1) / WEIGHT
    a_com = comz[assist_mask]
    total = foot[rel_mask].sum(1) + nf_signed[rel_mask]
    support = foot[rel_mask].sum(1) / WEIGHT
    with np.errstate(invalid='ignore', divide='ignore'):
        share = np.where(np.abs(total) > 1e-9, nf_signed[rel_mask] / np.where(np.abs(total) > 1e-9, total, 1.), 0.)
    accel = np.gradient(np.gradient(comz[rel_mask], t[rel_mask]), t[rel_mask])
    accel = np.convolve(accel, np.ones(9) / 9, mode='same')
    free_fall = ((total / WEIGHT < 0.2) & (accel <= -0.5 * 9801.)).mean()
    resid = np.abs(MASS * accel - (total - WEIGHT)) / WEIGHT
    bearing = (foot[rel_mask] > 1e-9).sum(1)
    checks = {
        'D1_no_nonfoot_ground_force': bool(nf_pos[rel_mask].max() <= 0.0),
        'D2_no_free_fall': bool(free_fall <= 0.0),
        'D3_foot_support_not_below_assist': bool(support.mean() >= a_foot.mean() - 0.10),
        'D4_foot_support_never_collapses': bool(np.percentile(support, 10) >= 0.5),
        'D4b_nonfoot_is_not_carrying_the_body': bool(np.nanmean(share) <= 0.2),
        'D5_com_height_not_below_assist': bool(comz[rel_mask].min() >= a_com.mean() - 0.10),
        'D6_force_balance_consistent': bool(np.percentile(resid, 95) <= 0.15),
        'D7_tripod_redundancy': bool(np.median(bearing) >= 3),
        'D8_still_above_absolute_floor': bool(comz[rel_mask].min() >= 0.8 * H0),
    }
    stats = dict(foot_support_mean=float(support.mean()),
                 foot_support_p10=float(np.percentile(support, 10)),
                 nonfoot_positive_max=float(nf_pos[rel_mask].max()),
                 nonfoot_share_mean=float(np.nanmean(share)),
                 com_z_min=float(comz[rel_mask].min()), com_z_mean=float(comz[rel_mask].mean()),
                 bearing_legs_median=float(np.median(bearing)),
                 free_fall_fraction=float(free_fall),
                 force_balance_residual_p95=float(np.percentile(resid, 95)),
                 assist_foot_mean=float(a_foot.mean()), assist_com_mean=float(a_com.mean()),
                 samples=int(rel_mask.sum()))
    return checks, stats


def replay_physics(model, body, commands, native, motor_cut=False):
    """Independent replay: own loop, own contact-force extraction, own PD.

    motor_cut reproduces the run() rule: ctrl is zero from the 1 s handover onward.
    Contact rows are recomputed from the replayed state every 8th native substep and
    compared with the saved 12.5 us records, so support numbers are not trusted from
    the saved file alone.
    """
    d = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, d, 0)
    mj.mj_forward(model, d)
    jids = model.actuator_trnid[:, 0]
    qa = model.jnt_qposadr[jids]
    va = model.jnt_dofadr[jids]
    kp, kd, limits = commands['kp'], commands['kd'], commands['limits']
    target = commands['q0'].copy()
    lo, hi = model.jnt_range[jids, 0], model.jnt_range[jids, 1]
    planes = model.geom_type == mj.mjtGeom.mjGEOM_PLANE
    thorax = model.body('Thorax').id
    legs = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
    feet = np.array([next((i for i, l in enumerate(legs)
                           if (model.geom(g).name or '').startswith(l)
                           and 'Tars' in (model.geom(g).name or '')), -1)
                     for g in range(model.ngeom)])

    def contact_row():
        f = np.zeros(6)
        nf = nfs = 0.0
        nc = 0
        v = np.zeros(6)
        for i in range(d.ncon):
            co = d.contact[i]
            a, b = int(co.geom1), int(co.geom2)
            if not (planes[a] or planes[b]):
                continue
            mj.mj_contactForce(model, d, i, v)
            z = float((co.frame.reshape(3, 3).T @ v[:3])[2]) * (1 if planes[a] else -1)
            li = feet[b if planes[a] else a]
            if li < 0:
                nf += max(0.0, z)
                nfs += z
                nc += 1
            else:
                f[li] += z
        return np.r_[d.time, f, nf, nfs, nc, d.subtree_com[thorax, 2], d.qpos[:7],
                     np.mean(d.ctrl ** 2), np.max(np.abs(d.qfrc_applied[:6]))]

    errs = dict(qpos=0., qvel=0., ctrl=0., time=0., contact=0.)
    applied_max = 0.
    for k, row in enumerate(commands['rows']):
        rate = np.clip(row[43:85] + row[85:127], -4., 4.)
        target = np.clip(target + 0.001 * rate, lo, hi)
        for s in range(NSUB):
            d.ctrl[:] = 0. if (motor_cut and k >= 1000) else np.clip(
                kp * (target - d.qpos[qa]) - kd * d.qvel[va], -limits, limits)
            mj.mj_step(model, d)
            mj.mj_forward(model, d)
            applied_max = max(applied_max, float(np.max(np.abs(d.qfrc_applied))))
            if s % 8 == 0:
                errs['contact'] = max(errs['contact'],
                                      float(np.max(np.abs(contact_row() - native[k * NSUB + s]))))
        if k % 5 == 4:
            i = k // 5
            for key, actual in (('qpos', d.qpos), ('qvel', d.qvel), ('ctrl', d.ctrl), ('time', d.time)):
                errs[key] = max(errs[key], float(np.max(np.abs(actual - body[key][i]))))
    return errs, applied_max


def ann_recompute(translator_path, frames, target_states, actions, start, frozen=False):
    """Own forward pass through exported weights; compare to saved actions.

    Windows are padded by repeating the oldest available frame, exactly like the
    runtime's history buffer does at t=0 (saved frames are float16 snapshots).
    """
    with np.load(translator_path, allow_pickle=False) as z:
        w = {k: z[k] for k in z.files if k != 'metadata_json'}
    hist = np.array([frames[np.maximum(0, np.arange(i - 3, i + 1))]
                     for i in range(start, len(frames))], np.float32)
    if frozen:
        hist[:] = hist[0]
    ts = target_states[start:].astype(np.float32)
    proj = hist @ w['projection.weight'].T + w['projection.bias']
    proj = proj.transpose(0, 2, 1)
    conv = np.einsum('bct,oct->bo', proj, w['temporal.weight']) + w['temporal.bias']
    conv = np.tanh(conv)
    hidden = np.tanh(np.concatenate((conv, ts), -1) @ w['hidden.weight'].T + w['hidden.bias'])
    out = np.tanh(hidden @ w['output.weight'].T + w['output.bias']) * 4.0
    saved = actions[start:]
    return float(np.max(np.abs(out - saved))), float(np.mean(np.abs(out - saved)))


def target_excursion(commands):
    """How far did the accumulated actuator target move from neutral?"""
    target = commands['q0'].copy()
    lo = np.full(42, -np.inf)
    hi = np.full(42, np.inf)
    span = np.zeros(42)
    maxdev = 0.
    for row in commands['rows']:
        rate = np.clip(row[43:85] + row[85:127], -4., 4.)
        target = target + 0.001 * rate
        span = np.maximum(span, np.abs(target - commands['q0']))
        maxdev = max(maxdev, float(np.max(np.abs(target - commands['q0']))))
    return dict(max_target_deviation_rad=maxdev,
                per_joint_max_deviation=span.tolist(),
                mean_joint_max_deviation=float(span.mean()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_model()
    phases = ['eval_ann_dt0.125', 'eval_brain_cut_dt0.125', 'eval_hold_dt0.125',
              'eval_motor_cut_dt0.125', 'eval_ann_dt0.0625',
              'extended_ann_5s', 'cold_ann', 'frozen_brain_history']
    checks = {}
    details = []
    for tag in phases:
        folder = DESKTOP / tag
        with np.load(folder / 'NATIVE.npz') as z:
            native = z['rows']
        with np.load(folder / 'BODY.npz') as z:
            body = {k: z[k] for k in z.files}
        with np.load(folder / 'COMMANDS.npz') as z:
            commands = {k: z[k] for k in z.files}
        with np.load(folder / 'DATA.npz') as z:
            data = {k: z[k] for k in z.files}
        saved = json.loads((folder / 'RESULT.json').read_text())
        gates, stats = support_gates(native)
        checks[tag + '_gate_recomputed'] = (gates == saved['validation']['checks'])
        checks[tag + '_passed_flag_matches'] = (all(gates.values()) == saved['validation']['passed'])
        checks[tag + '_native_clock'] = bool(np.max(np.abs(np.diff(native[:, 0]) - DT)) < 1e-12)
        # Physics replay for every saved run; motor_cut must switch ctrl off at 1 s.
        delta, applied_max = replay_physics(model, body, commands, native,
                                            motor_cut=(tag == 'eval_motor_cut_dt0.125'))
        checks[tag + '_replay_bitwise'] = max(delta.values()) < 1e-9
        checks[tag + '_no_external_force'] = applied_max == 0.0
        checks[tag + '_no_native_qfrc'] = float(np.abs(native[:, 19]).max()) == 0.0
        start = 0 if tag == 'cold_ann' else 200
        if tag in ('eval_hold_dt0.125', 'eval_motor_cut_dt0.125'):
            # hold/motor_cut never apply an ANN action after handover: pending is zero.
            # Before the 1 s handover the shared teacher preconditioning still applies.
            checks[tag + '_ann_action_reproduced'] = bool(np.all(np.abs(data['actions'][start:]) < 1e-12))
            ann_err, ann_mean = 0.0, 0.0
        else:
            if tag == 'eval_brain_cut_dt0.125':
                frames = np.zeros_like(data['frames'])
            else:
                frames = data['frames']
            ann_err, ann_mean = ann_recompute(DESKTOP / 'TRANSLATOR.npz', frames,
                                              data['target_states'], data['actions'], start,
                                              frozen=(tag == 'frozen_brain_history'))
            checks[tag + '_ann_action_reproduced'] = ann_err < 0.005
        exc = target_excursion(commands)
        details.append(dict(tag=tag, computed=gates, saved_passed=saved['validation']['passed'],
                            replay_max_error=delta, applied_force_max=applied_max,
                            ann_max_error_rad_s=ann_err, ann_mean_error_rad_s=ann_mean,
                            target_excursion=exc, stats=stats))
        print(json.dumps(dict(tag=tag, passed=saved['validation']['passed'],
                              replay_ok=checks[tag + '_replay_bitwise'],
                              max_target_dev_rad=exc['max_target_deviation_rad'])), flush=True)
    # Video provenance: source hash and frame count.
    vmeta = json.loads(Path('/home/sensen/Desktop/engine修复验收/s5-ann-incremental/20261107T010000Z/ANN_STANDING_4x_slow.json').read_text())
    src = Path(vmeta['source'])
    import hashlib
    src_sha = hashlib.sha256(src.read_bytes()).hexdigest()
    checks['video_source_hash'] = src_sha == vmeta['source_sha256']
    checks['video_frames'] = vmeta['frames'] == 720
    checks['video_visible_meshes'] = vmeta['visible_meshes'] == 71
    checks['video_replay_com_error_zero'] = vmeta['replay_com_max_error'] == 0.0
    result = dict(importance='核心必需',
                  description='independent recomputation of the claimed 5 s ANN standing from raw arrays',
                  checks=checks, passed=all(checks.values()), details=details,
                  video_meta=dict(source=src_sha[:16], slow_factor=vmeta['slow_factor']),
                  image_visual_inspection='NOT PERFORMED: model has no image input capability in this session; not claimed',
                  caveat='Replay confirms the saved trajectory is self-consistent physics; it does not by itself prove neural necessity. brain_cut/frozen/cold comparisons are reported separately.')
    dump(OUT / 'INDEPENDENT_VERIFICATION.json', result)
    print(json.dumps(dict(passed=result['passed'], n_checks=len(checks),
                          failed=[k for k, v in checks.items() if not v])), flush=True)
    if not result['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
