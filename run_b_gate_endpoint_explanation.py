"""B-gate offline end-to-end explanation: first-difference sample indices and the LF tibia moment sign.

Reads the two archived B-gate artifacts (the one bounded 25 us fixed-command replay and the frozen P1
control archive) and answers, per link of the chain, at WHICH sample the recomputed link first differs
from the archived one, at declared tolerances:

  L1 command            : archived per-step ctrl vs the P1 archive's 10 x 90 commands held 5 ms
  L2 activation         : archived act vs the archived independent activation reference (DOP853)
  L3 target generalised : archived qfrc_actuator vs the value recomputed from the archived state+ctrl
     force                (mj_forward per sample; no integration, no new trajectory)
  L4 total force terms  : per-sample L1/L2 aggregates of the same arrays, plus the plane of
                          qfrc_bias / qfrc_passive sums
  L5 LF tibia sign      : the signed contribution of the actuators that load joint_LFTibia_pitch, and
                          the first sample at which the recomputed sign disagrees with the archive

Offline: no mj_step, no new dynamics, no model/gain change. Every number is an archived-chain
consistency statement, NOT a behaviour, support, standing or biology claim.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent
DESK = Path('/home/sensen/Desktop/engine修复验收')
REPLAY = DESK / 'B-fixed-command-replay-25us-20261015T103243/replay.npz'
P1 = ROOT / 'outputs/experiment_platform/20261013_233237_7be93a1d'
TOLS = (0.0, 1e-12, 1e-9, 1e-6, 1e-4)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def first_over(series, tol):
    idx = np.nonzero(np.asarray(series) > tol)[0]
    return int(idx[0]) if idx.size else None


def main():
    out = DESK / ('B-gate-endpoint-explanation-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(status='RUNNING', manual_research_not_product=True, standing=False, walking=False,
                  scope=('B-gate offline end-to-end explanation: first-difference sample index per chain link '
                         'and the LF tibia moment sign'),
                  declared_calls=dict(project_mj_steps=0, project_model_loads=1),
                  tolerances=list(TOLS))
    inputs = [Path(__file__), REPLAY, P1 / 'traces.npz', P1 / 'result.json']
    hashes = {str(p): sha(p) for p in inputs}
    report['input_hashes'] = hashes
    try:
        import mujoco
        from engine import canonical_physics as cp
        m = cp.load_model()
        d = mujoco.MjData(m)
        report['canonical_xml'] = str(cp.CANONICAL_XML)
        report['canonical_xml_sha256'] = sha(cp.CANONICAL_XML)
        z = np.load(REPLAY)
        qpos, qvel, act, ctrl = z['qpos'], z['qvel'], z['act'], z['ctrl']
        qfrc = z['qfrc_actuator']
        act_ref = z['activation_reference']
        commands = z['commands']
        steps = qpos.shape[0]
        report['declared_calls']['project_forward_calls'] = steps

        # ---- L1 command chain
        hold = steps // commands.shape[0]
        expanded = np.repeat(commands, hold, axis=0)
        n = min(expanded.shape[0], ctrl.shape[0])
        cmd_diff = np.max(np.abs(ctrl[:n] - expanded[:n]), axis=1)
        report['L1_command'] = dict(
            per_step_max_abs_diff_max=float(cmd_diff.max()),
            bitwise_identical=bool(np.array_equal(ctrl[:n], expanded[:n])),
            first_difference_index={str(t): first_over(cmd_diff, t) for t in TOLS},
            nonzero_command_scalars=int(np.count_nonzero(commands)),
            note=('the command link is a copy of the frozen archive: at tolerance 0 the first difference index '
                  'is the one reported here'))

        # ---- L2 activation chain
        act_diff = np.max(np.abs(act - act_ref), axis=1)
        report['L2_activation'] = dict(
            per_step_max_abs_diff_max=float(act_diff.max()),
            rms=float(np.sqrt(np.mean((act - act_ref) ** 2))),
            first_difference_index={str(t): first_over(act_diff, t) for t in TOLS},
            note=('reference is the archived independent DOP853 activation reference on the native muscle ODE '
                  'with u_eff = clip(ctrl, 0.0001, 1); the replay integrates with RK4 at 25 us'))

        # ---- L3 target generalised force chain (recompute per sample, no integration)
        recomputed = np.zeros_like(qfrc)
        bias_sum = np.zeros(steps)
        passive_sum = np.zeros(steps)
        for i in range(steps):
            d.qpos[:] = qpos[i]
            d.qvel[:] = qvel[i]
            d.act[:] = act[i]
            d.ctrl[:] = ctrl[i]
            mujoco.mj_forward(m, d)
            recomputed[i] = d.qfrc_actuator
            bias_sum[i] = float(np.sum(np.abs(d.qfrc_bias)))
            passive_sum[i] = float(np.sum(np.abs(d.qfrc_passive)))
        gf_diff = np.max(np.abs(recomputed - qfrc), axis=1)
        report['L3_target_generalised_force'] = dict(
            per_step_max_abs_diff_max=float(gf_diff.max()),
            first_difference_index={str(t): first_over(gf_diff, t) for t in TOLS},
            worst_sample_index=int(np.argmax(gf_diff)),
            worst_sample_value=float(gf_diff.max()),
            note=('the known semantics caveat: after mj_step the archived force fields belong to an RK4 '
                  'intermediate stage, so a mismatch concentrated at command-switch samples is expected and '
                  'is a property of the archive, not of the chain identity'))

        # ---- L4 total force terms
        l1 = np.abs(qfrc).sum(axis=1)
        l1r = np.abs(recomputed).sum(axis=1)
        l2 = np.sqrt((qfrc ** 2).sum(axis=1))
        l2r = np.sqrt((recomputed ** 2).sum(axis=1))
        report['L4_total_force_terms'] = dict(
            l1_first_difference_index={str(t): first_over(np.abs(l1 - l1r), t) for t in TOLS},
            l2_first_difference_index={str(t): first_over(np.abs(l2 - l2r), t) for t in TOLS},
            l1_max_abs_diff=float(np.max(np.abs(l1 - l1r))),
            l2_max_abs_diff=float(np.max(np.abs(l2 - l2r))),
            archived_l1_first_last=[float(l1[0]), float(l1[-1])],
            recomputed_l1_first_last=[float(l1r[0]), float(l1r[-1])],
            bias_abs_sum_first_last=[float(bias_sum[0]), float(bias_sum[-1])],
            passive_abs_sum_first_last=[float(passive_sum[0]), float(passive_sum[-1])])

        # ---- L5 LF tibia moment sign (and the wider actuation-map question)
        def dof_arms(i):
            out = []
            for dof in range(m.nv):
                adr = int(d.moment_rowadr[dof]); nnz = int(d.moment_rownnz[dof])
                cols = np.asarray(d.moment_colind)[adr:adr + nnz]
                vals = np.asarray(d.actuator_moment)[adr:adr + nnz]
                for c, v in zip(cols, vals):
                    if int(c) == i and abs(float(v)) > 0:
                        out.append((dof, float(v)))
            return out

        jn = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
        dof_joint = {}
        for j in range(m.njnt):
            dof_joint[int(m.jnt_dofadr[j])] = jn(j)
        for dof in range(6):
            dof_joint[dof] = 'root_free_dof_%d' % dof
        an = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ''
        d.qpos[:] = qpos[0]; d.qvel[:] = qvel[0]; d.act[:] = act[0]; d.ctrl[:] = ctrl[0]
        mujoco.mj_forward(m, d)
        zero_arm = []
        arm_table = {}
        for i in range(m.nu):
            arms = dof_arms(i)
            arm_table[an(i)] = [dict(dof=d, joint=dof_joint.get(d), arm=a) for d, a in arms]
            if not arms:
                zero_arm.append(an(i))
        lf_joint = m.joint('joint_LFTibia_pitch').id
        lf_dof = int(m.jnt_dofadr[lf_joint])
        lf_tibia = [an(i) for i in range(m.nu) if an(i).startswith('LFTibia')]
        lf_tibia_arms = {n: arm_table[n] for n in lf_tibia}
        archived_dof = qfrc[:, lf_dof]
        recomputed_dof = recomputed[:, lf_dof]
        sgn_a = np.sign(archived_dof); sgn_r = np.sign(recomputed_dof)
        disagree = np.nonzero((sgn_a != sgn_r) & (sgn_a != 0) & (sgn_r != 0))[0]
        report['L5_LF_tibia_moment_sign'] = dict(
            joint='joint_LFTibia_pitch', dof_index=lf_dof,
            lf_tibia_actuators=lf_tibia,
            lf_tibia_moment_arms_at_sample_0=lf_tibia_arms,
            lf_tibia_arm_is_zero_everywhere=[n for n in lf_tibia if not arm_table[n]],
            archived_dof_min_max=[float(archived_dof.min()), float(archived_dof.max())],
            recomputed_dof_min_max=[float(recomputed_dof.min()), float(recomputed_dof.max())],
            dof_value_max_abs_difference=float(np.max(np.abs(archived_dof - recomputed_dof))),
            first_sample_with_sign_disagreement=(int(disagree[0]) if disagree.size else None),
            sign_is_undefined_because_arm_is_zero=bool(all(not arm_table[n] for n in lf_tibia)),
            interpretation=('the LF tibia muscles carry NO moment arm on any degree of freedom at this pose, '
                            'so no moment sign exists for them here and they cannot load the LF leg at all; the '
                            'archived and recomputed generalised force on joint_LFTibia_pitch agree in sign '
                            'wherever both are nonzero (see first_sample_with_sign_disagreement). This is a '
                            'property of the declared actuation map at this pose, not of the replay.'))
        report['actuation_map_at_the_seated_pose'] = dict(
            muscles_with_zero_moment_arm_on_every_dof=len(zero_arm),
            total_muscles=int(m.nu),
            zero_arm_names=zero_arm,
            tibia_muscles_total=len([i for i in range(m.nu) if 'Tibia' in an(i)]),
            tibia_muscles_with_nonzero_arm=[n for n in arm_table if 'Tibia' in n and arm_table[n]],
            cross_leg_arms={n: [a for a in arm_table[n]
                                if abs(a['arm']) > 1e-6
                                and not str(a['joint']).startswith('joint_' + n[:2])
                                and not str(a['joint']).startswith('root_free_dof')]
                            for n in arm_table
                            if arm_table[n] and [a for a in arm_table[n]
                                                 if abs(a['arm']) > 1e-6
                                                 and not str(a['joint']).startswith('joint_' + n[:2])
                                                 and not str(a['joint']).startswith('root_free_dof')]},
            root_dof_arms={n: [a for a in arm_table[n] if str(a['joint']).startswith('root_free_dof')]
                           for n in arm_table if arm_table[n]
                           and [a for a in arm_table[n] if str(a['joint']).startswith('root_free_dof')]},
            same_leg_residue_note=('arms below 1e-6 are treated as numerical residue, not as transmission: the '
                                   'earlier version of this field used a prefix test that such residue satisfied, '
                                   'which hid the two right-front tibia muscles'),
            note=('recorded because it bounds what any downstream force/activation analysis on this pose can '
                  'mean: a muscle with zero moment arm transmits nothing, whatever its activation or force.'))

        report['first_difference_summary'] = {
            'command_L1': report['L1_command']['first_difference_index']['0.0'],
            'activation_L2_at_1e-12': report['L2_activation']['first_difference_index']['1e-12'],
            'activation_L2_at_1e-04': report['L2_activation']['first_difference_index']['0.0001'],
            'target_generalised_force_L3_at_1e-12': report['L3_target_generalised_force'][
                'first_difference_index']['1e-12'],
            'total_force_terms_L4_L1_at_1e-12': report['L4_total_force_terms'][
                'l1_first_difference_index']['1e-12'],
            'LF_tibia_sign_L5': report['L5_LF_tibia_moment_sign']['first_sample_with_sign_disagreement'],
            'muscles_with_zero_moment_arm_everywhere': report['actuation_map_at_the_seated_pose'][
                'muscles_with_zero_moment_arm_on_every_dof'],
            'reading':('indices are sample numbers in the archived 2000-step replay (25 us, 50 ms). A null '
                       'index means the link stayed within that tolerance for the whole run. These are '
                       'consistency statements about the archived chain, not behaviour evidence.')}
        report['status'] = 'PASS_B_GATE_ENDPOINT_EXPLANATION_REPORTED'
        np.savez_compressed(out / 'endpoint_series.npz',
                            cmd_diff=cmd_diff, act_diff=act_diff, gf_diff=gf_diff,
                            l1=l1, l1r=l1r, archived_lf_tibia=archived_dof, recomputed_lf_tibia=recomputed_dof)

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
        for series, lab in ((cmd_diff, 'L1 command'), (act_diff, 'L2 activation'),
                            (gf_diff, 'L3 target gen. force')):
            axes[0].semilogy(np.arange(len(series)), np.maximum(series, 1e-18), lw=0.8, label=lab)
        axes[0].set_xlabel('sample (25 us)')
        axes[0].set_ylabel('max abs difference vs archive')
        axes[0].set_title('first-difference chains (log)')
        axes[0].legend(fontsize=8)
        axes[1].plot(np.arange(steps), l1, lw=0.8, label='archived L1')
        axes[1].plot(np.arange(steps), l1r, lw=0.8, ls='--', label='recomputed L1')
        axes[1].set_xlabel('sample (25 us)')
        axes[1].set_ylabel('sum |qfrc_actuator|')
        axes[1].set_title('L4 total force terms')
        axes[1].legend(fontsize=8)
        axes[2].plot(np.arange(steps), archived_dof, lw=0.8, label='archived LF tibia dof')
        axes[2].plot(np.arange(steps), recomputed_dof, lw=0.8, ls='--', label='recomputed LF tibia dof')
        axes[2].axhline(0, color='k', lw=0.6)
        axes[2].set_xlabel('sample (25 us)')
        axes[2].set_ylabel('generalised force on joint_LFTibia_pitch')
        axes[2].set_title('L5 LF tibia moment sign')
        axes[2].legend(fontsize=8)
        fig.suptitle('B-gate offline endpoint explanation - archived-chain consistency, NOT behaviour/standing')
        fig.tight_layout()
        fig.savefig(out / 'endpoint-explanation.png', dpi=130)
        plt.close(fig)
    except Exception as exc:
        import traceback
        report.update(status='FAIL_STOP', error=type(exc).__name__ + ': ' + str(exc),
                      traceback=traceback.format_exc()[-1500:])
    finally:
        report['input_hashes_after'] = {p: sha(p) for p in hashes}
        report['hashes_unchanged'] = report['input_hashes_after'] == hashes
        if not report['hashes_unchanged']:
            report['status'] = 'FAIL_INPUT_MUTATED'
        with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(Path(__file__), Path(__file__).name)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False,
                                                    default=lambda o: o.tolist()) + '\n')
        (out / 'delivery-hashes.json').write_text(json.dumps(
            {f.name: sha(f) for f in sorted(out.iterdir())
             if f.is_file() and f.name != 'delivery-hashes.json'}, indent=2) + '\n')
    print(out)
    print(report['status'])
    if report['status'].startswith('PASS'):
        print('summary:', json.dumps(report['first_difference_summary'], indent=1))
        print('LF tibia arms:', report['L5_LF_tibia_moment_sign']['lf_tibia_moment_arms_at_sample_0'])
        print('zero-arm muscles:', report['actuation_map_at_the_seated_pose']['muscles_with_zero_moment_arm_on_every_dof'], '/', report['actuation_map_at_the_seated_pose']['total_muscles'])
    else:
        print(report.get('error', ''))
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
