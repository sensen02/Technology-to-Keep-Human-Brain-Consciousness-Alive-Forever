"""S203 supplement: agonist/antagonist actuators for the six hard DOFs, via the moment-arm matrix.

The muscles drive joints through tendons, so actuator_trnid is a site, not a joint. The anatomically
meaningful definition of agonist/antagonist for a DOF is therefore the SIGN of the actuator's moment
on that DOF at the frozen reference pose: positive moment = pushes the DOF positive (agonist),
negative moment = pushes it negative (antagonist). No physics stepping here.
"""
import json
import numpy as np
import mujoco
from pathlib import Path

RUN = Path('/home/sensen/Desktop/engine修复验收/s203-attribution-chaos-audit/20261025T072017913193Z')
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
Q0 = np.load('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z/init80ms/TRAJECTORY.npz')['qpos'][0]
HARD = (19, 22, 26, 40, 42, 47)

m = mujoco.MjModel.from_xml_path(str(XML))
# The frozen S154 archive already holds the DENSE moment matrix at the release pose (frame 0),
# reconstructed from MuJoCo's sparse pattern by that round. Use it: no stepping, no forward.
TRAJ = np.load('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z/init80ms/TRAJECTORY.npz')
M = np.asarray(TRAJ['actuator_moment'][0], float)      # (nu, nv) at the frozen release pose
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or '' for i in range(M.shape[0])]
jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or '' for j in range(m.njnt)]
dof_of_joint = {j: int(m.jnt_dofadr[j]) for j in range(m.njnt)}
joint_of_dof = {v: k for k, v in dof_of_joint.items()}
print('moment matrix', M.shape, 'nonzero entries', int((np.abs(M) > 1e-9).sum()))

# ctrl state per arm, pre-collapse window
def ctrl_pre(arm, short=None):
    if short:
        z = np.load(RUN / short / 'TRACE.npz')
        return z['ctrl']
    z = np.load(RUN / arm / 'TRACE.npz')
    t = np.arange(z['ctrl'].shape[0]) * 0.005
    return z['ctrl'][t <= 0.020 + 1e-12]

pre = dict(A=ctrl_pre('A'), B=ctrl_pre('B'))   # 5 s arms: only A and B exist this round
pre_short = dict(A=ctrl_pre(None, 'short3_A'), B=ctrl_pre(None, 'short3_B'),
                 C=ctrl_pre(None, 'short3_C'), D=ctrl_pre(None, 'short3_D'),
                 T=ctrl_pre(None, 'short3_T'))
rowmap = json.load(open('/home/sensen/Desktop/engine修复验收/s179-targeted-legjoint-search/20261025T070000000000Z/ROW_DOF_MAP.json'))
dof_to_row = {r['dof']: r['joint_row'] for r in rowmap['rows']}

out = {}
for dof in HARD:
    j = joint_of_dof.get(dof)
    moment = M[:, dof]
    order = np.argsort(-np.abs(moment))
    acts = []
    for i in order:
        if abs(moment[i]) <= 1e-9:
            continue
        row = dict(actuator=str(names[i]), index=int(i), moment=float(moment[i]),
                   role='agonist' if moment[i] > 0 else 'antagonist')
        for arm, C in pre.items():
            row['%s_pre_at_zero' % arm] = float(np.mean(C[:, i] <= 1e-12))
            row['%s_pre_at_one' % arm] = float(np.mean(C[:, i] >= 1.0 - 1e-12))
        for arm, C in pre_short.items():
            row['%s_short_at_zero' % arm] = float(np.mean(C[:, i] <= 1e-12))
            row['%s_short_at_one' % arm] = float(np.mean(C[:, i] >= 1.0 - 1e-12))
        acts.append(row)
    n_ag = sum(1 for a in acts if a['role'] == 'agonist')
    out[dof] = dict(dof=int(dof), joint=jn[j] if j is not None else None,
                    joint_row_in_42row_map=dof_to_row.get(dof),
                    n_actuators_with_nonzero_moment=len(acts),
                    n_agonist=n_ag, n_antagonist=len(acts) - n_ag,
                    actuators=acts)

# compact answer to the question that was actually asked
summary = {}
for dof, v in out.items():
    A1 = [a for a in v['actuators'] if a['A_short_at_one'] >= 0.999]
    Z0 = [a for a in v['actuators'] if a['A_short_at_zero'] >= 0.999]
    summary[dof] = dict(joint=v['joint'], n_actuators=v['n_actuators_with_nonzero_moment'],
                        at_ceiling_in_A_precollapse=[a['actuator'] for a in A1],
                        at_floor_in_A_precollapse=[a['actuator'] for a in Z0],
                        note='pre-collapse = the 4 command intervals of the 0.02 s instrumented arms')
(RUN / 'HARD_DOF_AGONIST_ANTAGONIST.json').write_text(json.dumps(dict(
    definition='agonist = positive actuator_moment on that DOF at the frozen reference pose; antagonist = negative',
    moment_matrix_source='d.actuator_moment after one mj_forward at the frozen S154 release pose (no stepping)',
    hard_dofs=out, summary=summary), indent=2, default=str))
print(json.dumps(summary, indent=1, default=str))
print(json.dumps({d: dict(joint=out[d]['joint'], n=out[d]['n_actuators_with_nonzero_moment'],
                          ag=out[d]['n_agonist'], ant=out[d]['n_antagonist']) for d in HARD}, indent=1))
