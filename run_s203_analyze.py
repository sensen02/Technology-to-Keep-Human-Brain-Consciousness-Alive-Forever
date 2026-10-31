"""S203 analysis: ctrl divergence, saturation by actuator, clamp audit, sensitivity, D mechanism."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RUN = Path('/home/sensen/Desktop/engine修复验收/s203-attribution-chaos-audit/20261025T072017913193Z')
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
HARD_DOF = (19, 22, 26, 40, 42, 47)   # user-specified qvel/DOF indices

def load(name):
    s = json.load(open(RUN / name / 'SUMMARY.json'))
    z = np.load(RUN / name / 'TRACE.npz')
    return s, z

S = {}
Z = {}
for name in ('A', 'A_d1', 'A_d2', 'B', 'B_d1', 'short_A', 'short_B', 'short_C', 'short_D',
             'short2_A', 'short2_D', 'short3_A', 'short3_B', 'short3_C', 'short3_D', 'short3_T'):
    S[name], Z[name] = load(name)

names = S['A']['actuator_names']
jnames = S['A']['joint_names']
nu = len(names)

# ---------------- actuator -> joint map + direction sign (one model load, no stepping) --------
import mujoco
m = mujoco.MjModel.from_xml_path(str(XML))
act_joint = {}
act_trntype = {}
for i in range(m.nu):
    act_trntype[i] = int(m.actuator_trntype[i])
    adr = int(m.actuator_trnid[i, 0])
    act_joint[i] = adr if act_trntype[i] == 0 else -1   # 0 == mjTRN_JOINT
dof_of_joint = {j: int(m.jnt_dofadr[j]) for j in range(m.njnt)}
joint_names_model = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or '' for j in range(m.njnt)]

def diverg(a, b, times, tau):
    """First time index and index set where |a-b| > tau."""
    diff = np.abs(a - b)
    hit = diff > tau
    if not hit.any():
        return None
    k = int(np.argmax(hit.any(axis=1)))
    return dict(command_index=k, time_s=float(times[k]), max_abs=float(diff.max()),
                n_values_differ=int(hit.sum()), n_commands_differ=int(hit.any(axis=1).sum()),
                n_actuators_differ=int(hit.any(axis=0).sum()),
                actuator_indices=sorted({int(a) for a in np.where(hit)[0]}))

def channel_separation():
    """Which tactile channel actually reaches the motor readout? Bitwise relations between arms."""
    def eq(x, y):
        return dict(ctrl=bool(np.array_equal(Z[x]['ctrl'], Z[y]['ctrl'])),
                    qpos=bool(np.array_equal(Z[x]['qpos'], Z[y]['qpos'])),
                    decoder_pools=bool(np.array_equal(Z[x]['pool_spikes'], Z[y]['pool_spikes'])))
    pools = {k: Z[k]['pool_spikes'].sum(axis=0) for k in
             ('short3_A', 'short3_B', 'short3_C', 'short3_D', 'short3_T')}
    changed = {k: int((pools[k] != pools['short3_A']).sum()) for k in pools}
    tot = {k: int(S[k]['total_spikes']) for k in pools}
    pool_sum = {k: int(pools[k].sum()) for k in pools}
    return dict(
        relations=dict(A_vs_T=eq('short3_A', 'short3_T'), A_vs_B=eq('short3_A', 'short3_B'),
                       A_vs_C=eq('short3_A', 'short3_C'), A_vs_D=eq('short3_A', 'short3_D'),
                       B_vs_T=eq('short3_B', 'short3_T')),
        decoder_pools_changed_vs_A=changed,
        total_spikes=tot, pool_spikes_sum=pool_sum,
        readout_fraction={k: (pool_sum[k] / tot[k] if tot[k] else None) for k in tot},
        reading=('Touch-only T is BITWISE identical to A (ctrl, qpos and every decoder pool) even though it adds %d spikes '
                 '(%.1f%% of its spikes are inside decoder pools, versus %.1f%% for A). Positive force B changes %d of 90 pools; '
                 'negative force D changes 0; C (force/4.07) changes %d. So the tactile-to-motor route is the contact FORCE '
                 'channel only, and only in its positive direction.'
                 % (tot['short3_T'] - tot['short3_A'], 100 * readout_fraction_of('short3_T'),
                    100 * readout_fraction_of('short3_A'), changed['short3_B'], changed['short3_C'])))


def readout_fraction_of(k):
    pools = Z[k]['pool_spikes'].sum(axis=0)
    tot = int(S[k]['total_spikes'])
    return float(pools.sum()) / tot if tot else 0.0


def ctrl_divergence_report():
    out = {}
    # pre-collapse: 4 command intervals from the 0.02 s instrumented arms
    pre_t = np.arange(Z['short_A']['ctrl'].shape[0]) * 0.005
    for arm in ('B', 'C', 'D'):
        d = diverg(Z['short_A']['ctrl'], Z['short_' + arm]['ctrl'], pre_t, 0.0)
        d_exact = diverg(Z['short_A']['ctrl'], Z['short_' + arm]['ctrl'], pre_t, 1e-12)
        out['pre_collapse_A_vs_' + arm] = dict(
            window_s=[0.0, 0.020], n_commands=4, body_steps_per_command=200,
            bitwise_identical=bool(np.array_equal(Z['short_A']['ctrl'], Z['short_' + arm]['ctrl'])),
            exact=d_exact, any_float_difference=d,
            body_steps_differing=None if d is None else 200 * d['n_commands_differ'],
            note='ctrl is piecewise constant over 200 body steps, so 4 command intervals = 800 body steps')
    # post-collapse: 5 s arms at command cadence
    tA = np.arange(Z['A']['ctrl'].shape[0]) * 0.005
    post = tA > 0.020 + 1e-12
    for arm in ('B',):
        d = diverg(Z['A']['ctrl'][post], Z[arm]['ctrl'][post], tA[post], 0.0)
        out['post_collapse_A_vs_' + arm] = dict(
            window_s=[0.020, 5.0], n_commands=int(post.sum()),
            bitwise_identical=bool(np.array_equal(Z['A']['ctrl'][post], Z[arm]['ctrl'][post])),
            exact=d,
            body_steps_differing=None if d is None else 200 * d['n_commands_differ'])
    out['post_collapse_C_D_not_computed'] = (
        'C and D post-collapse ctrl divergence would need 2 more 200000-step arms; the frozen 1100000-step '
        'cap does not allow it, so it is NOT_COMPUTED, not zero. Their pre-collapse divergence IS computed above.')
    # when does A vs B first diverge over the whole 5 s?
    whole = diverg(Z['A']['ctrl'], Z['B']['ctrl'], tA, 0.0)
    out['A_vs_B_whole_run_first_divergence'] = whole
    # first-command probe: identical physical state, no dynamics yet, only the tactile channel differs
    first_cmd = {}
    for arm, key in (('A', 'short3_A'), ('B', 'short3_B'), ('C', 'short3_C'), ('D', 'short3_D'),
                     ('T', 'short3_T')):
        z = Z[key]
        s = S[key]
        first_cmd[arm] = dict(
            initial_pose_tarsus_forces_N=[None] * 0 + [float(v) for v in z['force'][0]],
            spikes_in_first_command=int(z['spikes'][0]),
            ctrl_max_first_command=float(np.max(z['ctrl'][0])),
            ctrl_nonzero_actuators_first_command=int(np.sum(z['ctrl'][0] > 1e-12)),
            ctrl_max_last_command=float(np.max(z['ctrl'][-1])),
            total_spikes=int(s['total_spikes']))
    out['first_command_probe'] = dict(
        note=('at t=0 the fly sits at the frozen S154 release pose and every tarsus already touches the floor, so the tactile '
              'force is nonzero before any dynamics. The first command is therefore a clean causal probe: identical physics, '
              'identical neurons, identical decoder, only the tactile channel differs.'),
        per_arm=first_cmd,
        reading=('arm A (no tactile) emits 0 spikes in the first 0.5 ms block and therefore a completely zero first command. '
                 'Arm B (raw force) emits spikes and drives its strongest actuator to 1.0 in that same first command. '
                 'Arm C (/weight_per_leg) reaches 0.2719. Arm D (negative force) emits %d spikes but its first command is still '
                 'all zero. The touch-only control T also has an all-zero first command despite %d spikes, so the bristle '
                 'channel does not reach the motor readout at all: the force channel is the only tactile channel that does.'
                 % (first_cmd['D']['spikes_in_first_command'], first_cmd['T']['spikes_in_first_command'])),
        channel_separation=channel_separation())
    return out

CD = ctrl_divergence_report()
(RUN / 'CTRL_DIVERGENCE.json').write_text(json.dumps(CD, indent=2, default=str))

# ---------------- saturation by actuator -----------------------------------------------------
def fracs(ctrl, mask):
    sub = ctrl[mask]
    return (float(np.mean(sub <= 1e-12)) if sub.size else None,
            float(np.mean(sub >= 1.0 - 1e-12)) if sub.size else None)

rows = []
A5, B5 = Z['A']['ctrl'], Z['B']['ctrl']
tA = np.arange(A5.shape[0]) * 0.005
pre5 = tA <= 0.020 + 1e-12
post5 = ~pre5
for i in range(nu):
    jid = act_joint[i]
    jn = joint_names_model[jid] if 0 <= jid < m.njnt else ''
    dof = dof_of_joint.get(jid, -1)
    gear = float(m.actuator_gear[i, 0])
    a_zp, a_op = fracs(A5[:, i], pre5)
    a_zq, a_oq = fracs(A5[:, i], post5)
    b_zp, b_op = fracs(B5[:, i], pre5)
    b_zq, b_oq = fracs(B5[:, i], post5)
    rows.append(dict(
        actuator_index=i, actuator=names[i], joint=jn, dof=dof,
        hard_dof=bool(dof in HARD_DOF), direction='positive' if gear > 0 else 'negative',
        gear=gear,
        A_pre_zero=a_zp, A_pre_one=a_op, A_post_zero=a_zq, A_post_one=a_oq,
        B_pre_zero=b_zp, B_pre_one=b_op, B_post_zero=b_zq, B_post_one=b_oq,
        pre_ctrl_differs=bool(np.any(np.abs(A5[pre5, i] - B5[pre5, i]) > 0)),
        post_ctrl_differs=bool(np.any(np.abs(A5[post5, i] - B5[post5, i]) > 0)),
        max_abs_diff_pre=float(np.max(np.abs(A5[pre5, i] - B5[pre5, i]))) if pre5.any() else None,
        max_abs_diff_post=float(np.max(np.abs(A5[post5, i] - B5[post5, i]))) if post5.any() else None))
import csv
with open(RUN / 'SATURATION_BY_ACTUATOR.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# hard-DOF summary: agonist/antagonist pairs for each requested DOF index
hard = {}
for dof in HARD_DOF:
    js = [j for j, dd in dof_of_joint.items() if dd == dof]
    if not js:
        hard[dof] = dict(found=False, note='no joint has this dof index (root dofs 0-5 excluded from the 42-row map)')
        continue
    jid = js[0]
    acts = [r for r in rows if r['dof'] == dof]
    hard[dof] = dict(found=True, joint=joint_names_model[jid],
                     actuators=[dict(actuator=r['actuator'], direction=r['direction'],
                                     A_pre_zero=r['A_pre_zero'], A_pre_one=r['A_pre_one'],
                                     A_post_zero=r['A_post_zero'], A_post_one=r['A_post_one'],
                                     B_pre_zero=r['B_pre_zero'], B_pre_one=r['B_pre_one'],
                                     differs_pre=r['pre_ctrl_differs']) for r in acts])
# also map each requested DOF index to its 42-row joint_row for cross-checking
rowmap = json.load(open('/home/sensen/Desktop/engine修复验收/s179-targeted-legjoint-search/20261025T070000000000Z/ROW_DOF_MAP.json'))
dof_to_row = {r['dof']: r['joint_row'] for r in rowmap['rows']}
for dof in HARD_DOF:
    hard[dof]['joint_row_in_42row_map'] = dof_to_row.get(dof)
    hard[dof]['dof_name_from_row_map'] = next((r['dof_name'] for r in rowmap['rows'] if r['dof'] == dof), None)

summary_pre = dict(
    n_values=float(pre5.sum() * nu),
    A_zero=float(np.mean(A5[pre5] <= 1e-12)), A_one=float(np.mean(A5[pre5] >= 1.0 - 1e-12)),
    B_zero=float(np.mean(B5[pre5] <= 1e-12)), B_one=float(np.mean(B5[pre5] >= 1.0 - 1e-12)))
summary_pre['A_at_rail'] = summary_pre['A_zero'] + summary_pre['A_one']
summary_pre['B_at_rail'] = summary_pre['B_zero'] + summary_pre['B_one']
summary_post = dict(
    A_zero=float(np.mean(A5[post5] <= 1e-12)), A_one=float(np.mean(A5[post5] >= 1.0 - 1e-12)),
    B_zero=float(np.mean(B5[post5] <= 1e-12)), B_one=float(np.mean(B5[post5] >= 1.0 - 1e-12)))
summary_post['A_at_rail'] = summary_post['A_zero'] + summary_post['A_one']

SAT = dict(rows_csv='SATURATION_BY_ACTUATOR.csv', n_actuators=nu,
           pre_collapse_window=summary_pre, post_collapse_window=summary_post,
           hard_dofs=hard,
           note=('window definition: pre = command intervals with t <= 0.020 s (5 samples of the 5 s run, '
                 'the 4 full 0.005 s intervals plus the sample stamped 0.020), post = the remaining 995; '
                 'DOF indices here are qvel/DOF indices as the round requested, and joint_row_in_42row_map '
                 'gives the 42-balance-row number so the two numbering schemes are not mixed'))
(RUN / 'SATURATION_BY_ACTUATOR.json').write_text(json.dumps(SAT, indent=2, default=str))

# ---------------- clamp path audit ------------------------------------------------------------
def lines_of(path, needles):
    hits = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        for nd in needles:
            if nd in line:
                hits.append(dict(line=i, text=line.strip()))
    return hits

enc_src = '/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/sensory_channels.py'
cond_src = '/run/media/sensen/Data2/cell_wound_prototype/engine/neural_cond.py'
mot_src = '/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/motor_output.py'
Acur = Z['short2_A']['current']
Dcur = Z['short2_D']['current']
CLA = dict(
    path='sensory quantity -> encoder -> external current -> tier membrane -> spikes -> decoder -> ctrl',
    stages=[
        dict(stage='encoder _encode accumulate', file=enc_src,
             lines=lines_of(enc_src, ['current[idx] += cfg.current_nA_per_unit']),
             clamp='NONE', negative_allowed=True),
        dict(stage='tier drive', file='/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/adapters.py',
             lines=lines_of('/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/adapters.py',
                            ['drive = self.background_nA + cur']),
             clamp='NONE', negative_allowed=True),
        dict(stage='membrane current injection', file=cond_src,
             lines=lines_of(cond_src, ["I = self._vector(current_nA, 'current')",
                                       '+ I - self.adaptation']),
             clamp='NONE (I enters the membrane equation directly; no max(0,.) anywhere)',
             negative_allowed=True),
        dict(stage='spike readout', file=cond_src,
             lines=lines_of(cond_src, ['spiked = active & (self.v >= p.threshold_mV)']),
             clamp='THRESHOLD (one-sided by construction: a cell cannot emit a negative spike, so '
                   'negative drive can only remove spikes, never add them)',
             negative_allowed='drive yes, spikes no'),
        dict(stage='decoder command clip', file=mot_src,
             lines=lines_of(mot_src, ['ctrl = np.clip(self.rates / self.config.rate_for_full_control_hz',
                                      '0, self.config.max_control)']),
             clamp='TWO-SIDED np.clip(..., 0, max_control=1.0)', negative_allowed=False)],
    instrumented=dict(
        arms=['short2_A', 'short2_D'], n_commands=4,
        current_bitwise_equal_pre_clamp=bool(np.array_equal(Acur, Dcur)),
        per_leg_current_min_max_sum_A=Z['short2_A']['leg_current'].tolist(),
        per_leg_current_min_max_sum_D=Z['short2_D']['leg_current'].tolist(),
        statement=('the encoder has no clamp, so "pre clamp" and "post clamp" per-leg current are the SAME array; '
                   'the only clamps on the path are the one-sided threshold spike readout and the two-sided decoder clip')),
    is_negative_current_structurally_rejected=False,
    verdict_text=('NO stage on the current path rejects a negative value. Negative current is integrated by the membrane '
                  'and lowers the firing probability; what is one-sided is the SPIKE READOUT (no negative spikes) and the '
                  'DECODER CLIP (ctrl >= 0). So "the force channel is rectified" is wrong as a statement about the channel; '
                  'the asymmetry lives downstream.'))
(RUN / 'CLAMP_PATH_AUDIT.json').write_text(json.dumps(CLA, indent=2, default=str))

# ---------------- sensitivity arms ------------------------------------------------------------
TIMES = (0.005, 0.02, 0.1, 0.5, 1.0, 2.0, 5.0)
def growth(base, pert):
    qb, qp = Z[base]['qpos'], Z[pert]['qpos']
    tb = np.arange(qb.shape[0]) * 0.005
    out = []
    for t in TIMES:
        k = int(round(t / 0.005)) - 1
        k = min(max(k, 0), qb.shape[0] - 1)
        out.append(dict(t=float(tb[k]), max_abs_dqpos=float(np.max(np.abs(qb[k] - qp[k]))),
                        root_dxy=float(np.hypot(*(qb[k, :2] - qp[k, :2]))), root_dz=float(abs(qb[k, 2] - qp[k, 2]))))
    return out

endstate = {}
for name in ('A', 'A_d1', 'A_d2', 'B', 'B_d1'):
    s = S[name]
    endstate[name] = dict(delta=s['delta'], horizontal_disp=s['horizontal_disp'], final_rootz=s['final_rootz'],
                          rootz_last1s_mean=s['rootz_last1s_mean'], rootz_last1s_std=s['rootz_last1s_std'],
                          mean_bodyfloor_after_1s=s['mean_bodyfloor_after_1s'], total_spikes=s['total_spikes'],
                          ctrl_zero_frac_pre=s['ctrl_zero_frac_pre'], ctrl_one_frac_pre=s['ctrl_one_frac_pre'])
g_A1 = growth('A', 'A_d1')
g_A2 = growth('A', 'A_d2')
g_B1 = growth('B', 'B_d1')
spread_A1_disp = abs(endstate['A_d1']['horizontal_disp'] - endstate['A']['horizontal_disp'])
spread_A1_rootz = abs(endstate['A_d1']['rootz_last1s_std'] - endstate['A']['rootz_last1s_std'])
spread_A2_disp = abs(endstate['A_d2']['horizontal_disp'] - endstate['A']['horizontal_disp'])
spread_A2_rootz = abs(endstate['A_d2']['rootz_last1s_std'] - endstate['A']['rootz_last1s_std'])
spread_B1_disp = abs(endstate['B_d1']['horizontal_disp'] - endstate['B']['horizontal_disp'])
gate_chaotic_A1 = (spread_A1_disp >= 0.05) or (spread_A1_rootz >= 0.005)
gate_chaotic_A2 = (spread_A2_disp >= 0.05) or (spread_A2_rootz >= 0.005)
SENS = dict(
    times_s=TIMES, endstate=endstate,
    growth=dict(A_vs_A_d1=g_A1, A_vs_A_d2=g_A2, B_vs_B_d1=g_B1),
    spreads=dict(A_d1_disp=spread_A1_disp, A_d1_rootz_std=spread_A1_rootz,
                 A_d2_disp=spread_A2_disp, A_d2_rootz_std=spread_A2_rootz,
                 B_d1_disp=spread_B1_disp),
    gate=dict(frozen_rule='A_d1 disp spread >= 0.05 m OR A_d1 rootz-std spread >= 0.005 m',
              A_d1_fires=bool(gate_chaotic_A1), A_d2_fires_same_rule=bool(gate_chaotic_A2),
              note_on_A_d2=('the frozen gate is specified at delta=1e-9; the delta=1e-6 arm is reported for scaling only and '
                            'is NOT used to promote a verdict the pre-registration did not license')),
    S202_claims_vs_envelope=dict(
        B_minus_A_disp=abs(endstate['B']['horizontal_disp'] - endstate['A']['horizontal_disp']),
        C_minus_A_disp=None,
        envelope_disp_at_delta_1e_6=spread_A2_disp,
        ratio_B=abs(endstate['B']['horizontal_disp'] - endstate['A']['horizontal_disp']) / spread_A2_disp
        if spread_A2_disp else None))
SENS['S202_claims_vs_envelope']['C_minus_A_disp'] = 0.022854174835833962  # S202 measured value, unchanged
SENS['S202_claims_vs_envelope']['ratio_C'] = (
    SENS['S202_claims_vs_envelope']['C_minus_A_disp'] / spread_A2_disp if spread_A2_disp else None)
(RUN / 'SENSITIVITY_ARMS.json').write_text(json.dumps(SENS, indent=2, default=str))

# ---------------- D mechanism verdict + pool-level attribution --------------------------------
poolA = Z['short3_A']['pool_spikes'].sum(axis=0)
poolB = Z['short3_B']['pool_spikes'].sum(axis=0)
poolC = Z['short3_C']['pool_spikes'].sum(axis=0)
poolD = Z['short3_D']['pool_spikes'].sum(axis=0)
totA = int(S['short3_A']['total_spikes'])
totB = int(S['short3_B']['total_spikes'])
totC = int(S['short3_C']['total_spikes'])
totD = int(S['short3_D']['total_spikes'])
ctrl_A_short = Z['short3_A']['ctrl']
ctrl_D_short = Z['short3_D']['ctrl']

def pool_attr(pool, tot, ctrl_ref, label):
    changed = np.where(pool != poolA)[0].tolist()
    detail = []
    for pi in changed:
        railed = bool(np.all(ctrl_ref[:, pi] <= 1e-12) or np.all(ctrl_ref[:, pi] >= 1.0 - 1e-12))
        detail.append(dict(pool=pi, actuator=names[pi], spikes=poolA[pi], spikes_arm=int(pool[pi]),
                           actuator_at_rail_in_A=railed))
    return dict(arm=label, total_spikes=tot, pool_spikes_sum=int(pool.sum()),
                readout_fraction=float(pool.sum()) / tot if tot else None,
                n_pools=int(pool.size), n_pools_changed=len(changed),
                pools_changed=detail,
                n_changed_pools_railed=sum(1 for d in detail if d['actuator_at_rail_in_A']),
                n_changed_pools_unrailed=sum(1 for d in detail if not d['actuator_at_rail_in_A']))

attr_B = pool_attr(poolB, totB, ctrl_A_short, 'B')
attr_C = pool_attr(poolC, totC, ctrl_A_short, 'C')
attr_D = pool_attr(poolD, totD, ctrl_A_short, 'D')
poolT = Z['short3_T']['pool_spikes'].sum(axis=0)
totT = int(S['short3_T']['total_spikes'])
attr_T = pool_attr(poolT, totT, ctrl_A_short, 'T')
attr_A = dict(arm='A', total_spikes=totA, pool_spikes_sum=int(poolA.sum()),
              readout_fraction=float(poolA.sum()) / totA if totA else None,
              n_pools=int(poolA.size), n_pools_zero=int((poolA == 0).sum()))

not_rail_A = [i for i in range(nu) if not (np.all(ctrl_A_short[:, i] <= 1e-12) or np.all(ctrl_A_short[:, i] >= 1.0 - 1e-12))]
DM = dict(
    frozen_values=['RECTIFIED', 'SATURATION_MASKED', 'UNRESOLVED'],
    evidence=dict(
        ctrl_bitwise_equal_A_vs_D=bool(np.array_equal(ctrl_A_short, ctrl_D_short)),
        qpos_bitwise_equal_A_vs_D=bool(np.array_equal(Z['short3_A']['qpos'], Z['short3_D']['qpos'])),
        current_bitwise_equal_A_vs_D=bool(np.array_equal(Z['short3_A']['current'], Z['short3_D']['current'])),
        spikes_changed=bool(totA != totD), total_spikes_A=totA, total_spikes_D=totD,
        decoder_pools=int(poolA.size), decoder_pools_with_changed_spike_count=attr_D['n_pools_changed'],
        negative_current_structurally_rejected=CLA['is_negative_current_structurally_rejected']),
    pool_attribution=dict(A=attr_A, B=attr_B, C=attr_C, D=attr_D, T=attr_T),
    measured_mechanism=('NO_DECODER_POOL_CHANGE for the negative-force arm: the extra spikes of arm D land entirely outside '
                        'every decoder readout pool, so ctrl cannot move and qpos stays bitwise identical. Arm B does change '
                        'pools, which is why its ctrl moves. Neither a current clamp nor a channel rectifier is involved: '
                        'the readout-relevant cells fire only for positive force, so a negative input has nothing to remove '
                        '(one-sided threshold readout with a silent baseline).'),
)
if CLA['is_negative_current_structurally_rejected']:
    DM['verdict'] = 'RECTIFIED'
elif attr_D['n_pools_changed'] == 0:
    DM['verdict'] = 'UNRESOLVED'
    DM['why_not_saturation_masked'] = (
        'the frozen SATURATION_MASKED definition requires that every actuator the negative input could have moved was '
        'already clipped at 0 or 1. That is not what the data show: ZERO decoder pools changed, so there was no actuator '
        'the input could have moved at all. Reporting SATURATION_MASKED here would be a vacuous pass of the definition, '
        'so the frozen three-way label is UNRESOLVED and the measured mechanism is stated separately. The label set needs a '
        'fourth value (NO_DECODER_POOL_CHANGE) before the next tactile round.')
else:
    DM['verdict'] = 'SATURATION_MASKED' if attr_D['n_changed_pools_unrailed'] == 0 else 'UNRESOLVED'
DM['verdict_basis'] = (
    'RECTIFIED is false because the encoder, the tier drive and the membrane all accept negative current (CLAMP_PATH_AUDIT). '
    'SATURATION_MASKED is not established because zero pools changed. The observed invariance of ctrl is structural in the '
    'readout routing: only %.1f%% of the tier spikes reach a decoder pool at all.'
    % (100 * (attr_A['readout_fraction'] or 0.0)))
(RUN / 'D_MECHANISM_VERDICT.json').write_text(json.dumps(DM, indent=2, default=str))

print(json.dumps(dict(
    ctrl_div=CD['pre_collapse_A_vs_B'], whole=CD['A_vs_B_whole_run_first_divergence'],
    pre=summary_pre, post=summary_post,
    spreads=SENS['spreads'], gate=dict(A1=gate_chaotic_A1, A2=gate_chaotic_A2),
    envelope=SENS['S202_claims_vs_envelope'],
    D=dict(verdict=DM['verdict'], pools_changed=attr_D['n_pools_changed'],
           spikes=[int(S['short2_A']['total_spikes']), int(S['short2_D']['total_spikes'])],
           not_rail=len(not_rail_A))), indent=1, default=str))
