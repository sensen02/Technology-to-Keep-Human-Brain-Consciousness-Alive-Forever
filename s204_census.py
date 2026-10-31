"""S204 Part A: readout census, decoder transfer function, saturation decomposition, reachability.

Uses the frozen engine objects and the already-stored S203 traces. No physics stepping.
"""
import json
import numpy as np
import mujoco
import scipy.sparse as sp
from pathlib import Path

import sys
sys.path.insert(0, '/run/media/sensen/Data2/cell_wound_prototype')
from engine.embodied import adapters as ad
from engine.embodied.motor_output import build_motor_manifest, MotorOutputConfig

RUN = Path('/home/sensen/Desktop/engine修复验收/s204-readout-calibration-audit/20261025T073450411660Z')
S203 = Path('/home/sensen/Desktop/engine修复验收/s203-attribution-chaos-audit/20261025T072017913193Z')
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
HARD_DOF = (19, 22, 26, 40, 42, 47)

# ------------------------------------------------------------------ A.1 readout census
tier_cfg = ad.TierConfig(background_mean_nA=0, background_sd_nA=0)
built = ad.build_neural_tier(tier_cfg)
n = int(built['n'])
assert n == 12000, n
pools = json.loads(Path('/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_rebuild/'
                        'neural_rebuild_11_mn_pools.json').read_text())
m = mujoco.MjModel.from_xml_path(str(XML))
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or '' for i in range(m.nu)]
man = build_motor_manifest(pools, built['tier']['root_ids'], names)
rows = man['rows']
assert len(rows) == m.nu == 90

# moment arms at the frozen release pose, for the hard-DOF cross-check
traj = np.load('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z/init80ms/TRAJECTORY.npz')
MOM = np.asarray(traj['actuator_moment'][0], float)      # (90, 48)
jname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or '' for j in range(m.njnt)]
dof_of_joint = {j: int(m.jnt_dofadr[j]) for j in range(m.njnt)}
joint_of_dof = {v: k for k, v in dof_of_joint.items()}

census_rows = []
for r in rows:
    i = int(r['actuator_index'])
    census_rows.append(dict(
        actuator=r['actuator'], actuator_index=i, leg=r.get('leg'), side=r.get('side'),
        muscle_target=r.get('muscle_target'), status=r['status'],
        pool_size=len(r['local_indices']), local_indices=[int(x) for x in r['local_indices']],
        pool_root_ids=[int(x) for x in r.get('pool_root_ids', [])],
        moment_on_hard_dofs={str(d): float(MOM[i, d]) for d in HARD_DOF if abs(MOM[i, d]) > 1e-9}))
empty_rows = [c for c in census_rows if c['pool_size'] == 0]
nonempty = [c for c in census_rows if c['pool_size'] > 0]
sizes = [c['pool_size'] for c in nonempty]
uniq_cells = sorted({int(x) for c in census_rows for x in c['local_indices']})
per_leg_nonempty = {L: sum(1 for c in nonempty if (c['leg'] or '').upper() == L.upper()) for L in LEGS}
per_leg_empty = {L: sum(1 for c in empty_rows if (c['leg'] or '').upper() == L.upper()) for L in LEGS}
size_hist = {int(k): int(v) for k, v in zip(*np.unique(sizes, return_counts=True))}
hard_members = {}
for dof in HARD_DOF:
    acts = [c for c in census_rows if abs(MOM[c['actuator_index'], dof]) > 1e-9]
    hard_members[dof] = dict(joint=jname[joint_of_dof[dof]],
                             n_actuators_with_moment=len(acts),
                             empty_row_members=[c['actuator'] for c in acts if c['pool_size'] == 0],
                             nonempty_members=[c['actuator'] for c in acts if c['pool_size'] > 0])
census = dict(
    n_tier=int(n), n_decoder_rows=len(rows), n_actuators=int(m.nu),
    n_empty_rows=len(empty_rows), n_nonempty_rows=len(nonempty),
    empty_actuators=[dict(actuator=c['actuator'], leg=c['leg'], status=c['status']) for c in empty_rows],
    empty_share_of_rows=len(empty_rows) / len(rows),
    pool_size_min=min(sizes), pool_size_mean=float(np.mean(sizes)), pool_size_max=max(sizes),
    pool_size_histogram=size_hist,
    unique_pool_cells=len(uniq_cells), coverage_fraction=len(uniq_cells) / n,
    per_leg_nonempty=per_leg_nonempty, per_leg_empty=per_leg_empty,
    hard_dofs=hard_members,
    rows=census_rows,
    statement=('Of the 90 decoder rows, %d can never receive a command: their muscle pool is unresolved, local_indices is '
               'empty, and motor_output maps them to no control index. The remaining %d rows read %d distinct tier cells out '
               'of %d (%.3f%%).' % (len(empty_rows), len(nonempty), len(uniq_cells), n, 100 * len(uniq_cells) / n)))
(RUN / 'READOUT_CENSUS.json').write_text(json.dumps(census, indent=2, default=str))

# ------------------------------------------------------------------ A.2 decoder transfer function
cfg = MotorOutputConfig()
H0 = float(cfg.rate_for_full_control_hz)
tau = float(cfg.filter_tau_s)
max_control = float(cfg.max_control)
spike_dt = 0.0005
decay = float(np.exp(-spike_dt / tau))
one_spike_inst = 1.0 / spike_dt          # a single spike on a k=1 pool -> 2000 Hz instantaneous
jump = (1.0 - decay) * one_spike_inst    # filtered contribution of exactly one spike
table = []
for k in range(1, 8):
    # steady single-cell rate f: instantaneous = f for any k (pool mean counts f*dt, divided by dt)
    need_1 = H0 * max_control
    table.append(dict(pool_size=k,
                      required_cell_rate_hz_for_ctrl_1=need_1,
                      required_cell_rate_hz_for_ctrl_0_5=0.5 * need_1,
                      instantaneous_hz_if_one_cell_fires_once=one_spike_inst,
                      filtered_rate_jump_per_single_spike_hz=jump * 1.0,
                      ctrl_after_one_spike_at_k1=jump / H0,
                      note=('the steady-rate requirement is independent of k; what depends on k is the VARIANCE: with k=1 a '
                            'single spike already gives %.3f of full scale at H0=%.0f Hz' % (jump / H0, H0))))
transfer = dict(source='engine/embodied/motor_output.py',
                config=dict(filter_tau_s=tau, rate_for_full_control_hz=H0, max_control=max_control),
                spike_dt_s=spike_dt, filter_decay_per_sample=decay,
                filtered_gain_per_single_spike=(1 - decay) * one_spike_inst,
                analytic=('rates <- decay*rates + (1-decay)*instantaneous ; instantaneous == mean over the pool of '
                          '(spikes_in_sample/spike_dt_s). For a steady per-cell rate f the pool mean is f*spike_dt_s, so '
                          'instantaneous == f and ctrl = clip(f/H,0,1): the steady requirement is f = H = %.0f Hz for ctrl=1 '
                          'and %.0f Hz for ctrl=0.5, INDEPENDENT of pool size. One spike, however, injects %.1f Hz of filtered '
                          'rate, i.e. ctrl jumps by %.3f at H=%.0f.' % (H0, 0.5 * H0, (1 - decay) * one_spike_inst,
                                                                          (1 - decay) * one_spike_inst / H0, H0)),
                table=table,
                exact_recompute='see DECODER_TRANSFER.json "bitwise_check" written after the H0/H1/H2 arms')
(RUN / 'DECODER_TRANSFER.json').write_text(json.dumps(transfer, indent=2, default=str))

# ------------------------------------------------------------------ A.3 saturation decomposition (S203 short arms)
def load203(name):
    return json.load(open(S203 / name / 'SUMMARY.json')), np.load(S203 / name / 'TRACE.npz')

dec_rows = {}
csv_lines = ['window,arm,class,actuator_index,actuator,pool_size,pool_spikes,ctrl,structural_zero']
for arm in ('short3_A', 'short3_B', 'short3_C', 'short3_D', 'short3_T'):
    s, z = load203(arm)
    ctrl = z['ctrl']
    ps = z['pool_spikes']
    per_arm = {}
    for w in range(1, min(4, ctrl.shape[0])):
        for c in census_rows:
            i = c['actuator_index']
            cval = float(ctrl[w, i])
            cnt = int(ps[w, i])
            if c['pool_size'] == 0:
                cls = 'structural_zero'
            elif cnt == 0:
                cls = 'pool_zero_spikes'
            elif cval >= 1.0 - 1e-12:
                cls = 'ctrl_at_one'
            else:
                cls = 'spiked_ctrl_below_one'
            per_arm.setdefault(w, {}).setdefault(cls, []).append(i)
            csv_lines.append(','.join(str(x) for x in (
                w, arm, cls, i, c['actuator'], c['pool_size'], cnt, '%.6f' % cval, int(c['pool_size'] == 0))))
    dec_rows[arm] = per_arm

summary = {}
for arm in ('short3_A', 'short3_B'):
    tot = {}
    tot_nonempty = {}
    for w in (1, 2, 3):
        counts = {k: len(v) for k, v in dec_rows[arm][w].items()}
        tot[w] = counts
        ne = {k: counts.get(k, 0) for k in ('pool_zero_spikes', 'spiked_ctrl_below_one', 'ctrl_at_one')}
        ne['total'] = sum(ne.values())
        tot_nonempty[w] = ne
    allc = {k: sum(tot[w].get(k, 0) for w in (1, 2, 3)) for k in
            ('structural_zero', 'pool_zero_spikes', 'spiked_ctrl_below_one', 'ctrl_at_one')}
    at_rail_incl = allc['structural_zero'] + allc['pool_zero_spikes'] + allc['ctrl_at_one']
    at_rail_excl = allc['pool_zero_spikes'] + allc['ctrl_at_one']
    summary[arm] = dict(
        counts=allc, n_values=3 * 90, n_nonempty_values=3 * 74,
        at_zero_incl_structural=(allc['structural_zero'] + allc['pool_zero_spikes']) / (3 * 90),
        at_one=allc['ctrl_at_one'] / (3 * 90),
        at_rail_incl_structural=at_rail_incl / (3 * 90),
        at_rail_excl_structural=at_rail_excl / (3 * 74),
        off_rail_excl_structural=(allc['spiked_ctrl_below_one']) / (3 * 74),
        structural_zero_share_of_the_rail_figure=allc['structural_zero'] / at_rail_incl if at_rail_incl else None,
        per_window_nonempty=tot_nonempty)
(RUN / 'SATURATION_DECOMPOSITION.csv').write_text('\n'.join(csv_lines) + '\n')
(RUN / 'SATURATION_DECOMPOSITION.json').write_text(json.dumps(dict(
    window='S203 short3 arms, command intervals 1-3 (the cold-start command 0 is excluded by construction)',
    classes=['structural_zero', 'pool_zero_spikes', 'spiked_ctrl_below_one', 'ctrl_at_one'],
    per_arm=summary,
    s203_numbers=dict(pre_collapse_at_rail_A=0.8472222222222223, at_zero_A=0.38333333333333336, at_one_A=0.4638888888888889,
                      window='S203 SATURATION_WINDOWS_REFINED A.pre_collapse_after_command0: 4 command windows '
                             '(t = 5,10,15,20 ms) over ALL 90 rows, so command 0 (the cold start) was ALREADY excluded',
                      note='this round uses commands 1-3 of the 0.02 s arms (t = 5,10,15 ms, 270 values) because those are the '
                           'windows for which per-pool spike counts were stored; the overlap with the S203 window is 3 of 4 '
                           'windows, so the two figures are comparable but not identical'),
    note=('at_rail here counts structural_zero + pool_zero_spikes + ctrl_at_one. The structural-zero rows are 16/90 = 17.8 pt '
          'of every "at zero" figure and they can never be anything else, so they must be removed before the number is used '
          'as evidence about the body.')),
    indent=2, default=str))

# ------------------------------------------------------------------ A.4 reachability
adj = (built['edges']['We'] + built['edges']['Wi']).tocsr()
adj.data[:] = 1.0                     # unweighted, direction preserved: adj[i, j] = an edge j -> i exists
pool_mask = np.zeros(n, bool)
pool_mask[uniq_cells] = True
non_pool = ~pool_mask
# The question is whether a NON-pool cell can INFLUENCE a pool cell, so propagate BACKWARDS:
# cells that reach a pool cell in <= 2 hops  <=>  (adj^T)^k applied to the pool mask.
in1 = (adj.T @ pool_mask.astype(float)) > 0
in2 = (adj.T @ in1.astype(float)) > 0
in3 = (adj.T @ in2.astype(float)) > 0
reach_in = in1 | in2 | pool_mask
# For completeness, the forward direction too (does a pool cell reach non-pool cells).
out1 = (adj @ pool_mask.astype(float)) > 0
out_reach = out1 | pool_mask
reachable_nonpool = int((reach_in & non_pool).sum())
share = reachable_nonpool / int(non_pool.sum())
reach = dict(
    adjacency='frozen tier edges We|Wi (12000x12000) from build_neural_tier, unweighted, direction preserved '
              '(adj[i,j] = edge j->i, matching self.ge += We @ arrivals); nothing inferred beyond these edges',
    direction_used='adj.T: cells that can reach a pool cell',
    n_tier=int(n), n_pool_cells=len(uniq_cells), n_non_pool_cells=int(non_pool.sum()),
    hop1_reaching_pool_nonpool=int((in1 & non_pool).sum()),
    hop2_only_reaching_pool_nonpool=int(((in2 & ~in1) & non_pool).sum()),
    reachable_nonpool_2hops=reachable_nonpool,
    reachable_nonpool_share=share,
    unreachable_nonpool=int(non_pool.sum()) - reachable_nonpool,
    forward_pool_reaches_nonpool_1hop=int((out1 & non_pool).sum()),
    hop3_only_reaching_pool_nonpool=int(((in3 & ~in2 & ~in1) & non_pool).sum()),
    reachable_nonpool_3hops=int(((in3 | in2 | in1) & non_pool).sum()),
    reachable_nonpool_share_3hops=float(((in3 | in2 | in1) & non_pool).sum()) / int(non_pool.sum()),
    margin_note=('the frozen rule is a 2-hop horizon; the share is %.4f against a 0.5 threshold, i.e. the label is decided '
                 'by %.2f percentage points. The 3-hop count is reported for transparency only and does NOT change the label.'
                 % (share, abs(share - 0.5) * 100)),
    mean_out_degree=float(adj.nnz / n),
    label=('SUBTHRESHOLD_REACHABLE' if share > 0.5 else 'TOPO_UNREACHABLE'))
(RUN / 'REACHABILITY.json').write_text(json.dumps(reach, indent=2, default=str))

print(json.dumps(dict(
    census=dict(n=n, rows=len(rows), empty=len(empty_rows), nonempty=len(nonempty),
                cells=len(uniq_cells), coverage_pct=100 * len(uniq_cells) / n,
                sizes=dict(min=min(sizes), mean=round(float(np.mean(sizes)), 4), max=max(sizes)),
                per_leg_nonempty=per_leg_nonempty, per_leg_empty=per_leg_empty),
    transfer=dict(H0=H0, decay=decay, per_spike_jump=jump, ctrl_after_one_spike=jump / H0),
    decomposition={a: {k: v for k, v in summary[a].items() if k != 'per_window_nonempty'} for a in summary},
    reachability=dict(label=reach['label'], share=round(reach['reachable_nonpool_share'], 6),
                      hop1=reach['hop1_reaching_pool_nonpool'], hop2=reach['hop2_only_reaching_pool_nonpool'])), indent=1, default=str))
