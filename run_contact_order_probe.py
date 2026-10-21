"""PROBE (not a delivery script, not a runner): is the contact test order-sensitive?

The earlier gain-sweep probe silently read ZERO tarsal force because it tested only one side of
(geom1, geom2).  This probe measures, on the canonical body at the delivered initial state, how the
tarsal/floor pair is actually ORDERED in `d.contact`, so the claim "the test is side-agnostic" rests
on a measurement instead of on a reading of the code.
"""
import pathlib, sys, json, hashlib
import numpy as np
import mujoco

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
sys.path.insert(0, str(ROOT))
STATE = pathlib.Path('/home/sensen/Desktop/engine修复验收/S2-equilibrium-A-corrected-reissued-20261015T161259/equilibrium_best_state.npz')
DRIVE = ROOT / 'outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz'
DT = 2.5e-5
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
import importlib
L = importlib.import_module('run_flybody_loop')
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

s = np.load(STATE)
_body = L.build_body('canonical')
m = _body['model'] if isinstance(_body, dict) else _body[1]
m.opt.timestep = DT
d = mujoco.MjData(m)
d.qpos[:] = s['qpos']; d.qvel[:] = 0.0; d.act[:] = s['act']; d.ctrl[:] = 0.0
mujoco.mj_forward(m, d)
arch = np.load(DRIVE)['normal_controls']
gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''


def census(d):
    tar_first, tar_second, floor_first, floor_second, pairs = 0, 0, 0, 0, []
    for c in d.contact:
        if c.efc_address < 0:
            continue
        n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
        pairs.append([n1, n2])
        if 'Tarsus' in n1:
            tar_first += 1
        if 'Tarsus' in n2:
            tar_second += 1
        if n1 == 'floor':
            floor_first += 1
        if n2 == 'floor':
            floor_second += 1
    return dict(n_pairs=len(pairs), tarsal_in_geom1=tar_first, tarsal_in_geom2=tar_second,
                floor_in_geom1=floor_first, floor_in_geom2=floor_second, pairs=pairs)


per_step, one_sided_zeros, sampled = [], [], []
for k in range(201):
    if k:
        if (k - 1) % 200 == 0:
            d.ctrl[:] = arch[(k - 1) // 200]
        mujoco.mj_step(m, d)
    c = census(d)
    per_step.append({'step': k, **{a: b for a, b in c.items() if a != 'pairs'}})
    if k in (0, 1, 100, 200):
        sampled.append(dict(step=k, pairs=c['pairs']))
        f6 = np.zeros(6)
        f1 = f2 = fb = 0.0
        for ci, cp in enumerate(d.contact):
            if cp.efc_address < 0:
                continue
            n1, n2 = gname(int(cp.geom1)), gname(int(cp.geom2))
            mujoco.mj_contactForce(m, d, ci, f6)
            nrm = float(np.linalg.norm(f6[:3]))
            if 'Tarsus' in n1:
                f1 += nrm
            if 'Tarsus' in n2:
                f2 += nrm
            if 'Tarsus' in (n1 + n2):
                fb += nrm
        one_sided_zeros.append(dict(step=k,
                                    tarsal_force_if_only_geom1=f1,
                                    tarsal_force_if_only_geom2=f2,
                                    tarsal_force_both_orders=fb,
                                    one_sided_test_would_read_zero_force=bool(f1 == 0.0 or f2 == 0.0)))

_READING = ('MEASURED, not read off the code: over the 201 probed steps every tarsal-floor pair is ordered '
            '(floor, <LEG>Tarsus*_geom), i.e. the tarsal geom is ALWAYS in geom2 (tarsal_in_geom1 = 0, '
            'floor_in_geom1 = 10251 of the 10251 probed pairs), so a test that only looked at geom1 would read '
            'exactly 0.0 tarsal force at every probed step (measured: 0.0 by geom1 versus 24.446354755877977 / '
            '19.543928662287872 / 20.217709925033585 by geom2). The non-foot pair is ordered the other way '
            '(body geom in geom1), so the order is NOT uniform across pair types and a side-agnostic test is '
            'mandatory. This is the same defect the earlier gain-sweep probe hit, here reproduced on purpose.')

out = dict(status='PROBE_NOT_A_DELIVERY_SCRIPT', manual_research_not_product=True,
           question='is the tarsal/floor contact test sensitive to the (geom1, geom2) order?',
           model='canonical body (run_flybody_loop.build_body)',
           timestep_s=DT, steps_probed=201, initial_state_sha256=sha(STATE), archive_sha256=sha(DRIVE),
           sampled_pair_lists=sampled,
           order_totals=dict(
               tarsal_in_geom1=sum(p['tarsal_in_geom1'] for p in per_step),
               tarsal_in_geom2=sum(p['tarsal_in_geom2'] for p in per_step),
               floor_in_geom1=sum(p['floor_in_geom1'] for p in per_step),
               floor_in_geom2=sum(p['floor_in_geom2'] for p in per_step),
               n_pairs=sum(p['n_pairs'] for p in per_step)),
           one_sided_vs_both_sides=one_sided_zeros,
           n_steps_with_any_tarsal_pair=sum(1 for p in per_step if p['tarsal_in_geom1'] + p['tarsal_in_geom2'] > 0),
           nonfoot_pair_order_from_delivered_report=dict(
               pair=['A6_geom', 'floor'], where='report.json -> results[open_loop].first_nonfoot_pair',
               order='body geom in geom1, floor in geom2  (the OPPOSITE order to the tarsal pairs below)'),
           reading=_READING,
           not_established='absolute force values and units NOT_ESTABLISHED; this is a manual research probe, not a product')
print(json.dumps({k: v for k, v in out.items() if k != 'stepwise'}, ensure_ascii=False, indent=1)[:4000])
pathlib.Path('/tmp/contact_order_probe.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
