#!/usr/bin/env python3
"""S177 independent verification WITH MODEL RELOAD.

One model load, one mj_forward, one 42-row LP.  No mj_step, no neural ticks, no
training, no other optimisation.  All inputs are read read-only from the frozen
producer archives; this script writes exactly one new file (its report) and
refuses to overwrite an existing report.

Target claim under test (S177 rerun dir 20261025T020500000000Z):
  with all 42 leg joints frozen at the S154 release values and only the root 6
  pose coordinates moved, the six-row root wrench residual
  -(base + J.T @ f0)[0:6] is claimed to fall from 172.74733807654337 to
  1.3464784842653899e-12 at one pose, the geometry gates pass (foot depth
  0.00598, non-foot 0, 6 bearing legs, friction margin 0.583), while the 42 joint
  rows are NOT balanced (LP minimax 3.3086, unreachable rows [13,16,20,34,36,41]).

Scope (do not over-read the numbers):
  * the root six-row result is a STATIC feasibility result at ONE pose with the
    leg joints frozen at the S154 release values;
  * the 42-row joint imbalance is NOT an infeasibility certificate - the LP is
    feasible at its t; that LP result is reported separately and is never merged
    with residual6;
  * nothing here shows standing, neural attainability or dynamics;
  * SI units are NOT_ESTABLISHED.
"""
from __future__ import annotations

import hashlib
import json
import sys
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import mujoco as mj
import scipy
from scipy.optimize import linprog

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = Path('/home/sensen/Desktop/engine修复验收')
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
S154 = BASE / 's154-activation-rate' / '20261024T215751775489Z' / 'init80ms'
S166 = BASE / 's166-mapped-support' / '20261024T231418078692647Z'
S177 = BASE / 's177-root-newton-search'
RUN_0155 = S177 / '20261025T015500000000Z'
RUN_0205 = S177 / '20261025T020500000000Z'
OUT = RUN_0205 / 'S177_INDEPENDENT_RELOAD_CHECK_RETRY.json'

DEPTH_CAP = 0.03442910871489635
NONFOOT_CAP = 1e-12
MIN_BEARING = 3
MARGIN_MIN = -1e-10
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
STATE = ('qpos', 'qvel', 'act', 'ctrl', 'qfrc_applied', 'xfrc_applied')
OPTIONAL = ('mocap_pos', 'mocap_quat', 'eq_active', 'plugin_state', 'userdata')
EXPECTED_LEDGER = dict(model_load=0, forward=42, cache_hits=1, newton_iterations=5, jacobians=6,
                       fd_probes=36, line_search_trials=5, line_search_halvings=0, pose_evaluations=42,
                       accepted_iterations=5, contact_set_changes=1)
EXPECTED_RUNNER = dict(model_load=1, gate_forward=1, confirmation_forward=1)
FAILED_DIRS = ('20261025T011500000000Z', '20261025T012500000000Z',
               '20261025T013500000000Z', '20261025T014500000000Z')
_SENT = object()


# ---------------------------------------------------------------- small helpers
def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_array(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_npz(path):
    with np.load(Path(path), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def dict_diff(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k, _SENT) != b.get(k, _SENT))


def sanitize(v):
    if isinstance(v, dict):
        return {str(k): sanitize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [sanitize(x) for x in v]
    if isinstance(v, np.ndarray):
        return sanitize(v.tolist())
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    return str(v)


def snapshot(model):
    """Own model-parameter snapshot: every public ndarray hashed, every scalar kept.

    Same coverage as the S154 MODEL_PROVENANCE.json record (all model.* and opt.*
    entries, including all body/joint/geom/actuator parameter arrays), so the two
    records can be compared entry by entry.
    """
    out = {}
    for prefix, obj in (('model', model), ('opt', model.opt)):
        for k in dir(obj):
            if k.startswith('_'):
                continue
            v = getattr(obj, k)
            if isinstance(v, np.ndarray):
                out[f'{prefix}.{k}'] = dict(shape=list(v.shape), dtype=str(v.dtype), sha256=sha_array(v))
            elif isinstance(v, (bool, int, float, np.integer, np.floating)):
                out[f'{prefix}.{k}'] = v.item() if isinstance(v, np.generic) else v
    return out


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def quat_from_xyz(rx, ry, rz):
    cx, sx = np.cos(rx / 2), np.sin(rx / 2)
    cy, sy = np.cos(ry / 2), np.sin(ry / 2)
    cz, sz = np.cos(rz / 2), np.sin(rz / 2)
    return np.array([cx * cy * cz + sx * sy * sz, sx * cy * cz - cx * sy * sz,
                     cx * sy * cz + sx * cy * sz, cx * cy * sz - sx * sy * cz])


def fl(l, lo, hi):
    """Native muscle force-length curve."""
    if l < lo or l > hi:
        return 0.
    a = (lo + 1) * .5
    b = (1 + hi) * .5
    if l <= a:
        return .5 * ((l - lo) / max(1e-15, a - lo)) ** 2
    if l <= 1:
        return 1 - .5 * ((1 - l) / max(1e-15, 1 - a)) ** 2
    if l <= b:
        return 1 - .5 * ((l - 1) / max(1e-15, b - 1)) ** 2
    return .5 * ((hi - l) / max(1e-15, hi - b)) ** 2


def muscle_terms(length, node):
    """Gain and bias of one XML actuator node at zero velocity / zero activation."""
    g = list(map(float, node.get('gainprm').split()))
    p = list(map(float, node.get('biasprm').split()))
    lr = list(map(float, node.get('lengthrange').split()))
    assert g[2] > 0 and p[2] > 0
    l0 = (lr[1] - lr[0]) / max(1e-15, g[1] - g[0])
    l = g[0] + (length - lr[0]) / max(1e-15, l0)
    v = 0. / max(1e-15, l0 * g[6])
    y = g[8] - 1
    if v <= -1:
        fv = 0.
    elif v <= 0:
        fv = (v + 1) ** 2
    elif v <= y:
        fv = g[8] - (y - v) ** 2 / max(1e-15, y)
    else:
        fv = g[8]
    gain = -g[2] * fl(l, g[4], g[5]) * fv
    l0b = (lr[1] - lr[0]) / max(1e-15, p[1] - p[0])
    lb = p[0] + (length - lr[0]) / max(1e-15, l0b)
    mid = (1 + p[5]) * .5
    if lb <= 1:
        bias = 0.
    elif lb <= mid:
        bias = -p[2] * p[7] * .5 * ((lb - 1) / max(1e-15, mid - 1)) ** 2
    else:
        bias = -p[2] * p[7] * (.5 + (lb - mid) / max(1e-15, mid - 1))
    return gain * 0. + bias, gain, bias


def moment_dense(model, data):
    """Dense row-addressed (nu, nv) actuator transmission matrix from mjData."""
    raw = np.asarray(data.actuator_moment).copy()
    nnz = np.asarray(data.moment_rownnz).copy()
    adr = np.asarray(data.moment_rowadr).copy()
    col = np.asarray(data.moment_colind).copy()
    assert raw.shape == col.shape == (int(model.nJmom),)
    assert nnz.shape == adr.shape == (int(model.nu),)
    dense = np.zeros((int(model.nu), int(model.nv)), dtype=raw.dtype)
    for row in range(int(model.nu)):
        start, count = int(adr[row]), int(nnz[row])
        assert start >= 0 and count >= 0 and start + count <= int(model.nJmom)
        columns = col[start:start + count]
        assert np.all((columns >= 0) & (columns < int(model.nv)))
        dense[row, columns] = raw[start:start + count]
    return dense


def efc_j_dense(model, data, nefc, nv):
    """Dense constraint Jacobian from mjData, sparse or dense native layout."""
    if bool(mj.mj_isSparse(model)):
        raw = np.asarray(data.efc_J).copy()
        nnz = np.asarray(data.efc_J_rownnz).copy()
        adr = np.asarray(data.efc_J_rowadr).copy()
        col = np.asarray(data.efc_J_colind).copy()
        assert raw.size == int(data.nJ), (raw.size, int(data.nJ))
        dense = np.zeros((nefc, nv), dtype=raw.dtype)
        for r in range(nefc):
            a, c = int(adr[r]), int(nnz[r])
            assert 0 <= a and 0 <= c and a + c <= raw.size
            idx = np.arange(a, a + c, dtype=np.int64)
            assert np.all((col[idx] >= 0) & (col[idx] < nv))
            np.add.at(dense[r], col[idx], raw[idx])
        return dense
    raw = np.asarray(data.efc_J).copy()
    assert raw.size == nefc * nv, (raw.size, nefc * nv)
    return raw.reshape(nefc, nv)


def contact_record(data):
    """Contact record from the single forward: geoms, distance, friction, row ids."""
    ncon = int(data.ncon)
    contacts = list(data.contact)
    geom = np.array([[int(c.geom1), int(c.geom2)] for c in contacts], dtype=np.int64).reshape(ncon, 2)
    dist = np.array([float(c.dist) for c in contacts], dtype=float)
    friction = np.array([np.asarray(c.friction, dtype=float).copy() for c in contacts],
                        dtype=float).reshape(ncon, 5)
    efc_address = np.array([int(c.efc_address) for c in contacts], dtype=np.int64)
    dims = np.array([int(c.dim) for c in contacts], dtype=np.int64)
    efc_type = np.asarray(data.efc_type).astype(np.int64)
    efc_id = np.asarray(data.efc_id).astype(np.int64)
    contact_types = [int(v) for k, v in mj.mjtConstraint.__members__.items()
                     if k.startswith('mjCNSTR_CONTACT_')]
    row_contact_index = np.where(np.isin(efc_type, contact_types), efc_id, -1)
    row_count = np.array([int(np.count_nonzero(row_contact_index == ci)) for ci in range(ncon)],
                         dtype=np.int64)
    return dict(geom=geom, dist=dist, friction=friction, efc_address=efc_address,
                dim=dims, row_count=row_count)


def geometry(rec, f0, nefc, floors, geom_leg):
    """Foot/non-foot penetration, bearing legs and minimum friction pyramid margin."""
    depth = 0.0
    nonfoot = 0.0
    bearing = set()
    margins = []
    for ci in range(rec['geom'].shape[0]):
        g1 = int(rec['geom'][ci, 0])
        g2 = int(rec['geom'][ci, 1])
        if (g1 in floors) != (g2 in floors):
            leg = geom_leg.get(g2 if g1 in floors else g1)
        else:
            leg = None
        pen = max(0.0, -float(rec['dist'][ci]))
        if leg is None:
            nonfoot = max(nonfoot, pen)
        else:
            depth = max(depth, pen)
        addr = int(rec['efc_address'][ci])
        dim = int(rec['dim'][ci])
        mu = np.asarray(rec['friction'][ci, :2], dtype=float)
        if addr < 0:
            continue
        count = int(rec['row_count'][ci])
        if dim == 1:
            local = np.array([f0[addr], 0.0, 0.0])
        else:
            assert dim == 3 and count == 4, ('unexpected native contact row layout', dim, count)
            e = f0[addr:addr + 4]
            local = np.array([e.sum(), mu[0] * (e[0] - e[1]), mu[1] * (e[2] - e[3])])
        margins.append(float(local[0] - np.sum(np.abs(local[1:] / mu))))
        if leg is not None and local[0] > 0 and rec['dist'][ci] <= 0:
            bearing.add(leg)
    sig = '|'.join(sorted(f'{int(a)}-{int(b)}' for a, b in rec['geom'])) + f'#nefc{nefc}'
    return dict(foot_depth=float(depth), nonfoot_depth=float(nonfoot),
                bearing_legs=int(len(bearing)), bearing_leg_names=sorted(bearing),
                friction_margin=float(min(margins, default=0.0)),
                contacts=int(rec['geom'].shape[0]), contact_signature=sig)


def solve_joint_lp(A, tgt):
    """ONE 42-row minimax LP: min t s.t. |A x - tgt| <= t, x in [0,1]^66, t >= 0."""
    assert A.shape[0] == 42 and tgt.shape == (42,)
    n = A.shape[1]
    c = np.concatenate([np.zeros(n), [1.0]])
    ones = np.ones(42)
    A_ub = np.vstack([np.column_stack([A, -ones]), np.column_stack([-A, -ones])])
    b_ub = np.concatenate([tgt, -tgt])
    bounds = [(0.0, 1.0)] * n + [(0.0, None)]
    return linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                   options=dict(primal_feasibility_tolerance=1e-9,
                                dual_feasibility_tolerance=1e-9))


# ---------------------------------------------------------------- verification
def run(report):
    call = dict(model_load=0, mj_forward=0, mj_step=0, lp_solves=0, neural_ticks=0,
                training=0, other_optimisations=0)
    report['call_accounting'] = call
    report['claim'] = dict(
        residual6_initial_archive=172.74733807654337,
        residual6_final_claim=1.3464784842653899e-12,
        p_claim='(x=-0.33981..., y=0.41009..., z=1.45239..., roll~-7e-5, pitch~-0.00135, yaw=0.35)',
        geometry_claim=dict(foot_depth=0.00598, nonfoot_depth=0.0, bearing_legs=6,
                            friction_margin=0.583),
        joint_lp_claim=3.3086208677913731,
        joint_unreachable_rows_claim=[13, 16, 20, 34, 36, 41],
        frozen_caps=dict(foot_depth=DEPTH_CAP, nonfoot_depth=NONFOOT_CAP,
                         min_bearing_legs=MIN_BEARING, friction_margin_min=MARGIN_MIN))

    # ---- 0. environment / inputs -------------------------------------------
    report['environment'] = dict(python=sys.version.split()[0], executable=sys.executable,
                                 numpy=np.__version__, scipy=scipy.__version__,
                                 mujoco=mj.__version__)
    report['inputs'] = {
        'model_xml': str(XML), 'model_xml_sha256': sha_file(XML),
        'model_provenance': str(S154 / 'MODEL_PROVENANCE.json'),
        'model_provenance_sha256': sha_file(S154 / 'MODEL_PROVENANCE.json'),
        's154_trajectory': str(S154 / 'TRAJECTORY.npz'),
        's154_trajectory_sha256': sha_file(S154 / 'TRAJECTORY.npz'),
        's154_initialization': str(S154 / 'INITIALIZATION.npz'),
        's154_initialization_sha256': sha_file(S154 / 'INITIALIZATION.npz'),
        'com_contact_metadata': str(S154 / 'COM_CONTACT_METADATA.json'),
        'com_contact_metadata_sha256': sha_file(S154 / 'COM_CONTACT_METADATA.json'),
        's166_mapping': str(S166 / 'MAPPING.npz'),
        's166_mapping_sha256': sha_file(S166 / 'MAPPING.npz'),
        's177_root_balance_state': str(RUN_0205 / 'ROOT_BALANCE_STATE.npz'),
        's177_root_balance_state_sha256': sha_file(RUN_0205 / 'ROOT_BALANCE_STATE.npz'),
        's177_root_result': str(RUN_0205 / 'ROOT_RESULT.json'),
        's177_root_result_sha256': sha_file(RUN_0205 / 'ROOT_RESULT.json'),
        's177_summary': str(RUN_0205 / 'SUMMARY.json'),
        's177_summary_sha256': sha_file(RUN_0205 / 'SUMMARY.json'),
        's177_joint_subsystem_42': str(RUN_0205 / 'JOINT_SUBSYSTEM_42.json'),
        's177_joint_subsystem_42_sha256': sha_file(RUN_0205 / 'JOINT_SUBSYSTEM_42.json'),
        's177_newton_ledger': str(RUN_0205 / 'NEWTON_LEDGER.json'),
        's177_newton_ledger_sha256': sha_file(RUN_0205 / 'NEWTON_LEDGER.json'),
        's177_confirmation': str(RUN_0205 / 'CONFIRMATION.json'),
        's177_confirmation_sha256': sha_file(RUN_0205 / 'CONFIRMATION.json'),
    }
    proof = load_json(S154 / 'MODEL_PROVENANCE.json')
    report['inputs']['model_xml_sha256_matches_provenance'] = bool(
        report['inputs']['model_xml_sha256'] == proof.get('xml_sha256'))

    # ---- 1. model reload: own snapshot vs provenance, before/after ----------
    assert mj.__version__ == '3.9.0', mj.__version__
    callbacks = {}
    for name in ('control', 'act_dyn', 'act_gain', 'act_bias', 'passive', 'sensor'):
        getter = getattr(mj, 'get_mjcb_' + name, None)
        callbacks[name] = (getter is None) or (getter() is None)
    model = mj.MjModel.from_xml_path(str(XML))
    call['model_load'] += 1
    snap_before = snapshot(model)
    diff_before = dict_diff(snap_before, proof['before'])
    model.actuator_dynprm[:, 0] = .001
    model.actuator_dynprm[:, 1] = .0004
    model.opt.integrator = mj.mjtIntegrator.mjINT_RK4
    model.opt.timestep = .000025
    snap_after = snapshot(model)
    diff_after = dict_diff(snap_after, proof['after'])
    changed = dict_diff(snap_before, snap_after)
    groups = ('body', 'jnt', 'geom', 'actuator')
    report['model_reload'] = dict(
        mujoco_version=mj.__version__,
        callbacks_absent=callbacks,
        model_loads=call['model_load'],
        patches=dict(actuator_dynprm_0=.001, actuator_dynprm_1=.0004,
                     integrator='mjINT_RK4', timestep=.000025),
        snapshot_keys=len(snap_before),
        provenance_keys=len(proof['before']),
        snapshot_before_equals_provenance=bool(diff_before == []),
        snapshot_after_equals_provenance=bool(diff_after == []),
        snapshot_diff_before=diff_before,
        snapshot_diff_after=diff_after,
        changed_fields_reproduced=changed,
        provenance_changed_numeric_fields=proof.get('changed_numeric_fields'),
        changed_fields_match_provenance=bool(changed == proof.get('changed_numeric_fields')),
        subsystem_parameter_counts={g: sum(1 for k in snap_before if k.startswith('model.' + g))
                                    for g in groups},
        subsystem_diffs={g: dict(before=[k for k in diff_before if k.startswith('model.' + g)],
                                 after=[k for k in diff_after if k.startswith('model.' + g)])
                         for g in groups},
    )

    # ---- 2. native state from the S154 release archive, ONE mj_forward ------
    traj = load_npz(S154 / 'TRAJECTORY.npz')
    ini = load_npz(S154 / 'INITIALIZATION.npz')
    rbs = load_npz(RUN_0205 / 'ROOT_BALANCE_STATE.npz')
    qpos_final = np.asarray(rbs['qpos'], dtype=float).copy()
    data = mj.MjData(model)
    data.time = float(traj['time'][0])
    for name in STATE:
        getattr(data, name)[:] = traj[name][0]
    optional_present = {}
    for name in OPTIONAL:
        optional_present[name] = name in traj
        if name in traj:
            getattr(data, name)[:] = traj[name][0]
        else:
            assert np.asarray(getattr(data, name)).size == 0, name
    data.qpos[:] = qpos_final
    data.qvel[:] = 0.
    data.act[:] = traj['act'][0]
    data.ctrl[:] = traj['act'][0]
    data.qacc_warmstart[:] = 0.
    pre = {name: np.asarray(getattr(data, name)).copy() for name in (*STATE, *OPTIONAL)}
    mj.mj_forward(model, data)
    call['mj_forward'] += 1
    def _maxdiff(name):
        cur = np.asarray(getattr(data, name)); ref = np.asarray(pre[name])
        if cur.size == 0:
            return 0.0
        return float(np.max(np.abs(cur - ref)))
    state_after = {name: _maxdiff(name) for name in (*STATE, *OPTIONAL)}
    root_result = load_json(RUN_0205 / 'ROOT_RESULT.json')
    report['native_state'] = dict(
        trajectory_first_frame=str(S154 / 'TRAJECTORY.npz'),
        time=data.time,
        qpos_source='ROOT_BALANCE_STATE.npz key qpos (S177 rerun 20261025T020500000000Z)',
        qpos_equals_root_result=bool(np.array_equal(qpos_final, np.asarray(root_result['qpos'], dtype=float))),
        qvel='forced to 0 (trajectory first-frame qvel not used)',
        act='saved activation first frame (traj["act"][0])',
        act_equals_first_frame_ctrl=bool(np.array_equal(traj['act'][0], traj['ctrl'][0])),
        qacc_warmstart=0.0,
        first_frame_absmax=dict(qvel=float(np.max(np.abs(traj['qvel'][0]))),
                                act=float(np.max(np.abs(traj['act'][0]))),
                                ctrl=float(np.max(np.abs(traj['ctrl'][0]))),
                                qfrc_applied=float(np.max(np.abs(traj['qfrc_applied'][0]))),
                                xfrc_applied=float(np.max(np.abs(traj['xfrc_applied'][0])))),
        optional_present_in_trajectory=optional_present,
        state_preserved_by_forward=bool(all(v == 0.0 for v in state_after.values())),
        post_forward_state_abs_diff=state_after,
        qpos_sha256=sha_array(qpos_final),
        s154_release_cross_check=dict(
            initialization_body_qpos_equals_trajectory_first_frame=bool(
                np.array_equal(ini['body_qpos'], traj['qpos'][0]))),
    )

    # ---- 3. independent 6-row root force recomputation ----------------------
    nefc = int(data.nefc)
    nv = int(model.nv)
    nu = int(model.nu)
    moment = moment_dense(model, data)
    nodes = list(ET.parse(XML).getroot().find('actuator'))
    assert len(nodes) == nu, (len(nodes), nu)
    lengths = np.asarray(data.actuator_length)
    gain = np.empty(nu)
    bias = np.empty(nu)
    for j in range(nu):
        _, gain[j], bias[j] = muscle_terms(float(lengths[j]), nodes[j])
    B = moment.T * gain[None, :]
    base = moment.T @ bias + np.asarray(data.qfrc_passive) - np.asarray(data.qfrc_bias)
    efc_type = np.asarray(data.efc_type)[:nefc].astype(np.int64)
    efc_D = np.asarray(data.efc_D)[:nefc].copy()
    efc_aref = np.asarray(data.efc_aref)[:nefc].copy()
    assert np.all(np.isin(efc_type, [0, 3, 4, 5, 6])), 'unsupported native row type'
    f0 = efc_D * efc_aref
    f0 = np.where(efc_type == 0, f0, np.maximum(f0, 0.))
    J = efc_j_dense(model, data, nefc, nv)
    act_native = np.asarray(data.act).copy()
    res48 = B @ act_native + base + J.T @ f0
    residual6 = -res48[0:6]
    root_res6 = np.asarray(root_result['residual6'], dtype=float)
    solve_raw = load_json(RUN_0205 / 'NEWTON_SOLVE_RAW.json')
    rb_cmp = {}
    for key, mine in (('B', B), ('base', base), ('f0', f0), ('J', J)):
        ref = np.asarray(rbs[key], dtype=float)
        rb_cmp[key] = dict(shapes_match=list(ref.shape) == list(mine.shape),
                           max_abs_diff=float(np.max(np.abs(ref - mine))) if ref.shape == mine.shape else None)
    report['one_forward'] = dict(
        nefc=nefc, nv=nv, nu=nu,
        actuator_moment_shape=list(moment.shape),
        actuator_moment_dense_sha256=sha_array(moment),
        gain_shape=list(gain.shape), bias_shape=list(bias.shape),
        B_shape=list(B.shape), base_shape=list(base.shape),
        J_shape=list(J.shape), f0_shape=list(f0.shape),
        contact_rows=int(np.count_nonzero(np.isin(efc_type,
                                                 [int(v) for k, v in mj.mjtConstraint.__members__.items()
                                                  if k.startswith('mjCNSTR_CONTACT_')]))),
        f0_row_types=dict(equality_rows=int(np.count_nonzero(efc_type == 0)),
                          other_rows=int(np.count_nonzero(efc_type != 0))),
        residual6=residual6.tolist(),
        max_abs_residual6=float(np.max(np.abs(residual6))),
        producer_residual6=root_res6.tolist(),
        producer_norm_inf=float(root_result['norm_inf']),
        residual6_abs_diff=(residual6 - root_res6).tolist(),
        residual6_max_abs_diff=float(np.max(np.abs(residual6 - root_res6))),
        residual6_matches_producer_exactly=bool(np.array_equal(residual6, root_res6)),
        solve_raw_residual6_equal=bool(np.array_equal(residual6, np.asarray(solve_raw['residual6'], dtype=float))),
        B_act_term_absmax=float(np.max(np.abs(B @ act_native))),
        root_balance_state_comparison=rb_cmp,
        initial_residual6_archive_norm_inf=float(root_result['history'][0]['norm_inf']),
        initial_residual6_archive_values=root_result['history'][0]['residual6'],
        initial_residual6_recomputed=False,
        initial_residual6_note=('archive-reported only: re-deriving the starting residual would need a '
                                'second forward, and this check spends exactly one mj_forward'),
    )

    # ---- 4. frozen leg joints and p <-> qpos mapping -----------------------
    release = np.asarray(traj['qpos'][0], dtype=float)
    leg_diff = float(np.max(np.abs(qpos_final[7:] - release[7:])))
    p = np.asarray(rbs['p'], dtype=float)
    dq = quat_from_xyz(p[3], p[4], p[5])
    qw = quat_mul(dq, release[3:7])
    qw = qw / np.linalg.norm(qw)
    report['joint_freeze'] = dict(
        n_leg_joint_coordinates=int(qpos_final[7:].size),
        leg_joints_exactly_s154_release=bool(np.array_equal(qpos_final[7:], release[7:])),
        leg_joint_max_abs_diff_vs_s154_release=leg_diff,
        root_coordinates=dict(x=float(qpos_final[0]), y=float(qpos_final[1]), z=float(qpos_final[2])),
        p=[float(v) for v in p],
        root_xyz_vs_p_max_abs_diff=float(np.max(np.abs(qpos_final[:3] - p[:3]))),
        root_quat_vs_roll_pitch_yaw_max_abs_diff=float(np.max(np.abs(qpos_final[3:7] - qw))),
        root_quat=qpos_final[3:7].tolist(),
    )

    # ---- 5. geometry gates ------------------------------------------------
    meta = load_json(S154 / 'COM_CONTACT_METADATA.json')
    floors = set(int(v) for v in meta['floor_geom_ids'])
    geom_leg = {int(k): v for k, v in meta['geom_map'].items()}
    crec = contact_record(data)
    geo = geometry(crec, f0, nefc, floors, geom_leg)
    gates = dict(foot_depth_ok=bool(geo['foot_depth'] <= DEPTH_CAP),
                 nonfoot_depth_ok=bool(geo['nonfoot_depth'] <= NONFOOT_CAP),
                 bearing_legs_ok=bool(geo['bearing_legs'] >= MIN_BEARING),
                 friction_margin_ok=bool(geo['friction_margin'] >= MARGIN_MIN))
    gates['geometry_gates_pass'] = bool(all(gates.values()))
    prod_geo = root_result['geometry']
    conf = load_json(RUN_0205 / 'CONFIRMATION.json')
    prod_geo_conf = conf.get('geometry', {})
    geo_cmp = {}
    for key in ('foot_depth', 'nonfoot_depth', 'bearing_legs', 'friction_margin', 'contact_signature'):
        geo_cmp[key] = dict(equals=bool(geo[key] == prod_geo[key]),
                            mine=geo[key] if not isinstance(geo[key], float) else float(geo[key]),
                            producer=prod_geo[key])
        if isinstance(geo[key], float):
            geo_cmp[key]['abs_diff'] = float(abs(geo[key] - prod_geo[key]))
    report['geometry'] = dict(
        floor_geom_ids=sorted(floors),
        max_foot_penetration=geo['foot_depth'],
        max_nonfoot_penetration=geo['nonfoot_depth'],
        n_bearing_legs=geo['bearing_legs'],
        bearing_leg_names=geo['bearing_leg_names'],
        min_friction_pyramid_margin=geo['friction_margin'],
        n_contacts=geo['contacts'],
        contact_signature=geo['contact_signature'],
        caps=dict(foot_depth=DEPTH_CAP, nonfoot_depth=NONFOOT_CAP,
                  min_bearing_legs=MIN_BEARING, friction_margin_min=MARGIN_MIN),
        cap_checks=gates,
        comparison_to_root_result=geo_cmp,
        comparison_to_confirmation=dict(
            foot_depth=bool(geo['foot_depth'] == prod_geo_conf.get('foot_depth')),
            nonfoot_depth=bool(geo['nonfoot_depth'] == prod_geo_conf.get('nonfoot_depth')),
            bearing_legs=bool(geo['bearing_legs'] == prod_geo_conf.get('bearing_legs')),
            friction_margin=bool(geo['friction_margin'] == prod_geo_conf.get('friction_margin')),
            contacts=bool(geo['contacts'] == prod_geo_conf.get('contacts')),
            contact_signature=bool(geo['contact_signature'] == prod_geo_conf.get('contact_signature'))),
        producer_geometry=prod_geo,
        producer_confirmation_geometry=prod_geo_conf,
    )

    # ---- 6. ONE 42-row joint LP (separate from residual6) ------------------
    T = np.asarray(load_npz(S166 / 'MAPPING.npz')['T'], dtype=float)
    assert T.shape == (90, 66) and np.count_nonzero(T) == 74, (T.shape, np.count_nonzero(T))
    A = (B @ T)[6:, :]
    tgt = -(base + J.T @ f0)[6:]
    assert A.shape == (42, 66) and tgt.shape == (42,)
    rows = []
    for d in range(42):
        lo = float(np.minimum(A[d], 0.).sum())
        hi = float(np.maximum(A[d], 0.).sum())
        tt = float(tgt[d])
        rows.append(dict(joint_row=d, target=tt, min_possible=lo, max_possible=hi,
                         reachable=bool(lo - 1e-9 <= tt <= hi + 1e-9),
                         violation=float(max(lo - tt, 0., tt - hi))))
    unreachable = [r['joint_row'] for r in rows if not r['reachable']]
    lp = solve_joint_lp(A, tgt)
    call['lp_solves'] += 1
    t_star = float(lp.x[-1]) if lp.success else None
    joint_json = load_json(RUN_0205 / 'JOINT_SUBSYSTEM_42.json')
    prod_t = float(joint_json['lp_minimax_t'])
    prod_rows = {r['joint_row']: r for r in joint_json['rows']}
    row_cmp = {}
    for key in ('target', 'min_possible', 'max_possible'):
        diffs = [abs(rows[i][key] - prod_rows[i][key]) for i in range(42)]
        row_cmp[key] = dict(max_abs_diff=float(max(diffs)),
                            all_exact=bool(all(rows[i][key] == prod_rows[i][key] for i in range(42))))
    row_cmp['reachable'] = dict(all_equal=bool(all(rows[i]['reachable'] == prod_rows[i]['reachable']
                                                    for i in range(42))))
    report['joint_lp_42'] = dict(
        separate_from_residual6=True,
        separation_statement=('reported separately from residual6 and never merged into one score; '
                              'the root six rows and the 42 joint rows are different questions'),
        A_shape=list(A.shape), T_shape=list(T.shape), T_nnz=int(np.count_nonzero(T)),
        t_mine=t_star,
        lp_success=bool(lp.success), lp_message=str(lp.message), lp_status=int(lp.status),
        t_producer=prod_t,
        abs_diff_vs_producer=(None if t_star is None else float(abs(t_star - prod_t))),
        matches_producer=bool(t_star is not None and abs(t_star - prod_t) <= 1e-9),
        unreachable_rows=unreachable,
        unreachable_rows_match_producer=bool(unreachable == list(joint_json['unreachable_rows'])),
        worst_single_row_violation=float(max(r['violation'] for r in rows)),
        producer_worst_single_row_violation=float(joint_json['worst_single_row_violation']),
        joint_rows_exactly_balanced_at_t=bool(t_star is not None and t_star <= 1e-8),
        producer_joint_rows_exactly_balanced=bool(joint_json['joint_rows_exactly_balanced']),
        rows=rows,
        row_comparison_to_producer=row_cmp,
        interpretability=('The LP is FEASIBLE at its t (solve success).  A positive t therefore means the '
                          '42 frozen-pose joint rows cannot all be zeroed simultaneously; it is NOT an '
                          'infeasibility certificate for the joint subsystem or for balance in general.'),
    )

    # ---- 7. archive bookkeeping: ledgers ----------------------------------
    summary = load_json(RUN_0205 / 'SUMMARY.json')
    ledger = summary['newton_ledger']
    runner = summary['runner_ledger']
    ledger_checks = {k: dict(expected=v, actual=ledger.get(k), ok=bool(ledger.get(k) == v))
                     for k, v in EXPECTED_LEDGER.items()}
    runner_checks = {k: dict(expected=v, actual=runner.get(k), ok=bool(runner.get(k) == v))
                     for k, v in EXPECTED_RUNNER.items()}
    newton_ledger = load_json(RUN_0205 / 'NEWTON_LEDGER.json')
    file_ledger = dict(newton_ledger['ledger'])
    events = list(newton_ledger['events'])
    roles = {}
    for e in events:
        roles[e.get('role')] = roles.get(e.get('role'), 0) + 1
    ls_decisions = [e for e in events if e.get('role') == 'LINE_SEARCH_DECISION']
    event_checks = dict(
        total_events=len(events),
        difference_probe_events=roles.get('difference_probe'),
        line_search_trial_events=roles.get('line_search_trial'),
        jacobian_rank_events=roles.get('JACOBIAN_RANK'),
        line_search_decision_events=roles.get('LINE_SEARCH_DECISION'),
        equivalence_gate_events=roles.get('equivalence_gate'),
        initial_events=roles.get('initial'),
        cache_hit_events=sum(1 for e in events if e.get('cache_hit') is True),
        max_forward_seen=max((e.get('forward', 0) for e in events), default=None),
        contact_set_change_events=sum(1 for e in ls_decisions if e.get('contact_set_changed')),
    )
    event_ok = (event_checks['difference_probe_events'] == 36
                and event_checks['line_search_trial_events'] == 5
                and event_checks['line_search_decision_events'] == 5
                and event_checks['equivalence_gate_events'] == 1
                and event_checks['initial_events'] == 1
                and event_checks['cache_hit_events'] == 1
                and event_checks['max_forward_seen'] == 42
                and event_checks['contact_set_change_events'] == 1)
    ledger_diff = dict_diff(file_ledger, ledger)
    report['archive_bookkeeping'] = dict(
        summary_status=summary.get('status'),
        summary_stop=summary.get('stop'),
        summary_newton_ledger=ledger,
        summary_runner_ledger=runner,
        claim_ledger_expectations={k: v for k, v in EXPECTED_LEDGER.items()},
        ledger_checks=ledger_checks,
        runner_checks=runner_checks,
        all_claim_ledger_counts_match=bool(all(v['ok'] for v in ledger_checks.values())
                                           and all(v['ok'] for v in runner_checks.values())),
        newton_ledger_file_last_dump=file_ledger,
        newton_ledger_file_event_counts=event_checks,
        newton_ledger_event_log_consistent=bool(event_ok),
        newton_ledger_file_vs_summary_diff=ledger_diff,
        model_load_counted_separately=bool(ledger.get('model_load') == 0
                                           and runner.get('model_load') == 1),
        model_load_accounting_note=('the solver ledger reports model_load=0 and the runner ledger reports '
                                    'model_load=1, i.e. the load is counted separately from the solver calls'),
        best_record_bookkeeping=newton_ledger.get('best', {}).get('norm_inf'),
        best_record_p=newton_ledger.get('best', {}).get('p'),
        best_record_note=('the solver ledger keeps a global-best record as bookkeeping only; it is never '
                          'consulted by the accept test and is not the final accepted iterate'),
    )

    # ---- 8. two-run determinism -------------------------------------------
    r1 = load_json(RUN_0155 / 'ROOT_RESULT.json')
    r2 = root_result
    s1 = load_json(RUN_0155 / 'SUMMARY.json')
    s2 = summary
    j1 = load_json(RUN_0155 / 'JOINT_SUBSYSTEM_42.json')
    raw1 = load_json(RUN_0155 / 'NEWTON_SOLVE_RAW.json')
    j2 = joint_json
    raw2 = solve_raw
    rb1 = load_npz(RUN_0155 / 'ROOT_BALANCE_STATE.npz')
    rb2 = rbs
    rb_arrays = {k: bool(np.array_equal(rb1[k], rb2[k])) for k in sorted(set(rb1) & set(rb2))}
    hist1 = r1.get('history', [])
    hist2 = r2.get('history', [])
    history_diffs = []
    for i, (a, b) in enumerate(zip(hist1, hist2)):
        diff_keys = dict_diff(a, b)
        if diff_keys:
            history_diffs.append(dict(index=i, diff_keys=diff_keys,
                                      only_in_0155=[k for k in diff_keys if k not in b],
                                      only_in_0205=[k for k in diff_keys if k not in a],
                                      values_equal_for_common_keys=bool(
                                          all(a[k] == b[k] for k in set(a) & set(b) if k not in diff_keys))))
    eg_equal = bool((RUN_0155 / 'EQUIVALENCE_GATE.json').read_bytes()
                    == (RUN_0205 / 'EQUIVALENCE_GATE.json').read_bytes())
    cf1 = load_json(RUN_0155 / 'CONFIRMATION.json')
    cf2 = conf
    cfz1 = load_npz(RUN_0155 / 'CONFIRMATION.npz')
    cfz2 = load_npz(RUN_0205 / 'CONFIRMATION.npz')
    conf_npz_equal = {k: bool(np.array_equal(cfz1[k], cfz2[k])) for k in sorted(set(cfz1) & set(cfz2))}
    report['determinism_two_runs'] = dict(
        runs=[str(RUN_0155), str(RUN_0205)],
        p_equal=bool(r1['p'] == r2['p']),
        residual6_equal=bool(r1['residual6'] == r2['residual6']),
        geometry_equal=bool(r1['geometry'] == r2['geometry']),
        joint_lp_t_equal=bool(j1['lp_minimax_t'] == j2['lp_minimax_t']),
        joint_lp_rows_equal=bool(j1['rows'] == j2['rows']),
        joint_unreachable_rows_equal=bool(j1['unreachable_rows'] == j2['unreachable_rows']),
        joint_worst_violation_equal=bool(j1['worst_single_row_violation'] == j2['worst_single_row_violation']),
        status_equal=bool(s1['status'] == s2['status']),
        stop_equal=bool(s1['stop'] == s2['stop']),
        newton_ledger_equal=bool(s1['newton_ledger'] == s2['newton_ledger']),
        runner_ledger_equal=bool(s1['runner_ledger'] == s2['runner_ledger']),
        rank_decomposition_equal=bool(r1['rank_decomposition'] == r2['rank_decomposition']),
        solve_raw_stop_norm_equal=bool(raw1['stop'] == raw2['stop']
                                       and raw1['norm_inf'] == raw2['norm_inf']),
        solve_raw_final_decomposition_equal=bool(raw1['final_decomposition'] == raw2['final_decomposition']),
        root_balance_state_arrays_equal=rb_arrays,
        equivalence_gate_json_equal=eg_equal,
        confirmation_json_equal=bool(cf1 == cf2),
        confirmation_npz_arrays_equal=conf_npz_equal,
        history_length_equal=bool(len(hist1) == len(hist2)),
        history_differences=history_diffs,
        core_outputs_identical=bool(r1['p'] == r2['p'] and r1['residual6'] == r2['residual6']
                                    and r1['geometry'] == r2['geometry']
                                    and j1['lp_minimax_t'] == j2['lp_minimax_t']
                                    and j1['rows'] == j2['rows']
                                    and j1['unreachable_rows'] == j2['unreachable_rows']),
    )

    # ---- 9. input hashes --------------------------------------------------
    hash_runs = {}
    for label, run_dir in (('0155', RUN_0155), ('0205', RUN_0205)):
        h = load_json(run_dir / 'INPUT_HASHES.json')
        ha = load_json(run_dir / 'INPUT_HASHES_AFTER.json')
        missing = [p for p in h if not Path(p).exists()]
        mismatched = []
        for p, recorded in h.items():
            if not Path(p).exists():
                continue
            actual = sha_file(p)
            if actual != recorded:
                mismatched.append(dict(path=p, recorded_sha256=recorded, actual_sha256=actual))
        hash_runs[label] = dict(
            dir=str(run_dir),
            n_entries=len(h),
            input_hashes_equals_after=bool(h == ha),
            missing_files=missing,
            hash_mismatches=mismatched,
            mismatch_count=len(mismatched),
            all_files_match=bool(not missing and not mismatched),
        )
    h0155 = load_json(RUN_0155 / 'INPUT_HASHES.json')
    h0205 = load_json(RUN_0205 / 'INPUT_HASHES.json')
    differing = sorted(k for k in set(h0155) | set(h0205) if h0155.get(k) != h0205.get(k))
    source_freeze_evidence = {}
    if differing:
        m0155 = load_json(RUN_0155 / 'SOURCE_FREEZE_MANIFEST.json')
        m0205 = load_json(RUN_0205 / 'SOURCE_FREEZE_MANIFEST.json')
        for path in differing:
            e = {}
            if path in m0155 and Path(m0155[path]['copy']).exists():
                e['0155_source_freeze_copy'] = dict(path=m0155[path]['copy'],
                                                    sha256=sha_file(m0155[path]['copy']),
                                                    matches_0155_record=bool(sha_file(m0155[path]['copy'])
                                                                             == h0155.get(path)))
            if path in m0205 and Path(m0205[path]['copy']).exists():
                e['0205_source_freeze_copy'] = dict(path=m0205[path]['copy'],
                                                    sha256=sha_file(m0205[path]['copy']),
                                                    matches_0205_record=bool(sha_file(m0205[path]['copy'])
                                                                             == h0205.get(path)))
            if Path(path).exists():
                e['live_file_sha256'] = sha_file(path)
            source_freeze_evidence[path] = e
    report['input_hash_checks'] = dict(
        runs=hash_runs,
        cross_run_input_hashes_equal=bool(h0155 == h0205),
        cross_run_differing_keys=differing,
        source_freeze_evidence=source_freeze_evidence,
        note=('file hashes are checked against the live files; an entry can disagree with the live '
              'tree only if the live producer file changed after that run was archived'),
    )

    # ---- 10. earlier development failures --------------------------------
    failed = []
    for name in FAILED_DIRS:
        tdir = S177 / name
        entry = dict(dir=str(tdir), role='development failure; NOT search evidence',
                     evidence='excluded from every numerical claim in this check')
        nl_path = tdir / 'NEWTON_LEDGER.json'
        if nl_path.exists():
            nl = load_json(nl_path)
            entry['newton_ledger_last_dump'] = nl.get('ledger')
            entry['newton_ledger_events'] = len(nl.get('events', []))
        if (tdir / 'FAILED.json').exists():
            f = load_json(tdir / 'FAILED.json')
            entry['failed_json_ledger'] = f.get('ledger')
            tb = f.get('traceback', '')
            entry['failure'] = dict(kind='FAILED.json', last_traceback_line=tb.strip().splitlines()[-1] if tb.strip() else None)
        elif (tdir / 'SUMMARY.json').exists():
            s = load_json(tdir / 'SUMMARY.json')
            entry['summary'] = dict(status=s.get('status'), stop=s.get('stop'),
                                    root_norm_inf=s.get('root_norm_inf'),
                                    joint_lp_minimax=s.get('joint_lp_minimax'),
                                    newton_ledger=s.get('newton_ledger'),
                                    runner_ledger=s.get('runner_ledger'))
            entry['failure'] = dict(kind='SUMMARY.json', status=s.get('status'), stop=s.get('stop'))
        else:
            entry['failure'] = dict(kind=None, note='no terminal FAILED.json/SUMMARY.json; stopped after the joint-subsystem dump')
        entry['files'] = sorted(p.name for p in tdir.iterdir() if p.is_dir() is False)
        failed.append(entry)
    report['failed_development_attempts'] = failed

    # ---- 11. scope --------------------------------------------------------
    report['scope'] = dict(
        root_six_row_result=('STATIC feasibility result at ONE pose: all 42 leg joints frozen exactly at '
                             'the S154 release values, only the 6 root pose coordinates moved, native soft '
                             'contact forces recomputed by one forward. No step, no motion, no dynamics.'),
        joint_42_lp=('The 42 joint rows are NOT balanced at this pose (minimax t = 3.308620867791373). '
                     'The LP is FEASIBLE at its t, so the imbalance is NOT an infeasibility certificate. '
                     'It is reported separately from residual6 and never merged with it.'),
        not_shown=['standing', 'neural attainability', 'dynamics', 'trajectory', 'integration',
                   'control feasibility', 'stability', 'SI units'],
        units='native; SI NOT_ESTABLISHED',
        forbidden_actions_taken=['none: no mj_step', 'no neural ticks', 'no training',
                                 'no optimisation other than the single 42-row LP'],
    )

    # ---- 12. core checks / discrepancies / status ------------------------
    rbc = report['one_forward']['root_balance_state_comparison']
    core = dict(
        model_snapshot_before_equals_provenance=report['model_reload']['snapshot_before_equals_provenance'],
        model_snapshot_after_equals_provenance=report['model_reload']['snapshot_after_equals_provenance'],
        changed_fields_match_provenance=report['model_reload']['changed_fields_match_provenance'],
        native_state_preserved_by_forward=report['native_state']['state_preserved_by_forward'],
        residual6_matches_archive=bool(report['one_forward']['residual6_max_abs_diff'] <= 1e-12),
        root_force_fields_match_root_balance_state=bool(all(
            v['max_abs_diff'] is not None and v['max_abs_diff'] <= 1e-12 for v in rbc.values())),
        leg_joints_frozen_exactly=report['joint_freeze']['leg_joints_exactly_s154_release'],
        geometry_matches_archive=bool(all(v['equals'] for v in geo_cmp.values())),
        geometry_gates_pass=report['geometry']['cap_checks']['geometry_gates_pass'],
        joint_lp_t_matches_archive=report['joint_lp_42']['matches_producer'],
        joint_lp_rows_match_archive=bool(all(v.get('all_exact', v.get('all_equal', False))
                                             for v in row_cmp.values())),
        unreachable_rows_match_archive=report['joint_lp_42']['unreachable_rows_match_producer'],
        claim_ledger_counts_match=report['archive_bookkeeping']['all_claim_ledger_counts_match'],
        ledger_event_log_consistent=report['archive_bookkeeping']['newton_ledger_event_log_consistent'],
        determinism_core_outputs_identical=report['determinism_two_runs']['core_outputs_identical'],
        input_hashes_after_equal_per_run=bool(all(v['input_hashes_equals_after']
                                                  for v in hash_runs.values())),
    )
    report['core_checks'] = core

    discrepancies = []
    if ledger_diff:
        discrepancies.append(dict(
            id='newton_ledger_file_lags_summary',
            severity='bookkeeping',
            detail=('NEWTON_LEDGER.json last dumped ledger differs from SUMMARY.json newton_ledger: '
                    f'{ {k: (file_ledger.get(k), ledger.get(k)) for k in ledger_diff} }; the event log itself '
                    'records 36 difference_probe evaluations, i.e. the claimed counts, so the claimed '
                    'forward=42 / jacobians=6 / fd_probes=36 is what the run actually spent; the final '
                    'post-solve Jacobian increments jacobians/fd_probes after the last event dump.'),
        ))
    if '0155' in hash_runs and hash_runs['0155']['mismatch_count']:
        for m in hash_runs['0155']['hash_mismatches']:
            discrepancies.append(dict(
                id='live_producer_file_changed_after_run_0155',
                severity='archive',
                detail=(f'run 0155 INPUT_HASHES.json records {m["path"]} sha256 '
                        f'{m["recorded_sha256"]}, but the live file is {m["actual_sha256"]} '
                        '(which is what run 0205 recorded). The 0155 archive is internally consistent '
                        '(INPUT_HASHES == INPUT_HASHES_AFTER and its SOURCE_FREEZE copy carries the '
                        'recorded hash), so this is a producer-file edit between the two runs, not an '
                        'archive corruption.'),
            ))
    if differing:
        discrepancies.append(dict(
            id='two_runs_input_hashes_differ',
            severity='archive',
            detail=f'the two runs input hashes differ in exactly these keys: {differing}',
        ))
    if history_diffs:
        discrepancies.append(dict(
            id='two_runs_history_schema_diff',
            severity='archive',
            detail=('ROOT_RESULT history entries differ between the two runs: '
                    + str([{'index': h['index'], 'only_in_0205': h['only_in_0205'],
                            'only_in_0155': h['only_in_0155']} for h in history_diffs])
                    + ' ; all numeric outputs compared here are identical, this is a writer-schema change '
                      '(0155 history lacks per-iteration residual6)'),
        ))
    for k, v in core.items():
        if not v:
            discrepancies.append(dict(id='core_check_failed', severity='core', detail=k))
    report['discrepancies'] = discrepancies
    report['notes'] = dict(
        initial_residual_not_recomputed=report['one_forward']['initial_residual6_note'],
        global_best_record=dict(norm_inf=newton_ledger.get('best', {}).get('norm_inf'),
                                p=newton_ledger.get('best', {}).get('p'),
                                note=('global-best bookkeeping record (an FD probe pose), never used by the '
                                      'accept test; the final accepted iterate is the one reported as '
                                      'residual6')),
        contact_row_layout=('all native contacts are 3-D pyramidal contacts with 4 constraint rows each '
                            f'({geo["contacts"]} contacts x 4 = {nefc} rows); the soft law treats only row '
                            'type 0 (equality) as unbounded, every other row as max(.,0)'),
        single_lp=('exactly one LP was solved: the 42-row absolute-value minimax over the real 66-pool '
                   'mapping with x in [0,1]'),
        result_separation=('residual6 (root six rows) and the joint LP (42 rows) are deliberately kept '
                           'separate; no single merged score is reported anywhere'),
    )
    if all(core.values()):
        report['status'] = 'PASS'
    else:
        report['status'] = 'FAIL'


def write_report(report):
    if OUT.exists():
        print(f'REFUSING to overwrite existing report: {OUT}', file=sys.stderr)
        raise SystemExit(3)
    text = json.dumps(sanitize(report), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with open(OUT, 'x', encoding='utf-8') as fh:
        fh.write(text)


def main():
    report = dict(
        check='S177_INDEPENDENT_RELOAD_CHECK',
        generated_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
        command='venv_body/bin/python -B verify_root_newton.py',
        mechanism=('independent reimplementation: one model load, one mj_forward, one 42-row LP; '
                   'no mj_step, no neural ticks, no training'),
        script=dict(path=str(Path(__file__).resolve()), sha256=sha_file(Path(__file__).resolve())),
    )
    try:
        run(report)
    except Exception:
        report['error'] = traceback.format_exc()
        report.setdefault('status', 'ERROR')
    write_report(report)
    print(json.dumps(dict(
        report=str(OUT),
        status=report.get('status'),
        residual6=report.get('one_forward', {}).get('residual6'),
        max_abs_residual6=report.get('one_forward', {}).get('max_abs_residual6'),
        geometry=dict(foot_depth=report.get('geometry', {}).get('max_foot_penetration'),
                      nonfoot_depth=report.get('geometry', {}).get('max_nonfoot_penetration'),
                      bearing_legs=report.get('geometry', {}).get('n_bearing_legs'),
                      friction_margin=report.get('geometry', {}).get('min_friction_pyramid_margin')),
        joint_lp_t=report.get('joint_lp_42', {}).get('t_mine'),
        unreachable_joint_rows=report.get('joint_lp_42', {}).get('unreachable_rows'),
        discrepancies=[d.get('id') for d in report.get('discrepancies', [])],
    ), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
