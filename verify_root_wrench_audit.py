"""S175 independent SAVED-DATA verification of the root-wrench audit + row-interval addendum.

Scope: saved arrays only. No engine import, no model load, no forward/step, no neural
component, no physics. The only solver use is 6 small box-constrained minimax LPs
(42 rows x 66/90 vars) re-solved from scratch. Every producer table is RECOMPUTED and
then compared; no saved table value is trusted.

Producer facts preserved as-is (not repaired, not overwritten):
  * audit_root_wrench.py attempt 1 -> exit code 1 at its LAST plotting line, AFTER all
    tables (ACTUATOR_AUTHORITY_BY_DOF.csv, ROOT_ROW_DECOMPOSITION.csv,
    SUBSYSTEM_42ROW_FEASIBILITY.json, CONTACT_WRENCH_AND_COM.json, S174_CONSISTENCY.json)
    had already been written. The interrupted figure was later produced by the addendum.
  * audit_row_intervals.py addendum -> ROW_INTERVAL_REACHABILITY.csv,
    ROW_INTERVAL_SUMMARY.json and ROOT_WRENCH_AUDIT.png.

Writes exactly one new file (refuses to overwrite it).
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

BASE = Path('/home/sensen/Desktop/engine修复验收')
S169 = BASE / 's169-near-initial-support/20261024T233428554259Z'
S173 = BASE / 's173-accounted-near-initial-search/20261024T235544196037334Z/RUN'
MAPPING = BASE / 's166-mapped-support/20261024T231418078692647Z/MAPPING.npz'
NATIVE_DIR = BASE / 's154-activation-rate/20261024T215751775489Z/init80ms'
S174_DIR = BASE / 's174-best-pose-interface-bound/20261025T000617427397Z'
OUT_DIR = BASE / 's175-root-wrench-audit/20261025T001707433958Z'
OUT_JSON = OUT_DIR / 'S175_SAVED_INDEPENDENT_CHECK.json'

INPUTS = [
    S169 / 'BEST_DIAGNOSTIC_NOT_CANDIDATE.npz',
    S173 / 'BEST_DIAGNOSTIC_NOT_CANDIDATE.npz',
    S169 / 'ORIGINAL_POSE_OPTIMIZED_POOLS_DIAGNOSTIC.npz',
    MAPPING,
    NATIVE_DIR / 'NATIVE_MODEL_ARRAYS.npz',
    NATIVE_DIR / 'COM_CONTACT_METADATA.json',
    S174_DIR / 'CASE_wide.npz',
]
POSES = [
    ('s169_best91', S169 / 'BEST_DIAGNOSTIC_NOT_CANDIDATE.npz'),
    ('s173_best303', S173 / 'BEST_DIAGNOSTIC_NOT_CANDIDATE.npz'),
    ('original_release_eval1', S169 / 'ORIGINAL_POSE_OPTIMIZED_POOLS_DIAGNOSTIC.npz'),
]
EXPECTED_UNREACHABLE = {
    ('s173_best303', 'real66'): [2, 3, 4, 5, 19, 20, 40, 42, 47],
    ('s173_best303', 'widest90'): [2, 3, 4, 5, 19, 40, 47],
}
EXPECTED_UNREACHABLE_COUNT = {('s169_best91', 'real66'): 11, ('original_release_eval1', 'real66'): 31}
EXPECTED_T = {'s169_best91': 3.3658522987215624, 's173_best303': 2.8716071056150922,
              'original_release_eval1': 20.095564476160046}
TOL_T = 1e-9          # LP minimax value agreement (task tolerance)
TOL_CSV = 1e-12       # producer-table agreement = our own roundoff allowance
TOL_INTERVAL = 1e-9   # interval-reachability flag boundary (matches producer semantics)

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
load = lambda p: {k: np.load(p)[k].copy() for k in np.load(p).files}


def main() -> int:
    if OUT_JSON.exists():
        print(f'REFUSING TO OVERWRITE {OUT_JSON}', file=sys.stderr)
        return 2

    disc: list[str] = []          # hard discrepancies -> verified False
    notes: list[str] = []         # observations, not discrepancies
    lp_calls = 0

    h_before = {str(p): sha(p) for p in INPUTS}
    native = load(NATIVE_DIR / 'NATIVE_MODEL_ARRAYS.npz')
    meta = json.loads((NATIVE_DIR / 'COM_CONTACT_METADATA.json').read_text())
    mass = native['body_mass']
    W = float(mass.sum() * 9801.)
    T_real = load(MAPPING)['T']
    SETS = [('real66', T_real), ('widest90', 0.5 * np.eye(90))]
    poses = {name: load(path) for name, path in POSES}

    # ---------------------------------------------------------------- 1. authority
    authority = []          # per pose/set/DOF max|A[d,:]|, A = B@T
    auth_summary = []
    for name, _ in POSES:
        z = poses[name]
        B = z['B']
        assert B.shape == (48, 90)
        for setname, Tm in SETS:
            A = B @ Tm
            row = np.abs(A).max(axis=1)
            root012 = row[0:3]
            root345 = row[3:6]
            joint = row[6:]
            autority_ok_012 = bool(np.all(root012 == 0.0))
            ok_345 = bool(np.all(root345 <= 1e-12))
            joint_ok = bool(np.all(joint >= 1e-1) and np.all(joint <= 1e2))
            if not autority_ok_012:
                disc.append(f'{name}/{setname}: DOF0-2 authority not exactly 0 (max {root012.max()!r})')
            if not ok_345:
                disc.append(f'{name}/{setname}: DOF3-5 authority > 1e-12 (max {root345.max()!r})')
            if not joint_ok:
                disc.append(f'{name}/{setname}: joint authority outside [1e-1,1e2] '
                            f'(min {joint.min()!r}, max {joint.max()!r})')
            if joint.max() > 10.0:
                notes.append(f'{name}/{setname}: joint max authority {joint.max():.4f} exceeds the '
                             f'10x upper end of "O(1e-1..1e1)" by a factor {joint.max() / 10.0:.2f}')
            for d in range(48):
                authority.append(dict(pose=name, activation_set=setname, dof=d,
                                      max_abs_A_row=float(row[d])))
            auth_summary.append(dict(
                pose=name, activation_set=setname,
                max_authority_dof0_2=float(root012.max()), dof0_2_exactly_zero=autority_ok_012,
                max_authority_dof3_5=float(root345.max()), dof3_5_le_1e_12=ok_345,
                joint_authority_min=float(joint.min()), joint_authority_max=float(joint.max()),
                joint_authority_o_0p1_to_10=joint_ok))

    # ------------------------------------------- 2. root rows (0-5) decomposition
    root_rows = []
    root_checks = []
    for name, _ in POSES:
        z = poses[name]
        B, act, base, J, f0, res = z['B'], z['act'], z['base'], z['J'], z['f0'], z['residual48']
        con = J.T @ f0
        muscle = B @ act
        for d in range(6):
            root_rows.append(dict(pose=name, dof=d, muscle=float(muscle[d]), base=float(base[d]),
                                  contact=float(con[d]), residual=float(res[d]),
                                  residual_minus_sum=float(res[d] - (muscle[d] + base[d] + con[d]))))
        recon = float(np.abs(res - (muscle + base + con)).max())
        if recon > 1e-8:
            disc.append(f'{name}: residual48 != muscle+base+contact (max dev {recon!r})')
        mus_abs = float(np.abs(muscle[:6]).max())
        if mus_abs > 1e-10:
            disc.append(f'{name}: root muscle term max {mus_abs!r} > 1e-10')
        # activation independence: root muscle term is B[:6] @ act, and B[:6] is ~0.
        # Worst-case change over the whole activation box act in [0,1]^90:
        box_bound = float(np.abs(B[:6]).sum(axis=1).max())
        probes = [np.zeros(90), np.ones(90)]
        rng = np.random.default_rng(20261025)
        probes += [rng.random(90) for _ in range(8)]
        dev = float(max(np.abs(B[:6] @ p - muscle[:6]).max() for p in probes))
        if dev > 1e-10:
            disc.append(f'{name}: root muscle term depends on activation (probe dev {dev!r})')
        root_checks.append(dict(pose=name, residual_reconstruction_max_abs_dev=recon,
                                root_muscle_max_abs=mus_abs, root_muscle_le_1e_10=mus_abs <= 1e-10,
                                box_bound_worst_case_root_muscle_change=box_bound,
                                probe_activation_change_max_abs=dev,
                                residual_independent_of_activation=dev <= 1e-10 and box_bound <= 1e-10,
                                root_dofs_with_exact_zero_muscle=[d for d in range(6) if muscle[d] == 0.0]))

    # ------------------------------------- 3. per-row interval reachability (48 rows)
    intervals = []
    interval_summary = []
    for name, _ in POSES:
        z = poses[name]
        B = z['B']
        bb = -(z['base'] + z['J'].T @ z['f0'])
        for setname, Tm in SETS:
            A = B @ Tm
            lo = np.minimum(A, 0.0).sum(axis=1)
            hi = np.maximum(A, 0.0).sum(axis=1)
            bad = []
            reach = []
            for d in range(48):
                ok = bool(lo[d] - TOL_INTERVAL <= bb[d] <= hi[d] + TOL_INTERVAL)
                reach.append(ok)
                if not ok:
                    bad.append(d)
                intervals.append(dict(pose=name, activation_set=setname, dof=d,
                                      target=float(bb[d]), min_possible=float(lo[d]),
                                      max_possible=float(hi[d]), interval_reachable=ok,
                                      gap_outside_interval=float(max(lo[d] - bb[d], bb[d] - hi[d], 0.0))))
            exp = EXPECTED_UNREACHABLE.get((name, setname))
            if exp is not None and bad != exp:
                disc.append(f'{name}/{setname}: unreachable DOF list {bad} != expected {exp}')
            expc = EXPECTED_UNREACHABLE_COUNT.get((name, setname))
            if expc is not None and len(bad) != expc:
                disc.append(f'{name}/{setname}: {len(bad)} unreachable rows != expected {expc}')
            interval_summary.append(dict(pose=name, activation_set=setname,
                                         rows_checked=48, rows_not_interval_reachable=len(bad),
                                         unreachable_dofs=bad, reachable_dofs=[d for d in range(48) if reach[d]]))

    # ------------------------------------------------ 4. 42-row minimax box LP (6 LPs)
    lp = []
    for name, _ in POSES:
        z = poses[name]
        bb42 = (-(z['base'] + z['J'].T @ z['f0']))[6:]
        ts = {}
        for setname, Tm in SETS:
            A = (z['B'] @ Tm)[6:, :]
            n = A.shape[1]
            # min t s.t. |A x - bb42|_inf <= t, x in [0,1], t >= 0
            A_ub = np.block([[A, -np.ones((42, 1))], [-A, -np.ones((42, 1))]])
            b_ub = np.concatenate([bb42, -bb42])
            bounds = [(0.0, 1.0)] * n + [(0.0, None)]
            c = np.concatenate([np.zeros(n), np.ones(1)])
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            lp_calls += 1
            if not res.success:
                disc.append(f'{name}/{setname}: LP failed (status {res.status})')
                t = float('nan')
            else:
                t = float(res.x[-1])
            x = res.x[:n] if res.success else np.full(n, np.nan)
            attained = float(np.abs(A @ x - bb42).max()) if res.success else float('nan')
            ts[setname] = t
            # solver-free certified lower bound: t* >= max_i dist(bb_i, [lo_i,hi_i]) over the 42 rows
            lo = np.minimum(A, 0.0).sum(axis=1)
            hi = np.maximum(A, 0.0).sum(axis=1)
            lower = float(np.max(np.maximum(np.maximum(lo - bb42, bb42 - hi), 0.0)))
            if not (lower - TOL_T <= t <= attained + TOL_T):
                disc.append(f'{name}/{setname}: interval lower bound {lower!r} > LP t {t!r} '
                            f'(LP not optimal / not feasible)')
            exp_t = EXPECTED_T[name]
            if not (abs(t - exp_t) <= TOL_T):
                disc.append(f'{name}/{setname}: minimax t {t!r} != expected {exp_t!r} (tol {TOL_T})')
            if abs(attained - t) > 1e-9:
                disc.append(f'{name}/{setname}: attained max|Ax-bb| {attained!r} != t {t!r}')
            lp.append(dict(pose=name, activation_set=setname, variables=n, joint_rows=42,
                           root_rows_excluded=6, solved=bool(res.success), status=int(res.status),
                           minimax_t=t, expected_t=exp_t, abs_diff_vs_expected=abs(t - exp_t),
                           attained_max_abs_residual=attained,
                           solver_free_lower_bound=lower,
                           lower_bound_minus_t=lower - t,
                           lower_bound_certifies_optimality=bool(abs(lower - t) <= TOL_T),
                           t_is_nonzero=bool(t > 1e-8)))
        same = ts['real66']
        diff = ts['widest90'] - same
        lp.append(dict(pose=name, real66_vs_widest90=dict(identical_exactly=bool(diff == 0.0),
                                                         difference=float(diff),
                                                         abs_difference=abs(diff))))
        if diff != 0.0:
            notes.append(f'{name}: real66 and widest90 minimax t differ by {diff!r} (not exactly equal)')

    # --------------------------------------------------------- 5. contact / COM / weight
    contact = []
    for name, _ in POSES:
        z = poses[name]
        xpos, f0, addr = z['xpos'], z['f0'], z['contact_efc_address']
        com = (mass[:, None] * xpos).sum(0) / mass.sum()
        root = z['qpos'][:3]
        load_z = float(z['foot_load'][:, 2].sum())
        pos = [i for i in range(len(z['contact_geom']))
               if int(addr[i]) >= 0 and f0[int(addr[i])] > 0.0]
        cp = np.array([z['contact_pos'][i] for i in pos])
        allcp = np.asarray(z['contact_pos'])
        inside_x = bool(cp[:, 0].min() <= com[0] <= cp[:, 0].max())
        inside_y = bool(cp[:, 1].min() <= com[1] <= cp[:, 1].max())
        if not (inside_x and inside_y):
            notes.append(f'{name}: COM not inside positive-force contact x/y ranges')
        contact.append(dict(
            pose=name, weight=W, body_mass_sum=float(mass.sum()), gravity=9801.0,
            total_normal_from_legs=load_z, ratio=load_z / W,
            ratio_matches_contact_row2=bool(abs(load_z - (z['J'].T @ f0)[2]) <= 1e-9 * max(1.0, abs(load_z))),
            com_world=com.tolist(), root_world=root.tolist(), com_minus_root=(com - root).tolist(),
            com_x_inside_contact_x=inside_x, com_y_inside_contact_y=inside_y,
            contact_x_range=[float(cp[:, 0].min()), float(cp[:, 0].max())],
            contact_y_range=[float(cp[:, 1].min()), float(cp[:, 1].max())],
            contacts_positive_force=len(pos),
            contacts_all=len(allcp),
            contact_x_range_all=[float(allcp[:, 0].min()), float(allcp[:, 0].max())],
            contact_y_range_all=[float(allcp[:, 1].min()), float(allcp[:, 1].max())],
            com_inside_all_contact_x=bool(allcp[:, 0].min() <= com[0] <= allcp[:, 0].max()),
            com_inside_all_contact_y=bool(allcp[:, 1].min() <= com[1] <= allcp[:, 1].max()),
            static_support_normal_force_ratio_ge_1=bool(load_z / W >= 1.0),
            positive_contact_floor_geoms=sorted({int(meta['floor_geom_ids'][0])}),
            note='root rows have zero activation authority; only gravity+contact can satisfy them'))

    # ------------------------------------------------------------------ 6. S174 consistency
    z173 = poses['s173_best303']
    wide = load(S174_DIR / 'CASE_wide.npz')
    A_ref = z173['B'] @ wide['T']
    b_ref = -(z173['base'] + z173['J'].T @ z173['f0'])
    same_A = bool(np.array_equal(wide['A'], A_ref))
    same_b = bool(np.array_equal(wide['b'], b_ref))
    half_id = bool(np.array_equal(wide['T'], 0.5 * np.eye(90)))
    cert = np.flatnonzero(wide['y']).tolist()
    if not same_A:
        disc.append('S174 CASE_wide A != B@T (not bitwise equal)')
    if not same_b:
        disc.append('S174 CASE_wide b != -(base+J.T@f0) (not bitwise equal)')
    if cert != [4]:
        disc.append(f'S174 certificate nonzero DOFs {cert} != [4]')
    s174 = dict(pose='s173_best303', A_bitwise_equal_B_at_T=same_A, b_bitwise_equal_neg_base_contact=same_b,
                T_is_half_identity=half_id, certificate_nonzero_dofs=cert,
                A_max_abs_dev=float(np.abs(wide['A'] - A_ref).max()),
                b_max_abs_dev=float(np.abs(wide['b'] - b_ref).max()),
                s174_saved_t=float(wide['t']),
                note='S174 wide minimax t INCLUDES the 6 root rows; item-4 t EXCLUDES them')

    # ------------------------------------------------------------------- 7. figure check
    fig_path = OUT_DIR / 'ROOT_WRENCH_AUDIT.png'
    fig = dict(path=str(fig_path), exists=fig_path.exists(),
               bytes=int(fig_path.stat().st_size) if fig_path.exists() else 0,
               png_magic=bool(fig_path.exists() and fig_path.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'),
               sha256=sha(fig_path) if fig_path.exists() else None, modified_by_this_check=False)
    if not (fig['exists'] and fig['bytes'] > 0 and fig['png_magic']):
        disc.append('ROOT_WRENCH_AUDIT.png missing/empty/not a PNG')

    # ------------------------------------------ comparison against producer saved tables
    def close(a, b, tol=TOL_CSV):
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))

    cmp_report = {}
    try:
        import csv
        with (OUT_DIR / 'ACTUATOR_AUTHORITY_BY_DOF.csv').open() as f:
            rows = list(csv.DictReader(f))
        mine = {(r['pose'], r['activation_set'], r['dof']): r['max_abs_A_row'] for r in authority}
        m = 0.0
        for r in rows:
            k = (r['pose'], 'real66', int(r['dof']))
            m = max(m, abs(mine[k] - float(r['authority_real66'])))
            k = (r['pose'], 'widest90', int(r['dof']))
            m = max(m, abs(mine[k] - float(r['authority_widest90'])))
        cmp_report['ACTUATOR_AUTHORITY_BY_DOF.csv'] = dict(rows=len(rows), max_abs_diff=m, ok=bool(m <= TOL_CSV))
        if not m <= TOL_CSV:
            disc.append(f'ACTUATOR_AUTHORITY_BY_DOF.csv mismatch, max abs diff {m!r}')

        with (OUT_DIR / 'ROOT_ROW_DECOMPOSITION.csv').open() as f:
            rows = list(csv.DictReader(f))
        mm = {(r['pose'], r['dof']): r for r in root_rows}
        m = 0.0
        for r in rows:
            q = mm[(r['pose'], int(r['dof']))]
            for k in ('muscle', 'base', 'contact', 'residual'):
                m = max(m, abs(q[k] - float(r[k])))
        cmp_report['ROOT_ROW_DECOMPOSITION.csv'] = dict(rows=len(rows), max_abs_diff=m, ok=bool(m <= TOL_CSV))
        if not m <= TOL_CSV:
            disc.append(f'ROOT_ROW_DECOMPOSITION.csv mismatch, max abs diff {m!r}')

        with (OUT_DIR / 'ROW_INTERVAL_REACHABILITY.csv').open() as f:
            rows = list(csv.DictReader(f))
        ii = {(r['pose'], r['activation_set'], r['dof']): r for r in intervals}
        m = 0.0
        flag_mismatch = []
        for r in rows:
            q = ii[(r['pose'], r['activation_set'], int(r['dof']))]
            for k in ('target', 'min_possible', 'max_possible'):
                m = max(m, abs(q[k] - float(r[k])))
            if str(q['interval_reachable']) != r['interval_reachable']:
                flag_mismatch.append((r['pose'], r['activation_set'], int(r['dof'])))
        cmp_report['ROW_INTERVAL_REACHABILITY.csv'] = dict(rows=len(rows), max_abs_diff=m,
                                                          flag_mismatches=flag_mismatch, ok=bool(m <= TOL_CSV and not flag_mismatch))
        if m > TOL_CSV or flag_mismatch:
            disc.append(f'ROW_INTERVAL_REACHABILITY.csv mismatch: max abs diff {m!r}, '
                        f'flag mismatches {flag_mismatch}')

        saved_sum = json.loads((OUT_DIR / 'ROW_INTERVAL_SUMMARY.json').read_text())
        sm = {(r['pose'], r['activation_set']): r for r in interval_summary}
        bad = []
        for r in saved_sum:
            q = sm[(r['pose'], r['activation_set'])]
            if r['rows_not_interval_reachable'] != q['rows_not_interval_reachable'] \
                    or r['reachable_dofs'] != q['reachable_dofs']:
                bad.append((r['pose'], r['activation_set']))
        cmp_report['ROW_INTERVAL_SUMMARY.json'] = dict(entries=len(saved_sum), mismatches=bad, ok=not bad)
        if bad:
            disc.append(f'ROW_INTERVAL_SUMMARY.json mismatch: {bad}')

        saved_lp = json.loads((OUT_DIR / 'SUBSYSTEM_42ROW_FEASIBILITY.json').read_text())
        dm = {(r['pose'], r['activation_set']): r['minimax_t'] for r in saved_lp['results']}
        m = 0.0
        for r in lp:
            if 'activation_set' in r:
                m = max(m, abs(r['minimax_t'] - dm[(r['pose'], r['activation_set'])]))
        cmp_report['SUBSYSTEM_42ROW_FEASIBILITY.json'] = dict(results=len(saved_lp['results']),
                                                              max_abs_t_diff=m, ok=bool(m <= TOL_T))
        if not m <= TOL_T:
            disc.append(f'SUBSYSTEM_42ROW_FEASIBILITY.json t mismatch, max abs diff {m!r}')

        saved_cw = json.loads((OUT_DIR / 'CONTACT_WRENCH_AND_COM.json').read_text())
        cw = {r['pose']: r for r in contact}
        m = 0.0
        mism = []
        for r in saved_cw:
            q = cw[r['pose']]
            for k in ('weight', 'total_normal_from_legs', 'ratio'):
                m = max(m, abs(q[k] - float(r[k])))
            for k in ('com_world', 'root_world', 'com_minus_root', 'contact_x_range', 'contact_y_range'):
                m = max(m, float(np.abs(np.array(q[k]) - np.array(r[k])).max()))
            for k in ('com_x_inside_contact_x', 'com_y_inside_contact_y'):
                if bool(q[k]) != bool(r[k]):
                    mism.append((r['pose'], k))
            if int(q['contacts_positive_force']) != int(r['contacts_positive_force']):
                mism.append((r['pose'], 'contacts_positive_force'))
        cmp_report['CONTACT_WRENCH_AND_COM.json'] = dict(poses=len(saved_cw), max_abs_diff=m,
                                                         mismatches=mism, ok=bool(m <= TOL_CSV and not mism))
        if m > TOL_CSV or mism:
            disc.append(f'CONTACT_WRENCH_AND_COM.json mismatch: max abs diff {m!r}, {mism}')

        saved_s174 = json.loads((OUT_DIR / 'S174_CONSISTENCY.json').read_text())
        ok = (saved_s174['same_A'] == same_A and saved_s174['same_b'] == same_b
              and saved_s174['wide_T_is_half_identity'] == half_id
              and saved_s174['certificate_dofs'] == cert)
        cmp_report['S174_CONSISTENCY.json'] = dict(saved=saved_s174, recomputed=s174, ok=bool(ok))
        if not ok:
            disc.append('S174_CONSISTENCY.json does not match recomputation')

        saved_ih = json.loads((OUT_DIR / 'INPUT_HASHES.json').read_text())
        mism = [p for p, v in h_before.items() if saved_ih.get(p) != v]
        cmp_report['INPUT_HASHES.json'] = dict(frozen_entries=len(saved_ih), mismatches=mism, ok=not mism)
        if mism:
            disc.append(f'INPUT_HASHES.json differs from current hashes for {mism}')
    except FileNotFoundError as exc:                       # pragma: no cover
        disc.append(f'producer table missing: {exc}')

    h_after = {str(p): sha(p) for p in INPUTS}
    unchanged = h_before == h_after
    if not unchanged:
        disc.append('INPUT HASHES CHANGED during verification')

    out = dict(
        schema='S175_SAVED_INDEPENDENT_CHECK/v1',
        generated_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
        verified=not disc,
        status='VERIFIED_SAVED_ONLY' if not disc else 'FAILED',
        verification_script=str(Path(__file__).resolve()),
        output_path=str(OUT_JSON),
        producer_status=dict(
            audit_root_wrench_py=dict(attempt=1, exit_code=1,
                                      failed_at='its last plotting line (fig.savefig of ROOT_WRENCH_AUDIT.png)',
                                      tables_written_before_failure=['ACTUATOR_AUTHORITY_BY_DOF.csv',
                                                                    'ROOT_ROW_DECOMPOSITION.csv',
                                                                    'SUBSYSTEM_42ROW_FEASIBILITY.json',
                                                                    'CONTACT_WRENCH_AND_COM.json',
                                                                    'S174_CONSISTENCY.json']),
            audit_row_intervals_py=dict(wrote=['ROW_INTERVAL_REACHABILITY.csv', 'ROW_INTERVAL_SUMMARY.json',
                                               'ROOT_WRENCH_AUDIT.png']),
            note='producer failure preserved as historical fact; nothing in the producer output was edited, '
                 'repaired or overwritten by this check'),
        scope=dict(saved_data_only=True, poses=[p for p, _ in POSES],
                   activation_sets=['real66 = MAPPING.T (66 cols)', 'widest90 = 0.5*I90'],
                   engine_import=0, note='no engine, no forward, no mj_step, no neural component; '
                                        'no physics; only 6 small box LPs on saved matrices'),
        calls=dict(model_load=0, forward=0, step=0, neural=0, physics=0, lp_calls=lp_calls),
        total_lp_calls=lp_calls,
        lp_formulation='min t s.t. |A[6:]x + (base+J.T@f0)[6:]|_inf <= t, x in [0,1]^n (HiGHS, default tolerances, '
                       're-solved from scratch; producer tightened feas tol to 1e-9)',
        input_hashes_before=h_before, input_hashes_after=h_after, inputs_unchanged=unchanged,
        checks=dict(
            authority_summary=auth_summary,
            authority_per_dof=authority,
            root_row_checks=root_checks,
            root_rows=root_rows,
            interval_summary=interval_summary,
            intervals=intervals,
            lp=lp,
            contact_com=contact,
            s174=s174,
            figure=fig,
            producer_table_comparison=cmp_report,
        ),
        discrepancies=disc,
        observations=notes,
        caveats=dict(
            interval_reachability='PER-ROW interval reachability is a NECESSARY condition ONLY. '
                                  'Each row is checked as if other rows were free; simultaneous satisfaction of '
                                  'all 48 rows is NOT established by it (that is what the 42-row LP and the S174 '
                                  'certificate address). Sufficiency requires simultaneous satisfaction.',
            conditioned_on='Conclusions are conditioned on these saved poses only '
                           '(s169_best91, s173_best303, original_release_eval1) and on the frozen saved arrays '
                           'they contain. No statement about other poses, future search results, dynamics, '
                           'standing, walking, control or neural attainability is made or implied.',
            units='Numbers are engine units from the saved arrays; physical SI calibration remains NOT_ESTABLISHED.',
            root_rows='Root rows 0-5 have zero actuator authority, so any imbalance there cannot be changed by '
                      'activation at all; the 42-row LP value is the residual that remains even at the LP optimum.'),
    )
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(verified=out['verified'], discrepancies=disc, lp_calls=lp_calls,
                          inputs_unchanged=unchanged, output=str(OUT_JSON)), ensure_ascii=False, indent=2))
    return 0 if out['verified'] else 1


if __name__ == '__main__':
    sys.exit(main())
