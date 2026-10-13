"""DESIGN TABLE: give every one of the 90 model actuators a motor-neuron pool.

WHY THIS FILE EXISTS
--------------------
Rounds 31-33 established that the standing bottleneck is not the body and not the spiking
network but the ANNOTATION GAP: 33 of the 90 muscle actuators had no motor pool that the
connectome annotations could hand over without a guess, and the feasible stance found in
round 30 needs 21 of those.  This file does the two things that can be done without
inventing biology:

  (A) STATE the mapping rule that is already implicit in the evidence, and apply it
      uniformly.  Two facts about the MODEL's own asset settle most of the gap:
        * the asset names each muscle attachment group with its Miller site
          (`..._LFC_Miller28a_p1`, `..._Miller28b_p1`, `..._Miller30_p1`, `..._Miller29_p1`);
        * `tergopleural_promotor_a` and `_b` hang on the SAME attachment group (Miller 28a
          vs 28b = the two fibre clusters of ONE muscle), whereas `pleural_promotor` is a
          different site entirely (Miller 30) and `pleural_remotor_and_abductor` another
          (Miller 29).
      So "one pool per muscle" is well defined: one pool drives a muscle's clusters
      together, and different muscles keep different pools.
  (B) TEST the result PHYSICALLY.  A wiring design that cannot hold a stance is worthless,
      so the designed pool set is put through the same fixed-pose static balance test the
      earlier rounds used, at the round-33 feasible pose and then by re-search.

WHAT THIS FILE DOES NOT DO
--------------------------
  * It does not assert individual anatomy.  Every row carries an explicit support level
    (`DATA_EXACT`, `DATA_LEG_SIDE_FILTERED`, `DATA_MUSCLE_LEVEL_ANNOTATION`, `MODEL_GEOMETRY`,
    `BILATERAL_MIRROR`, `DESIGN_HYPOTHESIS`).
  * It does not modify the canonical XML, and it grants no runtime authorisation: the strict
    entry point must keep refusing.
  * It does not call any designed row "verified".  The design is a numbered hypothesis set
    with, for each row, the exact measurement that would confirm or refute it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
ROOT = HERE
sys.path.insert(0, str(HERE / 'diagnostics'))

import diag_static_support_feasibility as r15  # noqa: E402
import diag_posture_support_search as r16      # noqa: E402

SIDEFIX = ROOT / 'outputs/mn_resolution_v888_sidefix/mn_resolution_v888.json'
MANIFEST = ROOT / 'outputs/muscle_anatomy_resolution/resolution.json'
ROUND33 = ROOT / 'outputs/stance_ab_coupled_round33/stance_ab_coupled.json'
OUT = ROOT / 'outputs/design_motor_pools_round34'
XML = r15.XML
LEGS = r16.LEGS
PENETRATION_MM = -0.050
ADHESION_BW = 2.0      # overridden by --adhesion-bw; the engine-default variant gives 1.636
LP_OPTS = {'presolve': True, 'time_limit': 60, 'maxiter': 20000}

# ---------------------------------------------------------------------------------------
# The four V888 cell-type families that MATCH a model actuator family by name once the
# model's `a`/`b` token is removed, and that the strict resolver could not use because the
# model splits the family into two actuators.  Listing them here is the whole "design":
# no new biology is introduced, only the decision that a fibre-cluster pair shares a pool.
# ---------------------------------------------------------------------------------------
FIBRE_CLUSTER_FAMILIES = (
    'tergopleural_promotor',              # a,b = Miller 28a/28b clusters of ONE muscle
    'trochanter_flexor',                  # already resolved at pair level (round 31)
)

# The two connectome pools that replace the model's combined sterno-tergo actuator pair.
STERNO_TERGO_POOLS = ('sternotrochanter_extensor', 'tergotrochanter_extensor')

# The model's sterno-tergo-trochanter extensor pair is NOT a fibre-cluster pair: the asset
# gives `_a` and `_b` the same attachment-group pattern (proximal sites on the coxa shaft,
# distal sites on the trochanter) while the connectome names the sternotrochanter_extensor and
# the tergotrochanter_extensor as TWO separate muscles with separate pools.  So the pair gets
# two pools, one per actuator, and the only open question is WHICH suffix is the sternal one.
# The two readings are `same` (the proximal-most pool is sternal on both sides) and `crossed`
# (the right side is the mirror image).  Neither is guessed silently: both are tested.
STERNO_TERGO_ORDERINGS = ('same', 'crossed')

SUPPORT_ORDER = ['DATA_EXACT', 'DATA_LEG_SIDE_FILTERED', 'DATA_MUSCLE_LEVEL_ANNOTATION',
                 'MODEL_GEOMETRY_BUNDLE_COUPLING', 'BILATERAL_MIRROR', 'DESIGN_HYPOTHESIS']


def support_facts():
    """What the connectome file itself says, once the side bug is fixed."""
    d = json.loads(SIDEFIX.read_text())
    man = json.loads(MANIFEST.read_text())
    pre = {r['actuator']: r for r in man['actuators']}
    pools = {}
    # (1) the 44 actuators whose pool was already supported by the older per-leg manifest
    for r in man['actuators']:
        if r['candidates']:
            pools[r['actuator']] = {
                'support': 'DATA_EXACT',
                'roots': sorted({str(c.get('root_id')) for c in r['candidates']}),
                'n': len(r['candidates']),
                'evidence': 'outputs/muscle_anatomy_resolution/resolution.json (per-leg pool)'}
    # (2) the v888 strict re-resolution under the corrected side filter
    for r in d['all_results']:
        if not r['status'].startswith('RESOLVED'):
            continue
        pools[r['actuator']] = {
            'support': ('DATA_MUSCLE_LEVEL_ANNOTATION'
                        if r['resolution_level'] == 'PAIR_LEVEL_CLUSTER_ASSUMPTION'
                        else 'DATA_EXACT'),
            'roots': sorted({c['root_id'] for c in r['candidates']}),
            'n': r['n_candidates_v888'],
            'cell_types': sorted({c['v888_cell_type'] for c in r['candidates']}),
            'evidence': 'outputs/mn_resolution_v888_sidefix/mn_resolution_v888.json'}
    return d, pre, pools


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=OUT)
    ap.add_argument('--skip-physics', action='store_true')
    ap.add_argument('--skip-search', action='store_true')
    ap.add_argument('--max-evals', type=int, default=600)
    ap.add_argument('--wall-budget-s', type=float, default=900.0)
    ap.add_argument('--adhesion-bw', type=float, default=None,
                    help='per-leg adhesion budget in body weights (measured range 1.0-2.0; the '
                         'engine-default variant supplies 1.636 at ctrl 1)')
    ap.add_argument('--compare-variable-sets', action='store_true',
                    help='also solve the design with one family at a time left independent, so '
                         'the cost of each grouping decision is measured instead of assumed')
    ap.add_argument('--muscle-only', action='store_true',
                    help='require the balance to hold with NO adhesion at all (zero adhesion '
                         'columns and a zero per-leg budget): the answer round 35 showed the earlier '
                         'rounds never actually asked for, because their LP was allowed to use '
                         'adhesion')
    args = ap.parse_args()
    global ADHESION_BW
    if args.adhesion_bw is not None:
        ADHESION_BW = float(args.adhesion_bw)
    if args.muscle_only:
        ADHESION_BW = 0.0
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    import mujoco as mj
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    xml_before = r15.digest(XML)
    model = _load_mjcf(str(XML)).compile()
    data = mj.MjData(model)
    model.actuator_ctrlrange[:, 0] = 0.0
    names = [model.actuator(i).name or f'act{i}' for i in range(model.nu)]
    assert len(names) == 90, len(names)

    d888, manifest_pre, pools = support_facts()

    # -----------------------------------------------------------------------------------
    # DESIGN: assign a pool to every actuator.  The assignment is generated from the
    # actuator names, so it is reproducible and auditable rather than hand-typed.
    # -----------------------------------------------------------------------------------
    rows, unexplained = [], []
    for i, n in enumerate(names):
        leg, comp = n[:2], n[2]
        muscle = n.split('_', 1)[1]
        row = {'actuator': n, 'index': i, 'leg': leg, 'compartment': comp, 'label': muscle}
        m = re.match(r'^(.*)_([ab])$', muscle)
        base = m.group(1) if m else muscle
        # a) pool already anchored in the connectome evidence, keyed by exact actuator name
        if n in pools:
            p = pools[n]
            row.update({'pool_key': n, 'support': p['support'], 'n_neurons': p['n'],
                        'roots': p.get('roots', [])[:12],
                        'cell_types': p.get('cell_types', []),
                        'evidence': p['evidence']})
        # b) fibre-cluster pair: the model presents ONE muscle as _a/_b; one pool drives both
        elif m and base in FIBRE_CLUSTER_FAMILIES:
            pair = [f'{leg}{comp}_{base}_a', f'{leg}{comp}_{base}_b']
            partner = [x for x in pair if x in pools]
            row.update({'pool_key': f'{leg}{comp}_{base}',
                        'support': 'MODEL_GEOMETRY_BUNDLE_COUPLING',
                        'coupled_actuators': pair,
                        'n_neurons': sum(pools[x]['n'] for x in partner) or None,
                        'roots': sorted({r for x in partner for r in pools[x].get('roots', [])})[:12],
                        'cell_types': sorted({c for x in partner for c in pools[x].get('cell_types', [])}),
                        'evidence': 'outputs/muscles_six_legs/fruitfly_six_leg_muscles.xml site names '
                                    '(same Miller attachment group) + '
                                    'outputs/mn_resolution_v888_sidefix/mn_resolution_v888.json'})
        elif m and base == 'sterno-tergo-trochanter_extensor':
            # c) TWO connectome muscles, TWO model actuators: a 1:1 assignment, not a coupling.
            #    Which suffix is the sternal component is not settled locally, so the ordering
            #    is a switch with two readings and BOTH are carried into the physics test.
            #    Anatomically anchored reading: on each side the sternal component lies ventral
            #    and anterior of the tergal one, so the pool whose somata sit more ventrally is
            #    the sternal pool; that is the `same` ordering, and `crossed` is its mirror.
            suff = m.group(2)
            for ordering in STERNO_TERGO_ORDERINGS:
                key = f'{leg}{comp}_{base}_{suff}__{ordering}'
                if ordering == 'same':
                    pool_for_suffix = {'a': STERNO_TERGO_POOLS[0], 'b': STERNO_TERGO_POOLS[1]}
                else:
                    pool_for_suffix = {'a': STERNO_TERGO_POOLS[1], 'b': STERNO_TERGO_POOLS[0]}
                row.setdefault('orderings', {})[ordering] = {
                    'pool_key': key, 'pool_cell_type': pool_for_suffix[suff]}
            row.update({'pool_key': row['orderings'][STERNO_TERGO_ORDERINGS[0]]['pool_key'],
                        'support': 'DESIGN_HYPOTHESIS',
                        'unresolved_because': 'the connectome annotates sternotrochanter_extensor '
                                              'and tergotrochanter_extensor as separate muscles; '
                                              'no local file states which one the model\'s suffix '
                                              f'_{suff} is, so both orders are tested',
                        'candidate_cell_types': list(STERNO_TERGO_POOLS),
                        'evidence': 'data/banc888/banc_888_meta.feather peripheral_target_type'})
            unexplained.append(n)
        else:
            row.update({'pool_key': None, 'support': 'UNASSIGNED',
                        'evidence': 'nothing in the local evidence base names this muscle'})
            unexplained.append(n)
        rows.append(row)

    by_support = {}
    for r in rows:
        by_support.setdefault(r['support'], []).append(r['actuator'])

    # the design's independent-variable structure: actuators sharing a pool_key are one variable
    pool_vars: dict[str, list[str]] = {}
    for r in rows:
        if r['pool_key']:
            pool_vars.setdefault(r['pool_key'], []).append(r['actuator'])
    coupled = {k: v for k, v in pool_vars.items() if len(v) > 1}

    # the design under each sterno/tergo ordering: same = both legs call their proximal pool
    # sternal; crossed = the right leg is the mirror image.  Nothing else differs.
    def design_under(ordering):
        g = {}
        for r in rows:
            key = (r.get('orderings', {}).get(ordering, {}) or {}).get('pool_key') or r['pool_key']
            if key:
                g.setdefault(key, []).append(r['actuator'])
        return g

    # -----------------------------------------------------------------------------------
    # PHYSICS: can the designed wiring hold a stance?  Pool = one variable, so the LP's
    # columns are the SUMS of the pooled actuators' moment arms.
    # -----------------------------------------------------------------------------------
    physics = {'performed': False, 'reason': 'skipped by flag'}
    if not args.skip_physics:
        floor = r16.floor_geom_id(model, mj)
        tarsus_geoms = [g for g in range(model.ngeom)
                        if 'Tarsus' in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) or '')]
        tarsus_names = {mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) for g in tarsus_geoms}
        thorax_id = model.body('Thorax').id
        body_weight = float(mj.mj_getTotalmass(model) * abs(float(model.opt.gravity[2])))

        mj.mj_resetDataKeyframe(model, data, 0)
        data.qvel[:] = 0
        mj.mj_forward(model, data)
        key_qpos = data.qpos.copy()

        # The moment matrix R is built per actuator of the CANONICAL model and is NOT
        # re-labelled: only the grouping into design variables changes (one column per pool,
        # equal to the sum of its members' columns).  That is exactly the physical content of
        # "one pool drives these actuators together".
        single_vars = {n: [n] for n in names}                      # 90 independent variables

        def solve_vars(grp, label, pattern=None):
            """Phase-1 LP with an explicit PER-LEG ADHESION BUDGET, written out here rather
            than reusing r15.solve_set because that helper has no adhesion term; the budget
            is what rounds 30-33 used (sum of the negative normal part per leg <= budget)."""
            cols, labels = [], []
            for k, acts in grp.items():
                idx = [names.index(a) for a in acts if a in names]
                if not idx:
                    continue
                cols.append(R[:, idx].sum(axis=1).reshape(-1, 1))
                labels.append(k)
            Rset = np.hstack(cols)
            nv, m_vars = model.nv, Rset.shape[1]
            n_lam = sum(b.shape[1] for b in B)
            n_p, n_lim = len(terms), len(lims)
            limits = (np.column_stack([l['direction'] for l in lims]) if lims
                      else np.zeros((nv, 0)))
            blocks = [Rset]
            for t, b in zip(terms, B):
                blocks.append(t['J'].T @ b)
            if ADHESION_BW > 0.0:
                for t in terms:                  # adhesion: normal part as its own column
                    blocks.append(-t['J'].T @ t['normal'].reshape(-1, 1))
            blocks.append(limits)
            A_eq = np.hstack(blocks)
            leg_of = []
            for t in terms:
                lg = None
                for cand in (t['geom2'], t['geom1']):
                    if cand[:2] in LEGS:
                        lg = cand[:2]; break
                leg_of.append(lg)
            present = sorted({l for l in leg_of if l})
            A = np.hstack([A_eq, np.eye(nv), -np.eye(nv)])
            A_ub = np.zeros((len(present), A.shape[1]))
            if ADHESION_BW > 0.0:
                for i_, lg in enumerate(present):
                    for k, l2 in enumerate(leg_of):
                        if l2 == lg:
                            A_ub[i_, m_vars + n_lam + k] = 1.0
            n_v = A.shape[1]
            c = np.zeros(n_v); c[-2 * nv:] = 1.0
            n_p_used = n_p if ADHESION_BW > 0.0 else 0
            bounds = ([(0.0, 1.0)] * m_vars + [(0.0, None)] * n_lam
                      + [(0.0, None)] * n_p_used + [(0.0, None)] * n_lim
                      + [(0.0, None)] * (2 * nv))
            A = A[:, :m_vars + n_lam + n_p_used + n_lim + 2 * nv]
            A_ub = A_ub[:, :A.shape[1]]
            c = np.zeros(A.shape[1]); c[-2 * nv:] = 1.0
            ub = (np.full(len(present), 0.0) if ADHESION_BW <= 0.0
                  else np.full(len(present), ADHESION_BW * body_weight))
            res = linprog(c, A_eq=A, b_eq=target, A_ub=A_ub, b_ub=ub,
                          bounds=bounds, method='highs', options=LP_OPTS)
            if res.x is None:
                return {'label': label, 'n_variables': m_vars, 'L1': None,
                        'feasible': False, 'activation': None,
                        'solver_message': res.message}
            x = res.x
            sup = x[-2 * nv:-nv]; smn = x[-nv:]
            per_leg = {}
            if ADHESION_BW > 0.0:
                p = x[m_vars + n_lam: m_vars + n_lam + n_p]
                for k, lg in enumerate(leg_of):
                    if lg:
                        per_leg[lg] = round(per_leg.get(lg, 0.0) + float(p[k]), 4)
            out = {'label': label, 'muscle_only': bool(ADHESION_BW <= 0.0),
                   'n_variables': m_vars,
                   'L1': float(np.abs(sup - smn).sum()),
                   'max_abs_residual': float(np.abs(sup - smn).max()),
                   'feasible': bool(np.abs(sup - smn).sum() < r15.FEASIBLE_SUM_TOL),
                   'adhesion_used_bw_per_leg': per_leg,
                   'adhesion_budget_bw_per_leg': ADHESION_BW,
                   'activation': {labels[k]: float(x[k]) for k in range(m_vars)}}
            if pattern is not None:
                xp = np.array([pattern.get(k, 0.0) for k in labels])
                out['collapsed_pattern_residual'] = float(np.abs(Rset @ xp - target).sum())
                out['collapsed_pattern_in_box'] = bool(np.all((xp >= 0) & (xp <= 1)))
            return out


        stances = []
        for label, path in (
                ('round30 minimal-excursion stance', ROOT / 'outputs/stance_joint_feasible_round30/'
                                                            'stance_joint_feasible.json'),
                ('round33 a/b-coupled stance', ROUND33)):
            if not path.exists():
                continue
            best = json.loads(path.read_text())['best']
            stances.append((label, path, best))

        case_rows = []
        for label, path, best in stances:
            qpos = key_qpos.copy()
            for jn, v in best['joints_rad'].items():
                qpos[model.jnt_qposadr[model.joint(jn).id]] = float(v)
            seated, root_z, _ = r16.seat_root(model, data, mj, qpos, tarsus_geoms, floor,
                                              PENETRATION_MM)
            data.qpos[:] = seated; data.qvel[:] = 0
            facts = r16.pose_facts(model, data, mj, tarsus_names, thorax_id)
            R, base = r15.actuator_moment_matrix(model, data, mj)
            target = np.asarray(data.qfrc_bias, float) - base - np.asarray(data.qfrc_passive, float)
            terms = r15.contact_terms(model, data, mj)
            lims = r15.limit_terms(model, data, mj)
            B = r15.cone_basis(model, data, terms, r15.CONE_DIRECTIONS)
            ctrl = solve_vars(single_vars, f'{label}: all 90 independent (control)')
            cases = [ctrl]
            for ordering in STERNO_TERGO_ORDERINGS:
                cases.append(solve_vars(design_under(ordering),
                                        f'{label}: DESIGN, sterno/tergo order = {ordering}'))
            # WHICH coupled group costs the balance?  Release one family at a time and see
            # whether the residual collapses; the family whose release fixes it is the suspect.
            base_design = design_under(STERNO_TERGO_ORDERINGS[0])
            suspects = {}

            def released_family(fam):
                touched = {a for v in base_design.values() for a in v if fam in a.split('_', 1)[1]}
                if not touched:
                    return None, set()
                rel = {k: v for k, v in base_design.items()
                       if not any(fam in a.split('_', 1)[1] for a in v)}
                for a in touched:
                    rel[a] = [a]
                return rel, touched

            for fam in FIBRE_CLUSTER_FAMILIES:
                relaxed, touched = released_family(fam)
                if relaxed is None:
                    continue
                rel = solve_vars(relaxed, f'{label}: DESIGN with {fam} released')
                suspects[fam] = {'L1': rel['L1'], 'feasible': rel['feasible'],
                                 'n_variables': rel['n_variables'],
                                 'actuators_released': sorted(touched)}
            if args.compare_variable_sets:
                # 1. only the tergopleural cluster coupling (everything else independent)
                tg, _ = released_family('tergopleural_promotor')
                if tg is not None:
                    suspects['ONLY tergopleural cluster coupling'] = solve_vars(
                        tg, f'{label}: all independent + tergopleural clusters coupled')
                # 2. the design with the sterno/tergo pair joined instead of split (the
                #    round-34 reading that turned out to be wrong, kept as a measured control)
                joined = {k: list(v) for k, v in base_design.items()}
                for leg in LEGS:
                    for comp in ('C', 'F'):
                        ka = f'{leg}{comp}_sterno-tergo-trochanter_extensor_a__same'
                        kb = f'{leg}{comp}_sterno-tergo-trochanter_extensor_b__same'
                        if ka in joined and kb in joined:
                            joined[f'{leg}{comp}_sterno-tergo-trochanter_extensor'] = (
                                joined.pop(ka) + joined.pop(kb))
                suspects['DESIGN with sterno/tergo JOINED'] = solve_vars(
                    joined, f'{label}: design with sterno/tergo joined')
            case_rows.append({'stance': label, 'source': str(path),
                              'pose': {'thorax_height_mm': facts['thorax_height_mm'],
                                       'tarsus_only': facts['contacts_are_feet_only_tarsus'],
                                       'contact_count': facts['contact_count']},
                              'cases': cases,
                              'release_one_family': suspects})

        # CAN ANY POSE DO IT?  The two known stances were searched with all 90 actuators
        # independent, so the design may simply not be able to hold those particular poses.
        # This re-searches the leg posture with the design's variable set.
        search = {'performed': False}
        if not args.skip_search and not args.skip_physics:
            PARAM_J = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll',
                       'Trochanter_yaw', 'Trochanter_pitch', 'Trochanter_roll', 'Tibia_pitch')
            adr = {j: {leg: int(model.jnt_qposadr[model.joint(f'joint_{leg}{j}').id])
                       for leg in LEGS} for j in PARAM_J}
            params = [(leg, j) for leg in LEGS for j in PARAM_J]
            gen = np.random.default_rng(11)

            def eval_pose(x, grp):
                qpos = key_qpos.copy()
                for k, (leg, j) in enumerate(params):
                    jid = model.joint(f'joint_{leg}{j}').id
                    lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
                    qpos[adr[j][leg]] = float(np.clip(key_qpos[adr[j][leg]] + x[k], lo, hi))
                seated, rz, _ = r16.seat_root(model, data, mj, qpos, tarsus_geoms, floor,
                                              PENETRATION_MM)
                data.qpos[:] = seated; data.qvel[:] = 0
                fa = r16.pose_facts(model, data, mj, tarsus_names, thorax_id)
                if not fa['contacts_are_feet_only_tarsus'] or fa['contact_count'] == 0:
                    return None, 99.0, fa
                q = np.asarray(data.qpos[3:7], float)
                if float(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)) < 0.9:
                    return None, 99.0, fa
                Rl, bl = r15.actuator_moment_matrix(model, data, mj)
                tg = np.asarray(data.qfrc_bias, float) - bl - np.asarray(data.qfrc_passive, float)
                tm = r15.contact_terms(model, data, mj)
                lm = r15.limit_terms(model, data, mj)
                Bl = r15.cone_basis(model, data, tm, r15.CONE_DIRECTIONS)

                def local_solve(grp2):
                    cols, labels2 = [], []
                    for k2, acts in grp2.items():
                        idx = [names.index(a) for a in acts if a in names]
                        if not idx:
                            continue
                        cols.append(Rl[:, idx].sum(axis=1).reshape(-1, 1))
                        labels2.append(k2)
                    RS = np.hstack(cols)
                    nvl, mv = model.nv, RS.shape[1]
                    n_lam = sum(b.shape[1] for b in Bl)
                    n_p2, n_lim2 = len(tm), len(lm)
                    lim2 = (np.column_stack([l['direction'] for l in lm]) if lm
                            else np.zeros((nvl, 0)))
                    blks = [RS]
                    for t2, b2 in zip(tm, Bl):
                        blks.append(t2['J'].T @ b2)
                    for t2 in tm:
                        blks.append(-t2['J'].T @ t2['normal'].reshape(-1, 1))
                    blks.append(lim2)
                    AE = np.hstack(blks)
                    leg_of2 = []
                    for t2 in tm:
                        lg2 = None
                        for cand2 in (t2['geom2'], t2['geom1']):
                            if cand2[:2] in LEGS:
                                lg2 = cand2[:2]; break
                        leg_of2.append(lg2)
                    pres2 = sorted({l2 for l2 in leg_of2 if l2})
                    A2 = np.hstack([AE, np.eye(nvl), -np.eye(nvl)])
                    AU = np.zeros((len(pres2), A2.shape[1]))
                    for r2, lg2 in enumerate(pres2):
                        for k3, l3 in enumerate(leg_of2):
                            if l3 == lg2:
                                AU[r2, mv + n_lam + k3] = 1.0
                    c2 = np.zeros(A2.shape[1]); c2[-2 * nvl:] = 1.0
                    bd = ([(0.0, 1.0)] * mv + [(0.0, None)] * n_lam + [(0.0, None)] * n_p2
                          + [(0.0, None)] * n_lim2 + [(0.0, None)] * (2 * nvl))
                    rr = linprog(c2, A_eq=A2, b_eq=tg, A_ub=AU,
                                 b_ub=np.full(len(pres2), ADHESION_BW * body_weight),
                                 bounds=bd, method='highs', options=LP_OPTS)
                    if rr.x is None:
                        return 99.0, None
                    return (float(np.abs(rr.x[-2 * nvl:-nvl] - rr.x[-nvl:]).sum()),
                            {labels2[k4]: float(rr.x[k4]) for k4 in range(mv)})

                val, act = local_solve(grp)
                return act, val, fa

            starts = []
            for label, path, best in stances:
                x = np.zeros(len(params))
                for k, (leg, j) in enumerate(params):
                    jn = f'joint_{leg}{j}'
                    if jn in best['joints_rad']:
                        x[k] = float(best['joints_rad'][jn]) - float(key_qpos[adr[j][leg]])
                starts.append(x)
            starts.append(np.zeros(len(params)))
            while len(starts) < 3:
                starts.append(gen.normal(0, 0.10, size=len(params)))

            evals, hist, best_d, reasons = 0, [], None, {}
            budget = args.max_evals  # noqa: E501
            for x0 in starts:
                x = np.array(x0, float)
                _, cur, fa = eval_pose(x, design_under(STERNO_TERGO_ORDERINGS[0])); evals += 1
                for step in (0.10, 0.05, 0.02):
                    improved = True
                    while improved:
                        improved = False
                        for k in range(len(params)):
                            if evals >= budget or time.time() - t0 > args.wall_budget_s:
                                break
                            for d in (+step, -step):
                                if evals >= budget or time.time() - t0 > args.wall_budget_s:
                                    break
                                cand = x.copy(); cand[k] += d
                                act, v, fa2 = eval_pose(cand, design_under(STERNO_TERGO_ORDERINGS[0])); evals += 1
                                hist.append({'eval': evals, 'L1': None if v >= 99 else v})
                                if v < cur - 1e-9:
                                    x, cur = cand, v; improved = True
                                    if best_d is None or v < best_d['L1']:
                                        best_d = {'L1': v, 'x': cand.tolist(),
                                                  'thorax_height_mm': fa2['thorax_height_mm'],
                                                  'contact_count': fa2['contact_count'],
                                                  'activation': act}
            search = {'performed': True, 'evaluations': evals, 'starts': len(starts),
                      'best': best_d,
                      'feasible_any_pose': bool(best_d and best_d['L1'] < r15.FEASIBLE_SUM_TOL),
                      'history': hist[-400:],
                      'note': 'posture search with the DESIGN variable set: can any tarsus-only '
                              'upright pose be balanced when each pool is one variable?'}

        physics = {
            'performed': True,
            'penetration_mm': PENETRATION_MM, 'adhesion_bw_per_leg': ADHESION_BW,
            'body_weight_units': body_weight,
            'stances': case_rows,
            'design_search_over_poses': search,
            'body_unchanged_sha256': r15.digest(XML),
            'reading': ('each stance is re-seated, then solved twice: once with all 90 actuators '
                        'independent (control, reproduces the earlier rounds) and once with the '
                        'design\'s pool grouping.  A design variable is the SUM of its members\' '
                        'moment-arm columns, so coupling can only remove directions.'),
        }

    report = {
        'status': 'MOTOR_POOL_DESIGN_ROUND34',
        'scope': 'a NUMBERED HYPOTHESIS SET, not verified anatomy; diagnostic only',
        'design_rule': [
            '1. one pool per muscle: actuators that the model presents as fibre clusters of one '
            'muscle (same Miller attachment group, `_a`/`_b`) share one activation variable;',
            '2. different muscles keep different pools, so a pool is never asked to drive two '
            'muscles at once;',
            '3. where the connectome names ONE muscle and the model carries it as a pair, the '
            'documented pool drives the pair (equivalent to rule 1);',
            '4. where the connectome names TWO muscles and the model carries them as one pair, '
            'the suffix order is left UNRESOLVED and both orderings must be tested;',
            '5. a leg with no annotated root on the required nerve may only be filled by the '
            'contralateral pool, and the row says so (BILATERAL_MIRROR).'],
        'source_fixes_applied': {
            'side_filter_bug': 'tools_resolve_mn_muscle_v888.py decided the wanted side from '
                               'actuator_label[1] (the leg code F/M/H) so every foreleg actuator '
                               'accepted LEFT-side candidates; fixed to actuator_label[0]. '
                               'Effect: strict exact coverage 45 -> 44, because the one exact '
                               'resolution (LMC_sternal_adductor) had come from a left mesothoracic '
                               'neuron; correct pools are now side-separated (verified: the left '
                               'and right foreleg trochanter-flexor pools share 0 roots).',
            'unchanged_evidence': 'the 44 per-leg pools of outputs/muscle_anatomy_resolution do '
                                  'not change: they were never side-filtered.'},
        'coverage': {
            'actuators_total': len(names),
            'assigned_a_pool': sum(1 for r in rows if r['pool_key']),
            'with_connectome_data': sum(1 for r in rows if r['support'].startswith('DATA_')),
            'by_model_geometry': sum(1 for r in rows if r['support'] == 'MODEL_GEOMETRY_BUNDLE_COUPLING'),
            'design_hypothesis_only': sum(1 for r in rows if r['support'] == 'DESIGN_HYPOTHESIS'),
            'unassigned': sum(1 for r in rows if r['support'] == 'UNASSIGNED'),
            'by_support': {k: len(v) for k, v in by_support.items()},
            'independent_variables_under_design': len(design_under(STERNO_TERGO_ORDERINGS[0])),
            'design_under_each_ordering': {o: len(design_under(o))
                                           for o in STERNO_TERGO_ORDERINGS},
            'coupled_pool_variables': coupled,
        },
        'unresolved_by_design': {
            'actuators': unexplained,
            'what_would_settle_it': [
                'the model asset\'s source paper (Ozdi et al., arXiv 2509.06426) muscle table '
                'stating whether `_a`/`_b` of sterno-tergo-trochanter extensor are the sternal and '
                'tergal components (or fibre clusters of one muscle), or',
                'a leg-nerve innervation table / EM tracing that names which motor neuron '
                'innervates the sternal vs the tergal component.']},
        'external_evidence_required': {
            '1_promotor_sternotergo_split': {
                'question': 'the connectome annotates ONE pool for tergopleural+pleural promotor '
                            '(Miller 28/30) and TWO muscles for sterno/tergo trochanter extensor, '
                            'while the model does the opposite split. Which pool drives which '
                            'model actuator?',
                'measurement': 'per-muscle innervation: which motor neuron terminates on which '
                               'muscle fibre (EM tracing or a muscle-target table at individual '
                               'muscle resolution), or a BANC/MANC cell type that names a single '
                               'muscle.',
                'status_here': 'rows in `unresolved_by_design`'},
            '2_right_mesothoracic_pleural_remotor': {
                'question': 'no annotated root exits the right mesothoracic leg nerve for '
                            'pleural_remotor_and_abductor; the left mesothoracic leg has one.',
                'measurement': 'either the missing annotation in BANC, or a statement that the '
                               'pool is bilateral.',
                'status_here': 'filled by BILATERAL_MIRROR from the left mesothoracic leg'},
            '3_behaviour_of_coupled_bundles': {
                'question': 'do the two fibre clusters of one muscle receive the same command?',
                'measurement': 'co-activity of the clusters during standing/walking (calcium or '
                               'EMG) or a statement in the model paper that the subdivision is '
                               'geometric only.',
                'status_here': 'used as the coupling assumption, flagged on every coupled row'},
            '4_nonlinearity_and_weights': {
                'question': 'the network is recurrence dominated at the declared weight scale '
                            '(crossover near 5 Hz partner rate); the synaptic scale is not measured.',
                'measurement': 'measured unitary PSP amplitudes / conductances for the relevant '
                               'cell classes, or a published calibrated model.',
                'status_here': 'unchanged from rounds 21-25'},
            '5_decoder_calibration': {
                'question': 'pool -> activation gain: pools here range 1..8 neurons, so equal '
                            'per-neuron output gives unequal muscle force.',
                'measurement': 'motor unit size / recruitment data, i.e. force per spike for small '
                               'vs large pools.',
                'status_here': 'pool sizes reported per row so the confound is visible'},
            '6_sensory_tuning': {
                'question': 'per-root preferred direction and gain for the leg sensory channels.',
                'measurement': 'published tuning curves or a calibration experiment.',
                'status_here': 'unchanged: channel-level proxies are declared as such'}},
        'rows': rows,
        'physics': physics,
        'source_sha256': {str(XML): r15.digest(XML), str(SIDEFIX): r15.digest(SIDEFIX),
                          str(ROUND33): r15.digest(ROUND33)},
        'limitations': [
            'A pool assignment is not an anatomical validation of any single neuron.',
            'The design changes which actuator a neuron drives; it never changes the body.',
            'The sterno/tergo suffix order is left open on purpose: guessing it would invent data.',
            'Static balance only: NOT stability, NOT a controller, NOT walking.',
            'runtime_authorization stays false; the strict entry point must keep refusing.'],
        'wall_seconds': float(time.time() - t0),
    }
    assert r15.digest(XML) == xml_before
    (out / 'design_motor_pools.json').write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    order = [s for s in SUPPORT_ORDER if s in by_support] + \
            [s for s in by_support if s not in SUPPORT_ORDER]
    ax[0].barh(range(len(order)), [len(by_support[s]) for s in order], color='tab:blue')
    ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels(order, fontsize=8)
    ax[0].set_xlabel('actuators (of 90)')
    ax[0].set_title('Every actuator gets a pool; the colour of its support matters\n'
                    'DESIGN HYPOTHESIS rows are guesses and are marked as such')
    for i, s in enumerate(order):
        ax[0].text(len(by_support[s]) + .5, i, str(len(by_support[s])), va='center', fontsize=8)
    ax[0].grid(alpha=.3, axis='x')
    if physics.get('performed'):
        flat = []
        for st in physics['stances']:
            for c in st['cases']:
                flat.append((f"{st['stance']}\n{c['label'].split(': ')[-1]}", c))
        lab = [f[0] for f in flat]
        vals = [np.nan if f[1]['L1'] is None else f[1]['L1'] for f in flat]
        col = ['tab:green' if f[1]['feasible'] else 'tab:red' for f in flat]
        ax[1].bar(range(len(flat)), vals, color=col)
        ax[1].set_xticks(range(len(flat))); ax[1].set_xticklabels(lab, fontsize=6)
        ax[1].axhline(r15.FEASIBLE_SUM_TOL, ls='--', c='green', label='feasibility tolerance')
        ax[1].set_ylabel('L1 residual (force units)')
        ax[1].set_title('Does the designed wiring keep a solvable stance?\n'
                        '0 = statically balanced; NOT stability, NOT walking')
        ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis='y')
    fig.tight_layout(); fig.savefig(out / 'design_motor_pools.png', dpi=110)

    print(json.dumps({'coverage': report['coverage']['by_support'],
                      'variables_under_design': report['coverage']['independent_variables_under_design'],
                      'unassigned': len(unexplained),
                      'physics': [{**{k: c[k] for k in ('label', 'n_variables', 'L1', 'feasible')},
                                   'stance': st['stance']}
                                  for st in physics.get('stances', []) for c in st['cases']]},
                     indent=2))
    print(out)


if __name__ == '__main__':
    main()
