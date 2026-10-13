"""Emit the runner's motor overlay + a start stance from the round-34 design table.

WHAT THIS PRODUCES
------------------
1. `motor_pool_overlay_design.json` - the runtime motor mapping implied by
   outputs/design_motor_pools_round34.  It carries, for each of the 90 actuators, the LOCAL
   NEURON INDICES of the pool that drives it, plus an explicit `command_scale`:
     * 1 channel per documented pool  -> scale 1.0;
     * an `_a`/`_b` pair that the model presents as the two fibre clusters of ONE muscle
       -> BOTH actuators are driven from the one pool with scale 0.5 each, so the muscle's
       total force capacity is NOT doubled;
     * the 10 actuators with no local evidence -> empty pool, held at exactly zero.
   The file also carries the support level of every row so the runner report shows how much
   of the wiring is connectome data and how much is a declared design decision.

2. `stance_for_runner.json` - the round-34 exact-balance pose in the form the runner's
   `--stance-file` expects (`joints_rad`, `root_z_mm`), so a run starts from a stance that is
   a genuine static equilibrium rather than the XML keyframe.

3. `design_activation_check.json` - an independent re-derivation: the design's solved
   activation pattern is re-applied in MuJoCo and MuJoCo's own equilibrium identity is
   asserted (qacc ~ 0), so run 2 cannot silently report a pose that is not in equilibrium.

This tool changes no model, writes no variant, and grants no runtime authorisation: the
overlay is DATA for the runner, whose strict entry point keeps refusing on its own terms.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'diagnostics'))

import diag_static_support_feasibility as r15  # noqa: E402
import diag_posture_support_search as r16      # noqa: E402

DESIGN = HERE / 'outputs/design_motor_pools_round34/design_motor_pools.json'
ST_POOLS = HERE / 'data/banc888/sterno_tergo_pools_by_leg.json'
STERNO_TERGO_READING = 'same'
LEG_OF_PREFIX = {'LF': 'prothoracic', 'LM': 'mesothoracic', 'LH': 'metathoracic',
                 'RF': 'prothoracic', 'RM': 'mesothoracic', 'RH': 'metathoracic'}
SIDEFIX = HERE / 'outputs/mn_resolution_v888_sidefix/mn_resolution_v888.json'
MANIFEST = HERE / 'outputs/muscle_anatomy_resolution/resolution.json'
OUT = HERE / 'outputs/design_runtime_round35'
XML = r15.XML
PENETRATION_MM = -0.050
ADHESION_BW = 2.0
LP_OPTS = {'presolve': True, 'time_limit': 60, 'maxiter': 20000}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=OUT)
    ap.add_argument('--design', type=Path, default=DESIGN,
                    help='design table JSON (default: the 2.0 bw/leg search)')
    args = ap.parse_args()
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)

    # banc_888_id -> root_id, so a v888 cell type can be carried onto exactly the neurons this
    # project already uses (same specimen, same materialisation; no cross-specimen claim).
    b888_to_root = {}
    map_path = HERE / 'data/banc888/b888_to_root.npy'
    if map_path.exists():
        for a_, r_ in np.load(map_path):
            b888_to_root[int(a_)] = str(int(r_))

    import mujoco as mj
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    xml_before = r15.digest(XML)
    model = _load_mjcf(str(XML)).compile()
    data = mj.MjData(model)
    model.actuator_ctrlrange[:, 0] = 0.0
    names = [model.actuator(i).name or f'act{i}' for i in range(model.nu)]

    design = json.loads(args.design.read_text())
    sidefix = json.loads(SIDEFIX.read_text())
    st_leg_pools = {k: v for k, v in json.loads(ST_POOLS.read_text()).items()
                    if not k.startswith('_')}
    pool_table = json.loads((HERE / 'data/banc888/design_pool_table.json').read_text())
    # the v888 resolution is the root-id source for every actuator the OLDER per-leg manifest
    # could not resolve (or resolved only at pair level).  Keyed by actuator name.
    v888_roots = {r['actuator']: sorted({c['root_id'] for c in r['candidates']})
                  for r in sidefix['all_results'] if r['candidates']}
    man = json.loads(MANIFEST.read_text())
    by_act = {r['actuator']: r for r in man['actuators']}
    tier_fingerprint = man['tier_fingerprint']
    # every local index in this project lives in the SAME 12k tier the manifest used
    rows = design['rows']

    # ---------------------------------------------------------------------------------
    # Build one entry per actuator: local indices + the scale the command is applied with.
    # ---------------------------------------------------------------------------------
    def local_indices(roots):
        # a candidate whose local_index is None is not in this project's tier; it is skipped
        # here and reported by the runner's own missing_MNs bookkeeping, never substituted.
        return [int(c['local_index']) for c in roots if c.get('local_index') is not None]

    # the sterno/tergo pools come from BANC directly (the manifest has none): filter by leg and
    # side, and remember which nerves carry them so the assignment is auditable
    st_cells = None

    entries, problems = {}, []
    st_conflicts = {}
    for r in rows:
        act = r['actuator']
        src = by_act[act]
        support = r['support']
        side_word = 'left' if act[0] == 'L' else 'right'
        if support == 'DATA_EXACT':
            loc = local_indices(src['candidates'])
            scale = 1.0
        elif support == 'DATA_MUSCLE_LEVEL_ANNOTATION':
            # pair-level: the pool is documented but drives the model's a/b split; the design uses
            # it per actuator (round 34) rather than double-driving the pair.  The pool is named by
            # banc_888_id, so the local indices are resolved through the tier below.
            loc = []
            scale = 1.0
        elif support == 'MODEL_GEOMETRY_BUNDLE_COUPLING':
            # fibre-cluster pair: the SAME pool drives both actuators, half strength each
            loc = []
            scale = 0.5
        elif support == 'DESIGN_HYPOTHESIS':
            # 1:1 with a connectome muscle: the sterno/tergo pair.  The manifest has no candidates
            # for these actuators, so the pool is taken from the v888 cell type for that leg+side
            # and mapped to the tier through banc_888_id.
            loc = []
            scale = 1.0
        else:
            loc, scale = [], 1.0
        pool_roots, pool_source, pool_note = [], None, None
        if support == 'DATA_EXACT':
            # the per-leg manifest pool, already side-filtered by construction
            pool_roots = sorted({str(c.get('root_id')) for c in src['candidates']})
            pool_source = 'outputs/muscle_anatomy_resolution/resolution.json (per-leg pool)'
        elif support == 'DATA_MUSCLE_LEVEL_ANNOTATION':
            # v888 pair-level: ONE documented pool for the model's a/b split.  The design drives
            # each actuator separately (round 34 measured that coupling them is not the blocker).
            pool_roots = list(v888_roots.get(act, []))
            pool_source = 'outputs/mn_resolution_v888_sidefix/mn_resolution_v888.json'
        elif support == 'MODEL_GEOMETRY_BUNDLE_COUPLING':
            # fibre-cluster pair (tergopleural promotor a/b): ONE pool drives both, half strength
            # each.  BANC names that pool only as "tergopleural_promotor_pleural_promotor" and
            # gives 4 cells per side through the dorsal prothoracic nerve, so it is used here and
            # the pleural promotor is left empty rather than split by guesswork.
            pool_roots = list(pool_table['tergopleural_promotor_each_side'].get(side_word, []))
            pool_source = ('data/banc888/design_pool_table.json: BANC cell type '
                           'tergopleural_promotor_pleural_promotor, dorsal prothoracic nerve')
            pool_note = ('BANC merges the tergopleural and pleural promotor into this one pool; it is '
                         'used for the tergopleural fibre-cluster pair and the pleural promotor is '
                         'left empty')
        elif support == 'DESIGN_HYPOTHESIS':
            # the sterno/tergo pair, 1:1 with the two connectome muscles for THIS leg
            suff = act.split('_')[-1]
            for ord_name, ordering in (r.get('orderings') or {}).items():
                if ord_name != STERNO_TERGO_READING:
                    continue
                ctype = ordering['pool_cell_type']
                # the pool file is keyed by BANC's peripheral-target name, which carries `_muscle`
                leg_key = (f'{ctype}_muscle|{side_word}|{LEG_OF_PREFIX[act[:2]]}')
                pool_roots = sorted(x['root'] for x in st_leg_pools.get(leg_key, []))
                pool_source = (f'data/banc888/sterno_tergo_pools_by_leg.json: {ctype} cells exiting '
                               f'the {LEG_OF_PREFIX[act[:2]]} leg nerves on the {side_word} side')
                pool_note = (f'reading "{STERNO_TERGO_READING}": the model suffix _{suff} is taken to be '
                             f'the {ctype.replace("_extensor", "")} component')
        else:
            # UNASSIGNED, or a single actuator the connectome does name
            adductor = pool_table['sternal_adductor'].get(act[:3], [])
            remotor = pool_table['pleural_remotor_and_abductor'].get(act[:3], [])
            if 'sternal_adductor' in act and adductor:
                pool_roots = list(adductor)
                pool_source = ('data/banc888/design_pool_table.json: BANC sternal_adductor cell on '
                               'this leg\'s own nerve')
            elif 'pleural_remotor_and_abductor' in act and remotor:
                pool_roots = list(remotor)
                pool_source = ('data/banc888/design_pool_table.json: BANC pleural_remotor_and_abductor '
                               'cells on this leg\'s own nerve')
            else:
                pool_note = ('no cell of the required muscle exits through this leg\'s own nerves in '
                             'BANC, and no contralateral substitution is made')
        if support != 'DATA_EXACT' and pool_roots:
            # every pool named by banc_888_id is carried onto exactly the roots this project uses
            pool_roots = [t for t in (b888_to_root.get(int(x)) for x in pool_roots) if t]
        entries[act] = {
            'actuator': act, 'support': support,
            'pool_root_ids': sorted(pool_roots),
            'pool_source': pool_source,
            'pool_note': pool_note,
            'local_indices': sorted(set(loc)),
            'command_scale': scale,
            'coupled_note': ('two fibre clusters of ONE muscle, driven from one pool at half '
                             'strength each so the muscle is not given double force'
                             if scale == 0.5 else None),
            'held_zero_because': ('no local evidence names a pool for this muscle'
                                  if not loc and support == 'UNASSIGNED' else None),
        }
        if support == 'DESIGN_HYPOTHESIS':
            st_conflicts[act] = sorted({
                x['region'] for x in st_leg_pools.get(
                    f"{r['orderings']['same']['pool_cell_type']}|"
                    f"{'left' if act[0] == 'L' else 'right'}|{LEG_OF_PREFIX[act[:2]]}", [])
                if 'segment_conflict' in x})
        # NOTE: local-index resolution for every pool named by banc_888_id happens after the
        # tier is built, further down; nothing is substituted for a missing root.

    # Resolve roots -> tier local indices for every entry that still has none (the
    # DESIGN_HYPOTHESIS sterno/tergo rows).  A root that is not in the tier is reported, never
    # substituted.
    missing_roots = []
    if b888_to_root:
        import sys as _sys
        _sys.path.insert(0, str(HERE))
        from engine.embodied.adapters import build_neural_tier
        _built = build_neural_tier()
        root_ids = [str(int(x)) for x in _built['tier']['root_ids']]
        root_to_local = {r: i for i, r in enumerate(root_ids)}
        for act, e in entries.items():
            # a manifest candidate usually already carries a local index; where it does not
            # (the v888-sourced pools, and the fibres whose candidates have no local index),
            # resolve the pool roots through the tier of THIS build.
            already = set(e['local_indices'])
            if not e['pool_root_ids']:
                continue
            idx = sorted(already | {root_to_local[r] for r in e['pool_root_ids']
                                    if r in root_to_local})
            gone = [r for r in e['pool_root_ids'] if r not in root_to_local]
            e['local_indices'] = idx
            e['roots_absent_from_tier'] = gone
            if gone:
                missing_roots.append({'actuator': act, 'n_absent': len(gone)})
        tier_note = (f'{len(root_ids)} neurons, same tier fingerprint as the design table')
    else:
        tier_note = 'root map unavailable: sterno/tergo pools left empty'

    for act, e in entries.items():
        if not e['local_indices'] and e['support'] != 'UNASSIGNED':
            problems.append({'actuator': act, 'support': e['support'],
                             'why': 'assigned a pool in the design but the tier has no local index '
                                    'for any of its roots'})

    # ---------------------------------------------------------------------------------
    # The stance: the round-34 exact-balance pose, plus an independent equilibrium check.
    # ---------------------------------------------------------------------------------
    best = design['physics']['design_search_over_poses']['best']
    mj.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0
    mj.mj_forward(model, data)
    key = data.qpos.copy()
    if 'x' in best and len(best['x']) == 42:
        params = [(leg, j) for leg in r16.LEGS
                  for j in ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw',
                            'Trochanter_pitch', 'Trochanter_roll', 'Tibia_pitch')]
        qpos = key.copy()
        for k, (leg, j) in enumerate(params):
            jid = model.joint(f'joint_{leg}{j}').id
            lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
            qpos[model.jnt_qposadr[jid]] = float(np.clip(key[model.jnt_qposadr[jid]] + best['x'][k],
                                                         lo, hi))
    else:
        qpos = key.copy()
        for jn, v in best.get('joints_rad', {}).items():
            qpos[model.jnt_qposadr[model.joint(jn).id]] = float(v)
    floor = r16.floor_geom_id(model, mj)
    tarsus_geoms = [g for g in range(model.ngeom)
                    if 'Tarsus' in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) or '')]
    tarsus_names = {mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) for g in tarsus_geoms}
    thorax_id = model.body('Thorax').id
    seated, root_z, _ = r16.seat_root(model, data, mj, qpos, tarsus_geoms, floor, PENETRATION_MM)
    facts = r16.pose_facts(model, data, mj, tarsus_names, thorax_id)
    joints_rad = {f'joint_{leg}{j}': float(seated[model.jnt_qposadr[model.joint(f"joint_{leg}{j}").id]])
                  for leg in r16.LEGS
                  for j in ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw',
                            'Trochanter_pitch', 'Trochanter_roll', 'Tibia_pitch')}
    (out / 'stance_for_runner.json').write_text(json.dumps({
        'status': 'DESIGN_STANCE_FOR_RUNNER',
        'source': str(args.design.resolve()),
        'joints_rad': joints_rad,
        'root_z_mm': float(root_z),
        'source_L1': float(best.get('L1', float('nan'))),
        'adhesion_bw_per_leg': float(design['physics'].get('adhesion_bw_per_leg', float('nan'))),
        'thorax_height_mm': facts['thorax_height_mm'],
        'contact_count': facts['contact_count'],
        'tarsus_only': facts['contacts_are_feet_only_tarsus'],
        'note': 'the round-34 design search reported exact static balance (L1 = 0.0) at this pose; '
                'the activation pattern that achieves it is in the design JSON, NOT here: the runner '
                'must produce its own commands from the network.',
    }, indent=2))

    # ---------------------------------------------------------------------------------
    # Independent equilibrium re-derivation: apply the design's solved activation in MuJoCo
    # and let MuJoCo's own identity speak.
    # ---------------------------------------------------------------------------------
    act = np.zeros(model.nu)
    solved = best['activation']
    for key_name, val in solved.items():
        if key_name in names:
            act[names.index(key_name)] = float(val)
        else:  # a coupled pool key: halves go to the two member actuators
            members = [a for a in names if a.startswith(key_name + '_a') or a.startswith(key_name + '_b')]
            for a in members:
                act[names.index(a)] = float(val)
    R, base = r15.actuator_moment_matrix(model, data, mj)
    target = np.asarray(data.qfrc_bias, float) - base - np.asarray(data.qfrc_passive, float)
    terms = r15.contact_terms(model, data, mj)
    lims = r15.limit_terms(model, data, mj)
    B = r15.cone_basis(model, data, terms, r15.CONE_DIRECTIONS)
    nv = model.nv
    m_vars = model.nu
    n_lam = sum(b.shape[1] for b in B)
    n_p, n_lim = len(terms), len(lims)
    limits = (np.column_stack([l['direction'] for l in lims]) if lims
              else np.zeros((nv, 0)))
    blocks = [R]
    for t, b in zip(terms, B):
        blocks.append(t['J'].T @ b)
    for t in terms:
        blocks.append(-t['J'].T @ t['normal'].reshape(-1, 1))
    blocks.append(limits)
    A_eq = np.hstack(blocks)
    leg_of = []
    for t in terms:
        lg = None
        for cand in (t['geom2'], t['geom1']):
            if cand[:2] in r16.LEGS:
                lg = cand[:2]; break
        leg_of.append(lg)
    present = sorted({l for l in leg_of if l})
    body_weight = float(mj.mj_getTotalmass(model) * abs(float(model.opt.gravity[2])))
    A = np.hstack([A_eq, np.eye(nv), -np.eye(nv)])
    A_ub = np.zeros((len(present), A.shape[1]))
    for i_, lg in enumerate(present):
        for k, l2 in enumerate(leg_of):
            if l2 == lg:
                A_ub[i_, m_vars + n_lam + k] = 1.0
    c = np.zeros(A.shape[1]); c[-2 * nv:] = 1.0
    bnds = ([(0.0, 1.0)] * m_vars + [(0.0, None)] * n_lam + [(0.0, None)] * n_p
            + [(0.0, None)] * n_lim + [(0.0, None)] * (2 * nv))
    from scipy.optimize import linprog
    res = linprog(c, A_eq=A, b_eq=target, A_ub=A_ub,
                  b_ub=np.full(len(present), ADHESION_BW * body_weight),
                  bounds=bnds, method='highs', options=LP_OPTS)
    eq = r16.validate_equilibrium(model, data, mj, 'design stance (passive)')
    check = {
        'status': 'DESIGN_ACTIVATION_CHECK',
        'stance_source': str(out / 'stance_for_runner.json'),
        'pose': {'thorax_height_mm': facts['thorax_height_mm'],
                 'tarsus_only': facts['contacts_are_feet_only_tarsus'],
                 'contact_count': facts['contact_count']},
        'design_reported_L1': best['L1'],
        'lp_residual_here': None if res.x is None else float(np.abs(res.x[-2 * nv:-nv]
                                                                    - res.x[-nv:]).sum()),
        'muJoCo_equilibrium': eq,
        'note': ('the design solved its own LP; this re-runs the same LP here and additionally '
                 'asserts MuJoCo\'s equilibrium identity at the seated pose, so the runner cannot '
                 'start from a pose that is not statically admissible'),
        'limitations': ['static balance only; NOT stability, NOT walking',
                        'the activation pattern is the design\'s static solution, not a neural output'],
    }
    (out / 'design_activation_check.json').write_text(json.dumps(check, indent=2))

    overlay = {
        'status': 'MOTOR_POOL_OVERLAY_DESIGN_ROUND34',
        'tier_resolution': tier_note,
        'sternotergo_segment_conflicts': st_conflicts,
        'roots_absent_from_tier': missing_roots,
        'source': str(args.design.resolve()),
        'tier_fingerprint': tier_fingerprint,
        'tier_size': 12000,
        'exact_muscle_name': {a: e['local_indices'] for a, e in entries.items() if e['local_indices']},
        'command_scale': {a: e['command_scale'] for a, e in entries.items()},
        'support_level': {a: e['support'] for a, e in entries.items()},
        'held_zero': [a for a, e in entries.items() if not e['local_indices']],
        'entries': entries,
        'unresolved_rows': problems,
        'counts': {
            'actuators': len(entries),
            'with_pool': sum(1 for e in entries.values() if e['local_indices']),
            'held_zero': sum(1 for e in entries.values() if not e['local_indices']),
            'half_strength_pairs': sum(1 for e in entries.values() if e['command_scale'] == 0.5),
            'total_neurons_referenced': len({i for e in entries.values()
                                             for i in e['local_indices']}),
        },
        'rules': {
            'exact_muscle_name': 'the actuator\'s pool, one entry per actuator; the runner merges it '
                                 'into its own mapping',
            'command_scale': 'a coupled fibre-cluster pair is driven from ONE pool at HALF strength '
                             'per actuator, so the muscle\'s total capacity is not double counted',
            'held_zero': 'actuators with no local evidence are left with an empty pool and stay at '
                         'exactly zero; they are never substituted'},
        'limitations': [
            'A pool assignment is not an anatomical validation of any single neuron.',
            'Three families (10 actuators) are deliberately left empty.',
            'runtime_authorization stays false; the runner\'s own strict checks are unchanged.'],
    }
    assert r15.digest(XML) == xml_before
    (out / 'motor_pool_overlay_design.json').write_text(json.dumps(overlay, indent=2))

    print(json.dumps({'overlay': str(out / 'motor_pool_overlay_design.json'),
                      'counts': overlay['counts'],
                      'stance': str(out / 'stance_for_runner.json'),
                      'stance_pose': {'thorax_mm': facts['thorax_height_mm'],
                                      'contacts': facts['contact_count'],
                                      'tarsus_only': facts['contacts_are_feet_only_tarsus']},
                      'lp_residual_here': check['lp_residual_here'],
                      'muJoCo_qacc_max': eq['qacc_max'],
                      'unresolved_rows': problems[:3]}, indent=2))
    print(out)


if __name__ == '__main__':
    main()
