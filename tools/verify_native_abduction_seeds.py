"""Independent pure-array recheck of the four-seed package: new process, no optimisation.

One model compile is used only for compiled joint names/ranges and for the model
parameter digest. There is no mj_forward, no mj_step, no contact evaluation and no
optimiser call here: every number is recomputed from the arrays already saved by the
preregistered run.
"""
from pathlib import Path
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys, json, hashlib, time
import numpy as np
import mujoco
from flygym.compose.fly.musculoskeletal import _load_mjcf

S = Path('/home/sensen/Desktop/engine修复验收/s5-detailed-handover-measurement/20261028T121000Z')
O = Path('/home/sensen/Desktop/engine修复验收/s5-equilibrium-abduction-seeds/20261101T201309Z')
R6 = Path('/home/sensen/Desktop/engine修复验收/s5-native-equilibrium-search/20261101T193630Z')
WRENCH = Path('/home/sensen/Desktop/engine修复验收/s5-equilibrium-wrench-diagnosis/20261101T200110Z/WRENCH.json')
EXPECTED = '3470c72f7e68f181ec1a80a5dc5fbff182b750091e7a1ddc7ccfbfb590b80cb6'
FACTORS = (0.0, 0.25, 0.5, 0.75)
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
REF_H_OVER_V = 0.19921933195031086


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,
                            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()) + '\n')


def parameter_digest(m):
    rows = {}
    for prefix, obj in [('model', m), ('option', m.opt)]:
        for name in dir(obj):
            if name.startswith('_'):
                continue
            value = getattr(obj, name)
            if isinstance(value, np.ndarray):
                a = np.ascontiguousarray(value)
                rows[prefix + '.' + name] = dict(shape=list(a.shape), dtype=str(a.dtype),
                                                 sha256=hashlib.sha256(a.tobytes()).hexdigest())
            elif isinstance(value, (int, float, bool, str)):
                rows[prefix + '.' + name] = value
    return hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest(), rows


def main():
    started = time.time()
    counts = dict(model_load_attempt=0, model_load=0, forward=0, mj_step=0, geom_distance=0,
                  contact_force=0, optimization=0)
    checks = {}
    details = {}
    variables = json.loads((O / 'VARIABLES.json').read_text())
    prereg = json.loads((O / 'PREREGISTRATION.json').read_text())
    table = json.loads((O / 'SEED_TABLE.json').read_text())
    comparison = json.loads((O / 'PER_SEED_COMPARISON.json').read_text())
    execution = json.loads((O / 'EXECUTION.json').read_text())
    identities = json.loads((O / 'INPUT_IDENTITIES.json').read_text())

    # ---- A. input identity, independently recomputed ----
    now = {str(p): sha(p) for p in [S / 'MODEL.xml', S / 'BODY.npz', S / 'MAPPING.json',
                                    O / 'PREREGISTRATION.json', O / 'SEED_TABLE.json']}
    checks['a_input_identities_match'] = all(identities[k] == v for k, v in now.items())
    checks['a_model_xml_sha_expected'] = now[str(S / 'MODEL.xml')] == EXPECTED
    checks['a_seed_table_sha_matches_prereg'] = now[str(O / 'SEED_TABLE.json')] == prereg['seed_table_sha256']
    tool = Path('/run/media/sensen/Data2/cell_wound_prototype/tools/search_native_equilibrium_abduction_seeds.py')
    executed = O / 'EXECUTED_search_native_equilibrium_abduction_seeds.py'
    checks['a_executed_source_equals_current_tool'] = tool.is_file() and sha(tool) == sha(executed)
    details['source_shas'] = dict(current_tool=str(tool), current=sha(tool) if tool.is_file() else None,
                                  executed=sha(executed))

    # ---- B. one compile, no forward ----
    counts['model_load_attempt'] += 1
    m = _load_mjcf(str(S / 'MODEL.xml')).compile()
    counts['model_load'] += 1
    digest, _ = parameter_digest(m)
    before = json.loads((O / 'MODEL_PARAMETERS_BEFORE.json').read_text())['sha256']
    after = json.loads((O / 'MODEL_PARAMETERS_AFTER.json').read_text())['sha256']
    stored = execution['model_parameters_before']
    checks['b_model_digest_recomputed_matches_stored'] = digest == before == after == stored
    checks['b_model_unchanged_before_after'] = execution['model_parameters_unchanged'] is True
    hinges = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    assert len(hinges) == 42
    qa = m.jnt_qposadr[hinges].astype(int)
    names = [m.joint(int(j)).name for j in hinges]
    hlo = m.jnt_range[hinges, 0]
    hhi = m.jnt_range[hinges, 1]
    checks['b_hinge_names_match_variables'] = names == variables['hinge_names']
    checks['b_joint_ranges_match_variables'] = bool(np.array_equal(hlo, variables['joint_lower'])
                                                    and np.array_equal(hhi, variables['joint_upper']))
    G = np.asarray(variables['group_matrix'], float)
    mapped = np.asarray(variables['mapped_actuator_mask'], bool)
    weight = float(variables['weight_native'])
    seed_pose = np.asarray(variables['seed_pose_43'], float)
    lower = np.asarray(variables['lower'], float)
    upper = np.asarray(variables['upper'], float)
    plo, phi = float(variables['pool_range'][0]), float(variables['pool_range'][1])
    checks['b_group_matrix_structure'] = bool(
        G.shape == (90, 66) and np.all((G == 0) | (G == 1)) and np.all(G.sum(0) >= 1)
        and int(G.sum()) == 74 and int(mapped.sum()) == 74 and int((~mapped).sum()) == 16
        and np.all(G.sum(1) <= 1) and np.array_equal(mapped, G.sum(1) > 0))

    # ---- C. seed table replay from the archived BODY trajectory ----
    with np.load(S / 'BODY.npz') as z:
        qseed = z['qpos'][int(np.argmin(abs(z['time'] - .15)))].copy()
    base = qseed[qa].copy()
    contract = [f'joint_{L}Coxa_{a}' for L in LEGS for a in ('roll', 'yaw')]
    cidx = np.array([names.index(n) for n in contract], int)
    seed_replay, ok = {}, True
    for factor, srow in zip(FACTORS, table['seeds']):
        raw = base.copy()
        raw[cidx] = factor * base[cidx]
        proj = np.clip(raw, hlo, hhi)
        x0 = np.r_[proj, qseed[2], np.full(66, .1)]
        diff = float(np.max(np.abs(x0 - np.asarray(srow['x0_109'], float))))
        nproj = int(np.sum(proj != raw))
        seed_replay[str(factor)] = dict(max_abs_diff=diff, n_projections=nproj,
                                        stored_n_projections=int(srow['n_projections']))
        ok = ok and diff == 0.0 and nproj == srow['n_projections']
    checks['c_seed_vectors_replay_exactly'] = bool(ok)
    checks['c_archived_hinge_values_replay'] = bool(
        np.array_equal(np.asarray(table['archived_hinge_values'], float), base))
    checks['c_only_the_12_coxa_hinges_contract'] = bool(
        all(np.array_equal(np.asarray(srow['raw_contracted_42'], float)[[i for i in range(42) if i not in cidx]],
                           base[[i for i in range(42) if i not in cidx]]) for srow in table['seeds']))
    details['seed_replay'] = seed_replay

    # ---- D. per-seed array recheck ----
    per_seed = {}
    for k in FACTORS:
        s = json.loads((O / ('SEED_%s_SUMMARY.json' % k)).read_text())
        with np.load(O / ('SEED_%s_FINAL_STATE.npz' % k)) as zz:
            arr = {name: zz[name] for name in zz.files}
        x, qpos, act, ctrl = arr['x'], arr['qpos'], arr['activation'], arr['ctrl']
        r_saved = arr['residual']
        r_qacc = arr['qacc'] / 9810.
        r_dist = 100 * np.maximum(0., .02 - arr['nonfoot_distances'])
        r_side = 100 * np.maximum(0., .02 - arr['body_side'])
        load = float(arr['foot_force'][:, 2].sum() / weight)
        r_load = np.array([10 * (load - 1)])
        r_reg = 1e-6 * (x[:43] - seed_pose)
        r_rebuilt = np.r_[r_qacc, r_dist, r_side, r_load, r_reg]
        blocks = dict(qacc48=float(r_qacc @ r_qacc), nonfoot41=float(r_dist @ r_dist),
                      side36=float(r_side @ r_side), load1=float(r_load @ r_load),
                      regularizer43=float(r_reg @ r_reg))
        side_rebuilt = ((arr['side_body_pos'] - arr['thorax_pos']) @ arr['thorax_yaxis']) * arr['side_sign']
        rows = arr['contact_rows']
        net = rows[:, 6:9].sum(0)
        about_com = np.cross(rows[:, 3:6] - arr['com_world'], rows[:, 6:9]).sum(0)
        about_root = np.cross(rows[:, 3:6] - arr['root_pos'], rows[:, 6:9]).sum(0)
        pool_values = (G @ x[43:])[mapped]
        recon_err = float(np.max(np.abs(arr['moment'].T @ arr['actuator_force'] - arr['qfrc_actuator'])))
        vertex_err = float(np.max(np.abs(arr['compiled_vertex_distances'] - arr['nonfoot_distances'])))
        q42 = qpos[qa]
        d_lo, d_hi = q42 - hlo, hhi - q42
        near = np.minimum(d_lo, d_hi)
        sides = np.where(d_lo < d_hi, 'lower', 'upper')
        coxa_mask = np.array([('Coxa' in n and n.split('_')[-1] in ('roll', 'yaw')) for n in names])
        f = s['final']
        gates_rebuilt = dict(
            all48_acceleration=bool(np.all(abs(arr['qacc']) <= .1)),
            nonfoot_clearance=bool(np.all(arr['nonfoot_distances'] >= .02)),
            nonfoot_force_zero=bool(np.max(np.abs(arr['nonfoot_force'])) == 0.),
            foot_load=bool(.99 <= load <= 1.01),
            joint_bounds=bool(np.all(q42 >= hlo) and np.all(q42 <= hhi) and np.array_equal(qpos[qa], x[:42])),
            root_z_bounds=bool(variables['root_z_range'][0] <= qpos[2] <= variables['root_z_range'][1]
                               and qpos[2] == x[42]),
            body_sides=bool(np.all(side_rebuilt >= .02)),
            activation_bounds=bool(np.all(act >= plo) and np.all(act <= phi)),
            pool_binding_66=bool(np.all(pool_values >= plo) and np.all(pool_values <= phi)
                                 and np.array_equal(act[mapped], pool_values)),
            unmapped_floor_16=bool(np.all(act[~mapped] == plo)),
            act_ctrl_equal=bool(np.array_equal(act, ctrl)),
            activation_steady=bool(np.max(abs(arr['act_dot'])) <= 1e-10),
            external_forces_zero=bool(not np.any(arr['qfrc_applied']) and not np.any(arr['xfrc_applied'])),
            moment_reconstruction=bool(recon_err <= 1e-9),
            finite=bool(all(np.isfinite(a).all() for a in [x, qpos, arr['qacc'], act, r_saved,
                                                           arr['actuator_force'], arr['foot_force']])),
            model_parameters_unchanged=bool(digest == before),
            xml_unchanged=bool(now[str(S / 'MODEL.xml')] == EXPECTED),
            geom_distance_cutoff_zero=bool(np.all(arr['nonfoot_distances'] < 1000.)),
            mesh_plane_crosscheck=bool(vertex_err <= 1e-8),
            all109_within_declared_bounds=bool(np.all(x >= lower) and np.all(x <= upper)))
        stored_gates = s['gate']['gates']
        per_seed[str(k)] = dict(
            residual_rebuild_max_abs_diff=float(np.max(np.abs(r_rebuilt - r_saved))),
            residual_rebuild_allclose=bool(np.allclose(r_rebuilt, r_saved, rtol=0, atol=1e-12)),
            block_sse_rebuild=blocks,
            block_sse_matches_summary=bool(all(abs(blocks[b] - f['blocks'][b]['sse'])
                                               <= 1e-12 * max(1., abs(blocks[b])) for b in blocks)),
            total_sse_rebuild=float(r_rebuilt @ r_rebuilt),
            total_sse_matches_summary=bool(abs(float(r_rebuilt @ r_rebuilt) - f['row']['cost'])
                                           <= 1e-12 * max(1., f['row']['cost'])),
            max_abs_qacc_rebuild=float(abs(arr['qacc']).max()),
            max_abs_qacc_matches_summary=bool(abs(float(abs(arr['qacc']).max()) - f['row']['max_abs_qacc'])
                                              <= 1e-9 * max(1., f['row']['max_abs_qacc'])),
            body_side_rebuild_max_abs_diff=float(np.max(np.abs(side_rebuilt - arr['body_side']))),
            load_rebuild=load,
            load_matches_summary=bool(abs(load - f['wrench']['vertical_over_weight']) <= 1e-12),
            horizontal_over_vertical=float(np.linalg.norm(net[:2]) / net[2]),
            hv_matches_summary=bool(abs(float(np.linalg.norm(net[:2]) / net[2])
                                        - f['wrench']['horizontal_over_vertical']) <= 1e-12),
            moment_over_weight_com=float(np.linalg.norm(about_com) / weight),
            moment_over_weight_root=float(np.linalg.norm(about_root) / weight),
            moment_matches_summary=bool(abs(float(np.linalg.norm(about_com) / weight)
                                            - f['wrench']['moment_norm_over_weight_com']) <= 1e-12),
            moment_rows_vs_legs_abs=float(np.max(np.abs(net - arr['foot_force'].sum(0)))),
            min_joint_limit_distance_rebuild=float(near.min()),
            min_joint_limit_joint_rebuild=str(names[int(np.argmin(near))]),
            min_joint_limit_side_rebuild=str(sides[int(np.argmin(near))]),
            coxa_roll_yaw_min_rebuild=float(near[coxa_mask].min()),
            joints_within_0p01_rebuild=int(np.sum(near <= .01)),
            joints_within_0p05_rebuild=int(np.sum(near <= .05)),
            limit_table_matches_summary=bool(
                abs(float(near.min()) - f['limit_table']['min_distance']) <= 1e-12
                and str(names[int(np.argmin(near))]) == f['limit_table']['argmin_joint']
                and str(sides[int(np.argmin(near))]) == f['limit_table']['argmin_side']
                and int(np.sum(near <= .01)) == f['limit_table']['joints_within_0p01']
                and int(np.sum(near <= .05)) == f['limit_table']['joints_within_0p05']
                and abs(float(near[coxa_mask].min()) - f['abduction']['coxa_roll_yaw_min_limit_distance']) <= 1e-12
                and abs(float(sum(abs(q42[names.index(n)]) for n in contract))
                        - f['abduction']['candidate_sum_abs_coxa_roll_yaw']) <= 1e-12),
            reconstruction_error=recon_err, vertex_crosscheck=vertex_err,
            gate_booleans_rebuilt=gates_rebuilt,
            gate_booleans_match_stored=bool(all(gates_rebuilt[g] == stored_gates[g] for g in gates_rebuilt)),
            gate_values_not_rebuildable_without_forward=['no_warnings'],
            stored_gate_passed=s['gate_passed'], stored_failed_gates=[g for g, v in stored_gates.items() if not v],
            n_contacts=int(len(rows)),
            zero_or_negative_leg_count=int(np.sum(arr['foot_force'][:, 2] <= 0)),
            per_leg_vertical_force=[float(v) for v in arr['foot_force'][:, 2]],
            net_contact_force=[float(v) for v in net],
            root_qacc=[float(v) for v in arr['qacc'][:6]],
            n_qacc_components_over_0p1=int(np.sum(np.abs(arr['qacc']) > .1)),
            coxa_room_to_more_abduction={n: float((hhi[names.index(n)] - q42[names.index(n)])
                                                 if n.startswith('joint_L') else
                                                 (q42[names.index(n)] - hlo[names.index(n)]))
                                         for n in contract})
    checks['d_all_residuals_rebuild'] = all(v['residual_rebuild_allclose'] for v in per_seed.values())
    checks['d_all_blocks_rebuild'] = all(v['block_sse_matches_summary'] for v in per_seed.values())
    checks['d_all_totals_rebuild'] = all(v['total_sse_matches_summary'] for v in per_seed.values())
    checks['d_all_max_qacc_rebuild'] = all(v['max_abs_qacc_matches_summary'] for v in per_seed.values())
    checks['d_all_body_side_rebuild'] = all(v['body_side_rebuild_max_abs_diff'] < 1e-12 for v in per_seed.values())
    checks['d_all_wrench_rebuild'] = all(v['load_matches_summary'] and v['hv_matches_summary']
                                         and v['moment_matches_summary'] for v in per_seed.values())
    checks['d_all_limit_tables_rebuild'] = all(v['limit_table_matches_summary'] for v in per_seed.values())
    checks['d_all_gate_booleans_rebuild'] = all(v['gate_booleans_match_stored'] for v in per_seed.values())
    checks['d_moment_crosscheck_1e-9'] = all(v['reconstruction_error'] <= 1e-9 for v in per_seed.values())
    checks['d_vertex_crosscheck_1e-8'] = all(v['vertex_crosscheck'] <= 1e-8 for v in per_seed.values())
    checks['d_moment_rows_vs_legs'] = all(v['moment_rows_vs_legs_abs'] < 1e-12 for v in per_seed.values())

    # ---- E. full evaluation-log replay ----
    log = O / 'EVALUATIONS.jsonl'
    per_seed_log = {str(k): dict(rows=0, optimizer=0, final=0, min_cost=float('inf'), min_row=None,
                                 max_abs_block_vs_cost=0., max_abs_reg_diff=0., topology_ok=True,
                                 seed_eval_contiguous=True, final_cost=None) for k in FACTORS}
    total_rows = 0
    last_seed_eval = {str(k): 0 for k in FACTORS}
    forward_ids = []
    with log.open() as fh:
        for line in fh:
            row = json.loads(line)
            key = str(row['seed_k'])
            rec = per_seed_log[key]
            rec['rows'] += 1
            total_rows += 1
            forward_ids.append(row['evaluation'])
            if row['role'] == 'final_validation':
                rec['final'] += 1
                rec['final_cost'] = row['cost']
            else:
                rec['optimizer'] += 1
                if row['cost'] < rec['min_cost']:
                    rec['min_cost'] = row['cost']
                    rec['min_row'] = dict(evaluation=row['evaluation'], cost=row['cost'],
                                          x=row['x'], ncon=row['ncon'], topology=row['contact_topology'],
                                          solver_niter=row['solver_niter'])
            if row['seed_evaluation'] != last_seed_eval[key] + 1:
                rec['seed_eval_contiguous'] = False
            last_seed_eval[key] = row['seed_evaluation']
            block_sum = sum(b['sse'] for b in row['blocks'].values())
            rec['max_abs_block_vs_cost'] = max(rec['max_abs_block_vs_cost'],
                                               abs(block_sum - row['cost']) / max(1., row['cost']))
            xr = np.asarray(row['x'], float)
            reg_rebuilt = float(1e-12 * np.sum((xr[:43] - seed_pose) ** 2))
            rec['max_abs_reg_diff'] = max(rec['max_abs_reg_diff'],
                                          abs(reg_rebuilt - row['blocks']['regularizer43']['sse']))
            sig = row['contact_topology']
            n_sig = 0 if sig.endswith('|') else sig.count(';') + 1
            if not (sig.startswith('ncon=%d|' % row['ncon']) and n_sig == row['ncon']):
                rec['topology_ok'] = False
    checks['e_forward_ids_strictly_increasing'] = bool(all(b > a for a, b in zip(forward_ids, forward_ids[1:])))
    checks['e_forward_ids_contiguous_from_1'] = bool(forward_ids == list(range(1, len(forward_ids) + 1)))
    checks['e_total_rows_equal_execution_forward_count'] = bool(total_rows == execution['counts']['forward'])
    checks['e_one_final_validation_per_seed'] = all(per_seed_log[str(k)]['final'] == 1 for k in FACTORS)
    checks['e_block_sum_equals_cost_every_row'] = all(per_seed_log[str(k)]['max_abs_block_vs_cost'] <= 1e-12
                                                      for k in FACTORS)
    checks['e_regularizer_block_rebuilt_every_row'] = all(per_seed_log[str(k)]['max_abs_reg_diff'] <= 1e-15
                                                          for k in FACTORS)
    checks['e_topology_signature_consistent'] = all(per_seed_log[str(k)]['topology_ok'] for k in FACTORS)
    checks['e_seed_evaluation_sequence_contiguous'] = all(per_seed_log[str(k)]['seed_eval_contiguous']
                                                          for k in FACTORS)
    selection, ok = {}, True
    for k in FACTORS:
        rec = per_seed_log[str(k)]
        s = json.loads((O / ('SEED_%s_SUMMARY.json' % k)).read_text())
        with np.load(O / ('SEED_%s_FINAL_STATE.npz' % k)) as zz:
            saved_x = zz['x']
        same_cost = abs(rec['min_cost'] - s['solver']['selected_sse']) <= 1e-12 * max(1., rec['min_cost'])
        same_x = rec['min_row'] is not None and float(np.max(np.abs(np.asarray(rec['min_row']['x'], float)
                                                                   - saved_x))) == 0.
        same_eval = rec['min_row']['evaluation'] == s['solver']['selected_evaluation']
        same_final = abs(rec['final_cost'] - s['final']['row']['cost']) <= 1e-12 * max(1., rec['final_cost'])
        nfev, njev = s['solver']['nfev'], s['solver']['njev']
        count_ok = rec['optimizer'] == nfev + njev * 109
        selection[str(k)] = dict(min_cost=rec['min_cost'], stored_selected_sse=s['solver']['selected_sse'],
                                 cost_matches=same_cost, x_identical=same_x, evaluation_id_matches=same_eval,
                                 fresh_repeat_cost_matches=same_final,
                                 optimizer_rows=rec['optimizer'], nfev=nfev, njev=njev,
                                 nfev_plus_109_probes=nfev + njev * 109, count_matches=count_ok,
                                 final_validation_ncon=rec['min_row']['ncon'],
                                 final_validation_solver_niter=rec['min_row']['solver_niter'])
        ok = ok and same_cost and same_x and same_eval and same_final and count_ok
    checks['e_selection_rule_replayed'] = bool(ok)

    # ---- F. round6 reference against the sealed wrench diagnosis ----
    sealed = json.loads(WRENCH.read_text())['results'][0]
    ref = comparison['round6_reference']
    with np.load(R6 / 'FINAL_STATE.npz') as z6:
        rows6 = z6['contact_rows']
        net6 = rows6[:, 6:9].sum(0)
        m6 = np.cross(rows6[:, 3:6] - z6['qpos'][:3], rows6[:, 6:9]).sum(0)
        q42_r6 = z6['qpos'][qa]
        ref_check = dict(vertical_over_weight=float(net6[2] / weight),
                         horizontal_over_vertical=float(np.linalg.norm(net6[:2]) / net6[2]),
                         moment_over_weight_root=float(np.linalg.norm(m6) / weight),
                         max_abs_qacc=float(abs(z6['qacc']).max()),
                         total_sse=float(z6['residual'] @ z6['residual']),
                         ncon=int(len(rows6)))
    checks['f_round6_reference_agrees_with_sealed_wrench'] = bool(
        abs(ref_check['horizontal_over_vertical'] - sealed['horizontal_over_vertical']) <= 1e-12
        and abs(ref_check['vertical_over_weight'] - sealed['vertical_over_weight']) <= 1e-12
        and abs(ref_check['moment_over_weight_root'] - sealed['moment_norm_over_weight_root']) <= 1e-12
        and abs(ref_check['max_abs_qacc'] - sealed['max_abs_qacc']) <= 1e-9)
    checks['f_round6_reference_agrees_with_this_package'] = bool(
        abs(ref_check['horizontal_over_vertical'] - ref['horizontal_over_vertical']) <= 1e-12
        and abs(ref_check['moment_over_weight_root'] - ref['moment_norm_over_weight_root']) <= 1e-12)

    # ---- G. classification rule replay ----
    avail = [r for r in comparison['rows'] if r['available']]
    h_min = min(r['horizontal_over_vertical'] for r in avail)
    any_pass = any(r['gate_passed'] for r in avail)
    if any_pass:
        label = 'A_supported_with_candidate'
    elif h_min <= .5 * REF_H_OVER_V:
        label = 'B_partial_support'
    elif all(r['horizontal_over_vertical'] >= .9 * REF_H_OVER_V for r in avail):
        label = 'C_rejected'
    else:
        label = 'D_weak_or_inconclusive'
    recurring = [r['retained_fraction_k'] for r in avail
                 if (r['lm_coxa_roll_side'] == 'upper' and r['lm_coxa_roll_distance'] <= .01)
                 or (r['rh_coxa_roll_side'] == 'lower' and r['rh_coxa_roll_distance'] <= .01)]
    checks['g_classification_label_reproduced'] = label == comparison['classification']['label']
    checks['g_recurring_binding_list_reproduced'] = recurring == \
        comparison['classification']['seeds_with_recurring_same_two_coxa_roll_binding']

    # ---- H. auxiliary coxa table recomputed from the saved states ----
    aux = {}
    for k in FACTORS:
        with np.load(O / ('SEED_%s_FINAL_STATE.npz' % k)) as zz:
            q42 = zz['qpos'][qa]
        aux[str(k)] = {n: dict(q=float(q42[names.index(n)]), lower=float(hlo[names.index(n)]),
                               upper=float(hhi[names.index(n)]),
                               room_more_abduction=float((hhi[names.index(n)] - q42[names.index(n)])
                                                         if n.startswith('joint_L') else
                                                         (q42[names.index(n)] - hlo[names.index(n)])))
                       for n in contract}
    aux['round6_search60'] = {n: dict(q=float(q42_r6[names.index(n)]), lower=float(hlo[names.index(n)]),
                                      upper=float(hhi[names.index(n)]),
                                      room_more_abduction=float((hhi[names.index(n)] - q42_r6[names.index(n)])
                                                                if n.startswith('joint_L') else
                                                                (q42_r6[names.index(n)] - hlo[names.index(n)])))
                              for n in contract}

    checks['verifier_did_no_physical_work'] = bool(counts['forward'] == 0 and counts['mj_step'] == 0
                                                   and counts['contact_force'] == 0
                                                   and counts['optimization'] == 0)
    passed = all(checks.values())
    dump(O / 'INDEPENDENT_ARRAY_RECHECK.json', dict(
        passed=passed, checks=checks, per_seed=per_seed, per_seed_log=per_seed_log, selection=selection,
        round6_reference_recheck=ref_check,
        sealed_wrench_reference=dict(horizontal_over_vertical=sealed['horizontal_over_vertical'],
                                     vertical_over_weight=sealed['vertical_over_weight'],
                                     moment_norm_over_weight_root=sealed['moment_norm_over_weight_root'],
                                     max_abs_qacc=sealed['max_abs_qacc']),
        auxiliary_coxa_table=aux, classification_replay=dict(label=label, h_min=float(h_min), recurring=recurring),
        details=details, counts=counts, wall_seconds=time.time() - started,
        scope='New process; one model compile for compiled names/ranges and the parameter digest; zero forward, '
              'zero step, zero contact evaluation, zero optimisation. Every reported number was rebuilt from arrays '
              'saved by the preregistered run or from the sealed round6 package.'))
    print(json.dumps(dict(passed=passed, failed=[k for k, v in checks.items() if not v],
                          h_min=h_min, label=label, recurring=recurring), indent=2), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
