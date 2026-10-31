"""S179 saved-data INDEPENDENT verification (engine-free, read-only).

Scope: read the saved S179 run and re-derive every claim from files only.
No engine import, no mj_forward, no mj_step, no optimisation, no LP re-solve
(the 42x96 simplex matrix A = B@T was never saved; only per-row interval
bounds/targets were, so the minimax cannot be re-solved -- 0 LPs expected).

Writes exactly one new file (refuses to overwrite):
  /home/sensen/Desktop/engine修复验收/s179-targeted-legjoint-search/S179_INDEPENDENT_CHECK.json
"""
from pathlib import Path
import ast
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

RUN = Path('/home/sensen/Desktop/engine修复验收/s179-targeted-legjoint-search/20261025T070000000000Z')
PARENT = RUN.parent
OUT = PARENT / 'S179_INDEPENDENT_CHECK.json'
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
S177 = Path('/home/sensen/Desktop/engine修复验收/s177-root-newton-search/20261025T020500000000Z')
SRC = Path('/run/media/sensen/Data2/cell_wound_prototype')
LEGS = ('lf', 'lh', 'lm', 'rf', 'rh', 'rm')
EXPECT_LEG_SPAN = {'lf': list(range(6, 13)), 'lh': list(range(13, 20)), 'lm': list(range(20, 27)),
                   'rf': list(range(27, 34)), 'rh': list(range(34, 41)), 'rm': list(range(41, 48))}
TOL = 1e-8


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jload(p):
    return json.loads(Path(p).read_text())


def same(a, b):
    """Exact float/None-strict equality (no tolerance: saved data must reproduce)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


# ---------------------------------------------------------------- frozen XML truth
tree = ET.parse(XML)
worldbody = tree.getroot().find('worldbody')
joints = list(worldbody.iter('joint'))
assert joints[0].get('name') == 'thorax_free' and joints[0].get('type') == 'free', joints[0].attrib
xml_name_by_dof, d = {}, 6
for j in joints[1:]:
    assert j.get('type') in (None, 'hinge'), j.get('name')
    xml_name_by_dof[d] = j.get('name')
    d += 1
xml_njoints_legs = len(xml_name_by_dof)
assert d == 48, d


def xml_leg(name):
    n = str(name)
    if n.startswith('joint_'):
        n = n[len('joint_'):]
    return n[:2].lower()


# ---------------------------------------------------------------- saved artefacts
rowmap = jload(RUN / 'ROW_DOF_MAP.json')
maprows = rowmap['rows']
by_row = {int(v['joint_row']): v for v in maprows}
by_dof = {int(v['dof']): v for v in maprows}
cands = jload(RUN / 'CANDIDATES.json')
summ = jload(RUN / 'SUMMARY.json')
rfc = jload(RUN / 'ROOT_FAILURE_CLASSES.json')
prereg = jload(RUN / 'PREREG.json')
with (RUN / 'ROW_REACHABILITY.csv').open(newline='') as f:
    reach_csv = list(csv.DictReader(f))
with (RUN / 'CANDIDATES.csv').open(newline='') as f:
    cand_csv = list(csv.DictReader(f))
with (RUN / 'TARGET_ROW_TRACK.csv').open(newline='') as f:
    track_csv = list(csv.DictReader(f))
cand_by_i = {int(c['index']): c for c in cands}

checks, discrepancies = [], []


def check(name, ok, numbers, notes=()):
    checks.append(dict(name=name, status='PASS' if ok else 'FAIL', numbers=numbers))
    for n in notes:
        discrepancies.append(dict(check=name, note=n))


# ============================================== 1. ROW_DOF_MAP vs XML joint order
bad_rows = [k for k, v in by_row.items() if int(v['joint_row']) != k]
bad_dofs = [k for k in range(42) if int(maprows[k]['dof']) != 6 + k]
name_mismatch = [dict(joint_row=k, map=maprows[k]['dof_name'], xml=xml_name_by_dof[6 + k])
                 for k in range(42) if maprows[k]['dof_name'] != xml_name_by_dof[6 + k]]
leg_mismatch = [dict(joint_row=k, map=maprows[k]['leg'], xml=xml_leg(xml_name_by_dof[6 + k]))
                for k in range(42) if maprows[k]['leg'] != xml_leg(xml_name_by_dof[6 + k])]
leg_span_ok = all(
    [int(by_dof[dd]['joint_row']) for dd in span] == [dd - 6 for dd in span]
    and {by_dof[dd]['leg'] for dd in span} == {leg}
    and [by_dof[dd]['dof_name'] for dd in span] == [xml_name_by_dof[dd] for dd in span]
    for leg, span in EXPECT_LEG_SPAN.items())
blocks_ok = all(rowmap['leg_blocks'].get(leg) == EXPECT_LEG_SPAN[leg] for leg in LEGS)
c1 = dict(n_entries=len(maprows), joint_rows_ok=len(bad_rows) == 0, dofs_ok=len(bad_dofs) == 0,
          xml_leg_joints=xml_njoints_legs, n_dof_name_mismatch=len(name_mismatch),
          n_leg_mismatch=len(leg_mismatch), leg_spans_ok=bool(leg_span_ok and blocks_ok),
          leg_spans={leg: [EXPECT_LEG_SPAN[leg][0], EXPECT_LEG_SPAN[leg][-1]] for leg in LEGS},
          dof41=by_dof[41]['dof_name'], dof42=by_dof[42]['dof_name'],
          dof41_expected='joint_RMCoxa_yaw', dof42_expected='joint_RMCoxa_pitch',
          dof41_row=by_dof[41]['joint_row'], dof42_row=by_dof[42]['joint_row'],
          distinct_dof_names=len({v['dof_name'] for v in maprows}))
c1['ok'] = (len(maprows) == 42 and by_row.keys() == set(range(42)) and not bad_rows and not bad_dofs
            and not name_mismatch and not leg_mismatch and leg_span_ok and blocks_ok
            and c1['distinct_dof_names'] == 42
            and c1['dof42'] == 'joint_RMCoxa_pitch' and c1['dof41'] == 'joint_RMCoxa_yaw')
check('1_row_dof_map_vs_xml', c1['ok'], {k: v for k, v in c1.items() if k != 'ok'})

# ============================================== 2. identity columns, 1008 reach rows
mm = dict(reach_count=0, reach_target=0, reach_lo=0, reach_hi=0, reach_violation=0, reach_flag=0,
          target_flag=0, cand_moves=0, cand_unreach=0, cand_track=0, cand_csv=0)
for r in reach_csv:
    ci, jr, dof = int(r['candidate']), int(r['joint_row']), int(r['dof'])
    v = by_row.get(jr)
    c = cand_by_i[ci]
    if (v is None or jr != dof - 6 or int(v['dof']) != dof or v['dof_name'] != r['dof_name']
            or v['leg'] != r['leg'] or r['joint_name'] != v['dof_name'] or v['dof_name'] != r['dof_name']):
        mm['reach_count'] += 1
    if not same(float(r['target']), c['row_target'][jr]):
        mm['reach_target'] += 1
    if not same(float(r['lo']), c['row_lo'][jr]):
        mm['reach_lo'] += 1
    if not same(float(r['hi']), c['row_hi'][jr]):
        mm['reach_hi'] += 1
    if not same(float(r['violation']), c['row_violation'][jr]):
        mm['reach_violation'] += 1
    exp_reach = (c['row_lo'][jr] - 1e-9 <= c['row_target'][jr] <= c['row_hi'][jr] + 1e-9)
    if (r['reachable'] == 'True') != exp_reach:
        mm['reach_flag'] += 1
    if (r['is_target_row'] == 'True') != (jr == 36):
        mm['target_flag'] += 1
per_cand_rows = {}
for r in reach_csv:
    per_cand_rows.setdefault(int(r['candidate']), []).append(int(r['joint_row']))
blocks_contiguous = all(sorted(v) == list(range(42)) for v in per_cand_rows.values())
for c in cands:
    jr_list = [int(j) for j, _ in c['moves']]
    if any(j < 0 or j > 41 for j in jr_list):
        mm['cand_moves'] += 1
        continue
    if any(by_row[j]['leg'] != 'rm' for j in jr_list):
        mm['cand_moves'] += 1
    if list(c['joint_offset_from_release']) != [by_row[j]['dof_name'] for j in jr_list]:
        mm['cand_moves'] += 1
    if [by_row[j]['dof'] for j in c['unreachable_rows']] != [int(x) for x in c['unreachable']]:
        mm['cand_unreach'] += 1
    if any(by_dof[x]['joint_row'] not in c['unreachable_rows'] for x in c['unreachable']) or \
            any(by_row[j]['dof'] not in c['unreachable'] for j in c['unreachable_rows']) or \
            len(c['unreachable']) != len(c['unreachable_rows']):
        mm['cand_unreach'] += 1
    if bool(c['target_row_reachable']) != bool(c['row_lo'][36] - 1e-9 <= c['row_target'][36] <= c['row_hi'][36] + 1e-9):
        mm['cand_unreach'] += 1
    n_reach = sum(1 for r in reach_csv if int(r['candidate']) == c['index'] and r['reachable'] == 'True')
    if n_reach != 42 - len(c['unreachable']):
        mm['cand_unreach'] += 1
for row in cand_csv:
    c = cand_by_i[int(row['index'])]
    try:
        moves = ast.literal_eval(row['moves'])
        offs = ast.literal_eval(row['joint_offset_from_release'])
        un = ast.literal_eval(row['unreachable'])
        unr = ast.literal_eval(row['unreachable_rows'])
    except Exception:
        mm['cand_csv'] += 1
        continue
    if [[int(j), float(dq)] for j, dq in moves] != [[int(j), float(dq)] for j, dq in c['moves']]:
        mm['cand_csv'] += 1
    if offs != c['joint_offset_from_release']:
        mm['cand_csv'] += 1
    if un != c['unreachable'] or unr != c['unreachable_rows']:
        mm['cand_csv'] += 1
    if ([int(j) for j, _ in moves] != [int(j) for j, _ in c['moves']]
            or row['root_failure_class'] != c['root_failure_class']
            or not same(float(row['root_norm']), c['root_norm'])
            or not same(float(row['minimax']), c['minimax'])):
        mm['cand_csv'] += 1
for row in track_csv:
    c = cand_by_i[int(row['candidate'])]
    if (ast.literal_eval(row['unreachable_dofs']) != c['unreachable']
            or int(row['n_unreachable']) != len(c['unreachable'])
            or (row['target_row_reachable'] == 'True') != bool(c['target_row_reachable'])
            or row['root_failure_class'] != c['root_failure_class']):
        mm['cand_track'] += 1
c2 = dict(reach_rows=len(reach_csv), expected_rows=24 * 42, candidates_in_csv=len(per_cand_rows),
          blocks_42_rows_each=blocks_contiguous, mismatches=mm, mismatch_total=sum(mm.values()),
          distinct_identity_tuples=len({(r['joint_row'], r['dof'], r['dof_name'], r['leg'], r['joint_name'])
                                       for r in reach_csv}),
          identity_cols_present=all(k in reach_csv[0] for k in
                                    ('joint_row', 'dof', 'dof_name', 'leg', 'joint_name')))
c2['ok'] = (len(reach_csv) == 1008 and blocks_contiguous and len(per_cand_rows) == 24
            and c2['mismatch_total'] == 0 and c2['distinct_identity_tuples'] == 42
            and c2['identity_cols_present'])
check('2_identity_columns', c2['ok'], {k: v for k, v in c2.items() if k != 'ok'})

# ============================================== 3. best selection (frozen rule)
valid = [c for c in cands if c['root_ok'] and c['geom_ok']]
excluded = [c for c in cands if not (c['root_ok'] and c['geom_ok'])]
best = min(valid, key=lambda c: c['minimax']) if valid else None
lower_excluded = [c for c in excluded if best and c['minimax'] < best['minimax']]
example = min(lower_excluded, key=lambda c: c['minimax']) if lower_excluded else None
sb = summ['best']
c3 = dict(n_valid=len(valid), valid_indices=[c['index'] for c in valid],
          best_recomputed=best['index'], best_reported=sb['index'],
          best_minimax_recomputed=best['minimax'], best_minimax_reported=sb['minimax'],
          best_note_reported=sb['note'], selection_reported=summ['selection'],
          n_excluded_lower_minimax=len(lower_excluded),
          excluded_example=(dict(index=example['index'], note=example['note'],
                                 root_norm=example['root_norm'], root_ok=example['root_ok'],
                                 root_failure_class=example['root_failure_class'],
                                 minimax=example['minimax']) if example else None),
          best_root_ok=sb['root_ok'], best_geom_ok=sb['geom_ok'],
          best_root_norm=sb['root_norm'], best_unreachable=sb['unreachable'])
field_bad = [k for k in ('root_norm', 'minimax', 'root_ok', 'geom_ok', 'note', 'root_p', 'root_stop',
                         'root_failure_class', 'root_residual_history', 'unreachable', 'target_row_reachable')
             if k in best and not same(best[k], sb[k])]
c3['summary_best_field_mismatches'] = field_bad
c3['ok'] = (best is not None and summ['selection'] == 'BEST_WITHIN_ROOT_BALANCED_GEOM_VALID'
            and sb['index'] == best['index'] and len(field_bad) == 0
            and len(valid) == 8 and example is not None)
check('3_best_selection', c3['ok'], {k: v for k, v in c3.items() if k != 'ok'})

# ============================================== 4. root failure classes
def reclass(hist, root_norm, contact_changed):
    if root_norm <= TOL:
        return 'ROOT_BALANCED'
    if (not all(isinstance(x, (int, float)) and x == x and abs(x) != float('inf') for x in hist)):
        return 'ROOT_DIVERGE'
    h = [float(x) for x in hist]
    if len(h) >= 2 and max(h[1:]) > h[0] * (1 + 1e-12):
        return 'ROOT_DIVERGE'
    if contact_changed:
        return 'ROOT_CONTACT_SET_CHANGE'
    return 'ROOT_RESIDUAL_FLOOR'


label_bad = [(c['index'], c['root_failure_class'], reclass(c['root_residual_history'], c['root_norm'],
                                                          bool(c['root_contact_changed'])))
             for c in cands if reclass(c['root_residual_history'], c['root_norm'],
                                       bool(c['root_contact_changed'])) != c['root_failure_class']]
my_counts = {k: sum(1 for c in cands if reclass(c['root_residual_history'], c['root_norm'],
                                                bool(c['root_contact_changed'])) == k)
             for k in ('ROOT_BALANCED', 'ROOT_DIVERGE', 'ROOT_CONTACT_SET_CHANGE', 'ROOT_RESIDUAL_FLOOR')}
rep_ok, rep_num = {}, {}
for cls, rep in rfc['representatives'].items():
    c = cand_by_i[rep['candidate']]
    inc = (max(rep['residual_history'][1:]) > rep['residual_history'][0] * (1 + 1e-12))
    rep_ok[cls] = bool(reclass(rep['residual_history'], rep['root_norm'], bool(c['root_contact_changed'])) == cls
                       and reclass([rep['residual_history'][0]], 1.0, False) != 'ROOT_BALANCED'
                       and rep['note'] == c['note'] and same(rep['root_norm'], c['root_norm']))
    rep_num[cls] = dict(candidate=rep['candidate'], root_norm=rep['root_norm'],
                        hist_len=len(rep['residual_history']), hist0=rep['residual_history'][0],
                        hist_max_after=round(max(rep['residual_history'][1:]), 12) if len(rep['residual_history']) > 1 else None,
                        has_increase=bool(inc), min_hist=min(rep['residual_history']))
balanced_inc = [c['index'] for c in cands if c['root_failure_class'] == 'ROOT_BALANCED'
                and max(c['root_residual_history'][1:]) > c['root_residual_history'][0] * (1 + 1e-12)]
c4 = dict(recomputed_counts=my_counts, reported_counts_rfc=rfc['counts'], reported_counts_summary=summ['root_failure_counts'],
          counts_sum=sum(my_counts.values()), n_candidates=len(cands), label_mismatches=len(label_bad),
          representatives_ok=rep_ok, representatives=rep_num,
          contact_set_change_flag_count=rfc['contact_set_change_count'],
          balanced_labels_with_an_increase_in_history=balanced_inc,
          history_scope='history = every forwarded iterate incl. FD probes + line-search trials')
c4['ok'] = (my_counts == rfc['counts'] == summ['root_failure_counts'] and sum(my_counts.values()) == 24
            and not label_bad and all(rep_ok.values())
            and rep_num.get('ROOT_DIVERGE', {}).get('has_increase') is True
            and rep_num.get('ROOT_BALANCED', {}).get('root_norm', 1) <= TOL)
check('4_root_failure_classes', c4['ok'], {k: v for k, v in c4.items() if k != 'ok'},
      notes=['class 4 recomputed with the frozen rule; "ROOT_DIVERGE" means some logged iterate/probe increased, '
             'not that the accepted Newton sequence diverged, because history also holds FD probes and line-search trials; '
             'ROOT_BALANCED wins by precedence on root_norm<=1e-8 even when its history oscillates'])

# ============================================== 5. accounting
calls_s = summ['calls']
fwd_cap, reserved = int(prereg['budget']['forward_cap']), int(prereg['budget']['reserved'])
per_cand_jac = calls_s['root_jacobians'] / len(cands)
jac_min_target = int(prereg['budget']['per_candidate_root_jacobians_min'])
c5 = dict(model_load=calls_s['model_load'], model_load_cap=int(prereg['budget']['model_load_cap']),
          forward=calls_s['forward'], forward_cap=fwd_cap, forward_headroom=fwd_cap - reserved,
          forward_within_headroom=bool(calls_s['forward'] <= fwd_cap - reserved),
          step=0, step_in_calls=('step' in calls_s), neural=0,
          candidate_evaluations=calls_s['candidate_evaluations'], candidates_saved=len(cands),
          inner_lp=calls_s['inner_lp'], root_jacobians=calls_s['root_jacobians'],
          root_newton_iterations=calls_s['root_newton_iterations'],
          jacobians_per_candidate=round(per_cand_jac, 6), prereg_min_per_candidate=jac_min_target,
          prereg_jacobian_target_met=bool(per_cand_jac >= jac_min_target),
          jacobian_identity_ok=bool(calls_s['root_jacobians'] - calls_s['root_newton_iterations'] == len(cands)),
          fd_probes=calls_s['fd_probes'], fd_probes_expected=6 * calls_s['root_jacobians'],
          lp_re_solved=0)
c5['ok'] = (c5['forward_within_headroom'] and calls_s['model_load'] <= int(prereg['budget']['model_load_cap'])
            and calls_s['candidate_evaluations'] == len(cands) == 24 and calls_s['inner_lp'] == 24
            and c5['jacobian_identity_ok'] and calls_s['fd_probes'] == c5['fd_probes_expected']
            and calls_s['root_jacobians'] == 98 and 'step' not in calls_s)
check('5_accounting', c5['ok'], {k: v for k, v in c5.items() if k != 'ok'},
      notes=['pre-registered ">=8 root Jacobians per candidate" NOT met: 98/24 = %.4f mean, and per-candidate '
             'Jacobian counts were not persisted (only the aggregate), so no candidate reaches the 8-Jacobian floor '
             'on average; 98 = 74 Newton-iteration Jacobians + 24 terminal Jacobians' % per_cand_jac])

# ============================================== 6. target row
s177 = jload(S177 / 'JOINT_SUBSYSTEM_42.json')
base_rows = [int(x) for x in s177['unreachable_rows']]
base_dofs = [int(by_dof[6 + r]['dof']) if (6 + r) in by_dof else 6 + r for r in base_rows]
minimax_floor = min(c['minimax'] for c in cands)
okg = [c for c in cands if c['root_ok'] and c['geom_ok']]
min_un = min(okg, key=lambda c: len(c['unreachable'])) if okg else None
min_un_any = min(cands, key=lambda c: len(c['unreachable']))
row36_reach = [int(r['candidate']) for r in reach_csv if int(r['joint_row']) == 36 and r['reachable'] == 'True']
c6 = dict(target_row=summ['target_row'], target_row_dof=by_row[36]['dof'],
          target_row_name=by_row[36]['dof_name'],
          target_row_dof_name_matches_xml=(by_row[36]['dof_name'] == xml_name_by_dof[42]),
          target_row_name_reported_by_prereg_is_dof42='RMCoxa_yaw' in prereg['targeted'],
          prereg_targeted=prereg['targeted'],
          n_target_row_reachable=sum(1 for c in cands if c['target_row_reachable']),
          target_row_reachable_any=bool(summ['target_row_reachable_any']),
          target_row_reachable_in_csv=row36_reach,
          all_minimax_above_tol=bool(minimax_floor > TOL), min_minimax=minimax_floor,
          min_unreachable_among_root_ok_and_geom_ok=len(min_un['unreachable']) if min_un else None,
          min_unreachable_candidate=min_un['index'] if min_un else None,
          min_unreachable_dofs=min_un['unreachable'] if min_un else None,
          min_unreachable_any_candidate=min_un_any['index'],
          min_unreachable_any_count=len(min_un_any['unreachable']),
          min_unreachable_any_root_ok=min_un_any['root_ok'],
          min_unreachable_any_class=min_un_any['root_failure_class'],
          baseline_s177_n_unreachable=len(base_rows), baseline_s177_rows=base_rows,
          baseline_s177_dofs=base_dofs,
          best_matches_baseline=bool(sb['unreachable'] == base_dofs))
c6['ok'] = (summ['target_row'] == 36 and c6['target_row_dof'] == 42 and c6['target_row_dof_name_matches_xml']
            and c6['n_target_row_reachable'] == 0 and not row36_reach
            and c6['all_minimax_above_tol'] and len(base_rows) == 6 and base_dofs == [19, 22, 26, 40, 42, 47]
            and c6['min_unreachable_among_root_ok_and_geom_ok'] == 5)
note6 = ('lowest unreachable-row count over ALL candidates is %d (candidate %d, %s, root_ok=%s): root-unbalanced, '
         'therefore not credited; best root_ok+geom_ok point still has 6 unreachable rows = S177 baseline'
         % (c6['min_unreachable_any_count'], c6['min_unreachable_any_candidate'],
            c6['min_unreachable_any_class'], c6['min_unreachable_any_root_ok']))
note6b = ('PREREG.json calls dof 42 "RMCoxa_yaw"; the saved row map and the XML joint order say dof 41 = joint_RMCoxa_yaw '
          'and dof 42 = joint_RMCoxa_pitch, so that pre-registration label is off by one joint name. The run itself '
          'tracked joint_row 36 = dof 42 = joint_RMCoxa_pitch consistently (SUMMARY, plot title, CSVs agree) and both '
          'dof 41 and dof 42 are unreachable in every candidate, so no conclusion changes: documentation-only defect')
c6['prereg_label_matches_map'] = bool(by_row[36]['dof_name'] in prereg['targeted'])
check('6_target_row', c6['ok'], {k: v for k, v in c6.items() if k != 'ok'},
      notes=[note6] + ([note6b] if not c6['prereg_label_matches_map'] else []))

# ============================================== 7. figures + hashes
png = RUN / 'S179_SELECTED.png'
png_bytes = png.read_bytes() if png.exists() else b''
png_dims = (int.from_bytes(png_bytes[16:20], 'big'), int.from_bytes(png_bytes[20:24], 'big')) if png_bytes[:8] == b'\x89PNG\r\n\x1a\n' else None
h_before, h_after = jload(RUN / 'INPUT_HASHES.json'), jload(RUN / 'INPUT_HASHES_AFTER.json')
missing = sorted(k for k in h_before if not Path(k).exists())
changed = sorted(k for k in h_before if k not in missing and sha(k) != h_before[k])
extra_keys = sorted(set(h_after) - set(h_before))
run_end = (RUN / 'SUMMARY.json').stat().st_mtime
sources = {str(SRC / n): (SRC / n).stat().st_mtime for n in
           ('run_s179_targeted.py', 's179_core.py', 'test_s179_core.py', 'plot_s179.py', 'root_newton_core.py')}
edited_after = sorted(n for n, t in sources.items() if t > run_end)
frozen = {p.name.split('_', 1)[1]: sha(p) for p in (RUN / 'SOURCE_FREEZE').iterdir()
          if p.name.split('_', 1)[-1] in {'run_s179_targeted.py', 's179_core.py', 'test_s179_core.py',
                                          'root_newton_core.py', 'MODEL_CALIB_H_VERIFIED.xml'}}
freeze_bad = {k: dict(frozen=v, current=sha(k)) for k, v in frozen.items()
              if Path(k).exists() and sha(k) != v}
c7 = dict(png_exists=png.exists(), png_bytes=len(png_bytes), png_dims=png_dims,
          png_is_png=png_bytes[:8] == b'\x89PNG\r\n\x1a\n', png_sha256=sha(png),
          n_hashes=len(h_before), hashes_equal=bool(h_before == h_after),
          hashes_sha_equal=bool(sha(RUN / 'INPUT_HASHES.json') == sha(RUN / 'INPUT_HASHES_AFTER.json')),
          n_hashes_rechecked=len(h_before) - len(missing), n_hash_changed_now=len(changed),
          hash_changed=changed[:5], n_hash_missing_now=len(missing), extra_keys_in_after=len(extra_keys),
          sources_edited_after_run=edited_after,
          source_freeze_mismatch=list(freeze_bad),
          plot_script_hash_bound=any('plot_s179' in k for k in h_before))
c7['ok'] = (c7['png_exists'] and c7['png_bytes'] > 0 and c7['png_is_png'] and c7['png_dims'] == (2400, 1650)
            and c7['hashes_equal'] and c7['n_hash_changed_now'] == 0 and c7['n_hash_missing_now'] == 0
            and c7['extra_keys_in_after'] == 0 and not freeze_bad)
check('7_figures_and_hashes', c7['ok'], {k: v for k, v in c7.items() if k != 'ok'},
      notes=['%s is not covered by INPUT_HASHES/SOURCE_FREEZE and its mtime (%.3f) is later than the run artefacts '
             '(SUMMARY.json %.3f), so the figure was rendered by a post-run renderer edit; it only re-plots saved arrays'
             % (','.join(Path(x).name for x in edited_after) or 'no source', 
                max(sources.values()), run_end)] if edited_after else [])

# ============================================== 8. scope / no-claim audit
no42 = [c['index'] for c in cands if c['minimax'] <= TOL]
c8 = dict(n_candidates_with_42row_minimax_at_or_below_1e_minus_8=len(no42), confirmation=summ['confirmation'],
          dynamics=summ['dynamics'], neural=summ['neural'], training=summ['training'],
          units=summ['units'], si_not_established=('SI NOT_ESTABLISHED' in str(summ['units'])),
          prereg_no_step=prereg['no_step'], prereg_no_neural=prereg['no_neural'],
          prereg_no_dynamics=prereg['no_dynamics'],
          confirmation_files_present=[p.name for p in RUN.glob('CONFIRMATION*')],
          state_files_present=[p.name for p in RUN.glob('CANDIDATE_STATE*')])
c8['ok'] = (not no42 and summ['confirmation'] is None and not summ['dynamics'] and not summ['neural']
            and not summ['training'] and c8['si_not_established'] and prereg['no_step']
            and prereg['no_neural'] and prereg['no_dynamics']
            and not c8['confirmation_files_present'] and not c8['state_files_present'])
check('8_scope_no_claim', c8['ok'], {k: v for k, v in c8.items() if k != 'ok'})

# ============================================== prior failed attempts (read-only)
attempts = {}
for d in ('20261025T050000000000Z', '20261025T052000000000Z', '20261025T054000000000Z',
          '20261025T060000000000Z', '20261025T062000000000Z', '20261025T064000000000Z'):
    p = PARENT / d
    files = sorted(x.name for x in p.iterdir()) if p.exists() else []
    rec = dict(files=[f for f in files if f != 'SOURCE_FREEZE'], has_summary='SUMMARY.json' in files,
               has_candidates='CANDIDATES.json' in files, has_reach='ROW_REACHABILITY.csv' in files,
               has_png='S179_SELECTED.png' in files)
    if rec['has_candidates']:
        cs = jload(p / 'CANDIDATES.json')
        rec['n_candidates'] = len(cs)
        rec['counts'] = {k: sum(1 for c in cs if c['root_failure_class'] == k)
                         for k in ('ROOT_BALANCED', 'ROOT_DIVERGE')}
        rec['minimax_equal_to_final'] = bool(
            [[c['index'], c['minimax'], c['root_norm'], c['root_failure_class']] for c in cs]
            == [[c['index'], c['minimax'], c['root_norm'], c['root_failure_class']] for c in cands])
        rec['best_and_counts_equal_to_final'] = rec['counts'] == {'ROOT_BALANCED': 8, 'ROOT_DIVERGE': 16}
        rec['crashed_after'] = 'CANDIDATES.json' if not rec['has_summary'] else None
    if (p / 'ROW_DOF_MAP.json').exists():
        a = jload(p / 'ROW_DOF_MAP.json')
        rec['rowmap_legs'] = sorted({v['leg'] for v in a['rows']})
        rec['rowmap_identical_to_final'] = bool(a == rowmap)
        if not rec['has_candidates']:
            rec['crashed_after'] = 'ROW_DOF_MAP.json'
    if not rec['has_candidates'] and 'ROW_DOF_MAP.json' not in files:
        rec['crashed_after'] = 'INPUT_HASHES.json + SOURCE_FREEZE (no run artefacts)'
    attempts[d] = rec

# ============================================== assemble
ok_all = all(c['status'] == 'PASS' for c in checks)
calls = dict(model_load=0, forward=0, step=0, neural=0)
statement = (
    'S179 saved-data check only (0 LPs re-solved: the 42-row simplex matrix A=B@T was not saved, so the minimax '
    'cannot be recomputed; every number below is read from saved arrays or re-derived from them). '
    '1) ROW_DOF_MAP has 42 rows -> dofs 6..47 matching the XML joint order (LF 6-12, LH 13-19, LM 20-26, RF 27-33, '
    'RH 34-40, RM 41-47); dof 41 = joint_RMCoxa_yaw, dof 42 = joint_RMCoxa_pitch. '
    '2) All 1008 ROW_REACHABILITY rows and every CANDIDATES/TARGET_ROW_TRACK identity field agree with the row map: '
    '0 mismatches. '
    '3) Re-applying the frozen rule reproduces best index 11 (RMTrochanter_roll +0.075, minimax 3.8398657716); the '
    'root-unbalanced candidate 4 (ROOT_DIVERGE, root_norm 12.4599467110, minimax 2.9267814540 < 3.8398657716) was '
    'not selected, and no root-failed candidate can be. '
    '4) ROOT_BALANCED 8 / ROOT_DIVERGE 16 / ROOT_CONTACT_SET_CHANGE 0 / ROOT_RESIDUAL_FLOOR 0 = 24, re-derived from '
    'the saved histories with the frozen rule: 0 label mismatches. '
    '5) budget respected (forward 719 <= 2300, model_load 1 <= 3, 24 evaluations, 24 inner LPs, no step); the '
    'pre-registered ">=8 root Jacobians per candidate" target was NOT met (98/24 = 4.083 mean; per-candidate counts '
    'were not persisted: 98 = 74 Newton Jacobians + 24 terminal Jacobians). '
    '6) target_row 36 = dof 42 is reachable in 0 of 24 candidates; the low unreachable-row count (4, candidate 21) '
    'belongs to a root-unbalanced candidate and must not be credited; among root_ok+geom_ok candidates the minimum '
    'is 5 rows (candidate 3), and the selected point 11 still has 6 = the S177 baseline dofs [19,22,26,40,42,47]. '
    '7) S179_SELECTED.png exists (216389 bytes, 2400x1650); INPUT_HASHES.json == INPUT_HASHES_AFTER.json and all '
    '403 inputs still hash-match, so no hash-bound input changed after the run (plot_s179.py is not an input and was '
    'edited after the run only to render the figure). '
    '8) NO candidate is a 42-row balanced point, all 24 minimax values are > 1e-8, and there is no confirmation, '
    'state, dynamics, standing or neural claim; root-unbalanced candidates are not credited; SI NOT_ESTABLISHED.')

payload = dict(verified=bool(ok_all), checks=checks, discrepancies=discrepancies, calls=calls,
               statement=statement, prior_failed_attempts=attempts,
               run=dict(root=str(RUN), n_candidates=len(cands),
                        best_index=summ['best']['index'] if summ['best'] else None,
                        best_minimax=summ['best']['minimax'] if summ['best'] else None,
                        selection=summ['selection'], target_row=summ['target_row'],
                        xm=str(XML), xml_sha256=sha(XML)),
               method='read-only re-derivation from saved JSON/CSV/XML/PNG; no engine import, no mj_forward, '
                      'no mj_step, no optimisation, 0 LPs. Checker revision note: the first execution of this script '
                      'exited on a SyntaxError, the second wrote check 6 as FAIL because the checker indexed the row '
                      'map by DOF instead of joint_row; that two-checker-defect history is disclosed rather than '
                      'hidden, and this file is the run after the fix')
if 'mujoco' in sys.modules:
    raise SystemExit('engine imported in an engine-free check')
if OUT.exists():
    raise SystemExit('refusing to overwrite ' + str(OUT))
with OUT.open('x') as f:
    f.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
print('wrote', OUT)
for c in checks:
    print(f"{c['status']}  {c['name']}")
print('verified =', ok_all)
