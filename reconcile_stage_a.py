"""S178 Stage A reconciliation from SAVED grid data only (no engine, no new forwards).

Fixes two labelling defects found after the Stage A run and re-derives the per-row verdict
without touching the physics: (1) leg names must come from the joint names, not from an
assumed block order; (2) the per-row baseline must use that row's own interval.
Also recomputes the sign-flip criterion against the one-sided displacement available
(+/-0.6 rad from release), not the full two-sided span.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); p = ap.parse_args().output
res = json.loads((p/'STAGE_A_RESULT.json').read_text())
ranges = json.loads((p/'FROZEN_RANGES.json').read_text())
base = json.loads((p/'STAGE_A_BASELINE_ROWS.json').read_text())
names = ranges['joint_names']; release = np.array(ranges['release'])
def leg_of(j):
    n = names[j]
    for pref, leg in (('LFCoxa','lf'),('LMCoxa','lm'),('LHCoxa','lh'),('RFCoxa','rf'),('RMCoxa','rm'),('RHCoxa','rh')):
        if n.startswith('joint_'+pref):
            return leg
    return 'unstaged'
base_rows = {r['dof']: r for r in base['rows']}
table = []
for t in res['table']:
    r = t['row']; j = t['joint']
    grid = t['grid']
    one_sided = 0.6
    reach_pts = [g for g in grid if g['reachable']]
    min_delta = min((abs(g['angle']-release[j]) for g in reach_pts), default=None)
    need = t['angle_needed_for_sign_flip']
    table.append(dict(row=r, dof=r, leg=leg_of(j), leg_block_joints=[names[k] for k in range((r-6)//7*7, (r-6)//7*7+7)][:1],
                      joint_index=j, joint_name=names[j], dominant_column=t['dominant_column'],
                      release_angle=float(release[j]), frozen_lower=t['range'][0], frozen_upper=t['range'][1],
                      baseline_target=base_rows[r]['target'], baseline_lo=base_rows[r]['lo'], baseline_hi=base_rows[r]['hi'],
                      baseline_reachable=base_rows[r]['reachable'],
                      grid_has_reachable_point=bool(reach_pts), min_abs_delta_rad_for_reachability=min_delta,
                      slope_dm_dq=t['slope_dm_dq'], angle_needed_for_sign_flip=need,
                      one_sided_window_rad=one_sided,
                      flip_possible_in_window=bool(need is not None and need <= one_sided),
                      empirical_reach_within_window=bool(reach_pts)))
with (p/'STAGE_A_SIGNFLIP_CORRECTED.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
per_row = []
for r in sorted({t['row'] for t in table}):
    sub = [t for t in table if t['row'] == r]
    emp = [t for t in sub if t['empirical_reach_within_window']]
    best = min(emp, key=lambda t: t['min_abs_delta_rad_for_reachability']) if emp else None
    per_row.append(dict(dof=r, leg=sub[0]['leg'], baseline=dict(target=sub[0]['baseline_target'], lo=sub[0]['baseline_lo'],
                      hi=sub[0]['baseline_hi'], reachable=sub[0]['baseline_reachable']),
                      any_point_in_window_makes_row_reachable=bool(emp),
                      cheapest_joint=None if best is None else dict(joint_index=best['joint_index'], joint_name=best['joint_name'],
                      min_abs_delta_rad=best['min_abs_delta_rad_for_reachability']),
                      diff_theory_estimate_min_rad=min([t['angle_needed_for_sign_flip'] for t in sub if t['angle_needed_for_sign_flip'] is not None], default=None),
                      joints_with_zero_channel=[t['joint_name'] for t in sub if t['slope_dm_dq'] == 0.0]))
json.dump(dict(per_row=per_row, note='reconciled offline from saved grid data; legs named from joint names; criterion uses the one-sided 0.6 rad window; zero new forwards'),
          (p/'STAGE_A_CORRECTED_SUMMARY.json').open('w'), indent=2)
for v in per_row:
    print(v['dof'], v['leg'], 'reach_in_window=', v['any_point_in_window_makes_row_reachable'],
          'cheapest=', None if not v['cheapest_joint'] else (v['cheapest_joint']['joint_name'], round(v['cheapest_joint']['min_abs_delta_rad'], 3)),
          'theory_min=', None if v['diff_theory_estimate_min_rad'] is None else round(v['diff_theory_estimate_min_rad'], 3),
          'baseline', round(v['baseline']['lo'], 4), round(v['baseline']['target'], 4), round(v['baseline']['hi'], 4))
