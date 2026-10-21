import json
import pathlib

D = pathlib.Path('/home/sensen/Desktop/engine修复验收/S2-accepted-equilibrium-CORRECTED-PROTOCOL-25us-20261015T171055')
r = json.loads((D / 'report.json').read_text())
for tag, x in r['runs'].items():
    print('==', tag)
    for k in ('first_nonfoot_contact', 'first_step_legs_in_contact_below_6', 'first_step_with_zero_tarsus_contacts',
              'first_step_max_u_diamond_ge_1', 'first_step_act_above_1', 'root_z', 'six_leg_steps', 'energy', 'tilt',
              'per_step_max_qacc'):
        print('  ', k, '=', json.dumps(x[k], ensure_ascii=False))
    print('   act_max', x['act_max'], '| ctrl_max', x['ctrl_max'])
L = r['localization']
print('=== localization')
print('  first leg loss:', L['first_leg_to_lose_tarsus_contact'])
print('  first leg force below 1pct:', L['first_leg_force_below_1pct'])
print('  first joint:', L['first_joint_past_0_01_rad'])
print('  frame stats:', json.dumps(L['drive_frame_stats'], ensure_ascii=False))
print('  amplitude gap mean:', json.dumps({k: v['mean_abs_gap_vs_first_nonzero_frame'] for k, v in L['drive_amplitude_gap_per_leg'].items()},
                                          ensure_ascii=False))
print('  per-leg first steps:', json.dumps({k: {kk: vv for kk, vv in v.items() if 'first_step' in kk}
                                          for k, v in L['per_leg'].items()}, ensure_ascii=False))
