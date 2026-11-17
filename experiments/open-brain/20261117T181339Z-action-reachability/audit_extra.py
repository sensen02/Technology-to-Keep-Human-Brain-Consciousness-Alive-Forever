"""Validate supplemental stored physical traces and collect final diagnostic status."""
from context import *
from collections import Counter

main=json.loads((HERE/'SUMMARY.json').read_text());verified=json.loads((HERE/'VERIFIED.json').read_text())
lift=json.loads((HERE/'LIFT_RESULT.json').read_text());amp=json.loads((HERE/'AMPLITUDE_RESULT.json').read_text())
pad=json.loads((HERE/'ADHESION_RESULT.json').read_text());teacher=json.loads((HERE/'FOOT_TEACHER_RESULT.json').read_text())
for row in amp['results']:
    path=HERE/f"AMPLITUDE_{row['leg']}_bound{row['bound_rad']:g}.npz"
    with np.load(path,allow_pickle=False) as a:
        x=a['trace'];assert len(x)==2000 and np.isfinite(x).all()
        assert float(x[600:1200,1].mean())==row['hold_lift_mean_native']
        assert (x[600:1200,1].mean()>=.15 and x[:,2].min()>=.8 and x[:,4].max()/main['results'][0]['model']['weight_native']<=.2)==row['pass_']
for row in lift['results']:
    with np.load(HERE/f"LIFT_{row['leg']}_bound{row['bound_rad']:g}.npz",allow_pickle=False) as a:
        assert len(a['trace'])==2000 and float(a['trace'][600:1200,1].mean())==row['actual_lift_hold_mean_native']
for row in pad['results']:
    with np.load(HERE/f"pad_{row['mode']}_dt{row['dt']:g}.npz",allow_pickle=False) as a:
        x=a['trace'];assert len(x)==row['steps'] and np.isfinite(x).all()
        assert float(x[-1,2])==row['vertical_displacement_native']
for row in teacher['results']:
    with np.load(HERE/'FOOT_TEACHER'/f"{row['scene']}_{row['seed']}"/'TRACE.npz',allow_pickle=False) as a:
        assert len(a['position'])==17000
for name in ('LIFT_CONFIG.json','AMPLITUDE_CONFIG.json','ADHESION_CONFIG.json','FOOT_TEACHER_CONFIG.json'):
    config=json.loads((HERE/name).read_text())
    assert all(sha(p)==v for p,v in config['source_sha256'].items())
assert sha(CURRENT['checkpoint'])==CURRENT['sha256']
known_steps=(main['physical_steps']+verified['replay_physical_steps']+lift['physical_steps']+amp['physical_steps']+
             pad['physical_steps']+teacher['physical_steps']+36000+17000)
out=dict(status='ACTION_REACHABILITY_DIAGNOSIS_COMPLETE_NO_TERRAIN_ACCEPTANCE',project='P1_open_source_brain',
   main_flat_diagnostic_pass=sum(r['pass_'] for r in main['results'] if r['scene']=='flat'),main_flat_total=8,
   main_obstacle_diagnostic_pass=sum(r['pass_'] for r in main['results'] if r['scene']!='flat'),main_obstacle_total=16,
   configured_active_joint_angle_limits=0,active_joints=42,configured_position_target_limits=0,
   actuator_force_limits=42,anatomical_joint_compliance='UNKNOWN',
   main_independent_physics_replays=2,main_metric_checks=24,
   amplitude_lift_pass=sum(r['pass_'] for r in amp['results']),amplitude_lift_total=18,
   zero_adhesion_derivative_pass=pad['derivative']['zero_adhesion_pass'],
   single_pad_pass=sum(r['pass_'] for r in pad['results']),single_pad_total=len(pad['results']),
   foot_teacher_flat_pass=sum(r['pass_'] for r in teacher['results'] if r['scene']=='flat'),
   foot_teacher_obstacle_pass=sum(r['pass_'] for r in teacher['results'] if r['scene']!='flat'),
   known_saved_physics_steps=known_steps,initial_interrupted_extra_steps='UNKNOWN_0_TO_16999',
   training_updates=0,new_neural_ticks=0,production_body_sha256=CURRENT['sha256'],production_unchanged=True,
   caveats=['diagnostic gates do not establish missing anatomical angle limits',
            '48-output derivative failed zero-adhesion dynamics equivalence',
            'primitive single-pad bench is not whole-fly or anatomical foot validation',
            'original non-Python assets were not individually hashed before all native runs; compiled model artifacts retained',
            'initial interrupted trial count unknown; not silently treated as zero'])
write(HERE/'FINAL_DIAGNOSTIC.json',out)
print(json.dumps(out,ensure_ascii=False,indent=2))
