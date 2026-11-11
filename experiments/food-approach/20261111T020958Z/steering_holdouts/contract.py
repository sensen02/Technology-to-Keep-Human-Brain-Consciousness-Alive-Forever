"""Pure schedules and fail-closed qualification. No physical/backend imports."""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
CASES=('halfleft','halfright','left_to_right','right_to_left','pulse_left','pulse_right')

def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def smooth(t):
 u=np.clip(t,0.,1.);return float(u*u*(3-2*u))
def command_at(case,k):
 if case not in CASES or type(k)is not int or not 0<=k<2880:raise ValueError('registeredcase anddecision0..2879 required')
 if k<960:return np.zeros(2)
 t=(k-960)*.005
 initial={'halfleft':.5,'halfright':-.5,'left_to_right':.5,'right_to_left':-.5,'pulse_left':1.,'pulse_right':-1.}[case]
 turn=initial*smooth(t)
 if case in ('left_to_right','right_to_left') and t>=4.8:turn=initial+( -2*initial)*smooth(t-4.8)
 return np.array([0.,turn],dtype=np.float64)
def pulse(case):
 if case not in CASES:raise ValueError('unknowncase')
 if not case.startswith('pulse'):return np.zeros(42),None
 seed=20261112 if case=='pulse_left' else 20261113
 return np.clip(np.random.default_rng(seed).normal(0,.02,42),-1,1),seed

def yaw_score(case,geometry):
 y=np.unwrap(np.asarray(geometry,float));shift=20 if case.startswith('pulse') else 0
 if y.shape!=(15400+shift,):raise ValueError('complete endpointyaw required')
 a,b,c=y[5799+shift],y[10599+shift],y[15399+shift]
 turns={'halfleft':(1,1),'halfright':(-1,-1),'left_to_right':(1,-1),'right_to_left':(-1,1),'pulse_left':(1,1),'pulse_right':(-1,-1)}[case]
 parts=[float(b-a),float(c-b)];whole=float(c-a)
 passed=all(v*s>=.002 for v,s in zip(parts,turns)) if 'to_' in case else whole*turns[0]>=(.01 if case.startswith('pulse') else .005)
 return dict(whole_rad=whole,first4p8_rad=parts[0],second4p8_rad=parts[1],passed=bool(passed))

def verify_qualification(path,checkpoint,sha):
 path=Path(path).resolve();q=json.loads(path.read_text())
 expected=HERE.parent/'train/TRAIN/checkpoint_005000.pt'
 if Path(checkpoint).resolve()!=expected:raise ValueError('only original nonlinear5000 loader ancestry; changed candidate needs explicit new amendment')
 if not q.get('all_conditions_eligible') or not q.get('saved_trajectory_verification'):raise ValueError('allthree actualconditions must pass')
 if Path(q['checkpoint']).resolve()!=Path(checkpoint).resolve() or q['checkpoint_sha256']!=sha or digest(checkpoint)!=sha:raise ValueError('qualificationcheckpoint mismatch')
 if set(q['conditions'])!={'neutral','left','right'}:raise ValueError('exactthreeconditions required')
 for case,entry in q['conditions'].items():
  if not entry.get('engineering_pass') or not entry.get('integrity_all_pass'):raise ValueError('unqualifiedcondition')
  rp=Path(entry['result']);ap=Path(entry['audit'])
  if digest(rp)!=entry['result_sha256'] or digest(ap)!=entry['audit_sha256']:raise ValueError('qualificationartifact changed')
  r=json.loads(rp.read_text());audit=json.loads(ap.read_text())
  if r['condition']!=case or r['checkpoint_sha256']!=sha or not r['engineering_pass'] or not all(r['criteria'].values()):raise ValueError('result binding')
  if not audit['integrity_all_pass'] or not audit['engineering_pass'] or not all(audit['checks'].values()) or audit['checkpoint_sha256']!=sha or audit['condition']!=case or Path(audit['run']).resolve()!=rp.parent.resolve():raise ValueError('independentaudit binding')
  for name,key in [('trace_1ms.npz','trace'),('neural_decisions.npz','neural_decisions'),('geometry_1ms.npz','geometry'),('runtime_geometry.npz','runtime_geometry')]:
   if digest(rp.parent/name)!=r[key]['sha256']:raise ValueError('qualifiedtrajectory changed')
  for filename,h in r['source_hashes'].items():
   if digest(filename)!=h:raise ValueError('qualifiedsource drift')
 return q

def validate_launch(args):
 for key in ('qualification','checkpoint','out','run_plan'):
  if not Path(getattr(args,key)).is_absolute():raise ValueError('absolute paths required')
 if not args.launch_authorized:raise ValueError('code-only: explicitfuturelaunch required')
 plan=json.loads(Path(args.run_plan).read_text())
 if plan.get('status')!='REGISTERED_FOR_ACTUAL_QUALIFICATION' or plan.get('draft_sha256')!=digest(HERE/'PREREG_DRAFT.json') or plan.get('checkpoint_sha256')!=args.sha256 or plan.get('cases')!=list(CASES):raise ValueError('futurefixedrunplan required')
 if Path(plan['checkpoint']).resolve()!=Path(args.checkpoint).resolve() or Path(plan['qualification']).resolve()!=Path(args.qualification).resolve():raise ValueError('plan binding mismatch')
 out=Path(args.out).resolve()
 if not out.is_relative_to(HERE) or out.exists():raise ValueError('freshscopedoutput required')
 return verify_qualification(args.qualification,args.checkpoint,args.sha256)
