"""Dminimal Csource adapters; interventionsenvironment-only, same288neuralchain."""
from contracts import *
def namespace():
 cr=load_C_runtime();g=cr.namespace();g.update(ROOT=ROOT,IDS=tuple(c['id'] for c in trials()),verify_frozen=verify,verify_all=verify);return g

def actor_input(trial,elapsed,z):
 import numpy as np
 cfg=config(trial)
 if cfg['control']=='cut' and elapsed>=3.-1e-10:return np.zeros(2)
 if cfg['control']=='swap':return np.asarray(z)[::-1].copy()
 return np.asarray(z).copy()
def applied_command(trial,k,raw,project):
 cfg=config(trial)
 if cfg['control']!='replay':return project(raw)
 import numpy as np,torch
 binding=read(ROOT/('REPLAY_BINDING_'+trial+'.json'));path=Path(binding['neural_decisions']);require(sha(path)==binding['source_hashes'][str(path)],'donor neuralhash')
 # Small1920/2880 decision table; cache perprocess read once, never use donor42actions directly.
 if trial not in _replay_cache:
  with np.load(path,allow_pickle=False) as z:_replay_cache[trial]=z['command'].copy()
 t=float(_replay_cache[trial][k,1]);return project(torch.tensor([[0.,t]],dtype=torch.float64))
_replay_cache={}
def concentration_at(trial,elapsed,sites,center):
 from spatial import concentration_at as inherited
 return inherited('C_noodor' if config(trial)['control']=='noodor' else trial,elapsed,sites,center)

def source(script):
 cr=load_C_runtime();src=cr.source(script)
 def once(old,new):
  nonlocal src
  require(src.count(old)==1,'Dadaptation '+old);src=src.replace(old,new)
 if script=='runner.py':
  once("stage='C'","stage='D'")
  once("sensory['receptors'][None]","actor_input(args.trial,elapsed,sensory['receptors'])[None]")
  once("command=project(raw)\n    mean,ms=", "command=applied_command(args.trial,k,raw,project)\n    mean,ms=")
  once("rows.append(row)","row.update(actor_receptors=np.zeros(2) if warm else actor_input(args.trial,elapsed,sensory['receptors']),command_override=(not warm and config(args.trial)['control']=='replay'),control_mode_code=control_code(args.trial))\n   rows.append(row)")
  once("result['spatial_metrics']=spatial_metrics(g);", "result['control']=config(args.trial)['control'];result['layout_id']=config(args.trial)['layout_id'];result['replay_closed_loop']=config(args.trial)['control']!='replay';result['cut_precondition']=cut_precondition(rows,args.trial);result['spatial_metrics']=spatial_metrics(g);")
 elif script=='audit.py':
  once("s['receptors'][None]","actor_input(trial,(k-960)*.005,s['receptors'])[None]")
  once("u=project(raw);mean,ms=", "u=applied_command(trial,k,raw,project) if k>=960 else project(raw);mean,ms=")
  once("audit_spatial(check,g,d,r);","audit_spatial(check,g,d,r);audit_interventions(check,t,d,r);")
 return src

def control_code(trial):return ('normal','cut','swap','noodor','replay').index(config(trial)['control'])
def cut_precondition(rows,trial):
 if config(trial)['control']!='cut':return dict(applicable=False)
 import numpy as np
 qualified=[]
 for k in range(960,1560):
  z=np.asarray(rows[k]['receptors']);turn=float(rows[k]['raw_turn']);difference=z[0]-z[1]
  if z.mean()>=.05 and abs(difference)>=.001 and turn*difference>0:qualified.append((k-960)*.005)
 return dict(applicable=True,cut_elapsed_s=3.,effective_response_before_cut=bool(qualified),first_effective_correct_response_elapsed_s=None if not qualified else qualified[0],no_reset=True,filter_continues=True)
def spatial_metrics(g):
 cr=load_C_runtime();return cr.spatial_metrics(g)
def audit_spatial(check,g,d,r):
 import numpy as np
 from spatial import anchor_from_head,environment_record
 local=np.asarray(read(AB.parent/'20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json')['sensor_offsets']['local_xyz']);center=anchor_from_head(g['head_xyz'][5799],g['head_rotation'][5799],local,config(r['condition']))
 check('Dsource_anchor_actualhead_exact',np.array_equal(center,g['source_center_xy'][5799]));check('Dfixedworldsource',np.array_equal(g['source_center_xy'][5799:],np.broadcast_to(center,(9601,2))));check('Dprime_source_invalid',not np.any(g['source_anchor_valid'][:5799]) and np.max(np.abs(g['source_center_xy'][:5799]))==0)
 check('Dactual1msdistanceinside',all(environment_record(g['sensor_xyz'][i],center,True)['center_distance']==g['center_distance'][i] and environment_record(g['sensor_xyz'][i],center,True)['inside_patch']==g['inside_patch'][i] for i in range(5799,15400)))
 check('D201spanarrival_recomputed',spatial_metrics(g)==r['spatial_metrics']);check('Dactual5mssamples',np.array_equal(d['sensor_xyz_now'][960:],g['sensor_xyz'][5799:15399:5]))

def audit_interventions(check,t,d,r):
 import numpy as np
 cfg=config(r['condition']);trial=r['condition'];expected=np.zeros((2880,2))
 for k in range(960,2880):expected[k]=actor_input(trial,(k-960)*.005,d['receptors'][k])
 check('actual_actor_receptors_intervention_exact',np.array_equal(expected,d['actor_receptors']));check('cut_keeps_actual_filter_chain',np.array_equal(d['filter_before'][961:],d['filter_after'][960:-1]))
 check('control_mode_codes',np.all(d['control_mode_code']==control_code(trial)));check('override_ONLY_explicit_replay',np.array_equal(d['command_override'],np.r_[np.zeros(960,dtype=bool),np.full(1920,cfg['control']=='replay')]))
 check('cut_precondition_reported_not_cherrypicked',cut_precondition([{k:d[k][i] for k in ('receptors','raw_turn')} for i in range(2880)],trial)==r['cut_precondition'])
 if cfg['control']=='replay':
  binding=read(ROOT/('REPLAY_BINDING_'+trial+'.json'));folder=Path(binding['donor_result']).parent
  with np.load(folder/'trace_1ms.npz',allow_pickle=False) as z:
   maxerr=max(float(np.max(np.abs(t[k]-z[k]))) for k in ('qpos','qvel','target','action'));bitexact=all(np.array_equal(t[k],z[k]) for k in ('qpos','qvel','target','action'))
  check('replay_actualbody_matches_donor_tol1e10',maxerr<=1e-10);check('replay_not_new_closedloop',r['replay_closed_loop'] is False)
  with (ROOT/'trials'/trial/'REPLAY_COMPARISON.json').open('x') as f:json.dump(dict(donor=cfg['donor'],recipient=trial,max_abs_error=maxerr,body_arrays_bitexact=bitexact,declared_tolerance=1e-10,not_newclosedloop=True),f,indent=2);f.write('\n')
  # Actual audit log retains comparator magnitude separately; strictchecks never fabricate equality.
  check('replay_donor_stable_bound',read(binding['donor_result'])['functional']['stable_body'] is True)
 else:check('normal_cut_swap_notcommandreplay',r['replay_closed_loop'] is True)

def entry(script,name):
 import types,sys
 sys.path.insert(0,str(C));from spatial import anchor_from_head,environment_record
 g=namespace();ns={'__name__':'_D_fromfrozen_C_'+script,'__file__':str(ROOT/script)};exec(compile(source(script),str(ROOT/script)+'::explicitDinterventions','exec'),ns);ns.update({k:v for k,v in g.items() if not k.startswith('__')});ns.update(layout=config,config=config,anchor_from_head=anchor_from_head,environment_record=environment_record,concentration_at=concentration_at,actor_input=actor_input,applied_command=applied_command,control_code=control_code,cut_precondition=cut_precondition,spatial_metrics=spatial_metrics,audit_spatial=audit_spatial,audit_interventions=audit_interventions)
 return types.FunctionType(ns[name].__code__,ns,ns[name].__name__,ns[name].__defaults__,ns[name].__closure__)
