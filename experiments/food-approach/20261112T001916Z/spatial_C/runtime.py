"""Minimal frozen B namespace reuse; source concentration is actual world geometry."""
from contracts import *
import sys
def namespace():
 sys.path.insert(0,str(B));br=module('_spatial_inherited_B_runtime',B/'runtime.py');g=br.namespace();g.update(ROOT=ROOT,IDS=tuple(c['id'] for c in read(ROOT/'PREREG_PROPOSAL.json')['layouts']),verify_frozen=verify,verify_all=verify)
 # Frozen B strictload is intentionally reused; C adds admission/source proof instead of relaxing it.
 return g

def source(script):
 src=(AB/script).read_text()
 def once(old,new):
  nonlocal src
  require(src.count(old)==1,'Csourceadaptation '+old);src=src.replace(old,new)
 if script=='runner.py':
  once("stage='A' if args.trial in A_IDS else 'B'","stage='C'")
  once("tr=sw.Trace(full=True)\n  def tick(action,source):","tr=sw.Trace(full=True);sourcecenter=None\n  def tick(action,source):")
  once("control_source=int(source)))","control_source=int(source),**environment_record(sensors,sourcecenter,sourcecenter is not None)))")
  once("if k==960:\n    before=","if k==960:\n    sourcecenter=anchor_from_head(c.d.xpos[head],c.d.xmat[head].reshape(3,3),local,layout(args.trial))\n    geometry[-1].update(environment_record(geometry[-1]['sensor_xyz'],sourcecenter,True))\n    before=")
  once("sensory=filter.advance(concentration(args.trial,elapsed))","sensory=filter.advance(concentration_at(args.trial,elapsed,c.d.xpos[head]+local@c.d.xmat[head].reshape(3,3).T,sourcecenter))")
  once("rows.append(row)","row.update(sensor_xyz_now=(c.d.xpos[head]+local@c.d.xmat[head].reshape(3,3).T).copy(),source_center_xy=np.zeros(2) if sourcecenter is None else sourcecenter.copy(),source_anchor_valid=sourcecenter is not None)\n   rows.append(row)")
  once("'CONCENTRATION_CLAMP_NOT_SPATIAL_NAVIGATION'","'SPATIAL_GAUSSIAN_ACTUAL_HEAD_FEEDBACK_DEVELOPMENT'")
  once("identity.verify_runtime(ident,c)","result['spatial_metrics']=spatial_metrics(g);result['functional_pass']=None;result['functional']['stimulus_comparison']='DEVELOPMENT_DISTANCES_NOT_B_FUNCTIONAL_GATE'\n  identity.verify_runtime(ident,c)")
 elif script=='audit.py':
  once("s=filter.advance(concentration(trial,(k-960)*.005))","s=filter.advance(concentration_at(trial,(k-960)*.005,g['sensor_xyz'][999+5*k],g['source_center_xy'][5799]))")
  once("identity.verify(ident);verify_frozen()","audit_spatial(check,g,d,r);identity.verify(ident);verify_frozen()")
 else:raise ValueError(script)
 return src

def entry(script,name):
 import types
 from spatial import layout,anchor_from_head,concentration_at,environment_record,arrival_metrics
 g=namespace();ns={'__name__':'_spatial_frozen_'+script,'__file__':str(ROOT/script)};exec(compile(source(script),str(ROOT/script)+'::exact_B_physics_adapter','exec'),ns)
 ns.update({k:v for k,v in g.items() if not k.startswith('__')});ns.update(layout=layout,anchor_from_head=anchor_from_head,concentration_at=concentration_at,environment_record=environment_record,spatial_metrics=spatial_metrics,audit_spatial=audit_spatial)
 return types.FunctionType(ns[name].__code__,ns,ns[name].__name__,ns[name].__defaults__,ns[name].__closure__)

def spatial_metrics(g):
 import numpy as np
 from spatial import arrival_metrics
 ix=slice(5799,None);time=g['time'][ix];distance=g['center_distance'][ix];edge=g['edge_distance'][ix];elapsed=time-5.8
 return dict(d_start_center=float(distance[0]),d_end_center=float(distance[-1]),d_min_center=float(distance.min()),d_start_edge=float(edge[0]),d_end_edge=float(edge[-1]),d_min_edge=float(edge.min()),distance_reduction=float(distance[0]-distance[-1]),**arrival_metrics(time,g['inside_patch'][ix],elapsed))
def audit_spatial(check,g,d,r):
 import numpy as np
 from spatial import layout,anchor_from_head,concentration_at,environment_record
 local=np.asarray(read(AB.parent/'20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json')['sensor_offsets']['local_xyz']);center=anchor_from_head(g['head_xyz'][5799],g['head_rotation'][5799],local,layout(r['condition']))
 check('source_anchor_actual_head_exact',np.array_equal(center,g['source_center_xy'][5799]))
 check('source_fixed_world_after_handover',np.array_equal(g['source_center_xy'][5799:],np.broadcast_to(center,(9601,2))))
 check('prime_source_invalid_not_futureleak',not np.any(g['source_anchor_valid'][:5799]) and np.max(np.abs(g['source_center_xy'][:5799]))==0)
 checks=[]
 for i in range(5799,15400):
  env=environment_record(g['sensor_xyz'][i],center,True);checks.append(env['center_distance']==g['center_distance'][i] and env['edge_distance']==g['edge_distance'][i] and env['inside_patch']==g['inside_patch'][i])
 check('all1ms_distance_inside_exact',all(checks));check('arrival_and_distances_recomputed',spatial_metrics(g)==r['spatial_metrics'])
 check('actual5ms_sensor_sites',np.array_equal(d['sensor_xyz_now'][960:],g['sensor_xyz'][5799:15399:5]))
 check('Cdevelopment_not_arrivalqualification',r['functional_pass'] is None and r['food_success'] is False)
