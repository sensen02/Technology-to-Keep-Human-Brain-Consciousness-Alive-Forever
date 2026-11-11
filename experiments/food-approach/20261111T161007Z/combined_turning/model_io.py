"""Strict component-bound constant COMBINED_TURN_V1 storage. Never import in CODE_ONLY."""
from pathlib import Path
import importlib.util,sys,json,hashlib,math
import torch,numpy as np
from model import CombinedTurning,FORMAT
HERE=Path(__file__).resolve().parent
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
REFERENCE=BASE.parent/'20261110T152754Z/NATIVE_ROLE_2500/neural_decisions.npz'
BASE_SHA='246f9fab7d21e2a1332a745fafd4a343b80ad5a70d2cafeeb900e876f7933809'
def sha256(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def exact(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return type(a)==type(b) and a==b
def verify_hashes(hashes):
 for path,h in hashes.items():
  if not Path(path).is_absolute() or sha256(path)!=h:raise ValueError('bound input/source drift '+path)
def jsonable(v):
 if torch.is_tensor(v):return v.detach().cpu().tolist()
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,Path):return str(v)
 if isinstance(v,dict):return {str(k):jsonable(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [jsonable(x) for x in v]
 return v
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
def actual_evidence(spec,digest,direction):
 folder=Path(spec['actual']);r=json.loads((folder/'RESULT.json').read_text());a=json.loads(Path(spec['audit']).read_text())
 if r['checkpoint_sha256']!=digest or a['checkpoint_sha256']!=digest or not a['integrity_all_pass'] or not all(a['checks'].values()):raise ValueError('actual component/checkpoint/audit binding required')
 if 'result_sha256' in a and a['result_sha256']!=sha256(folder/'RESULT.json'):raise ValueError('component audit result drift')
 for key,name in [('trace','trace_1ms.npz'),('neural_decisions','neural_decisions.npz'),('geometry','geometry_1ms.npz'),('runtime_geometry','runtime_geometry.npz')]:
  if sha256(folder/name)!=r[key]['sha256']:raise ValueError('actual component artifact drift '+name)
 native=r['native'];yaw=math.degrees(r['yaw_change_rad']);signed=yaw if direction=='left' else -yaw
 stable=all(r['autonomous']['gates'][k] for k in ('support','nonfoot','com','forward')) and r['autonomous_cycles']['all_legs_completed'] and native['post_nonfoot_peak_native']==0. and native['post_native_steps']==768000 and native['actual_native_steps']==1232000 and native['extra_force_max']==0. and signed>0.
 if not stable:raise ValueError(direction+' component lacks audited stable directional motion; composition execution deferred')
 formal=bool(r['engineering_pass'])
 if bool(a['engineering_pass'])!=formal:raise ValueError('component qualification audit mismatch')
 if spec.get('require_formal_PASS') and not formal:raise ValueError('optional RIGHT replacement must pass every original engineering gate')
 return dict(condition=direction,demonstration_stable_motion=True,angle30_qualified=bool(signed>=30.),engineering_qualified=formal,signed_yaw_deg=yaw,max_slip=r['autonomous']['slip_ratio_max'],source_result=str(folder/'RESULT.json'),source_result_sha256=sha256(folder/'RESULT.json'),source_audit=str(spec['audit']),source_audit_sha256=sha256(spec['audit']),reference_stream=str(folder/'neural_decisions.npz'),reference_stream_sha256=sha256(folder/'neural_decisions.npz'))
def load_component(role,spec,digest):
 if digest is None or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('future component SHA must be supplied explicitly after parent outcome decision')
 if spec.get('expected_sha256') is not None and spec['expected_sha256']!=digest:raise ValueError('fixed component SHA mismatch')
 if sha256(spec['checkpoint'])!=digest:raise ValueError('actual component SHA mismatch')
 io=module('_combined_'+role+'_strict_io',Path(spec['loader']));model,p=io.load_checked(spec['checkpoint'],digest,require_final=True)
 if p['curve'][-1]['step']!=spec['source_step']:raise ValueError('component source training lineage differs from selected fixedfinal')
 if model.identity().get('baseline_sha256')!=BASE_SHA:raise ValueError('component validated original walking base identity required')
 if tuple(model.encoder.weight.shape)!=(64,242) or tuple(model.head.weight.shape)!=(42,64) or tuple(model.head.bias.shape)!=(42,):raise ValueError('canonical branch242->64/GRU64/head42 interface required')
 evidence=actual_evidence(spec,digest,'right' if role=='RIGHT' else 'left')
 hashes={str(Path(spec['loader'])):sha256(spec['loader']),str(Path(spec['checkpoint'])):digest}
 hashes.update(p['config']['source_hashes']);hashes.update(p['config']['data_hashes']);hashes[evidence['source_result']]=evidence['source_result_sha256'];hashes[evidence['source_audit']]=evidence['source_audit_sha256'];hashes[evidence['reference_stream']]=evidence['reference_stream_sha256']
 info=dict(component_selection_scope=spec.get('selection_scope','fixedcomponent_or_strictqualifiedvariant'),loader=str(spec['loader']),loader_sha256=sha256(spec['loader']),checkpoint=str(spec['checkpoint']),sha256=digest,source_training_step=spec['source_step'],source_identity=jsonable(model.identity()),evidence=evidence)
 return model,info,hashes

def components(right_option,left_sha,right_sha=None):
 plan=json.loads((HERE/'COMPONENT_PLAN.json').read_text())
 if right_option not in plan['right_options']:raise ValueError('unregistered right source option')
 rspec=plan['right_options'][right_option];rsha=rspec['expected_sha256'] if right_sha is None else right_sha
 right,rinfo,rhash=load_component('RIGHT',rspec,rsha);left,linfo,lhash=load_component('LEFT',plan['left'],left_sha)
 scope_paths=[HERE/n for n in ('PREREG.json','AMENDMENT_01_LEFT_PENDING_FEATURE_REFINE.json','AMENDMENT_02_CONDITIONAL_RIGHT_FEATURE_REFINE.json','AMENDMENT_03_HONEST_DEMONSTRATION_COMPONENTS.json','AMENDMENT_04_POSTHOC_VISUAL_RIGHT_COMPONENT.json','COMPONENT_PLAN.json')]
 bindings=dict(format_version=FORMAT,right_option=right_option,RIGHT=rinfo,LEFT=linfo,baseline_sha256=BASE_SHA,composition_scope_hashes={str(p):sha256(p) for p in scope_paths},training_performed=False,natural_joint_training=False,qualification=dict(RIGHT_engineering_qualified=rinfo['evidence']['engineering_qualified'],LEFT_engineering_qualified=linfo['evidence']['engineering_qualified'],composition_actual_qualified_at_assembly='NOT_EVALUATED'))
 return right,left,bindings,{**rhash,**lhash}

def load_checked(path,digest,require_final=True):
 path=Path(path).resolve()
 if path!=HERE/'COMBINED.pt' or sha256(path)!=digest:raise ValueError('exact composed checkpoint/SHA required')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p.get('format_version')!=FORMAT or p.get('optimizer') is not None or p.get('training_performed') is not False or 'curve' in p or 'training_updates' in p:raise ValueError('strict constant composition format; no fabricated optimizer/training history')
 verify_hashes(p['source_hashes']);verify_hashes(p['component_source_hashes'])
 if p['bindings_path']!=str(HERE/'COMPONENT_BINDINGS.json') or sha256(p['bindings_path'])!=p['bindings_sha256']:raise ValueError('immutable component bindings required')
 bindings=json.loads(Path(p['bindings_path']).read_text());right,left,current,hashes=components(bindings['right_option'],bindings['LEFT']['sha256'],bindings['RIGHT']['sha256'])
 if current!=bindings or hashes!=p['component_source_hashes']:raise ValueError('component/source/evidence bindings drift')
 model=CombinedTurning(right,left,bindings)
 if p['identity']!=model.identity() or not exact(model.state_dict(),p['state_dict']):raise ValueError('composed tensors must equal exactly assembled trained components')
 model.load_state_dict(p['state_dict'],strict=True)
 if any(q.requires_grad for q in model.parameters()) or not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('all composed weights frozen and finite')
 return model.eval(),p
