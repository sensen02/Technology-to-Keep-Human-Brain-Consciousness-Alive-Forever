"""CODE_ONLY-safe conditional same-COMBINED-SHA left/right renderer."""
from pathlib import Path
import hashlib,json,math,runpy
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PARENT_RENDER=ROOT.parent/'neural_right/visual_right/render.py'
REFERENCE=ROOT.parent.parent/'20261111T072012Z/render.py'

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def require_contract(result,audit,run,checkpoint,digest):
 condition=run.name.removeprefix('EVAL_')
 if condition not in ('left','right') or result['condition']!=condition or audit['condition']!=condition:raise ValueError('direction/run mismatch')
 if Path(audit['run']).resolve()!=run.resolve():raise ValueError('audit run mismatch')
 if not audit['integrity_all_pass'] or not audit['checks'] or not all(v is True for v in audit['checks'].values()):raise ValueError('incomplete integrity audit')
 for key in ('full2880_threehidden_decisions','full256_hidden_chain','actual2880_checkpoint_threehidden_exact'):
  if audit['checks'].get(key) is not True:raise ValueError('required256 audit missing: '+key)
 if Path(result['checkpoint']).resolve()!=checkpoint.resolve():raise ValueError('exact COMBINED.pt required')
 if result['checkpoint_sha256']!=digest or audit['checkpoint_sha256']!=digest:raise ValueError('checkpoint identity mismatch')
 if audit['engineering_pass'] is not result['engineering_pass']:raise ValueError('audit status mismatch')
 if result['native']['actual_native_steps']!=1232000 or result['native']['post_native_steps']!=768000:raise ValueError('incomplete native recorder')
 if not result['transition']['all_exact'] or not result['transition']['no_reset']:raise ValueError('handover mismatch')
 return dict(target_degrees=30.,actual_signed_degrees=math.degrees(result['yaw_change_rad'])*(1 if condition=='left' else -1),engineering_pass=result['engineering_pass'])

def validate_pair(root,digest):
 # Runtime only. Both independent actual trajectories must use this one composition SHA.
 loaded={}
 for condition in ('left','right'):
  run=root/('EVAL_'+condition);r=json.loads((run/'RESULT.json').read_text());a=json.loads((root/('VERIFY_'+condition+'.json')).read_text())
  require_contract(r,a,run,root/'COMBINED.pt',digest)
  if 'result_sha256' in a and a['result_sha256']!=sha(run/'RESULT.json'):raise ValueError('audit result drift')
  loaded[condition]=(r,a)
 return loaded

def build_source():
 prereg=json.loads((HERE/'PREREG.json').read_text())
 if sha(PARENT_RENDER)!=prereg['parent_renderer_sha256'] or sha(REFERENCE)!=prereg['reference_sha256']:raise ValueError('frozen renderer drift')
 parent=runpy.run_path(str(PARENT_RENDER),run_name='_frozen_renderer_codeonly')
 source=parent['build_source']()
 def replace(old,new):
  nonlocal source
  if source.count(old)!=1:raise ValueError('combined adaptation mismatch: '+old)
  source=source.replace(old,new)
 replace("assert run==ROOT/'EVAL_right' and Path(a.audit).resolve()==ROOT/'VERIFY_right.json' and a.condition=='right'", "assert a.condition in ('left','right') and run==ROOT/('EVAL_'+a.condition) and Path(a.audit).resolve()==ROOT/('VERIFY_'+a.condition+'.json')")
 replace("checkpoint=ROOT/'TRAIN/checkpoint_002000.pt'", "checkpoint=ROOT/'COMBINED.pt'")
 replace("spec=importlib.util.spec_from_file_location('_bigturn_right_checked_model',ROOT/'model.py');io=importlib.util.module_from_spec(spec);sys.modules[spec.name]=io;spec.loader.exec_module(io)", "sys.path.insert(0,str(ROOT))\n spec=importlib.util.spec_from_file_location('_combined_turn_checked_io',ROOT/'model_io.py');io=importlib.util.module_from_spec(spec);sys.modules[spec.name]=io;spec.loader.exec_module(io)")
 replace("assert payload['format_version']=='RIGHT_ONLY_V1' and payload['curve'][-1]['step']==2000\n assert 'yaw/.06' in payload['identity']['command_domain'] and payload['identity']['command_domain'].startswith('forward0; turn[-1,0]')", """assert payload['format_version']=='COMBINED_TURN_V1' and payload['optimizer'] is None and payload['training_performed'] is False
 assert 'curve' not in payload and 'training_updates' not in payload
 assert payload['identity']==checked_model.identity()
 assert payload['identity']['state_dims']==[128,64,64] and payload['identity']['state_total']==256
 pair=validate_pair(ROOT,r['checkpoint_sha256'])
 assert pair[a.condition][0]==r
 pair_paths=[]
 for condition in ('left','right'):
  pair_paths.extend([ROOT/('EVAL_'+condition)/'RESULT.json',ROOT/('VERIFY_'+condition+'.json')])""")
 replace("REFERENCE,ROOT/'model.py',ROOT/'EVAL_PREREG.json']", "REFERENCE,PARENT_RENDER,PARENT_RENDER.parent/'PREREG.json',ROOT/'model.py',ROOT/'model_io.py',ROOT/'EVAL_PREREG.json']+pair_paths")
 replace("files=[run/n for n in ('RESULT.json','IDENTITY.json','trace_1ms.npz','neural_decisions.npz','geometry_1ms.npz','runtime_geometry.npz')]", "handover=json.loads((run/'HANDOVER.json').read_text())\n assert handover['state_dims']==[128,64,64]\n files=[run/n for n in ('RESULT.json','IDENTITY.json','trace_1ms.npz','neural_decisions.npz','geometry_1ms.npz','runtime_geometry.npz','HANDOVER.json')]")
 replace("with np.load(run/'neural_decisions.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}",  """with np.load(run/'neural_decisions.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
 assert d['hidden_before'].shape==d['hidden_after'].shape==(2880,256)
 assert np.array_equal(d['hidden_before'][1:],d['hidden_after'][:-1])
 assert np.array_equal(d['control_source'],np.r_[np.full(960,2),np.full(1920,3)])""")
 replace("f'BIG RIGHT TURN | VISUAL REVIEW | {phase}'", "f'LARGE {a.condition.upper()} NEURAL TURN | VISUAL REVIEW | {phase}'")
 replace("f'ANN slip {r[\"autonomous\"][\"slip_ratio_max\"]:.6f} (limit .5) | RIGHT {-np.degrees(r[\"yaw_change_rad\"]):.3f} deg (target 30)' ", "f'ANN slip {r[\"autonomous\"][\"slip_ratio_max\"]:.6f} (limit .5) | actual yaw {np.degrees(r[\"yaw_change_rad\"]):+.3f} deg | target 30'")
 replace("target_right_degrees=30.,actual_right_degrees=-float(np.degrees(r['yaw_change_rad'])),checkpoint_sha256=r['checkpoint_sha256'],model_format='RIGHT_ONLY_V1',model_domain='yaw/.06'", "target_degrees=30.,actual_yaw_degrees=float(np.degrees(r['yaw_change_rad'])),checkpoint_sha256=r['checkpoint_sha256'],model_format='COMBINED_TURN_V1',model_domain='coupled forward=relu(turn)/3; signed turn[-1,1]; yaw/.06',state_dims=[128,64,64],state_total=256,same_combined_SHA_both_actual_runs=True")
 return source

def main():
 source=build_source()
 namespace=dict(__name__='_combined_turn_render',__file__=str(Path(__file__).resolve()),ROOT=ROOT,REFERENCE=REFERENCE,PARENT_RENDER=PARENT_RENDER,require_contract=require_contract,validate_pair=validate_pair)
 exec(compile(source,str(Path(__file__).resolve())+'::frozen_renderer_adapted','exec'),namespace)
 namespace['main']()
if __name__=='__main__':main()
