"""Stdlib AST/exact adaptation/pure mock metadata only; no eval/model/Core reads."""
from pathlib import Path
import ast,copy,json,runpy,sys
HERE=Path(__file__).resolve().parent
for name in ('render.py','check_codeonly.py'):ast.parse((HERE/name).read_text())
ns=runpy.run_path(str(HERE/'render.py'),run_name='_codeonly_combined')
source=ns['build_source']();ast.parse(source)
for text in ["checkpoint=ROOT/'COMBINED.pt'","io.load_checked(checkpoint,r['checkpoint_sha256'],require_final=True)","payload['format_version']=='COMBINED_TURN_V1'","payload['identity']['state_dims']==[128,64,64]","payload['identity']['state_total']==256","(2880,256)","np.array_equal(d['hidden_before'][1:],d['hidden_after'][:-1])","pair=validate_pair(ROOT,r['checkpoint_sha256'])","LARGE {a.condition.upper()} NEURAL TURN","m.geom_rbound[mesh]","mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden","m.vis.global_.offwidth,m.vis.global_.offheight=original_framebuffer"]:assert text in source,text
assert "handover['state_dims']==[128,64,64]" in source
assert not any(isinstance(n,ast.Constant) and n.value==192 for n in ast.walk(ast.parse(source))) and 'dualhidden' not in source
assert "payload['curve'][-1]" not in source and '6a6c23feb9' not in source and "model_format='RIGHT_ONLY_V1'" not in source
root=HERE.parent;ck=root/'COMBINED.pt';digest='a'*64;contract=ns['require_contract'];cases=0
for condition,yaw in [('left',.517),('right',-.535)]:
 run=root/('EVAL_'+condition)
 r=dict(condition=condition,checkpoint=str(ck),checkpoint_sha256=digest,engineering_pass=False,native=dict(actual_native_steps=1232000,post_native_steps=768000),transition=dict(all_exact=True,no_reset=True),yaw_change_rad=yaw)
 a=dict(condition=condition,run=str(run),integrity_all_pass=True,checks={k:True for k in ('full2880_threehidden_decisions','full256_hidden_chain','actual2880_checkpoint_threehidden_exact')},checkpoint_sha256=digest,engineering_pass=False)
 assert contract(r,a,run,ck,digest)['engineering_pass'] is False
 for key,value in [('condition','neutral'),('checkpoint_sha256','b'*64),('integrity_all_pass',False),('checks',{}),('checks',{'chain':False}),('run',str(root/'EVAL_neutral')),('engineering_pass',True),('checks',{'full2880_threehidden_decisions':True,'full256_hidden_chain':True})]:
  wrong=copy.deepcopy(a);wrong[key]=value
  try:contract(r,wrong,run,ck,digest)
  except ValueError:cases+=1
  else:raise AssertionError('accepted bad audit '+key)
 wrong=copy.deepcopy(r);wrong['native']['actual_native_steps']=1
 try:contract(wrong,a,run,ck,digest)
 except ValueError:cases+=1
 else:raise AssertionError('accepted incomplete native')
assert not any(k in sys.modules for k in ('torch','numpy','mujoco','physics','model','model_io'))
result=dict(status='CODEONLY_PASS',syntax=True,exact_adaptation=True,mock_negative_cases=cases,left_and_right_behavioral_failure_renderable=True,same_SHA_pair_guard=True,explicit256_hidden_chain_guard=True,no_combined_training_curve=True,heavy_modules_imported=False,model_loads=0,Core_constructions=0,physics_steps=0,render_frames=0,eval_files_read=0,claim='Conditional renderer preparation only; final component selection, assembled checkpoint and actual results have not been read or executed.')
(HERE/'CODEONLY_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
