"""Stdlib only: syntax, exact adaptation, pure fake metadata contracts; no model/Core."""
from pathlib import Path
import ast,runpy,copy,json,sys
HERE=Path(__file__).resolve().parent
for name in ('render.py','check_codeonly.py'):ast.parse((HERE/name).read_text())
ns=runpy.run_path(str(HERE/'render.py'),run_name='_codeonly_import')
source=ns['build_source']();ast.parse(source)
assert '6a6c23feb9' not in source and "io.load_checked(checkpoint,r['checkpoint_sha256'],require_final=True)" in source
assert "payload['format_version']=='RIGHT_ONLY_V1'" in source and "'yaw/.06' in payload['identity']['command_domain']" in source
assert "cam.distance=max(11.,radius/" in source and 'm.geom_rbound[mesh]' in source
assert "mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden" in source
assert "m.vis.global_.offwidth,m.vis.global_.offheight=original_framebuffer" in source
root=HERE.parent;run=root/'EVAL_right';ck=root/'TRAIN/checkpoint_002000.pt';digest='a'*64
r=dict(condition='right',checkpoint=str(ck),checkpoint_sha256=digest,engineering_pass=False,native=dict(actual_native_steps=1232000,post_native_steps=768000),transition=dict(all_exact=True,no_reset=True),yaw_change_rad=-.4)
a=dict(condition='right',run=str(run),integrity_all_pass=True,checks={'fake_integrity':True},checkpoint_sha256=digest,engineering_pass=False)
contract=ns['require_contract'];assert contract(r,a,run,ck,digest)['engineering_pass'] is False
bad=[('condition','left'),('checkpoint_sha256','b'*64),('integrity_all_pass',False),('checks',{}),('checks',{'one':False}),('run',str(root/'EVAL_neutral')),('engineering_pass',True)]
for key,value in bad:
 wrong=copy.deepcopy(a);wrong[key]=value
 try:contract(r,wrong,run,ck,digest)
 except ValueError:pass
 else:raise AssertionError('accepted bad audit '+key)
wrong=copy.deepcopy(r);wrong['native']['actual_native_steps']=1
try:contract(wrong,a,run,ck,digest)
except ValueError:pass
else:raise AssertionError('accepted incomplete native')
assert not any(k in sys.modules for k in ('torch','numpy','mujoco','physics','model'))
result=dict(status='CODEONLY_PASS',syntax=True,frozen_reference_sha256=ns['sha'](ns['REFERENCE']),exact_adaptation=True,mock_negative_cases=8,behavioral_failure_preserved=True,heavy_modules_imported=False,model_loads=0,Core_constructions=0,physics_steps=0,render_frames=0,eval_files_read=0,claim='Preparation only. Actual final checkpoint/evaluation/audit/render not verified or executed.')
(HERE/'CODEONLY_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
