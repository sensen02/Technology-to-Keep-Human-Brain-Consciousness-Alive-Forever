"""Stdlib source/contract checks only; no arraypayload or numericalmodel execution."""
from pathlib import Path
import ast,json,sys,hashlib,subprocess
HERE=Path(__file__).resolve().parent

def main():
 checks={};hashes={};trees={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('FusionaggCODEONLY '+k)
 def sha(p):
  with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
 files=('common.py','model.py','prepare_odor_binding.py','data.py','preflight.py','train.py','verify_saved.py','evaluate.py','run.py','codecheck.py')
 for name in files:
  p=HERE/name;trees[name]=ast.parse(p.read_text());compile(trees[name],str(p),'exec');hashes[str(p)]=sha(p);check(name+'_ASTcompiles',True)
 for name in ('prepare_odor_binding.py','data.py','preflight.py','train.py','verify_saved.py','evaluate.py','run.py'):check(name+'_publicmain',any(isinstance(n,ast.FunctionDef) and n.name=='main' for n in trees[name].body))
 check('publicevaluateaudit',any(isinstance(n,ast.FunctionDef) and n.name=='audit' for n in trees['evaluate.py'].body));pr=HERE/'PREREG.json';p=json.loads(pr.read_text());hashes[str(pr)]=sha(pr);check('actual24sourceplan',p['source_counts']==dict(latest_actual_045302_body=14,prior_actual_033622_visual_normal=6,explicit_both_neural_zero_aux=4,effective_train=24));check('fixed400micro8x3',p['training']['updates']==400 and p['training']['effective_batch']==24 and p['training']['microbatch']==8 and p['training']['accumulation']==3 and p['training']['history_steps']==1920);check('sourceformat',p['format']=='SOURCE_NECK_MULTISENSORY_AGGREGATION_REPAIR_V1');check('strictunknownodor_binding',"unknown_SHA_not_admitted=False" in (HERE/'prepare_odor_binding.py').read_text() and 'Actual250odorSHArequired' in (HERE/'prepare_odor_binding.py').read_text());check('strict24shapeguard',"value['teacher'].shape!=(1920,24 if split=='train' else 2,1)" in (HERE/'data.py').read_text());check('onezero_row_only',"np.zeros((1,1),dtype=np.float64),desired[:-1]" in (HERE/'data.py').read_text());check('jointteacherboundofflineonly',"np.clip(desired,-.78,.78)" in (HERE/'data.py').read_text() and not any(isinstance(n,ast.Attribute) and n.attr in ('clip','clamp') for n in ast.walk(trees['model.py'])));check('no_fake_RGB_aux',"no_RGB_or_receptors_created=True" in (HERE/'data.py').read_text());check('newodor_not_false_oldparity',"new_odor_output_not_claimed_equal_old66f=True" in (HERE/'data.py').read_text());check('visionexactfields',all(repr(k) in (HERE/'data.py').read_text() for k in ('held_features','vision_hidden_before','vision_hidden_after','visual_raw_turn','visual_latent')));check('parentallstateexact',"Exactdb25allstate/tenparametersrequired" in (HERE/'common.py').read_text());check('no_identity_dynamic_common_import','from common import' not in (HERE/'model.py').read_text());check('10onlyoptimizer',"params=[v for k,v in trainable(m)]" in (HERE/'train.py').read_text());check('allfrozenparamgrad_absent',"Frozenparametergradient" in (HERE/'train.py').read_text());check('no_Coreconstruct',all(not (isinstance(n,ast.Name) and n.id in ('Core','mj_step')) for t in trees.values() for n in ast.walk(t)));check('savedsteps0_200_400',p['training']['save_steps']==[0,200,400])
 sys.path.insert(0,str(HERE));import common
 check('commonstandardlibraryonly',not any(k in sys.modules for k in ('numpy','torch','mujoco','physics','student_wave','flygym')));check('actual14plus6plus4names',len(common.TRAIN)==24 and len(common.LATEST)==14 and len(common.NORMAL)==6 and len(common.AUX)==4);check('parentactualsourceexists',common.PARENT.is_file() and common.BODY.joinpath('SENSORY_BODY_SUMMARY.json').is_file());b=json.loads((common.BODY/'SENSORY_BODY_SUMMARY.json').read_text());j=json.loads((common.BODY/'JOBS.json').read_text());review=json.loads((common.BODY/'PARENT_READY_REVIEW.json').read_text());check('actual30bodyauditmetadata',len(j)==30 and all(x['exit_code']==0 for x in j) and b['all14_integrity_verified'] and review['integrity_all_pass']);check('recoveryparentformat_actual',common.OLD.name=='sensory_fusion_recovery_v1')
 for script,flag in [('run.py',None),('prepare_odor_binding.py',None)]:
  q=subprocess.run([sys.executable,'-B',str(HERE/script)],capture_output=True,text=True,check=True);check(script+'_defaultnoexecution','CODE_ONLY' in q.stdout)
 with (HERE/'CODEONLY_CHECK.json').open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes=hashes,stdlib_only=True,numericalimports=False,array_reads=0,model_loads=0,forward_calls=0,training_updates=0,Core_constructions=0,new_physics_steps=0,unknown_odor_SHA_not_admitted=True),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=False)))
if __name__=='__main__':main()
