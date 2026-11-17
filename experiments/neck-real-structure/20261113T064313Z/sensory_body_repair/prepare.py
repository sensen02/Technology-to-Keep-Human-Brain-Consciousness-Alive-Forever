"""Stdlib bind actual independentlyverified finalodor250 andfusion400; unknownSHAs rejected."""
import argparse,ast
from runtime import *
def declared_format(folder,expected):
 tree=ast.parse((folder/'common.py').read_text());values=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FORMAT' for t in n.targets)]
 if values!=[expected]:raise ValueError('StrictactualsourceFORMAT '+str(folder))
def completed_proof(folder,step,digest,expected_format):
 checkpoint=folder/'TRAIN'/('checkpoint_'+str(step).zfill(6)+'.pt');savedpath=folder/'TRAIN'/('VERIFY_'+str(step).zfill(6)+'.json');declared_format(folder,expected_format)
 if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest) or sha(checkpoint)!=digest:raise ValueError('Actual final checkpoint bytes '+str(folder))
 jobs=read(folder/'JOBS.json');result=read(folder/'EVALUATION/RESULT.json');audit=read(folder/'EVALUATION/VERIFY.json');saved=read(savedpath)
 if len(jobs)!=7 or not all(x['exit_code']==0 for x in jobs):raise ValueError('Actual seven pipeline stages must complete '+str(folder))
 if result['checkpoint_sha256']!=digest or audit['checkpoint_sha256']!=digest or saved['sha256']!=digest or audit['result_sha256']!=sha(folder/'EVALUATION/RESULT.json'):raise ValueError('Stored model/eval SHA identity '+str(folder))
 if not audit['integrity_all_pass'] or not all(audit['checks'].values()) or not saved['all_passed'] or not all(saved['checks'].values()):raise ValueError('Independentstorage/metric proof '+str(folder))
 hashes={}
 for proof in (read(folder/'FROZEN.json'),result,audit,saved):
  for key in ('source_hashes','data_hashes'):
   for path,h in proof.get(key,{}).items():
    if sha(path)!=h:raise ValueError('Ancestor input drift '+path)
    hashes[path]=h
 for p in (checkpoint,savedpath,folder/'common.py',folder/'model.py',folder/'FROZEN.json',folder/'JOBS.json',folder/'EVALUATION/RESULT.json',folder/'EVALUATION/VERIFY.json'):hashes[str(p)]=sha(p)
 return hashes

def main():
 p=argparse.ArgumentParser();p.add_argument('--bind-completed-calibrations',action='store_true');p.add_argument('--brain-sha256');p.add_argument('--odor-sha256');a=p.parse_args()
 if not a.bind_completed_calibrations:print('CODE_ONLY_UNBOUND no weights admitted');return
 if any((ROOT/n).exists() for n in ('INPUT_BINDINGS.json','FROZEN.json')):raise FileExistsError('No blindsource rebind')
 proof=completed_proof(ODOR_DIR,250,a.odor_sha256,ODOR_FORMAT);proof.update(completed_proof(CAL,400,a.brain_sha256,FORMAT))
 binding=read(CAL/'PARENT_ODOR_BINDING.json')
 if binding['format']!=ODOR_FORMAT or binding['step']!=250 or binding['sha256']!=a.odor_sha256 or Path(binding['checkpoint']).resolve()!=ODOR or binding['unknown_SHA_not_admitted'] is not False:raise ValueError('Fusion used exactactualnewodor250')
 for path,h in binding['source_hashes'].items():
  if sha(path)!=h:raise ValueError('Fusion odorlineage drift '+path)
  proof[path]=h
 proof[str(CAL/'PARENT_ODOR_BINDING.json')]=sha(CAL/'PARENT_ODOR_BINDING.json')
 if sha(VISION)!=VISION_SHA or sha(MOTOR)!=MOTOR_SHA or sha(BINDING)!=BINDING_SHA:raise ValueError('Unchangedvisual/motor/neckidentity')
 checked=read(ROOT/'CODEONLY_CHECK.json')
 if not checked['all_passed'] or not all(checked['checks'].values()):raise ValueError('ActualCODEONLY contracts required')
 for path,h in checked['source_hashes'].items():
  if sha(path)!=h:raise ValueError('Body code drift '+path)
  proof[path]=h
 names=('PREREG.json','runtime.py','adapter.py','prepare.py','runner.py','audit.py','preflight.py','summary.py','review.py','orchestrate.py','check_codeonly.py','CODEONLY_CHECK.json','OLD_SOURCE_IDENTITIES.json')
 paths=[ROOT/n for n in names]+[OLD_BODY/n for n in ('PREREG.json','runtime.py','runner.py','audit.py','preflight.py','summary.py','review.py','orchestrate.py','SENSORY_BODY_SUMMARY.json','PARENT_READY_REVIEW.json','FROZEN.json','JOBS.json')]+[VISION,MOTOR,BINDING,PEAK_SOURCE,HEAD_AUDIT]+[NEUTRAL/n for n in ('IDENTITY.json','RESULT.json','trace_1ms.npz','neural_decisions.npz')]+list(BASE.glob('*.py'))+list((BASE/'sources').glob('*.py'))
 paths.extend(ODOR_DIR.glob('*.py'));paths.extend(CAL.glob('*.py'))
 for path in paths:
  if not path.is_file():raise FileNotFoundError(path)
  proof[str(path)]=sha(path)
 for row in read(BINDING)['source_files']:
  if sha(row['path'])!=row['sha256']:raise ValueError('Source neck drift')
  proof[row['path']]=row['sha256']
 save(ROOT/'INPUT_BINDINGS.json',dict(kind=KIND,summary_kind=SUMMARY_KIND,brain_checkpoint=str(BRIDGE),brain_sha256=a.brain_sha256,brain_final_step=400,brain_format=FORMAT,visual_checkpoint=str(VISION),visual_sha256=VISION_SHA,odor_checkpoint=str(ODOR),odor_sha256=a.odor_sha256,odor_final_step=250,odor_format=ODOR_FORMAT,motor_sha256=MOTOR_SHA,neck_sha256=BINDING_SHA,unknown_SHA_not_admitted=False,functional_pass_claim=False,new_physics_steps=0));proof[str(ROOT/'INPUT_BINDINGS.json')]=sha(ROOT/'INPUT_BINDINGS.json');save(ROOT/'FROZEN.json',dict(kind='MULTISENSORY_NEURAL_REPAIR_BOUND_BEFORE_PHYSICS',source_hashes=proof,model_loads=0,Core_constructions=0,new_physics_steps=0));print('ACTUAL_NEWODOR250_FUSION400_BOUND_UNCHANGED_BODY_NO_PHYSICS')
if __name__=='__main__':main()
