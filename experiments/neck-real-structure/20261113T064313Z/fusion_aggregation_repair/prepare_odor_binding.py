"""Parent-only qualify actual completedodor250 before any fusion arrays/model execution."""
import argparse,ast
from common import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--bind-completed-odor',action='store_true');p.add_argument('--odor-sha256');a=p.parse_args()
 if not a.bind_completed_odor:print('CODE_ONLY unknownodorSHA notadmitted');return
 digest=a.odor_sha256;path=ODORCAL/'TRAIN/checkpoint_000250.pt'
 if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest) or sha(path)!=digest:raise ValueError('Actual250odorSHArequired')
 if (HERE/'PARENT_ODOR_BINDING.json').exists():raise FileExistsError('Inspect before rebind')
 tree=ast.parse((ODORCAL/'common.py').read_text());formats=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and n.value=='ODOR_NEURAL_OFFSET_REPAIR_V1']
 if not formats:raise ValueError('ActualodorcommonFORMATnotestablished')
 jobs=read(ODORCAL/'JOBS.json');trained=read(ODORCAL/'TRAIN/RESULT.json');saved=read(ODORCAL/'TRAIN/VERIFY_000250.json');r=read(ODORCAL/'EVALUATION/RESULT.json');v=read(ODORCAL/'EVALUATION/VERIFY.json')
 if len(jobs)!=7 or not all(j['exit_code']==0 for j in jobs) or trained['updates']!=250 or trained['sha256']!=digest or saved['sha256']!=digest or not saved['all_passed'] or not all(saved['checks'].values()) or r['checkpoint_sha256']!=digest or v['checkpoint_sha256']!=digest or not v['integrity_all_pass'] or not all(v['checks'].values()) or v['result_sha256']!=sha(ODORCAL/'EVALUATION/RESULT.json'):raise ValueError('Actualodor250allstorage/evaluationproofrequired')
 hashes={}
 for proof in (read(ODORCAL/'FROZEN.json'),saved,r,v):
  for key in ('source_hashes','data_hashes'):
   verify(proof.get(key,{}));hashes.update(proof.get(key,{}))
 for p in [ODORCAL/n for n in ('common.py','model.py','PREREG.json','FROZEN.json','JOBS.json','TRAIN/RESULT.json','TRAIN/VERIFY_000250.json','EVALUATION/RESULT.json','EVALUATION/VERIFY.json')]+[path,HERE/'prepare_odor_binding.py']:hashes[str(p)]=sha(p)
 save(HERE/'PARENT_ODOR_BINDING.json',dict(format='ODOR_NEURAL_OFFSET_REPAIR_V1',step=250,checkpoint=str(path),sha256=digest,source_hashes=hashes,unknown_SHA_not_admitted=False,actualall7stagesexit0=True,saved_and_evaluation_independently_verified=True,new_fusion_model_loads=0,new_physics_steps=0));print(json.dumps(dict(path=str(HERE/'PARENT_ODOR_BINDING.json'),odor_sha256=digest,model_loads=0)))
if __name__=='__main__':main()
