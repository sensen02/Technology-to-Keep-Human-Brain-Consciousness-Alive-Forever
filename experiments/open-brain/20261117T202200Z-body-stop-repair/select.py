from common import *
from fit_plus_fast import load
from policy_repair import read

torch.set_num_threads(1)
for group in ('DEV_ROUTING','DEV_FIT'):assert (HERE/group/'RESULT.json').exists()
dev,_=load(PARENT/'TEACHER_DEV');x=torch.from_numpy(dev['x']);y=torch.from_numpy(dev['y'])
records=[]
for file,group in (('ROUTING_FIXED.pt','DEV_ROUTING'),('FIT_REPAIR/epoch030.pt','DEV_FIT')):
 model=read(HERE/file);base=torch.from_numpy(model.rhythmic(dev['phase']))
 with torch.inference_mode():pred,_,_=model(x,base);rmse=float(torch.sqrt(((pred-y)**2).mean()))
 results=json.loads((HERE/group/'RESULT.json').read_text())['results']
 records.append(dict(checkpoint=str(HERE/file),sha256=sha(HERE/file),group=group,all_pass=all(r['pass_'] for r in results),
   pass_count=sum(r['pass_'] for r in results),dev_joint_rmse=rmse))
choice=min(records,key=lambda r:(not r['all_pass'],-r['pass_count'],r['dev_joint_rmse']))
write(HERE/'SELECTION.json',dict(**choice,candidates=records,selection='Explicit pre-held revision SELECTION_REVISION: all physical gates, then physical pass count, then supervised dev RMSE. Neither candidate is globally qualified if all_pass=false. Frozen before new held2039/2711/3253.'))
print(choice)
