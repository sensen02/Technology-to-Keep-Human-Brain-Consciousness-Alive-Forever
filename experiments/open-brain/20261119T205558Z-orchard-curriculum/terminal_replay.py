"""Mix verified ingestion transitions from the original food task into the final SAC stage."""
import json
from pathlib import Path
import numpy as np
import learn_stage
from rewards import analyze as orchard

def read_rows(folder,phase):
    if (folder/'TRAIN_MEASURES.npz').exists():
        return orchard(folder,phase)
    assert folder.parent==learn_stage.ODOR and folder.name in ('LIVE_DETOUR_ODOR_L','LIVE_DETOUR_ODOR_R')
    result=json.loads((folder/'RESULT.json').read_text())
    assert phase==3 and result['meal_pass'] and abs(result['intake']-result['stock_loss'])<1e-9
    with np.load(folder/'BODY.npz') as b:
        obs,meal=b['observation'],b['meal']
        assert np.all(meal[meal[:,4]>0,3]>0)
    first=result['first_intake_s']
    rows=[]
    total=0.
    for i,row in enumerate(obs):
        if row[0]<first-.48:continue
        last=i==len(obs)-1
        end=result['duration_s'] if last else obs[i+1,0]
        amount=float(meal[(meal[:,0]>=row[0])&(meal[:,0]<end),4].sum())
        total+=amount
        reward=12.*amount/.18+4.*last
        rows.append((row[1:15],row[15:17],reward,row[1:15] if last else obs[i+1,1:15],float(last)))
    assert abs(total-result['intake'])<1e-10
    return rows,dict(source='intact controlled wall, archived terminal experience',intake=total,transitions=len(rows))

if __name__=='__main__':
    learn_stage.analyze=read_rows
    learn_stage.main()
