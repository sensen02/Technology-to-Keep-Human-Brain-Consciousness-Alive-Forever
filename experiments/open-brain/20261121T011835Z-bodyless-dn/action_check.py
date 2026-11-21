"""Inject the paired decoded command into the same observed pose and 5 ms input window."""
from pathlib import Path
import sys,json
import numpy as np
import torch
here=Path(__file__).resolve().parent
body=here.parent/'20261120T155833Z-body-surrogate'
sys.path.insert(0,str(body))
from skeleton import ObservedBody
from train import load
rec=json.loads((here/'results.json').read_text())
io=json.loads((body/'IO.json').read_text())
state=Path('/run/media/sensen/Data1/omnisim-results/implant-site/banc-neck/patch-pitch5/full5/acute-state.npz')
with np.load(state) as z: roots=z['registered_roots'].astype(str).tolist()
net=ObservedBody(body/'weights.pt').eval()
_,pos,touch,hist,_=next(r for r in load(11701) if r[0].endswith('_intact'))
lookup={roots.index(r):c for c,r in enumerate(io['dn_roots'])}
rows=[]
with torch.inference_mode():
 for prob in [1,.8]:
    changes=[];dist=[];flips=[]
    for seed in [713,1211,1847]:
        cases=sorted([r for r in rec['rows'] if r['p']==prob and r['seed']==seed],key=lambda r:r['dn_on'])
        true=np.repeat([roots.index(r) for r in io['dn_roots']],6)
        ix=np.random.default_rng(seed).integers(4,len(pos)-1,len(true))
        preds=[]; cmds=[]
        for case in cases:
            g=np.asarray(case['predicted'])[256:]
            cmd=np.array([lookup.get(i,-1) for i in g])
            obs=hist[ix].clone()
            for i,c in enumerate(cmd):
                if c>=0:obs[i,-1,c]+=1
            preds.append(net(pos[ix-1],pos[ix],touch[ix],obs)[0])
            cmds.append(cmd)
            if case['dn_on']:flips.extend((np.array(cases[0]['predicted'][256:])==true).astype(int)-(g==true).astype(int))
        changes.extend(cmds[0]!=cmds[1])
        dist.extend(torch.linalg.vector_norm(preds[0]-preds[1],dim=-1).mean(1).tolist())
    rows.append(dict(p=prob,command_changes=int(sum(changes)),trials=len(dist),max_marker_mean_um=float(np.max(dist)*1000),
        mean_marker_mean_um=float(np.mean(dist)*1000),formerly_correct_lost=int((np.asarray(flips)>0).sum()),
        formerly_wrong_recovered=int((np.asarray(flips)<0).sum())))
(here/'action.json').write_text(json.dumps(rows,indent=2));print(rows)
