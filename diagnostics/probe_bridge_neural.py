"""Measure engineered stimulation dose on annotated BANC DNa01 pair."""
from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from engine.embodied.adapters import TierConfig,build_neural_tier,NeuralTier,OwnershipLedger
cfg=TierConfig(background_sd_nA=0,weight_scale_uS_per_synapse=1e-6)
b=build_neural_tier(cfg)
ids=np.asarray(b['data']['root_ids']).astype(str)
roots={'left':'720575941535862506','right':'720575941432123640'}
idx={}
for side,r in roots.items():
    g=np.flatnonzero(ids==r)
    assert g.size==1
    local=int(b['local_of_global'][int(g[0])]);assert local>=0
    idx[side]=local
rows=[]
for amp in (.01,.05,.1,.2,.5,1.):
    tier=NeuralTier(b,cfg,OwnershipLedger())
    cur=np.zeros(b['n']);cur[idx['left']]=amp
    sp=tier.advance(600,cur)
    row={'current_nA':amp,'left_hz':float(sp[:,idx['left']].sum()/.3),'right_hz':float(sp[:,idx['right']].sum()/.3)}
    rows.append(row);print(row,flush=True)
out=ROOT/'outputs/neural_motor_bridge/neural_dose.json'
out.write_text(json.dumps({'scope':'engineered conductance dynamics; not measured biological transfer','indices':idx,'fingerprint':b['fingerprint'],'rows':rows},indent=2))
