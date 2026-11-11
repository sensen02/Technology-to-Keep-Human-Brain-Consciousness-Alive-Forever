"""Reconstruct actual RNG transition caused by save-time strictload; no optimization."""
import json
import torch,numpy as np
from model import *
def main():
 torch.set_num_threads(1)
 a=HERE/'TRAIN/checkpoint_001000.pt';b=HERE/'TRAIN/checkpoint_002000.pt'
 p=torch.load(a,map_location='cpu',weights_only=False);q=torch.load(b,map_location='cpu',weights_only=False)
 torch.set_rng_state(p['rng']['torch']);np.random.set_state(p['rng']['numpy'])
 m,_=load_checked(a,sha256(a),require_final=False)
 consumed=dict(torch=torch.get_rng_state(),numpy=np.random.get_state())
 curve=[json.loads(r) for r in (HERE/'TRAIN/CURVE.jsonl').read_text().splitlines()]
 checks=dict(saved1000_rng_to_final2000_exact_after_actual_loader=exact(consumed,q['rng']),numpy_unmodified=exact(p['rng']['numpy'],q['rng']['numpy']),actual2000_updates=len(curve)==2000,final_curve_exact=exact(curve,q['curve']),no_dropout=not any(isinstance(v,torch.nn.modules.dropout._DropoutNd) for v in m.modules()))
 r=dict(all_passed=all(checks.values()),checks=checks,loader_consumed_torch_rng=not torch.equal(p['rng']['torch'],consumed['torch']),maximum_actual_gradient_norm=max(z['gradient_norm'] for z in curve),clip_active_updates=sum(z['gradient_norm']>1. for z in curve),limitation='save-time model construction consumes RNG and live RNG was not restored; actual deterministic training has no stochastic operations; saved RNG itself reloads exactly, but this is not a claim of arbitrary stochastic exact continuation',physics_steps=0,optimizer_updates=0,checkpoint_hashes={str(v):sha256(v) for v in (a,b)})
 (HERE/'TRAIN/RNG_AUDIT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
 if not r['all_passed']:raise ValueError('actual RNG transition mismatch')
if __name__=='__main__':main()
