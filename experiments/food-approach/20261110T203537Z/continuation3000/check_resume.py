"""Preflight exact checkpoint resume + sole-lr change, no training/physics."""
from pathlib import Path
import sys,json
import torch,numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from fast_motor import load,sha256
from fast_data import prepare,metrics
from check_fast import equal


def main():
 torch.set_num_threads(1)
 path=ROOT/'TRAIN_FAST_1000/checkpoint_001000.pt';digest='3dd80394239504ec8b88f4cfe07c79569cf08329a9320c749c3c7b25035fda28'
 m,p=load(path,digest);eps,data=prepare(m)
 opt=torch.optim.Adam(m.adapter_parameters(),lr=.001);opt.load_state_dict(p['optimizer'])
 checks=dict(model_exact=equal(m.state_dict(),p['model']),optimizer_exact=equal(opt.state_dict(),p['optimizer']),counters1000=all(float(s['step'])==1000 for s in opt.state.values()),data_same=data==p['config']['data_hashes'],predictions_same=metrics(m,eps)==p['curve'][-1]['metrics_after_update'])
 opt.param_groups[0]['lr']=.0001
 expected=p['optimizer'];expected['param_groups'][0]['lr']=.0001
 checks['sole_lr_change']=equal(opt.state_dict(),expected)
 torch.set_rng_state(p['rng']['torch']);np.random.set_state(p['rng']['numpy'])
 checks['rng_exact']=equal(dict(torch=torch.get_rng_state(),numpy=np.random.get_state()),p['rng'])
 report=dict(all_passed=all(checks.values()),checks=checks,physical_steps=0,optimizer_updates=0,parent_checkpoint=str(path),parent_sha256=digest,source_sha256=sha256(__file__))
 (HERE/'PREFLIGHT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
 if not report['all_passed']:raise ValueError('preflight fail')
if __name__=='__main__':main()
