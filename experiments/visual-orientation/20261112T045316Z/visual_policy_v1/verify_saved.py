"""Fresh-process actual RGB production replay, optimizer and pair RNG proof."""
import argparse
from data import *
from model import production
def main():
 cpu1();frozen();ap=argparse.ArgumentParser();ap.add_argument('--step',type=int,choices=(0,250,500),required=True);args=ap.parse_args();path=HERE/'TRAIN'/f'checkpoint_{args.step:06d}.pt';m,p=load_checked(path,sha(path),require_final=False);rgb,_=library();d=arrays('train');x,_,_=batch(rgb,d,p['replay_indices']);checks={}
 with torch.no_grad():turn,latent,h=production(m,x)
 for k,v in [('turn',turn),('latent',latent),('hidden',h)]:checks['same_shape_production_'+k+'_bitexact']=torch.equal(v,p['replay'][k])
 generator=np.random.Generator(np.random.PCG64(20261112));ids=[generator.choice(12,2,replace=False).tolist() for _ in range(args.step)]
 checks['all_recorded_pairIDs_exact']=ids==p['pair_schedule'];checks['all_expanded_batches_exact']=[expand(v) for v in ids]==p['batch_schedule'];checks['sampler_rng_exact']=exact(generator.bit_generator.state,p['rng']['numpy']);checks['pair_table_exact']=p['pair_table']==pair_table();checks['actual500_only_final']=p['step']==args.step
 if not all(checks.values()):raise ValueError('saved visual replay failed '+str(checks))
 save(HERE/'TRAIN'/f'VERIFY_{args.step:06d}.json',dict(all_passed=True,step=args.step,sha256=sha(path),checks=checks,Core_constructions=0,motor_model_loads=0,production_replay_path='same batch4/framebyframeCNN/5msGRU',source_hashes={str(HERE/'verify_saved.py'):sha(HERE/'verify_saved.py')}));print(json.dumps(dict(all_passed=True,step=args.step,checks=len(checks))))
if __name__=='__main__':main()
