"""Fresh-process actual behavior production replay and optimizer/RNG/data audit."""
import argparse
from common import *
from data import training_data,arrays,split
from engine import make,production

def main():
 p=argparse.ArgumentParser();p.add_argument('--step',type=int,choices=(0,500,1000),required=True);args=p.parse_args();cpu1();freeze_proof();path=HERE/'TRAIN'/f'checkpoint_{args.step:06d}.pt';m,payload=load_checked(path,sha(path),require_final=False);checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('saved check failed '+k)
 opt=torch.optim.Adam(m.parameters(),lr=1e-4);opt.load_state_dict(payload['optimizer']);check('Adam_exact_reload',exact(opt.state_dict(),payload['optimizer']))
 torch.set_rng_state(payload['rng']['torch']);generator=np.random.Generator(np.random.PCG64());generator.bit_generator.state=payload['rng']['numpy_generator'];check('RNG_exact_reload',exact(dict(torch=torch.get_rng_state(),numpy_generator=generator.bit_generator.state),payload['rng']))
 check('curve_all_steps',list(range(args.step+1))==[r['step'] for r in payload['curve']]);check('recorded_batch_schedule_length',len(payload['batch_schedule'])==args.step)
 if args.step:
  fresh=np.random.Generator(np.random.PCG64(20261111));expected=[fresh.choice(24,4,replace=False).tolist() for _ in range(args.step)];check('actual_uniform_batch_schedule_exact',expected==payload['batch_schedule']);check('Adam8_exact_counters',len(opt.state)==8 and all(float(v['step'])==args.step for v in opt.state.values()))
 data=training_data()
 with torch.no_grad():turn,latent,hidden=production(m,data['receptors']);scan_turn,scan_latent,scan_hidden=make(m)(data['receptors'])
 for name,a in [('turn',turn),('latent',latent),('hidden',hidden)]:check('production_'+name+'_bitexact',torch.equal(a,payload['replay_train'][name]))
 errors=dict(turn=float((turn-scan_turn).abs().max()),latent=float((latent-scan_latent).abs().max()),hidden=float((hidden-scan_hidden).abs().max()));check('production_scan_tol1e12',max(errors.values())<=1e-12)
 check('same_original_architecture',m.identity()==payload['identity'] and m.identity()['inputs']=='actual local receptor[left,right] only')
 check('parent_scope_preserved',payload['parent_sha256']==PARENT_SHA and payload['deployment_qualified'] is False and payload['config']['lr']==.0001)
 # Regenerate both deterministic long datasets from frozen configs, no model validation on heldout.
 for which in ('train','validation'):
  regenerated,meta=split(which);saved=arrays(HERE/(which+'.npz'));check(which+'_regenerated_arrays_exact',all(np.array_equal(regenerated[k],saved[k]) for k in saved))
 check('motor_checkpoint_unmodified',sha(MOTOR)==MOTOR_SHA);check('no_physics_import',not any(n in sys.modules for n in ('physics','mujoco','student_wave','flygym')))
 save(HERE/'TRAIN'/f'VERIFY_{args.step:06d}.json',dict(all_passed=True,checks=checks,max_errors=errors,step=args.step,checkpoint=str(path),sha256=sha(path),source_hashes={str(p):sha(p) for p in (HERE/'verify_saved.py',HERE/'common.py',HERE/'engine.py',HERE/'data.py')},Core_constructions=0,physics_steps=0,optimizer_updates=0,validation_outputs_evaluated=False))
 print(json.dumps(dict(step=args.step,all_passed=True,checks=len(checks),max_error=max(errors.values()),sha256=sha(path))),flush=True)
if __name__=='__main__':main()
