"""Parent-only fixed600 effective8/micro4x2 ANN residual calibration, zero physics."""
import argparse
import json
import subprocess
import sys
import time
from sources import ROOT,read,save,verify,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-parent-training',action='store_true');a=p.parse_args()
    if not a.execute_parent_training:p.error('Parent finite600 release required')
    from learning import cpu1,rng_snapshot,rng_equal,objective,adapter_anchor,metrics,resource_gate
    cpu1()
    import torch,numpy as np,random
    import common,model
    from data import prepare
    binding=read(ROOT/'INPUT_BINDINGS.json');pre=read(ROOT/'PREFLIGHT.json')
    if not pre['all_passed'] or pre['timing_status']!='READY_FIXED600' or pre['source_hashes']!=binding['source_hashes'] or pre['data_hashes']!=binding['data_hashes']:raise ValueError('Actual same-input FULL1920preflight required')
    verify(binding['source_hashes']);verify(binding['data_hashes'])
    out=ROOT/'TRAIN';out.mkdir(exist_ok=False)
    torch.manual_seed(20261113);np.random.seed(20261113);random.seed(20261113)
    m,oldpayload=common.initialize();scan=model.make_scan(m);data=prepare(m);params=list(m.adapter_parameters());anchor=adapter_anchor(m);opt=torch.optim.Adam(params,lr=3e-4)
    config=dict(kind='fixed600_independent_body_command_calibration',steps=600,lr=3e-4,effective_batch=8,microbatch=4,seed=20261113,source_hashes={**binding['source_hashes'],str(ROOT/'INPUT_BINDINGS.json'):sha(ROOT/'INPUT_BINDINGS.json'),str(ROOT/'PREFLIGHT.json'):sha(ROOT/'PREFLIGHT.json')},data_hashes=binding['data_hashes'],initial_anchor=anchor,prereg=read(ROOT/'PREREG.json'))
    save(out/'IDENTITY.json',dict(kind=config['kind'],steps=600,source_hashes=config['source_hashes'],data_hashes=config['data_hashes'],architecture=m.identity(),actual8data_proofs=data['proofs'],physical_steps=0))
    started=time.monotonic();curve=[dict(step=0,metrics=metrics(scan,data),wall_s=0.)]
    saved_rng=rng_snapshot();digest=common.save_checkpoint(out/'checkpoint_000000.pt',m,opt,config,curve,0,saved_rng)
    from preflight import max_error
    restored,stored=common.load_checkpoint(out/'checkpoint_000000.pt',digest,final=False)
    if max_error(m.state_dict(),restored.state_dict())!=0 or max_error(opt.state_dict(),stored['optimizer_state_dict'])!=0 or not rng_equal(saved_rng,stored['rng']):raise ValueError('Saved0 model/Adam/RNG roundtrip exactrequired')
    del restored,stored
    subprocess.run([sys.executable,'-B',str(ROOT/'verify_saved.py'),'--checkpoint',str(out/'checkpoint_000000.pt'),'--sha256',digest,'--execute-parent-saved-verify'],check=True)
    with (out/'TRAINING.jsonl').open('x') as log:
        log.write(json.dumps(curve[0])+'\n');log.flush()
        for step in range(1,601):
            resource_gate(started);opt.zero_grad(set_to_none=True);value=0.;components={}
            for lo in (0,4):
                residual,h=scan(data['features'][:,lo:lo+4]);loss,terms=objective(residual,data['oldmean'][:,lo:lo+4],data['y'][:,lo:lo+4],data['v1'][lo:lo+4],m,anchor);loss.backward();value+=float(loss.detach())
                for key,number in terms.items():components[key]=components.get(key,0.)+number
                del residual,h,loss
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):raise ValueError('All8 finite gradients required')
            if any(p.requires_grad or p.grad is not None for p in m.old.parameters()):raise ValueError('Frozen old789f gradient violation')
            opt.step();row=dict(step=step,loss=value,**components,wall_s=time.monotonic()-started,peak_RSS_MiB=resource_gate(started))
            curve.append(row);log.write(json.dumps(row)+'\n');log.flush()
            if step%25==0:print(json.dumps(row),flush=True)
            if step%50==0:verify(config['source_hashes']);verify(config['data_hashes'])
            if step in (300,600):
                row['metrics_after_update']=metrics(scan,data);checkpoint=out/f'checkpoint_{step:06d}.pt';saved_rng=rng_snapshot();digest=common.save_checkpoint(checkpoint,m,opt,config,curve,step,saved_rng)
                restored,stored=common.load_checkpoint(checkpoint,digest,final=False)
                if max_error(m.state_dict(),restored.state_dict())!=0 or max_error(opt.state_dict(),stored['optimizer_state_dict'])!=0 or not rng_equal(saved_rng,stored['rng']):raise ValueError('Saved model/Adam/RNG roundtrip exactrequired')
                del restored,stored
                subprocess.run([sys.executable,'-B',str(ROOT/'verify_saved.py'),'--checkpoint',str(checkpoint),'--sha256',digest,'--execute-parent-saved-verify'],check=True)
                resource_gate(started)
    verify(config['source_hashes']);verify(config['data_hashes'])
    save(out/'RESULT.json',dict(kind='TRAINED_NOT_BODY_VERIFIED_INDEPENDENT_MOTOR_V1',updates=600,checkpoint=str(checkpoint),sha256=digest,final=curve[-1],wall_s=time.monotonic()-started,source_hashes=config['source_hashes'],data_hashes=config['data_hashes'],physical_steps=0,new_neck_deployment=False))
    print(json.dumps(dict(training_updates=600,checkpoint=str(checkpoint),sha256=digest,physical_steps=0)),flush=True)


if __name__=='__main__':main()
