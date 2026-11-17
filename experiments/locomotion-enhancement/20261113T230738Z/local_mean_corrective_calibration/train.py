"""One fixed600 masked12sequence localmeanANN calibration, never physical."""
import argparse
import json
import subprocess
import sys
import time
from sources import ROOT,read,save,verify,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-parent-training',action='store_true');a=p.parse_args()
    if not a.execute_parent_training:p.error('Explicitparent fixed600trainingrequired')
    started=time.monotonic()
    from learning import cpu1,rng_snapshot,rng_equal,objective,adapter_anchor,metrics,resource_gate
    cpu1()
    import torch,numpy as np,random
    import common,model
    from adapter import max_error
    from data import prepare
    b=read(ROOT/'INPUT_BINDINGS.json');pre=read(ROOT/'PREFLIGHT.json')
    if not pre['all_passed'] or pre['timing_status']!='READY_FIXED600_3600S' or pre['source_hashes']!=b['source_hashes'] or pre['data_hashes']!=b['data_hashes']:raise ValueError('Actualfullhistorymigrationgradientrange/timing preflightrequired')
    verify(b['source_hashes']);verify(b['data_hashes']);out=ROOT/'TRAIN';out.mkdir(exist_ok=False)
    torch.manual_seed(20261113);np.random.seed(20261113);random.seed(20261113)
    m,parent=common.initialize();scan=model.make_scan(m);data=prepare(m);params=list(m.adapter_parameters());anchor=adapter_anchor(m);opt=torch.optim.Adam(params,lr=3e-4)
    config=dict(kind='onefixed600_localmean_masked12_corrective_calibration',steps=600,source_hashes={**b['source_hashes'],str(ROOT/'INPUT_BINDINGS.json'):sha(ROOT/'INPUT_BINDINGS.json'),str(ROOT/'PREFLIGHT.json'):sha(ROOT/'PREFLIGHT.json')},data_hashes=b['data_hashes'],prereg=read(ROOT/'PREREG.json'),initial_anchor=anchor,effective_batch=12,microbatch=4,seed=20261113,lr=3e-4)
    save(out/'IDENTITY.json',dict(kind=config['kind'],steps=600,architecture=m.identity(),source_hashes=config['source_hashes'],data_hashes=config['data_hashes'],actual12data_proofs=data['proofs'],masked_rows_preserved=True,DEV_is_training=False,physical_steps=0))
    resource_gate(started);curve=[dict(step=0,metrics=metrics(scan,data),wall_s=time.monotonic()-started)]
    def publish(step):
        checkpoint=out/f'checkpoint_{step:06d}.pt';rng=rng_snapshot();digest=common.save_checkpoint(checkpoint,m,opt,config,curve,step,rng)
        restored,payload=common.load_checkpoint(checkpoint,digest,final=False)
        if max_error(m.state_dict(),restored.state_dict())!=0. or max_error(opt.state_dict(),payload['optimizer_state_dict'])!=0. or not rng_equal(rng,payload['rng']):raise ValueError('Savedmodel/Adam/RNG exactroundtripfailed')
        del restored,payload
        subprocess.run([sys.executable,'-B',str(ROOT/'verify_saved.py'),'--checkpoint',str(checkpoint),'--sha256',digest,'--execute-parent-saved-verify'],check=True)
        resource_gate(started)
        return checkpoint,digest
    checkpoint,digest=publish(0)
    with (out/'TRAINING.jsonl').open('x') as log:
        log.write(json.dumps(curve[0])+'\n');log.flush()
        for step in range(1,601):
            resource_gate(started);opt.zero_grad(set_to_none=True);total=0.;components={}
            for lo in (0,4,8):
                residual,h=scan(data['features'][:,lo:lo+4]);loss,terms=objective(residual,data['oldmean'][:,lo:lo+4],data['y'][:,lo:lo+4],data['v1'][lo:lo+4],m,anchor,valid=data['valid'][:,lo:lo+4]);loss.backward();total+=float(loss.detach())
                for key,value in terms.items():components[key]=components.get(key,0.)+value
                del residual,h,loss
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):raise ValueError('All8finitegradientsrequired')
            if any(p.requires_grad or p.grad is not None for p in m.old.parameters()):raise ValueError('Frozen789fgradientviolation')
            opt.step();row=dict(step=step,loss=total,**components,wall_s=time.monotonic()-started,peak_RSS_MiB=resource_gate(started));curve.append(row);log.write(json.dumps(row)+'\n');log.flush()
            if step%25==0:print(json.dumps(row),flush=True)
            if step%50==0:verify(config['source_hashes']);verify(config['data_hashes'])
            if step in (300,600):row['metrics_after_update']=metrics(scan,data);checkpoint,digest=publish(step)
    verify(config['source_hashes']);verify(config['data_hashes'])
    save(out/'RESULT.json',dict(kind='LOCALMEAN_CORRECTIVE_TRAINED_NOT_BODY_VERIFIED_V1',updates=600,checkpoint=str(checkpoint),sha256=digest,final=curve[-1],wall_s=time.monotonic()-started,source_hashes=config['source_hashes'],data_hashes=config['data_hashes'],physical_steps=0,new_neck_deployment=False))


if __name__=='__main__':main()
