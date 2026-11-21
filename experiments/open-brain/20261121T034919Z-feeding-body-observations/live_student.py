"""Freeze brain and head policy. Drive and sense the virtual student through the electrode only."""
import subprocess,sys,json,math
from pathlib import Path
import numpy as np
import torch
from fit_feeding import OUT,OLD,BASE,ObservedBody,FlatFloor,FeedbackSampler,T,local_axes
from array_loop import Electrode
sys.path.insert(0,str(BASE/'20261119T190005Z-odor-sac'))
from proximity import FittedNose,ANTENNA_OFFSETS
from odor_env import course,odor_field,sample,intent
sys.path.insert(0,str(BASE/'20261119T073818Z-sac-terrain-brain-pair'))
from policy_runtime import actor_mean
sys.path.insert(0,'/home/sensen/.codex/worktrees/4528/cell_wound_prototype')
from engine.embodied.feeding import FeedingBody,FeedingState,Food
from voltages import r
@torch.inference_mode()
def run(seed,on,seconds,variant):
    frames=round(seconds/.005)
    el=Electrode(frames);net=ObservedBody(OUT/variant/'weights.pt').eval()
    floor=FlatFloor().eval();floor.load_state_dict(torch.load(OUT/variant/'floor.pt' if variant=='adapted' else OLD/'floor.pt',weights_only=True))
    mouth_off=T(json.loads((OUT/variant/'mouth_offset.json').read_text()))
    gen=torch.Generator().manual_seed(88211)
    sampler=FeedbackSampler(1)
    with np.load(OUT/f'aligned-{seed}'/'external.npz') as obs:
        a,b=T(obs['markers_mm'][:1]),T(obs['markers_mm'][1:2])
    sensor=FittedNose();cfg=course(seed,'open',1);field=odor_field(cfg)
    with np.load(BASE/'20261119T190005Z-odor-sac/TRAIN_DETOUR/POLICY.npz') as z:policy=dict(z)
    feeder=FeedingBody();state=FeedingState(Food(.9,.81),reserve=.26,circulating=.23)
    total=0.;first=None;prop=(0.,0.);last=floor(a,a);hist=torch.zeros(1,4,17)
    requests=np.zeros((5,2),bool)
    trajectory=[a[0].numpy(),b[0].numpy()];counts=[];neural=[];ascent=[];contact_count=0;mouth_seq=[]
    out=OUT/('live-%d-%s'%(seed,'on' if on else 'off'))
    if variant=='adapted':out=out.with_name(out.name+'-adapted')
    out.mkdir(exist_ok=True)
    with (out/'worker.log').open('w') as log,subprocess.Popen([sys.executable,'-Bu',str(OUT/'eon_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True) as worker:
        def exchange(v):
            worker.stdin.write(json.dumps(v)+'\n');worker.stdin.flush()
            msg=worker.stdout.readline()
            if not msg:raise RuntimeError((out/'worker.log').read_text()[-2500:])
            return json.loads(msg)
        identity=exchange(dict(ticks=frames*50))
        assert identity['an']==el.targets
        reorder=[identity['roots'].index(k) for k in r.IO['dn_roots']]
        for j in range(frames):
            el.stimulate(j,requests,on)
            actual=el.pending[j*50:(j+1)*50]
            rot=local_axes(b)[0].numpy();face=rot[:,0];side=rot[:,1];center=b[0,0].numpy()
            mouth=center+rot@mouth_off.numpy();mouth_seq.append(mouth)
            force=floor(a,b)
            contact=bool((force[0]>=.5).sum()>=3 and rot[2,2]>=.88 and -.002<=mouth[2]/10-.004<=.14 and np.hypot(mouth[0]/10-cfg.goal_x,mouth[1]/10-cfg.goal_y)<=.49-.035)
            contact_count+=contact
            if j%16==0 and j>=32:
                antenna=np.stack([center[:2]/10+fw*face[:2]+lr*side[:2] for fw,lr in ANTENNA_OFFSETS])
                c=sample(field,antenna)
                pace=float((b[0,0,:2]-a[0,0,:2]).norm()/.005/12)
                prev=local_axes(a)[0,:,0].numpy()
                yaw=math.atan2(prev[0]*face[1]-prev[1]*face[0],np.dot(prev[:2],face[:2]))/.005
                obs=sensor.observe(c,np.clip(pace,0,1),np.clip(yaw,-.9,.9),(0.,0.))
                prop=intent(obs,actor_mean(policy,obs))
            packet=exchange(dict(intent=prop,circulating=state.circulating,contact=contact,arrivals=actual.astype(int).tolist()))
            spikes=np.asarray(packet['source'],bool)[:,reorder]
            calls=el.read_dn(j,spikes);counts.append(calls);neural.append(spikes);ascent.append(packet['encoded'])
            hist=torch.cat((hist[:,1:],T(calls)[None,None,:]),1)
            nxt,logits=net(a,b,torch.stack((last,force),1),hist)
            prob=logits.sigmoid();requests=sampler.sample(prob,torch.rand(prob.shape,generator=gen))[0].numpy()
            for feed in packet['feed'][4::5]:
                row=feeder.advance(state,.001,feed=bool(feed),contact=contact)
                state=row.state;total+=row.ingested.volume
                if first is None and total>0:first=(j+1)*.005
            a,b=b,nxt;last=force;trajectory.append(b[0].numpy())
        worker.stdin.close()
        if worker.wait()!=0:raise RuntimeError('neural worker failed')
    trajectory=np.asarray(trajectory)
    result=dict(seed=seed,current_on=on,seconds=seconds,requests=el.requests,arrivals=el.pending[:frames*50].sum(0).tolist(),encoded=np.asarray(ascent).sum((0,1)).tolist(),decoded_dn=np.sum(counts,0).tolist(),source_dn=np.sum(neural,(0,1)).tolist(),old_pattern_fired=el.before,registered_recalibration_left_scale=1.25,contact_time_s=contact_count*.005,intake=total,stock_loss=.9-state.patch.volume,first_bite_s=first,net_mm=float(np.linalg.norm(trajectory[-1,0,:2]-trajectory[0,0,:2])),tail_rms_mm_s=float(np.sqrt((np.diff(trajectory[-41:,0,:2],axis=0)**2).sum(1).mean())/.005))
    assert abs(result['stock_loss']-total)<1e-9
    if not on:assert sum(result['arrivals'])==0 and sum(result['encoded'])==0
    np.savez_compressed(out/'replay.npz',markers=trajectory,mouth=mouth_seq,dn_read=counts,brain_dn=np.concatenate(neural),brain_input=np.concatenate(ascent),brain_end=el.pending[:frames*50],event_log=np.array(el.events,dtype=str))
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(result,flush=True)
if __name__=='__main__':run(int(sys.argv[1]),sys.argv[2]=='on',float(sys.argv[3]),sys.argv[4] if len(sys.argv)>4 else 'student')
