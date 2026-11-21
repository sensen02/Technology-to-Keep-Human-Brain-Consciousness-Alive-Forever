"""Freeze brain and head policy. Drive and sense the virtual student through the electrode only."""
import subprocess,sys,json,math
from pathlib import Path
import numpy as np
import torch
from settling import Settling,ContactGate,OUT as EXP
from fit_feeding import OUT,OLD,BASE,ObservedBody,FlatFloor,FeedbackSampler,T,local_axes
from array_loop import Electrode
sys.path.insert(0,str(BASE/'20261119T190005Z-odor-sac'))
from proximity import FittedNose,ANTENNA_OFFSETS
from odor_env import Course,odor_field,sample,intent
sys.path.insert(0,str(BASE/'20261119T073818Z-sac-terrain-brain-pair'))
from policy_runtime import actor_mean
sys.path.insert(0,'/home/sensen/.codex/worktrees/4528/cell_wound_prototype')
from engine.embodied.feeding import FeedingBody,FeedingState,Food
from voltages import r
@torch.inference_mode()
def run(seed,on,seconds,variant):
    frames=round(seconds/.005)
    weights_dir=(OUT.parent/'20261121T040639Z-feeding-body-practice'/'long-rollout'
                 if variant in ('previous','settling') else OUT/variant)
    el=Electrode(frames);net=ObservedBody(weights_dir/'weights.pt').eval()
    floor=FlatFloor().eval();floor.load_state_dict(torch.load(weights_dir/'floor.pt' if (weights_dir/'floor.pt').is_file() else OLD/'floor.pt',weights_only=True))
    mouth_off=T(json.loads((weights_dir/'mouth_offset.json').read_text()))
    gen=torch.Generator().manual_seed(88211)
    sampler=FeedbackSampler(1)
    correction=Settling().eval()
    correction.load_state_dict(torch.load(EXP/'settling.pt',weights_only=True))
    gate=ContactGate();enabled=[]
    demo=OUT/f'aligned-{88211 if seed==91001 else seed}'
    if not demo.exists():demo=OUT.parent/'20261121T040639Z-feeding-body-practice'/demo.name
    if not demo.exists():demo=OUT.parent/'20261121T034919Z-feeding-body-observations'/demo.name
    with np.load(demo/'external.npz') as obs:
        a,b=T(obs['markers_mm'][:1]),T(obs['markers_mm'][1:2])
    sensor=FittedNose();cfg=Course(**json.loads((demo/'RESULT.json').read_text())['geometry'])
    if seed==91001:
        from dataclasses import replace
        cfg=replace(cfg,goal_x=.765,goal_y=.215)
    field=odor_field(cfg)
    with np.load(BASE/'20261119T190005Z-odor-sac/TRAIN_DETOUR/POLICY.npz') as z:policy=dict(z)
    feeder=FeedingBody();state=FeedingState(Food(.9,.81),reserve=.26,circulating=.23)
    total=0.;first=None;prop=(0.,0.);last=floor(a,a);hist=torch.zeros(1,4,17)
    requests=np.zeros((5,2),bool)
    trajectory=[a[0].numpy(),b[0].numpy()];counts=[];neural=[];ascent=[];contact_count=0;mouth_seq=[];contacts=[];intakes=[];supports=[];upright=[]
    out=EXP/('live-%d-%s'%(seed,'on' if on else 'off'))
    out=out.with_name(out.name+'-'+variant)
    out.mkdir(exist_ok=False)
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
            supports.append(int((force[0]>=.5).sum()));upright.append(float(rot[2,2]))
            contact=bool(supports[-1]>=3 and rot[2,2]>=.88 and -.002<=mouth[2]/10-.004<=.14 and np.hypot(mouth[0]/10-cfg.goal_x,mouth[1]/10-cfg.goal_y)<=.49-.035)
            contact_count+=contact;contacts.append(contact)
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
            active=gate.step(contact) and variant=='settling'
            enabled.append(active)
            if active:
                nxt=correction(a,b,force)
            prob=logits.sigmoid();requests=sampler.sample(prob,torch.rand(prob.shape,generator=gen))[0].numpy()
            for feed in packet['feed'][4::5]:
                row=feeder.advance(state,.001,feed=bool(feed),contact=contact)
                state=row.state;total+=row.ingested.volume
                if first is None and total>0:first=(j+1)*.005
            intakes.append(total)
            if (j+1)%200==0:print(f"{(j+1)*.005:.1f}s intake={total:.4f} contact={contact_count}",flush=True)
            a,b=b,nxt;last=force;trajectory.append(b[0].numpy())
        worker.stdin.close()
        if worker.wait()!=0:raise RuntimeError('neural worker failed')
    trajectory=np.asarray(trajectory)
    result=dict(goal_cm=[cfg.goal_x,cfg.goal_y],seed=seed,current_on=on,seconds=seconds,requests=el.requests,arrivals=el.pending[:frames*50].sum(0).tolist(),encoded=np.asarray(ascent).sum((0,1)).tolist(),decoded_dn=np.sum(counts,0).tolist(),source_dn=np.sum(neural,(0,1)).tolist(),old_pattern_fired=el.before,registered_recalibration_left_scale=1.25,contact_time_s=contact_count*.005,min_upright=min(upright),tail_min_supports=min(supports[-28:]),tail_min_upright=min(upright[-28:]),closest_oral_center_mm=float(np.linalg.norm(np.asarray(mouth_seq)[:,:2]-np.array([cfg.goal_x,cfg.goal_y])*10,axis=1).min()),intake=total,stock_loss=.9-state.patch.volume,first_bite_s=first,net_mm=float(np.linalg.norm(trajectory[-1,0,:2]-trajectory[0,0,:2])),tail_rms_mm_s=float(np.sqrt((np.diff(trajectory[-41:,0,:2],axis=0)**2).sum(1).mean())/.005))
    assert abs(result['stock_loss']-total)<1e-9
    if not on:assert sum(result['arrivals'])==0 and sum(result['encoded'])==0
    np.savez_compressed(out/'replay.npz',gate=enabled,supports=supports,upright=upright,markers=trajectory,mouth=mouth_seq,dn_read=counts,brain_dn=np.concatenate(neural),brain_input=np.concatenate(ascent),brain_end=el.pending[:frames*50],mouth_contact=contacts,intake=intakes,event_log=np.array(el.events,dtype=str))
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(result,flush=True)
if __name__=='__main__':run(int(sys.argv[1]),sys.argv[2]=='on',float(sys.argv[3]),sys.argv[4] if len(sys.argv)>4 else 'student')
