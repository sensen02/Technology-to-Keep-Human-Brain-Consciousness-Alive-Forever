"""Frozen Eon, fixed two-way BANC transport, matrix and learned replacement body."""
import json
import sys
import numpy as np
import torch
from paths import ARCHIVE, BASE, HERE, read
from electrode import SAMPLES, WIDTH, library, extract
from model import FeedbackSampler
from skeleton import ObservedBody
from floor import FlatFloor

sys.path.insert(0,str(BASE/'20261118T002600Z-ascending-behavior-comparison'))
from neural_runtime import ContextBrain, FrozenCPUBrain, context_groups, eon, load_connectome
sys.path.insert(0,str(BASE/'20261118T020425Z-front-touch-timing-repair'))
from timing import CanonicalEvents, FixedCircuit

class SourceNeck(FixedCircuit):
    def update_port_sources(self, drive, arrivals):
        self._spikes[:,self.front_cells] = self.body_events

class Matrix:
    def __init__(self, wave, chol, seed, n):
        self.wave, self.chol = wave,chol
        self.rng = np.random.default_rng(seed)
        self.raw = (self.rng.normal(size=(n*SAMPLES+WIDTH*SAMPLES+2,64)).astype('f')@chol.T)*2

    def record(self, tick, events):
        for col in np.flatnonzero(events):
            start = tick*SAMPLES+self.rng.integers(3)
            self.raw[start:start+WIDTH*SAMPLES] += self.wave[col]*(
                self.rng.uniform(.87,1.13)*self.rng.uniform(.92,1.08,64))

    def completed_counts(self, end):
        raw = self.raw[max(0,end-150)*SAMPLES:end*SAMPLES]
        calls = extract(np.rint(raw/.02).astype(np.int16),self.wave,self.chol)
        hi=len(calls)-WIDTH+1
        return calls[max(0,hi-50):hi,:17].sum(0)

@torch.inference_mode()
def main():
    n,lanes=20000,3
    torch.set_num_threads(1)
    seed=11701
    meta=read(ARCHIVE/f'seed{seed}/INPUTS.json')
    bindings=read(BASE/'20261117T235930Z-ascending-reference-fit/BINDING.json')['records']
    dn=meta['dn_roots']; an=[x['banc_root'] for x in bindings]
    catalog=read(BASE/'20261117T051249Z-bidirectional/CATALOG.json')['ports']
    neck=SourceNeck({'ports':{r:catalog[r] for r in meta['neck_ports']}},batch=lanes)
    neck.front_cells=[neck.index[r] for r in an]
    cols=[neck.port_index[r] for r in dn+an]
    delay=neck.neck_delay[cols]
    wave,chol=library(dn+an)
    matrices=[Matrix(wave,chol,44217,n) for _ in range(lanes)]
    cut=np.zeros(len(neck.ports),bool)
    drive=np.zeros((lanes,len(neck.roots)),dtype='f')
    external=np.zeros((lanes,len(neck.ports)),bool)
    index,_=eon.get_hash_tables(eon.path_comp)
    admitted=read(BASE/'20261117T051249Z-bidirectional/EON/RESULT.json')['admitted']
    dn_idx=[index[int(admitted[r])] for r in dn]
    brain=ContextBrain(FrozenCPUBrain(lanes,load_connectome(index)),[b['flywire_index'] for b in bindings],range(lanes))
    ctx=np.array(context_groups(index)[meta['context']])
    with np.load(ARCHIVE/f'seed{seed}/NEURAL.npz') as z:
        stimulus=z['central_hits']
    codec=CanonicalEvents((lanes,2))
    with np.load(HERE/'data/11701_intact/external.npz') as z:
        obs=torch.from_numpy(z['markers_mm']); measured=torch.from_numpy(z['force_uN'])
    net=ObservedBody(HERE/'weights.pt').eval()
    floor=FlatFloor().eval()
    floor.load_state_dict(torch.load(HERE/'floor.pt',weights_only=True))
    pos=torch.zeros((400,lanes,21,3))
    pos[:2]=obs[:2,None]
    forces=torch.zeros((400,lanes,6))
    counts=torch.zeros((lanes,4,17))
    emitted=np.zeros((n,lanes,19),bool)
    arrived=np.zeros_like(emitted)
    brain_in=np.zeros((n,lanes,2),bool)
    schedule=np.zeros((n,lanes,2),bool)
    rng=np.random.default_rng(614)
    sampler=FeedbackSampler(lanes)
    for tick in range(n):
        external[:,cols[:17]]=brain.spikes[:,dn_idx].numpy().astype(bool)
        neck.body_events=schedule[tick].copy()
        neck.body_events[2]=False  # paired control, silence the two body-side AN sources
        em,ar,_,_=neck.step(drive,cut,eon_output=external)
        emitted[tick],arrived[tick]=em[:,cols],ar[:,cols]
        for lane,matrix in enumerate(matrices):
            matrix.record(tick,np.r_[arrived[tick,lane,:17],emitted[tick,lane,17:]])
        brain_in[tick]=codec.step(ar[:,cols[17:]])
        brain.step([ctx[stimulus[tick]].tolist()]*lanes,brain_in[tick])
        if (tick+1)%50:
            continue
        frame=tick//50
        nd=np.stack([m.completed_counts(tick+1) for m in matrices])
        counts=torch.cat((counts[:,1:],torch.tensor(nd,dtype=torch.float32)[:,None]),1)
        forces[frame,0]=measured[frame]
        forces[frame,1:]=floor(pos[max(0,frame-1),1:],pos[frame,1:])
        if 0<frame<399:
            pair=forces[frame-1:frame+1].transpose(0,1)
            nxt,logit=net(pos[frame-1],pos[frame],pair,counts)
            pos[frame+1]=nxt
            # Paired uniform draws. One event per 1 ms bin at its fixed midpoint.
            u=torch.from_numpy(rng.random((1,5,2))).expand(lanes,-1,-1)
            flags=sampler.sample(logit.sigmoid(),u).numpy().transpose(1,0,2)
            schedule[tick+6:tick+51:10]=flags
        if (tick+1)%5000==0: print('ticks',tick+1,flush=True)
    verified=[bool(np.array_equal(arrived[d:,:,j],emitted[:-d,:,j]) and not arrived[:d,:,j].any()) for j,d in enumerate(delay)]
    assert all(verified) and np.array_equal(schedule[:,:2],emitted[:,:2,17:]) and not emitted[:,2,17:].any()
    for lane in range(lanes):
        for side in range(2):
            gaps=np.diff(np.flatnonzero(schedule[:,lane,side]))
            assert np.all(gaps>=30)
    ref=obs.numpy(); p=pos.numpy()
    speeds=np.sqrt(np.mean(np.sum((np.diff(p[-41:,:,0,:2],axis=0)/.005)**2,axis=-1),axis=0))
    result=dict(lanes=['recorded_touch','pose_driven_flat_floor','floor_silence_front_an'],seed=seed,ticks=n,delay_ticks=delay.tolist(),
        roots=dn+an,delay_matched=verified,warm_start_frames=2,
        an_source=emitted[:,:,17:].sum(0).tolist(),an_arrived=arrived[:,:,17:].sum(0).tolist(),
        brain_events=brain_in.sum(0).tolist(),dn_arrived=arrived[:,:,:17].sum((0,2)).tolist(),
        individual_dn_counts=arrived[:,:,:17].sum(0).tolist(),
        autonomous_vs_silenced_dn_disagreements=int(np.count_nonzero(arrived[:,1,:17]!=arrived[:,2,:17])),
        last_100ms_center_net_speed_mm_s=(np.linalg.norm(p[-1,:,0]-p[-21,:,0],axis=-1)/.1).tolist(),
        teacher_last_100ms_center_net_speed_mm_s=float(np.linalg.norm(ref[-1,0]-ref[-21,0])/.1),
        final_center_error_mm=np.linalg.norm(p[-1,:,0]-ref[-1,0],axis=-1).tolist(),
        final_200ms_marker_center_xy_rms_speed_mm_s=speeds.tolist(),
        teacher_200ms_marker_center_xy_rms_speed_mm_s=float(np.sqrt(np.mean(np.sum((np.diff(ref[-41:,0,:2],axis=0)/.005)**2,axis=-1)))),
        finite=bool(np.isfinite(p).all()))
    assert result['finite']
    np.savez_compressed(HERE/'loop.npz',poses=p,reference=ref,touch=forces.numpy(),emitted=emitted,arrived=arrived,brain_in=brain_in)
    (HERE/'loop.json').write_text(json.dumps(result,indent=2))
    print(result,flush=True)

if __name__=='__main__':main()
