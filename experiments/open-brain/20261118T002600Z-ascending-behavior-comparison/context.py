"""Frozen original Eon equations with declared AN source interventions."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2];BASE=HERE.parent
FIT=BASE/'20261117T235930Z-ascending-reference-fit'
FOUNDATION=BASE/'20261117T045901Z-foundation'
BIDIR=BASE/'20261117T051249Z-bidirectional'
FIXED=BASE/'20261117T164207Z-fixed-neck'
sys.path[:0]=[str(FIT),str(FIXED),str(BIDIR),str(FOUNDATION)]
from circuit import eon
from run_probe import FrozenCPUBrain,load_connectome
from sources import verify_upstream
DT=.0001


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def sources(extra=()):
    paths=set(HERE.glob('*.py'))|{HERE/'PROTOCOL.zh-CN.md'}|set(map(Path,extra))
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and Path(f).is_file() and Path(f).resolve().is_relative_to(ROOT):paths.add(Path(f).resolve())
    return {str(p.resolve()):sha(p) for p in paths}


def verify(pins):
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('input changed: '+p)


def context_groups(index):
    old=read(FOUNDATION/'BENCH/INPUTS.json')['external_input_flywire_roots']
    left=old['PFL3_to_left_DNa02'];right=old['PFL3_to_right_DNa02']
    return {'pfl_bilateral':[index[int(r)] for r in sorted(set(left+right))],
            'pfl_left':[index[int(r)] for r in left],
            'pfl_right':[index[int(r)] for r in right],
            'original_p9':[index[int(r)] for r in old['DNp09_bilateral']]}


class ContextBrain:
    def __init__(self,model,source_indices=(),external_lanes=()):
        self.model=model;self.source_indices=list(source_indices);self.external_lanes=list(external_lanes)
        self.g,self.queue,self.spikes,self.v,self.refrac=model.state_init()
        self.slot=0;self.stim=torch.zeros_like(self.v);self.ref_steps=model.neurons.refrac_steps.unsqueeze(0)

    def step(self,stimulus_indices,events=None,extra_recurrent=None):
        activity=self.spikes.clone()
        if events is not None:
            for lane in self.external_lanes:
                activity[lane,self.source_indices]=torch.from_numpy(events[lane].astype(np.float32))
        self.stim.zero_()
        for lane,indices in enumerate(stimulus_indices):
            self.stim[lane,indices]=self.model.scale*eon.MODEL_PARAMS['scalePoisson']
        self.refrac=torch.where(self.spikes>0,0.,self.refrac+1.)
        following=self.g*(1-self.model.neurons.synapse.time_factor)+self.queue[:,self.slot,:]*(self.refrac>=self.ref_steps).float()
        volley=self.model.recurrent_volley(activity)
        if extra_recurrent is not None:volley+=torch.from_numpy(extra_recurrent)
        self.queue[:,self.slot,:]=volley*self.model.scale
        self.slot=(self.slot+1)%self.queue.shape[1]
        self.spikes,self.v=self.model.neurons.neuron(self.g,self.stim,self.v)
        self.g=following-following*self.spikes
        for lane in self.external_lanes:
            self.spikes[lane,self.source_indices]=0.;self.v[lane,self.source_indices]=eon.MODEL_PARAMS['vRest']
            self.g[lane,self.source_indices]=0.;self.queue[lane,:,self.source_indices]=0.


@torch.inference_mode()
def preflight():
    torch.set_num_threads(1);verify_upstream();index,_=eon.get_hash_tables(eon.path_comp);weights=load_connectome(index)
    groups=context_groups(index);names=list(groups)
    model=FrozenCPUBrain(len(names),weights);brain=ContextBrain(model)
    admitted=read(BIDIR/'EON/RESULT.json')['admitted'];catalog=read(BIDIR/'CATALOG.json')['ports']
    roots=[r for r in admitted if catalog[r]['group'] not in ('takeoff','wing_amplitude','flight_steer','landing')]
    columns=[index[int(admitted[r])] for r in roots]
    rng=np.random.default_rng(18031)
    ticks=3200;inputs={n:rng.random((ticks,len(indices)))<.01 for n,indices in groups.items()}
    rows=[];population=[];began=time.monotonic();out=HERE/'PREFLIGHT';out.mkdir()
    pins=sources({FOUNDATION/'BENCH/INPUTS.json',eon.path_comp,eon.path_con,BIDIR/'EON/RESULT.json',BIDIR/'CATALOG.json'})
    for tick in range(ticks):
        active=[np.asarray(groups[n])[inputs[n][tick]].tolist() if 300<=tick<2700 else [] for n in names]
        brain.step(active);rows.append(brain.spikes[:,columns].numpy().copy());population.append(brain.spikes.sum(1).numpy().copy())
        if time.monotonic()-began>180:raise TimeoutError('context preflight budget')
    spikes=np.asarray(rows);np.savez_compressed(out/'TRACE.npz',emitted=spikes,population=np.asarray(population))
    counts={n:{r:int(spikes[:,i,j].sum()) for j,r in enumerate(roots)} for i,n in enumerate(names)}
    # Select only among upstream PFL3 contexts, in fixed preference order. P9 is
    # a published direct-DN diagnostic and never eligible as task background.
    chosen=next((n for n in ('pfl_bilateral','pfl_left','pfl_right') if sum(counts[n].values())>=20),None)
    verify(pins)
    write(out/'RESULT.json',dict(names=names,counts=counts,chosen=chosen,source_sha256=pins,
       trace_sha256=sha(out/'TRACE.npz'),brain_batch_ticks=ticks,brain_lane_ticks=ticks*len(names),
       original_p9_is_diagnostic_not_context=True,optimizer_updates=0,wall_seconds=time.monotonic()-began))
    print('upstream context',chosen,'ground DN counts',{n:sum(c.values()) for n,c in counts.items()},flush=True)


if __name__=='__main__':preflight()
