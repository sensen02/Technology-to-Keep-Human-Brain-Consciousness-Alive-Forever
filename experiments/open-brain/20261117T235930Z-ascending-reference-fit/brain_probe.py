"""Frozen original brain response to explicitly prescribed source boundary events.

Two ANs are external source identities, not a second simulated soma. Their
incoming brain feedback is excluded and counted, never silently called intact.
"""
import argparse
import time
from shared import *
from encoder import load
from run_probe import FrozenCPUBrain,load_connectome


class BoundaryBrain:
    def __init__(self,model,source_indices):
        self.model=model;self.sources=source_indices
        self.g,self.queue,self.spikes,self.v,self.refrac=model.state_init()
        self.slot=0;self.stim=torch.zeros_like(self.v)
        self.ref_steps=model.neurons.refrac_steps.unsqueeze(0)

    def step(self,events):
        activity=self.spikes.clone()
        activity[:,self.sources]=torch.from_numpy(events.astype(np.float32))
        self.refrac=torch.where(self.spikes>0,0.,self.refrac+1.)
        syn=self.model.neurons.synapse
        following=self.g*(1-syn.time_factor)+self.queue[:,self.slot,:]*(self.refrac>=self.ref_steps).float()
        self.queue[:,self.slot,:]=self.model.recurrent_volley(activity)*self.model.scale
        self.slot=(self.slot+1)%self.queue.shape[1]
        self.spikes,self.v=self.model.neurons.neuron(self.g,self.stim,self.v)
        self.g=following-following*self.spikes
        # No second local AN LIF or brain->externally-owned-AN feedback.
        self.spikes[:,self.sources]=0.;self.v[:,self.sources]=eon.MODEL_PARAMS['vRest']
        self.g[:,self.sources]=0.;self.queue[:,:,self.sources]=0.


def encode_events(rates):
    phase=np.zeros(rates.shape[1:]);out=np.zeros(rates.shape,dtype=bool)
    for tick,row in enumerate(rates):
        phase+=row*DT;out[tick]=phase>=1.;phase-=out[tick]
    return out


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    parser.add_argument('--ticks',type=int,default=6000);a=parser.parse_args()
    torch.set_num_threads(1);start=time.monotonic();a.output.mkdir(parents=True)
    b=read(HERE/'BINDING.json');verify(b['source_sha256'])
    fit=read(HERE/'FIT_RESULT.json');verify(fit['source_sha256'])
    data,_=load_data('held_left');model,_=load(fit['candidate']);initial,_=load(HERE/'FIT/update000.pt')
    x=torch.from_numpy(data['features_hz'][:a.ticks]).double()
    # All lanes share physical source data. The all-time left-cut lane is
    # computed only from the actually interrupted, archived arrival stream.
    with np.load(PREP/'DATA/held_left_cut_left/INPUTS.npz',allow_pickle=False) as z:
        cut_arrivals=z['arrivals'][:a.ticks].copy()
    alpha=-np.expm1(-DT/np.array([.005,.020,.050]));cut_x=np.zeros((a.ticks,2,3));f=np.zeros((2,3))
    for tick in range(a.ticks):f+=alpha*(cut_arrivals[tick,:,None]/DT-f);cut_x[tick]=f
    rates=np.stack((data['teacher_rate_hz'][:a.ticks],model(x).numpy(),initial(x).numpy(),
                    model(torch.from_numpy(cut_x).double()).numpy()),axis=1)
    events=encode_events(rates)
    index,_=eon.get_hash_tables(eon.path_comp);weights=load_connectome(index)
    source_idx=[index[int(r['flywire_root'])] for r in b['records']]
    assert len(set(source_idx))==2
    targets=np.unique(weights[:,source_idx].nonzero()[0]);targets=targets[~np.isin(targets,source_idx)]
    dn=read(BIDIR/'EON/RESULT.json')['admitted'];dn_idx=[index[int(w)] for w in dn.values()]
    # The declared external source intervention excludes every incoming edge
    # onto these two source identities, including inter-AN feedback.
    feedback=int(weights[source_idx,:].nnz)
    population=[];voltages=[];dn_events=[]
    brain=BoundaryBrain(FrozenCPUBrain(4,weights),source_idx)
    pins=sources({HERE/'BINDING.json',HERE/'FIT_RESULT.json',Path(fit['candidate']),HERE/'DATA/held_left/PAIRED.npz',
                 PREP/'DATA/held_left_cut_left/INPUTS.npz',eon.path_comp,eon.path_con})
    for tick in range(a.ticks):
        brain.step(events[tick])
        voltages.append(brain.v[:,targets].numpy().copy())
        dn_events.append(brain.spikes[:,dn_idx].numpy().copy())
        population.append(brain.spikes.sum(1).numpy().copy())
        if (tick+1)%2000==0:print('brain batch ticks',tick+1,round(time.monotonic()-start,2),flush=True)
        if time.monotonic()-start>300:raise TimeoutError('discrete brain verification budget')
    voltages=np.asarray(voltages);dn_events=np.asarray(dn_events);population=np.asarray(population)
    signal=voltages[:,0]-eon.MODEL_PARAMS['vRest']
    error=np.mean((voltages[:,1]-voltages[:,0])**2)
    baseline=np.mean((voltages[:,2]-voltages[:,0])**2)
    nrmse=float((error/np.mean(signal*signal))**.5)
    gate=bool(error<=.8*baseline and nrmse<=.25)
    assert not events[:,3,0].any()
    assert np.array_equal(events[:,1,1],events[:,3,1])
    np.savez_compressed(a.output/'BRAIN.npz',rates=rates,boundary_events=events,receiver_voltage=voltages,
                         dn_events=dn_events,population=population,source_indices=source_idx,target_indices=targets)
    verify(pins)
    write(a.output/'RESULT.json',dict(status='FROZEN_BRAIN_ENGINEERING_INPUT_PROBE',brain_batch_ticks=a.ticks,
          brain_lane_ticks=a.ticks*4,lanes=['reference','trained','initial','actual_left_cut'],
          neurons=len(index),edges=weights.nnz,external_source_identities=2,excluded_feedback_edges=feedback,
          receiving_neurons=len(targets),receiver_nrmse=nrmse,initial_receiver_mse=float(baseline),
          trained_receiver_mse=float(error),reference_agreement_pass=gate,
          actual_left_cut_new_events_zero=True,uncut_encoded_events_exact=True,
          population_events_by_lane=population.sum(0).tolist(),dn_events_by_lane=dn_events.sum((0,2)).tolist(),
          own_native_AN_membrane_simulations=0,brain_weights_modified=False,
          boundary_scope=b['boundary'],trace_sha256=sha(a.output/'BRAIN.npz'),source_sha256=pins,
          wall_seconds=time.monotonic()-start,behavior_acceptance='NOT_RUN_NO_AUTONOMOUS_BEHAVIOR_CLAIM'))
    print('brain receiving nrmse',nrmse,'improvement',1-error/baseline,'DN',dn_events.sum((0,2)).tolist(),flush=True)


if __name__=='__main__':main()
