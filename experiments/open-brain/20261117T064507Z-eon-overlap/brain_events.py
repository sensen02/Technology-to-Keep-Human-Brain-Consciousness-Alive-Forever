"""Record real emitted events from the frozen, full-size Eon dynamics."""
import argparse
import json
import sys
from hashlib import file_digest
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BIDIR = HERE.parent/'20261117T051249Z-bidirectional'
FOUNDATION = HERE.parent/'20261117T045901Z-foundation'
sys.path[:0] = [str(BIDIR),str(FOUNDATION)]
from circuit import eon
from run_probe import FrozenCPUBrain, load_connectome
from sources import COMMIT, digest, verify_upstream, write

TICKS = 17000
ON = 2300
STEER = 3800
OFF = 11800
LOW = 82
HIGH = 182
TURN = 132
NAMES = ('cruise','fast','a01_left','a01_right','a02_left','a02_right')
GROUPS = {'forward_speed','walking_steer','walking_steer_context'}


def inputs():
    catalog = json.loads((BIDIR/'CATALOG.json').read_text())['ports']
    binding = json.loads((BIDIR/'EON/RESULT.json').read_text())
    # The first Eon run predates the addition of abdomen ascending ports.
    reviewed_path = BIDIR/'CATALOG_PRE_ABDOMEN.json'
    assert binding['catalog_sha256'] == digest(reviewed_path)
    reviewed = json.loads(reviewed_path.read_text())['ports']
    assert all(catalog[root] == reviewed[root] for root in binding['admitted'])
    admitted = binding['admitted']
    roots = list(admitted)
    index, _ = eon.get_hash_tables(eon.path_comp)
    wires = [admitted[r] for r in roots]
    assert len(set(wires)) == len(roots) == 25
    assert all(admitted[r] == catalog[r]['fafb_match'] and catalog[r]['direction']=='DN' for r in roots)
    groups = {(catalog[r]['group'],catalog[r]['side']):r for r in roots if catalog[r]['group'] in GROUPS}
    assert len(groups)==6
    return catalog, admitted, roots, index, groups


class RingBrain:
    """Frozen Eon state equations, using a ring in place of the full delay roll."""
    def __init__(self, model):
        self.model = model
        self.conductance, self.queue, self.spikes, self.voltage, self.refrac = model.state_init()
        self.slot = 0
        self.stim = torch.zeros_like(self.voltage)
        self.ref_steps = model.neurons.refrac_steps.unsqueeze(0)

    def step(self, active):
        self.stim.zero_()
        self.stim[0,active] = self.model.scale * eon.MODEL_PARAMS['scalePoisson']
        syn = self.model.neurons.synapse
        self.refrac = torch.where(self.spikes>0,0.,self.refrac+1.)
        potential = self.conductance*(1-syn.time_factor) + self.queue[:,self.slot,:]*(self.refrac>=self.ref_steps).float()
        self.queue[:,self.slot,:] = self.model.recurrent_volley(self.spikes)*self.model.scale
        self.slot = (self.slot+1)%self.queue.shape[1]
        self.spikes, self.voltage = self.model.neurons.neuron(self.conductance, self.stim, self.voltage)
        self.conductance = potential-potential*self.spikes


@torch.inference_mode()
def preflight(model, indices):
    candidate = RingBrain(model)
    reference = model.state_init()
    rng = np.random.default_rng(813)
    active_ticks = 0
    for tick in range(100):
        active = [i for i, use in zip(indices,rng.random(len(indices))<.11,strict=True) if use]
        candidate.step(active)
        ref_in = model.recurrent_volley(reference[2])*model.scale
        reference = model.neurons(ref_in,candidate.stim,*reference)
        active_ticks += bool(torch.any(reference[2]))
        order = list(range(candidate.slot,candidate.queue.shape[1]))+list(range(candidate.slot))
        actual = (candidate.conductance,candidate.queue[:,order,:],candidate.spikes,candidate.voltage,candidate.refrac)
        assert all(torch.equal(a,r) for a,r in zip(actual,reference,strict=True)), tick
    assert active_ticks
    return dict(steps=100,active_ticks=active_ticks,exact_all_arrays=True)


def schedule(name, groups, archived_stimulation):
    drive = [(groups['forward_speed',side], LOW, ON, OFF) for side in ('left','right')]
    if name == 'off':
        return [], []
    if name == 'fast':
        return [(root,HIGH,a,b) for root,_,a,b in drive], []
    if name.startswith('pfl_'):
        side = name.rsplit('_',1)[1]
        return drive, [(root,100,STEER,OFF) for root in archived_stimulation[f'PFL3_to_{side}_DNa02']]
    if name.startswith(('a01','a02')):
        family, side = name.split('_')
        group = 'walking_steer_context' if family=='a01' else 'walking_steer'
        drive.append((groups[group,side], TURN, STEER, OFF))
    return drive, []


@torch.inference_mode()
def export(name, seed, out):
    started = perf_counter()
    torch.set_num_threads(1)
    source = verify_upstream()
    _, admitted, roots, index, groups = inputs()
    archive = json.loads((FOUNDATION/'BENCH/INPUTS.json').read_text())['external_input_flywire_roots']
    model = FrozenCPUBrain(1,load_connectome(index))
    model_idx = [index[int(admitted[r])] for r in roots]
    parity = preflight(model,[index[int(admitted[r])] for r in groups.values()]) if name=='off' else None
    params = schedule(name, groups, archive)
    spec = [(admitted[root],rate,on,off) for root,rate,on,off in params[0]]+params[1]
    generators = []
    for root, rate, first, last in spec:
        generator = np.random.default_rng(np.random.SeedSequence([seed,int(root)]))
        hits = generator.random(TICKS)<(rate*eon.DT/1000.)
        hits[:first] = False
        hits[last:] = False
        generators.append(hits)
    stimuli = np.array(generators,dtype=bool).reshape(-1,TICKS).T
    indices = np.array([index[int(row[0])] for row in spec], dtype=np.int64)
    out.mkdir()
    write(out/'INPUT.json',dict(name=name,seed=seed,dt_ms=eon.DT,ticks=TICKS,neurons=len(index),
          connections=model.weights.nnz,roots=roots,flywire_roots=[admitted[r] for r in roots],
          stimulus=[dict(flywire_root=r,rate_hz=hz,first_tick=first,last_tick_exclusive=last)
                    for r,hz,first,last in spec], equivalent_ring=parity, upstream_commit=COMMIT,upstream=source,
          file_sha256={str(path):digest(path) for path in (HERE/'brain_events.py', BIDIR/'CATALOG.json',
                       BIDIR/'CATALOG_PRE_ABDOMEN.json', BIDIR/'EON/RESULT.json', BIDIR/'circuit.py', FOUNDATION/'run_probe.py', FOUNDATION/'sources.py',
                       HERE/'PROTOCOL.zh-CN.md')}))
    state = RingBrain(model)
    spike = np.zeros((TICKS,len(roots)), dtype=bool)
    population = np.zeros(TICKS,dtype=np.int32)
    voltage = np.zeros(spike.shape,dtype=np.float32)
    for tick in range(TICKS):
        state.step(indices[stimuli[tick]].tolist())
        spike[tick] = state.spikes[0,model_idx].numpy().astype(bool)
        voltage[tick] = state.voltage[0,model_idx].numpy()
        population[tick] = int(torch.count_nonzero(state.spikes))
        if (tick+1)%5000==0:
            print(name,tick+1,float(perf_counter()-started),flush=True)
    assert np.isfinite(voltage).all() and not population[:ON].any()
    trace=out/'BRAIN.npz'
    np.savez_compressed(trace,emitted=spike,voltage=voltage,population=population,stimuli=stimuli)
    write(out/'RESULT.json',dict(trace_sha256=digest(trace),input_sha256=digest(out/'INPUT.json'),
          counts={r:int(n) for r,n in zip(roots,spike.sum(0),strict=True)},
          population_spikes=int(population.sum()),seconds=perf_counter()-started))
    print(name,seed,'population',int(population.sum()),'target export',int(spike.sum()),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name',choices=('off',*NAMES,'pfl_left','pfl_right'))
    parser.add_argument('seed',type=int)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    export(args.name,args.seed,args.out)
