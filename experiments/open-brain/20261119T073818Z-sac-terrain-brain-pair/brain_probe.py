"""Identify brain/neck responses and calibrate a bounded PFL3→DN synaptic gain."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OPEN = HERE.parent / '20261118T231627Z-open-fruit-loop'
sys.path.insert(0,str(OPEN))
from motor import FastBrain, MotorAdapter, Neck, NeuralInputs, DT, eon, load_connectome

COMMANDS = [(p,y) for p in (.28,.55,.76) for y in (-.22,0.,.22)]
GAINS = (1., .90, 1.10)
TRAIN_SEED = 46323
HELDOUT_SEED = 78217


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def read_weights():
    index,_ = eon.get_hash_tables(eon.path_comp)
    return index, load_connectome(index).T.tocsr()


def patchable(index, graph, sensory, neck):
    reverse = {int(i):str(root) for root,i in index.items()}
    output = np.zeros(graph.shape[0],dtype=bool)
    output[neck.dn_ix] = True
    masks, edges = [],[]
    for side in sensory.pfl:
        locations = []
        for source in side:
            a,b = graph.indptr[source:source+2]
            offsets = a+np.flatnonzero(output[graph.indices[a:b]] & (graph.data[a:b]!=0))
            locations.extend(offsets.tolist())
            edges.extend(dict(source=reverse[source],target=reverse[graph.indices[i]],
                              csr_offset=int(i),initial=float(graph.data[i])) for i in offsets)
        masks.extend(locations)
    assert masks and len(set(masks))==len(masks)
    return np.asarray(masks),edges


def case(graph, identity, neck_factory, drive, rates, rng_seed):
    model = FastBrain(graph,drive.an_ix)
    neck = neck_factory()
    ticks = 1800
    draws = drive.masks(ticks,rng_seed)
    p = drive.rates(rates,(0.,0.),0.,0.) * DT
    emitted = np.zeros((ticks,len(neck.columns)),dtype=bool)
    arrivals = np.zeros_like(emitted)
    command = np.zeros((ticks,2))
    for i,u in enumerate(draws):
        model.step(drive.assembled[u<p],neck.an_inputs)
        emitted[i] = model.spikes[neck.dn_ix]
        command[i] = neck.update(i,emitted[i],np.zeros(2))
        arrivals[i] = neck.last_dn
    delays = neck.model.neck_delay[neck.columns]
    for j,shift in enumerate(delays):
        assert not arrivals[:shift,j].any()
        assert np.array_equal(arrivals[shift:,j],emitted[:-shift,j])
    mean = command[850:].mean(axis=0)
    return dict(intent=list(identity), rates_hz=list(map(float,rates)), drive=mean.tolist(),
                emitted=int(emitted.sum()), arrived=int(arrivals.sum()),
                transport=True, delay_ticks=delays.tolist(), ticks=ticks)


def calibration_loss(rows):
    errors = []
    for row in rows:
        advance,turn = row['intent']
        left,right = row['drive']
        errors.append((.5*(left+right)-advance)**2 + .75*(.5*(right-left)-turn)**2)
    return float(np.mean(errors))


def run(output):
    torch.set_num_threads(1)
    output.mkdir()
    index, graph = read_weights()
    drive = NeuralInputs(index)
    create_neck = lambda: Neck(index,drive,tactile=True)
    neck = create_neck()
    positions, edges = patchable(index,graph,drive,neck)
    # Save the gains as a sidecar. The connectivity graph on disk stays immutable.
    original = graph.data[positions].copy()
    unchanged_hash = digest(graph.data)
    adapter = MotorAdapter(json.loads((OPEN/'TRANSLATED_MOTOR.json').read_text()))
    print('Audited PFL3→registered ground DN synapses',len(edges),flush=True)
    trials = {}
    for gain in GAINS:
        graph.data[positions] = original*gain
        rows = []
        for i,(advance,turn) in enumerate(COMMANDS):
            row = case(graph,(advance,turn),create_neck,drive,adapter.rates(advance,turn,0.),TRAIN_SEED+i)
            rows.append(row)
            print(gain,advance,turn,np.round(row['drive'],4).tolist(),row['emitted'],flush=True)
        trials[str(gain)] = dict(weights_sha256=digest(graph.data), rows=rows,
                                 calibration_mse=calibration_loss(rows))
    selected = min(GAINS[1:],key=lambda g:trials[str(g)]['calibration_mse'])
    held = {}
    for gain in (1.,selected):
        graph.data[positions] = original*gain
        rows = [case(graph,(a,t),create_neck,drive,adapter.rates(a,t,0.),HELDOUT_SEED+i)
                for i,(a,t) in enumerate(((.34,-.16),(.34,.16),(.68,0.)))]
        held[str(gain)] = dict(rows=rows,calibration_mse=calibration_loss(rows))
    graph.data[positions]=original
    assert digest(graph.data)==unchanged_hash
    assert trials['1.0']['weights_sha256']==unchanged_hash
    assert trials[str(selected)]['weights_sha256']!=unchanged_hash
    result = dict(frozen_gain=1., tuned_gain=selected, original_weights_sha256=unchanged_hash,
                  tuned_weights_sha256=trials[str(selected)]['weights_sha256'],
                  synapse_count=len(positions),edges=edges, trials=trials, heldout=held,
                  fitting='matched physical Eon/BANC calibration with bounded gain selection',
                  objective='post-neck intended common/differential squared error',
                  seed_train=TRAIN_SEED,seed_heldout=HELDOUT_SEED)
    (output/'CALIBRATION.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('selected gain',selected,'remaining probes complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    run(parser.parse_args().output)
