"""CPU implementation of the frozen Eon numerical step, with archive parity."""
import argparse
import os
from pathlib import Path
import sys

import numpy as np
from numba import njit
import torch

ARCHIVE = Path(os.environ.get('P1_ARCHIVE_ROOT', '/run/media/sensen/Data2/cell_wound_prototype'))
BASE = ARCHIVE / 'experiments/open-brain'
sys.path.insert(0, str(BASE/'20261118T002600Z-ascending-behavior-comparison'))
from neural_runtime import ContextBrain, FIT, FrozenCPUBrain, context_groups, eon, load_connectome, read, verify_upstream, write


@njit
def step_state(v, g, refractory, queue, slot, last, n_active, active, spikes, volley, stim,
               offsets, targets, values, sensor_ix, sensor_mask, decay, k_mem, scale,
               refrac_steps, v_rest, v_reset, threshold):
    volley[:] = 0.
    for j in range(n_active):
        source = last[j]
        for edge in range(offsets[source], offsets[source + 1]):
            volley[targets[edge]] += values[edge]
    for s in range(len(sensor_ix)):
        if sensor_mask[s]:
            source = sensor_ix[s]
            for edge in range(offsets[source], offsets[source + 1]):
                volley[targets[edge]] += values[edge]
    eligible = 0
    for i in range(len(v)):
        refr = np.float32(0.) if spikes[i] else refractory[i] + np.float32(1.)
        refractory[i] = refr
        follow = g[i] * decay + (queue[slot, i] if refr >= refrac_steps else np.float32(0.))
        queue[slot, i] = volley[i] * scale
        u = v[i] + stim[i]
        u = u + k_mem * (g[i] - (u - v_rest))
        if u > threshold:
            v[i] = u - (u - v_reset)
            g[i] = np.float32(0.)
            spikes[i] = True
        else:
            v[i] = u
            g[i] = follow
            spikes[i] = False
    for i in sensor_ix:
        v[i] = v_rest
        g[i] = np.float32(0.)
        spikes[i] = False
        queue[:, i] = np.float32(0.)
    for i in range(len(spikes)):
        if spikes[i]:
            active[eligible] = i
            eligible += 1
    return eligible


class FastBrain:
    def __init__(self, outgoing, sensor_ix=()):
        cfg = eon.MODEL_PARAMS
        n = outgoing.shape[0]
        self.outgoing = outgoing
        self.sensor_ix = np.asarray(sensor_ix, dtype=np.int64)
        self.v = np.full(n, cfg['v0'], dtype=np.float32)
        self.g = np.zeros(n, dtype=np.float32)
        self.refrac = np.full(n, round(cfg['tRefrac']/eon.DT), dtype=np.float32)
        self.queue = np.zeros((int(cfg['tDelay']/eon.DT) + 1, n), dtype=np.float32)
        self.spikes = np.zeros(n, dtype=bool)
        self.stim = np.zeros_like(self.g)
        self.volley = np.zeros_like(self.g)
        self.last = np.empty(n, dtype=np.int64)
        self.active = np.empty(n, dtype=np.int64)
        self.slot = 0
        self.n_active = 0
        self.k = tuple(np.float32(x) for x in (1.-eon.DT/cfg['tauSyn'], eon.DT/cfg['tauMem'],
                       cfg['wScale'], round(cfg['tRefrac']/eon.DT),
                       cfg['vRest'], cfg['vReset'], cfg['vThreshold']))
        self.strength = np.float32(cfg['wScale']*cfg['scalePoisson'])
        self.no_sensor = np.zeros(len(self.sensor_ix), dtype=bool)

    def step(self, stimulus, sensor=None):
        self.stim[:] = 0.
        self.stim[stimulus] = self.strength
        graph = self.outgoing
        self.n_active = step_state(self.v, self.g, self.refrac, self.queue, self.slot, self.last,
            self.n_active, self.active, self.spikes, self.volley, self.stim,
            graph.indptr, graph.indices, graph.data, self.sensor_ix,
            self.no_sensor if sensor is None else sensor, *self.k)
        self.last, self.active = self.active, self.last
        self.slot = (self.slot + 1) % self.queue.shape[0]


@torch.inference_mode()
def parity(output):
    torch.set_num_threads(1)
    verify_upstream()
    index, _ = eon.get_hash_tables(eon.path_comp)
    weights = load_connectome(index)
    drive = np.array(context_groups(index)['pfl_bilateral'] +
                     [index[n] for n in eon.get_experiment('sugar')['neu_exc']])
    sources = [row['flywire_index'] for row in read(FIT/'BINDING.json')['records']]
    original = ContextBrain(FrozenCPUBrain(1, weights), sources, (0,))
    fast = FastBrain(weights.T.tocsr(), sources)
    rng = np.random.default_rng(471)
    n = 2400
    positive, active_ticks = 0, 0
    for i in range(n):
        inputs = drive[rng.random(len(drive)) < .018] if i < 1900 else np.empty(0, dtype=int)
        events = rng.random(2) < .008
        original.step([inputs.tolist()], events[None, :])
        fast.step(inputs, events)
        for field in ('spikes', 'v', 'g', 'refrac', 'queue'):
            if not np.array_equal(getattr(original, field).numpy()[0], getattr(fast, field)):
                raise AssertionError((i, field))
        positive += fast.n_active
        active_ticks += bool(fast.n_active)
    assert positive > 0
    write(output, dict(status='BIT_EXACT_STOCK_EON_PARITY', tested_steps=n,
                       active_ticks=active_ticks, population_spikes=positive,
                       fields=['spikes', 'v', 'g', 'refrac', 'queue']))
    print(n, 'steps at bit exact parity', positive, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    parity(p.parse_args().output)
