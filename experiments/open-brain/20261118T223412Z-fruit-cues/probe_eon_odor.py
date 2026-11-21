"""Deliver rendered-arena antenna samples to named frozen Eon olfactory neurons."""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ARCHIVE = Path(os.environ.get('P1_ARCHIVE_ROOT', '/run/media/sensen/Data2/cell_wound_prototype'))
BASE = ARCHIVE / 'experiments/open-brain'
sys.path.insert(0, str(BASE/'20261118T002600Z-ascending-behavior-comparison'))
from neural_runtime import (
    BIDIR, ContextBrain, FrozenCPUBrain, context_groups, eon, load_connectome,
    read, sha, sources, verify, verify_upstream, write,
)


DT = .0001
TICKS = 3500
STIMULUS = slice(350, 2950)
CASES = ('no_odor', 'behind_wall', 'open', 'past_left_edge', 'PFL3_positive')


def table(path):
    with gzip.open(path, 'rt') as stream:
        return {int(row['root_id']): row for row in csv.DictReader(stream)}


@torch.inference_mode()
def run(preflight, output):
    torch.set_num_threads(1)
    verify_upstream()
    index, _ = eon.get_hash_tables(eon.path_comp)
    type_file = ARCHIVE/'data/flywire/fafb_consolidated_cell_types.csv.gz'
    source_file = ARCHIVE/'data/flywire/fafb_classification.csv.gz'
    types, classification = table(type_file), table(source_file)
    known_orn = {'ORN_DM1', 'ORN_VA2'}
    sensory = sorted(root for root, row in types.items() if row['primary_type'] in known_orn and root in index)
    assert all(classification[root]['super_class'] == 'sensory' for root in sensory)
    side = np.array([('left', 'right').index(classification[root]['side']) for root in sensory])
    receptors = np.array([index[r] for r in sensory])
    pn_roots = sorted(root for root, row in types.items() if row['primary_type'] in ('DM1_lPN','VA2_adPN')
                      and root in index)
    projection = np.array([index[r] for r in pn_roots])
    assert len(receptors) >= 120 and len(projection) == 4
    positive = np.array(context_groups(index)['pfl_bilateral'])
    catalog = read(BIDIR/'CATALOG.json')['ports']
    eon_binding_file = BIDIR/'EON/RESULT.json'
    admitted = read(eon_binding_file)['admitted']
    ground = {root: int(match) for root, match in admitted.items()
              if catalog[root]['group'] not in ('takeoff', 'wing_amplitude', 'landing', 'flight_steer')}
    dn = np.array([index[root] for root in ground.values()])
    values = read(preflight)['sensor_stations']
    concentrations = np.array([[0.,0.], *[values[name]['odor'] for name in CASES[1:4]]])
    # Relative engineering concentration and transducer gain. No measured ppm assignment.
    rates = 100. * concentrations / (.01 + concentrations)
    common = np.random.default_rng(2761)
    odor = common.random((TICKS, len(receptors)))
    pfl = common.random((TICKS, len(positive)))
    brain = ContextBrain(FrozenCPUBrain(len(CASES), load_connectome(index)))
    oral_input = np.zeros((TICKS, 4, len(receptors)), dtype=bool)
    projection_spikes = np.zeros((TICKS, len(CASES), len(projection)), dtype=bool)
    dn_spikes = np.zeros((TICKS, len(CASES), len(dn)), dtype=bool)
    pop = np.zeros((TICKS, len(CASES)), dtype=np.int32)
    output.mkdir()
    pins = sources([__file__, preflight, type_file, source_file, eon.path_comp, eon.path_con,
                    eon_binding_file, BIDIR/'CATALOG.json'])
    for tick in range(TICKS):
        if STIMULUS.start <= tick < STIMULUS.stop:
            oral_input[tick] = odor[tick] < rates[:, side] * DT
            stimuli = [receptors[active].tolist() for active in oral_input[tick]]
            stimuli.append(positive[pfl[tick] < .01].tolist())
        else:
            stimuli = [[] for _ in CASES]
        brain.step(stimuli)
        firing = brain.spikes.numpy()
        projection_spikes[tick] = firing[:, projection] > 0.
        dn_spikes[tick] = firing[:, dn] > 0.
        pop[tick] = firing.sum(axis=1)
    assert not pop[:, 0].any() and dn_spikes[:, -1].any()
    assert projection_spikes[:, 1:4].any(axis=(0, 2)).all()
    result = {
        name: dict(projection_spikes=int(projection_spikes[:, i].sum()),
                   total_dn_spikes=int(dn_spikes[:, i].sum()), population_spikes=int(pop[:, i].sum()),
                   dn_by_port={root: int(dn_spikes[:, i, j].sum()) for j, root in enumerate(ground)})
        for i, name in enumerate(CASES)}
    trace = output/'TRACE.npz'
    np.savez_compressed(trace, receptor_stimulus=oral_input, pn_spikes=projection_spikes,
                        dn_spikes=dn_spikes, population=pop)
    verify(pins)
    write(output/'RESULT.json', dict(status='NATIVE_OLFACTORY_INPUT_PROBE',
        brain_lane_steps=TICKS*len(CASES), actual_body_episodes=0, training_updates=0,
        cases=CASES, seed=2761, rates_hz=rates.tolist(), concentrations=concentrations.tolist(),
        odor_root_ids=[str(r) for r in sensory], pn_root_ids=[str(r) for r in pn_roots],
        trace_sha256=sha(trace), source_sha256=pins, metrics=result))
    print({name: (r['projection_spikes'], r['total_dn_spikes']) for name, r in result.items()}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('preflight', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    run(args.preflight, args.output)
