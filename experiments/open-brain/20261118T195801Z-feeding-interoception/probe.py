"""Screen paper-defined StN candidates against frozen Eon, with a matched route cut."""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow.feather as feather
import torch

HERE = Path(__file__).resolve().parent
ARCHIVE = Path(os.environ.get('P1_ARCHIVE_ROOT', '/run/media/sensen/Data2/cell_wound_prototype'))
RUNS = ARCHIVE / 'experiments/open-brain'
sys.path[:0] = [str(HERE.parents[2]), str(RUNS / '20261118T002600Z-ascending-behavior-comparison')]
from neural_runtime import (
    BIDIR, DT, ContextBrain, FrozenCPUBrain, eon, load_connectome, read, sha,
    sources, verify, verify_upstream, write,
)
from engine.embodied.feeding import FeedingBody, FeedingState, Food


MN9 = 720575940660219265
TICKS = 4000
SUGAR_ON, SUGAR_OFF = 300, 3600
STN_ON = 1400
DELAY = 15
WINDOW = slice(1700, 3600)
AUTHOR_COMMIT = '9a82bcc1877e1285ac3a6db3c2417b8bb507d161'


def bind(index):
    files = sorted((HERE / 'sources').glob('set_*.csv'))
    assert len(files) == 6
    groups = {}
    for path in files:
        with path.open() as stream:
            groups[path.stem] = [int(row['root_id']) for row in csv.DictReader(stream)]
    roots = [root for group in groups.values() for root in group]
    assert len(set(roots)) == len(roots) == 71
    metadata = ARCHIVE / 'data/flywire/fafb_classification.csv.gz'
    with gzip.open(metadata, 'rt') as stream:
        annotations = {int(row['root_id']): row for row in csv.DictReader(stream)}
    catalog_path = BIDIR / 'CATALOG.json'
    catalog = read(catalog_path)['ports']
    neck_matches = {str(row['fafb_match']) for row in catalog.values() if row['fafb_match']}
    # Audit the larger v888 metadata as well as the currently installed BANC ports.
    banc_path = ARCHIVE / 'data/banc888/banc_888_meta.feather'
    banc = feather.read_table(banc_path, columns=['banc_888_id', 'fafb_match', 'cell_type', 'super_class'])
    banc_matches = {}
    root_strings = {str(root) for root in roots}
    for row in banc.to_pylist():
        root = str(row['fafb_match'])
        if root in root_strings:
            banc_matches.setdefault(root, []).append(row)
    audit = {}
    for name, group in groups.items():
        rows = []
        for root in group:
            label = annotations[root]
            assert (label['super_class'], label['nerve']) == ('sensory', 'PhN')
            assert root in index
            rows.append(dict(flywire_root=str(root), eon_index=index[root], nerve=label['nerve'],
                             classification=label['class'], banc_matches=banc_matches.get(str(root), []),
                             installed_banc_port_match=str(root) in neck_matches))
        audit[name] = rows
    return groups, audit, [metadata, catalog_path, banc_path, *files]


def body_signal():
    body = FeedingBody()
    state = FeedingState(Food(1., 1.), crop=Food(.75, .1))
    sample = np.empty(TICKS)
    for tick in range(TICKS):
        step = body.advance(state, DT, feed=False, contact=True)
        assert step.crop_stretch.route == 'body_ascending'
        state = step.state
        sample[tick] = step.crop_stretch.value
    return sample


@torch.inference_mode()
def run(output, seed):
    torch.set_num_threads(1)
    verify_upstream()
    index, _ = eon.get_hash_tables(eon.path_comp)
    groups, audit, binding_files = bind(index)
    sugar = eon.get_experiment('sugar')
    sugar_indices = np.array([index[root] for root in sugar['neu_exc']])
    groups_index = {name: np.array([index[root] for root in roots])
                    for name, roots in groups.items()}
    assert not set(sugar_indices) & {i for ids in groups_index.values() for i in ids}
    output.mkdir()
    pins = sources([__file__, HERE.parents[2] / 'engine/embodied/feeding.py',
                    eon.path_comp, eon.path_con, *binding_files])
    write(output / 'BINDING.json', dict(
        species='adult female Drosophila melanogaster, FlyWire FAFB v783',
        anatomy='body StN -> PhN -> subesophageal zone',
        source_paper='https://doi.org/10.1016/j.isci.2026.117359',
        author_repository='https://github.com/ZepengYao-Lab/Giakoumas-et-al',
        author_commit=AUTHOR_COMMIT, molecular_piezo_identity='unresolved',
        uses_banc_vnc_connective=False, groups=audit, source_sha256=pins))
    names = ['blank', 'sugar_cut', *groups]
    brain = ContextBrain(FrozenCPUBrain(len(names), load_connectome(index)))
    rng = np.random.default_rng(seed)
    sugar_events = rng.random((TICKS, len(sugar_indices))) < sugar['stim_rate'] * DT
    sugar_events[:SUGAR_ON] = False
    sugar_events[SUGAR_OFF:] = False
    stretch = body_signal()
    source_rate = 80. * stretch  # Engineering screening intensity, in Hz.
    source_rate[:STN_ON] = 0.
    emitted = {name: rng.random((TICKS, len(ids))) < source_rate[:, None] * DT
               for name, ids in groups_index.items()}
    arrived = {name: np.concatenate((np.zeros((DELAY, trace.shape[1]), dtype=bool), trace[:-DELAY]))
               for name, trace in emitted.items()}
    cut_arrived = {name: arrived[name] & False for name in groups_index}
    population = np.zeros((TICKS, len(names)), dtype=np.int32)
    mn9 = np.zeros_like(population, dtype=bool)
    for tick in range(TICKS):
        common = sugar_indices[sugar_events[tick]].tolist()
        arrivals = [ids[arrived[name][tick]].tolist() for name, ids in groups_index.items()]
        cut_inputs = [ids[cut_arrived[name][tick]] for name, ids in groups_index.items()]
        inputs = [[], common + np.concatenate(cut_inputs).tolist(), *(common + events for events in arrivals)]
        brain.step(inputs)
        population[tick] = brain.spikes.sum(1).numpy()
        mn9[tick] = brain.spikes[:, index[MN9]].numpy()
        if (tick + 1) % 1000 == 0:
            print(seed, 'whole-brain steps', tick + 1, flush=True)
    assert not population[:, 0].any() and mn9[WINDOW, 1].any()
    assert np.all(population[:STN_ON + DELAY, 2:] == population[:STN_ON + DELAY, [1]])
    assert np.all(mn9[:STN_ON + DELAY, 2:] == mn9[:STN_ON + DELAY, [1]])
    trace_path = output / 'TRACE.npz'
    np.savez_compressed(trace_path, mn9=mn9, population=population, sugar=sugar_events,
                        crop_stretch=stretch, stn_cut_arrived=np.concatenate(list(cut_arrived.values()), axis=1),
                        **{name + '_emitted': trace for name, trace in emitted.items()},
                        **{name + '_arrived': trace for name, trace in arrived.items()})
    metrics = {name: dict(mn9_window_spikes=int(mn9[WINDOW, j].sum()),
                          population_window_spikes=int(population[WINDOW, j].sum()),
                          arriving_stn_events=int(arrived[name].sum()) if name in arrived else 0)
               for j, name in enumerate(names)}
    verify(pins)
    write(output / 'RESULT.json', dict(
        status='ANATOMICAL_CANDIDATE_SCREEN_ONLY', seed=seed, conditions=names, dt_s=DT,
        ticks=TICKS, lane_ticks=len(names) * TICKS, window=[WINDOW.start, WINDOW.stop],
        donor_delay_ticks=DELAY, delay_source='engineering assumption',
        first_possible_arrival_tick=STN_ON + DELAY, brain_to_body_motion=False,
        satiety_specific_mapping_verified=False, native_physics_steps=0, training_updates=0,
        trace_sha256=sha(trace_path), metrics=metrics))
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('seed', type=int)
    args = parser.parse_args()
    run(args.output, args.seed)
