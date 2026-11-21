"""Do not turn a cross-sample match candidate into an admitted sensory teacher."""
from collections import Counter
import platform

from common import *
from sources import flywire_annotations, verify_upstream
from circuit import eon
import benchmark
import numpy as np
import torch
import pyarrow.parquet as pq


def main():
    original = verify_upstream()
    catalog = read(BIDIR / 'CATALOG.json')
    classification = flywire_annotations('classification')
    types = flywire_annotations('consolidated_cell_types')
    index, _ = eon.get_hash_tables(eon.path_comp)
    ascending = {r: p for r, p in catalog['ports'].items() if p['direction'] != 'DN'}
    multiplicity = Counter(p['fafb_match'] for p in ascending.values() if p['fafb_match'])
    sugar = {str(r) for r in eon.get_experiment('sugar')['neu_exc']}
    # Sparse edge list only; no full-brain integration in this audit.
    table = pq.read_table(eon.path_con, columns=['Postsynaptic_Index', 'Presynaptic_Index',
                                                 'Excitatory x Connectivity'])
    post, pre, weights = [c.to_numpy() for c in table.columns]
    assert len(weights) == 15091983
    descending = read(BIDIR / 'EON/RESULT.json')['admitted']
    ground = {r: w for r, w in descending.items()
              if catalog['ports'][r]['group'] not in ('takeoff', 'wing_amplitude', 'flight_steer', 'landing')}
    ground_indices = {index[int(w)] for w in ground.values()}
    records = {}
    for root, p in ascending.items():
        wire = p['fafb_match']
        cl = classification.get(wire, {})
        native_index = index.get(int(wire)) if wire else None
        flags = []
        if native_index is None:
            flags.append('NO_ORIGINAL_BRAIN_INDEX')
        if wire and multiplicity[wire] != 1:
            flags.append('NONUNIQUE_CROSS_SAMPLE_IDENTITY')
        if wire and cl.get('side') != p['side']:
            flags.append('SIDE_ANNOTATION_DISAGREEMENT_NOT_AUTO_SWAPPED')
        expected = 'ascending' if p['direction'] == 'AN' else 'sensory_ascending'
        if wire and cl.get('super_class') != expected:
            flags.append('SUPERCLASS_DISAGREEMENT')
        if wire not in sugar:
            flags.append('NO_ARCHIVED_ORIGINAL_SENSORY_ENCODER_TARGET')
        flags.append('CROSS_SAMPLE_TYPE_AND_BOUNDARY_REVIEW_NOT_COMPLETED')
        hits = np.flatnonzero(pre == native_index) if native_index is not None else np.empty(0, dtype=int)
        records[root] = dict(group=p['group'], direction=p['direction'], banc_type=p['cell_type'],
                            banc_side=p['side'], flywire_root=wire, flywire_index=native_index,
                            flywire_classification=cl, flywire_type=types.get(wire),
                            match_multiplicity=multiplicity.get(wire, 0),
                            original_outgoing_edges=int(len(hits)),
                            direct_ground_dn_targets=sum(int(x in ground_indices) for x in post[hits]),
                            in_original_sugar_teacher=wire in sugar,
                            eligible_for_original_teacher_distillation=False, reasons=flags)
    pins = pin(imported_sources() | {BIDIR/'CATALOG.json', BIDIR/'EON/RESULT.json', FIXED/'PARAMETERS.json',
                ROOT/'data/flywire/fafb_classification.csv.gz', ROOT/'data/flywire/fafb_consolidated_cell_types.csv.gz',
                HERE/'PROTOCOL.zh-CN.md', BODY/'FIT/epoch040.pt'})
    result = dict(project='P1_open_source_brain', upstream=original, sources=pins, environment=dict(
                  python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
                  archived_experiments=list(benchmark.EXPERIMENTS),
                  original_sugar_neurons=[dict(root=r, classification=classification.get(r)) for r in sorted(sugar)],
                  sugar_neck_overlap=sorted(sugar & {p['fafb_match'] for p in ascending.values()}),
                  ascending_ports=len(records), records=records,
                  eligible_common_channels=[], training_gate='NO_REVIEWED_PAIRED_TEACHER',
                  scope='fixed archived upstream fragment; not a claim that no other Eon input implementation exists',
                  interpretation='Side fields may use different anatomical conventions; no automatic relabeling. '
                                 'Edges indicate structural access, not functional behavior or accepted homology.',
                  brain_steps=0, optimizer_updates=0)
    verify(pins)
    write(HERE/'SOURCE_AUDIT.json', result)
    print('audited', len(records), 'ascending ports; sugar overlap', result['sugar_neck_overlap'], flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
