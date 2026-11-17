"""Build the fixed cross-neck circuit from published BANC v888 source files."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.parquet as parquet

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
NECK = PROJECT / 'experiments/neck-real-structure/20261112T214803Z'
META = PROJECT / 'data/banc888/banc_888_meta.feather'
MARKERS = NECK / 'geometry_downloads/neck_connective_y92500.parquet'
EDGE = PROJECT / 'experiments/visual-orientation/20261116T023521Z/banc_888_edgelist_simple_v2.feather'
PLANES = (370000.0, 484000.0)
BRAIN = {'central_brain', 'optic_lobe'}
VNC = 'ventral_nerve_cord'
SENSORS = {
    'front_touch': 'front_leg_bristle_neuron',
    'front_load': 'front_leg_campaniform_sensillum_neuron',
    'leg_vibration': 'front_leg_club_chordotonal_organ_neuron',
    'leg_vibration_alt': 'front_leg_club_chordotonal_organ_neuron',
    'joint_position': 'front_leg_claw_chordotonal_organ_neuron',
    'joint_position_alt': 'front_leg_claw_chordotonal_organ_neuron',
    'joint_movement': 'front_leg_hook_chordotonal_organ_neuron',
}
CLASS = {'DN': 'descending', 'AN': 'ascending', 'SA': 'sensory_ascending'}
SIGN = {'acetylcholine': 1, 'gaba': -1}

sys.path.insert(0, str(NECK / 'geometry_sources'))
from validate_swc import parse_swc


def read(path):
    return json.loads(path.read_text())


def hash_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def record(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def crossings(path, markers):
    nodes, _, roots = parse_swc(path)
    assert len(roots) == 1
    hits = {plane: {} for plane in PLANES}
    for key, (_, position, _, parent) in nodes.items():
        if parent == -1:
            continue
        start = nodes[parent][1]
        for y in PLANES:
            if min(start[1], position[1]) < y <= max(start[1], position[1]):
                fraction = (y - start[1]) / (position[1] - start[1])
                hits[y][key] = [a + fraction * (b - a) for a, b in zip(start, position, strict=True)]
    anterior, posterior = (hits[p] for p in PLANES)
    pairs = []
    for first, second in ((anterior, posterior), (posterior, anterior)):
        for child in second:
            ancestor = nodes[child][3]
            while ancestor != -1:
                if ancestor in first:
                    front, rear = (ancestor, child) if first is anterior else (child, ancestor)
                    point = anterior[front]
                    if markers:
                        pairs.append((min(math.dist(point, m) for m in markers), front, rear))
                ancestor = nodes[ancestor][3]
    assert pairs, 'no matched continuous branch and anterior marker'
    distance, front, rear = min(pairs)
    assert distance <= 1500, f'closest continuous branch {distance:.1f} nm from marker'
    return {'nodes': len(nodes), 'crossing_children': [front, rear], 'front_nm': anterior[front],
            'rear_nm': posterior[rear], 'front_marker_distance_nm': distance}


def signed(row):
    verified = row['neurotransmitter_verified']
    transmitter = verified or row['neurotransmitter_predicted']
    return SIGN.get(transmitter), transmitter, 'verified' if verified else 'predicted'


def main():
    pins = {str(p): hash_file(p) for p in (EDGE, META, MARKERS, HERE / 'PORTS.json', HERE / 'GEOMETRY_RECEIPTS.json',
                                         HERE / 'build_catalog.py')}
    assert pins[str(EDGE)] == '363fdef3813b72a5e45a42f17034cd5a544b654c838929cecf6e1ce5f60625cb'
    assert pins[str(META)] == '86ccf5df0c67419f8c5f43e93a7ed38d23a080e9f7fde26737290252f3780098'
    cols = ['banc_888_id', 'root_888', 'region', 'cell_type', 'cell_sub_class', 'super_class', 'side',
            'tract', 'neuromere', 'body_part_sensory', 'body_part_effector', 'neurotransmitter_predicted',
            'neurotransmitter_verified', 'fafb_match']
    rows = feather.read_table(META, columns=cols).to_pylist()
    meta = {}
    for row in rows:
        root = row['banc_888_id']
        if root:
            assert root not in meta
            meta[root] = row
    marks = defaultdict(list)
    for row in parquet.read_table(MARKERS).to_pylist():
        if row['valid']:
            marks[str(row['pt_root_id'])].append([float(p) * scale for p, scale in zip(row['pt_position'], (4, 4, 45), strict=True)])
    receipts = read(HERE / 'GEOMETRY_RECEIPTS.json')
    ports, rejected = {}, {}
    for key, (label, direction, *roots) in read(HERE / 'PORTS.json').items():
        for root in roots:
            row = meta[root]
            try:
                assert row['root_888'] == root and row['tract'] == 'cervical_connective'
                assert row['super_class'] == CLASS[direction]
                assert row['cell_sub_class' if direction == 'SA' else 'cell_type'] == label
                assert row['side'] in ('left', 'right')
                assert marks[root], 'no valid v888 anterior marker for this root'
                assert signed(row)[0] in (-1, 1), 'sign unestablished for this model'
                tried = []
                for asset in receipts[root]['assets']:
                    path = HERE / 'geometry' / asset['filename']
                    assert hash_file(path) == asset['sha256']
                    try:
                        geom = crossings(path, marks[root])
                    except (AssertionError, ValueError) as error:
                        tried.append(f'{path.name}: {error}')
                    else:
                        break
                else:
                    raise ValueError(tried or 'official skeleton unavailable')
            except (AssertionError, ValueError) as error:
                rejected[root] = {'group': key, 'reason': str(error)}
                continue
            sign, nt, basis = signed(row)
            ports[root] = dict(row, group=key, direction=direction, geometry=geom,
                               object=asset, sign=sign, transmitter=nt, transmitter_basis=basis)
    outgoing, incoming = defaultdict(list), defaultdict(list)
    data = feather.read_table(EDGE, columns=['pre', 'post', 'count'], memory_map=True)
    root_keys = pa.array(list(ports))
    hits = data.filter(pc.or_(pc.is_in(data['pre'], value_set=root_keys), pc.is_in(data['post'], value_set=root_keys)))
    for pre, post, count in zip(*(column.to_pylist() for column in hits.columns), strict=True):
        assert type(count) is int and count > 0
        if pre in ports:
            target = meta.get(post)
            if target and (target['region'] == VNC if ports[pre]['direction'] == 'DN' else target['region'] in BRAIN):
                outgoing[pre].append((post, count))
        if post in ports and ports[post]['direction'] == 'AN':
            receptor = meta.get(pre)
            spec = ports[post]
            if (receptor and receptor['cell_sub_class'] == SENSORS[spec['group']]
                    and receptor['side'] == spec['side'] and receptor['super_class'] == 'sensory'
                    and signed(receptor)[0] == 1):
                incoming[post].append((pre, count))
    for root, port in list(ports.items()):
        if not outgoing[root] or (port['direction'] == 'AN' and not incoming[root]):
            rejected[root] = {'group': port['group'], 'reason': 'no proven source or post on the required side'}
            del ports[root]
            continue
        port['outgoing'] = sorted(outgoing[root])
        port['incoming_receptors'] = sorted(incoming[root])
        port['targets_by_effector'] = dict(sorted(summarize(outgoing[root], meta).items()))
    groups = defaultdict(list)
    for root, port in ports.items():
        groups[port['group']].append(root)
    for group, roots in groups.items():
        print(group, [(ports[r]['side'], len(outgoing[r]), sum(c for _, c in incoming[r])) for r in roots], flush=True)
    print('excluded:', rejected, flush=True)
    record(HERE / 'CATALOG.json', {'source': 'BANC v888 official simple v2', 'files': pins,
          'planes_nm': PLANES, 'ports': ports, 'groups': groups, 'rejected': rejected})


def summarize(edges, meta):
    totals = defaultdict(int)
    for target, count in edges:
        row = meta[target]
        totals[row['body_part_effector'] or row['super_class'] or 'unclassified'] += count
    return totals


if __name__ == '__main__':
    main()
