"""Export source-pinned IDs and permissions for a later experimental learner."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from pyarrow import feather

from repair_graph import HERE, META, ROOT, sha

OLD = ROOT / 'experiments/open-brain/20261120T031017Z-minimal-stance'
SURVEY = ROOT / 'experiments/open-brain/20261120T032302Z-peripheral-sources'
LEG_AREAS = {'F': 'front_leg', 'M': 'middle_leg', 'H': 'hind_leg'}


def export():
    paths = dict(network=HERE / 'NETWORK.npz', muscles=OLD / 'results/MUSCLE_CHAIN.json',
                 source_metadata=META, body=OLD / 'body/fruitfly_six_leg_standing.xml',
                 sensory=SURVEY / 'BANC888_CANDIDATES.json')
    sources = {str(path): sha(path) for path in paths.values()}
    with np.load(paths['network'], allow_pickle=False) as saved:
        roots = {str(root): i for i, root in enumerate(saved['root_ids'])}
    source_muscle_rows = json.loads(paths['muscles'].read_text())
    columns = ['banc_888_id', 'super_class', 'region', 'flow', 'side',
               'body_part_effector', 'peripheral_target_type']
    meta = feather.read_table(META, columns=columns).to_pylist()
    leg_motors = defaultdict(list)
    exact_target = defaultdict(list)
    for row in meta:
        key = (row['body_part_effector'], row['side'])
        if (key[0] in LEG_AREAS.values() and key[1] in ('left', 'right') and row['super_class'] == 'motor'
                and row['flow'] == 'efferent' and row['region'] == 'ventral_nerve_cord'):
            root = row['banc_888_id']
            exact_target[(*key, row['peripheral_target_type'])].append(root)
            if root in roots:
                leg_motors[key].append(root)
    muscles = []
    for row in source_muscle_rows:
        assert row['actuator_index'] == len(muscles)
        key = (LEG_AREAS[row['leg'][1]], row['side'])
        prior = set(row['pool_root_ids'])
        annotated = set(exact_target[(*key, row['muscle_target'])])
        same_leg = set(leg_motors[key])
        candidates = sorted(annotated & same_leg, key=int)
        hypotheses = sorted(same_leg - annotated, key=int)
        muscles.append(dict(actuator=row['actuator'], actuator_index=row['actuator_index'], leg=row['leg'],
            side=row['side'], muscle_target=row['muscle_target'], previous_status=row['status'],
            previous_candidate_roots=sorted(prior, key=int), annotated_candidate_roots=candidates,
            annotated_roots_absent_from_network=sorted(annotated - set(roots), key=int),
            old_roots_not_exactly_supported_in_v888=sorted(prior - annotated, key=int),
            inactive_functional_hypothesis_roots=hypotheses, weights=None))
    senses = []
    observation_links = {'leg_FeCO_claw': 'tibia_angle_rad', 'leg_FeCO_hook': 'tibia_velocity_rad_s',
        'leg_FeCO_club': None, 'leg_campaniform': None, 'leg_hair_plate': None, 'leg_tactile_bristle': None}
    for row in json.loads(paths['sensory'].read_text()):
        root, subtype = row['banc_888_id'], row['group']
        if subtype in observation_links and root in roots and row['side'] in ('left', 'right'):
            senses.append(dict(root_id=root, root_index=roots[root], group=subtype, leg=row['leg'],
                side=row['side'], nerve=row['nerve'], body_part=row['body_part_sensory'],
                candidate_observation=observation_links[subtype], emitter_enabled=False))
    result = dict(project='independent_banc_muscle_body', graph_sha256=sources[str(paths['network'])],
        body_sha256=sources[str(paths['body'])], source_sha256=sources,
        model_link_strengths=None, trainable_layer_enabled=False, actuator_command_range=[0., 1.],
        muscles=muscles, sensory_candidates=senses,
        scope='Candidate anatomy and same-leg functional hypotheses are separate. All weights unspecified and disabled.')
    assert len(muscles) == 90
    assert len(senses) == len({s['root_id'] for s in senses})
    assert sources == {str(path): sha(path) for path in paths.values()}
    (HERE / 'INTERFACE.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(len(senses), sum(bool(r['annotated_candidate_roots']) for r in muscles),
          sum(len(r['inactive_functional_hypothesis_roots']) for r in muscles))


if __name__ == '__main__':
    export()
