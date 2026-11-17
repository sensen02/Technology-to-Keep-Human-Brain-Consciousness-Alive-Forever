"""Read-only provenance/count audit; no neural or physical simulation."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / 'experiments/open-brain'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


current_path = BASE / 'BODY_CURRENT.json'
current = json.loads(current_path.read_text())
acceptance_path = Path(current['evidence'])
acceptance = json.loads(acceptance_path.read_text())
assert digest(acceptance_path) == current['evidence_sha256']
assert digest(Path(current['checkpoint'])) == current['sha256'] == acceptance['weight_sha256']
assert (acceptance['regression_pass'], acceptance['new_condition_pass'],
        acceptance['independent_replays']) == (57, 12, 69)
catalog_path = BASE / '20261117T051249Z-bidirectional/CATALOG.json'
catalog = json.loads(catalog_path.read_text())
groups = defaultdict(list)
for identity, port in catalog['ports'].items():
    groups[port['group']].append({'id': identity, 'direction': port['direction'],
                                 'cell_type': port['cell_type'], 'side': port['side']})
flight = {'takeoff', 'wing_amplitude', 'flight_steer', 'landing'}
ground_groups = {g: ports for g, ports in groups.items()
                 if ports[0]['direction'] == 'DN' and g not in flight}
counts = Counter(p['direction'] for p in catalog['ports'].values())
assert dict(counts) == {'DN': 27, 'AN': 11, 'SA': 14}
assert len(ground_groups) == 9 and sum(map(len, ground_groups.values())) == 19
assert sum(len(groups[g]) for g in flight) == 8
source_paths = [current_path, acceptance_path, catalog_path,
    Path(current['checkpoint']), Path(current['frozen_neck_parameters']),
    BASE/'20261117T170820Z-body-readaptation/tasks.py',
    BASE/'20261117T170820Z-body-readaptation/SENSITIVITY.json',
    BASE/'20261117T062447Z-body-commands/expansion.py',
    BASE/'20261117T062447Z-body-commands/PHASE/phase.py',
    BASE/'20261117T062447Z-body-commands/PHASE/FINAL.json',
    BASE/'20261117T051249Z-bidirectional/physical.py',
    BASE/'20261117T064507Z-eon-overlap/overlap.py',
    BASE/'20261117T164207Z-fixed-neck/adapters.py',
    HERE/'REPORT.zh-CN.md', Path(__file__).resolve()]
metadata = list((ROOT/'venv_body/lib').glob('python*/site-packages/flygym-*.dist-info/METADATA'))
assert len(metadata) == 1
version = next(row.removeprefix('Version: ') for row in metadata[0].read_text().splitlines()
               if row.startswith('Version: '))
assert version == '2.1.0'
source_paths += metadata
result = dict(project='P1_open_source_brain', audit_type='READ_ONLY_GAP_INVENTORY',
              body_checkpoint=current['checkpoint'], body_sha256=current['sha256'],
              flygym_version=version, counts=dict(counts),
              ground_groups=ground_groups,
              flight_groups={g: groups[g] for g in sorted(flight)},
              ascending_groups={g: ports for g, ports in groups.items()
                                if ports[0]['direction'] != 'DN'},
              baseline_result=dict(regression_pass=57, new_conditions_pass=12,
                                   independent_replays=69, sensitivity_pass=29,
                                   sensitivity_total=30),
              gaps=['continuous_speed_and_transitions', 'speed_turn_combinations',
                    'brake_phases_and_restart', 'low_translation_pivot',
                    'local_terrain_recovery', 'CPG_frequency_modulation_unvalidated',
                    'channel_specific_combination_semantics',
                    'Eon_full_ground_readout_and_ascending_feedback'],
              new_physics_steps=0, new_neural_ticks=0, new_training_updates=0,
              source_sha256={str(p): digest(p) for p in source_paths})
(HERE/'AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(dict(ground_groups=len(ground_groups), ground_axons=19,
                      flight_groups=4, flight_axons=8, ascending_axons=25,
                      version=version, baseline_identity_verified=True,
                      new_simulation=False), ensure_ascii=False))
