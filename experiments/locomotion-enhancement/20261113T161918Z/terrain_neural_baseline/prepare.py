"""Stdlib bind already-calibrated four NN identities to actual tested terrain model.

No new calibration, scientific imports, model construction or array reads.
"""
from pathlib import Path
import argparse
import hashlib
import json

HERE = Path(__file__).resolve().parent
PROJECT = Path('/run/media/sensen/Data2/cell_wound_prototype')
UPSTREAM = PROJECT/'experiments/neck-real-structure/20261113T064313Z/sensory_body_repair'
TERRAIN = HERE.parent/'terrain_body_baseline'
LOOP = PROJECT/'experiments/neck-real-structure/20261113T045302Z/sensory_body'
MOTOR = PROJECT/'experiments/food-approach/20261111T161007Z/combined_turning/COMBINED.pt'
EXPECTED = dict(brain_sha256='92dc0d243dad42b1df284ab797fa3dcaa13ca2d9ebfbb4cb434cebba7c9fe490',
    odor_sha256='e21dfb68e38760a6c74a871cfa3686cb79d1512b1ca60b5432397a456e473bfc',
    visual_sha256='996980806ac107137d6d5f4743de207fb3350e84a8a145dc5f52b8807398bb87',
    motor_sha256='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd',
    neck_sha256='9e88a04dd91aa6f9c1c1c68e0562325205e0f36f5cdd9ae41d1dd46c8171dc23')
KIND = 'SOURCE_NECK_ROUGH_TERRAIN_BASELINE_V1'
CASES = ('plane_control','flat_box_control','mild_blocks_seed17')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bind-completed-core-and-existing-policies',action='store_true')
    args = parser.parse_args()
    if not args.bind_completed_core_and_existing_policies:
        print('SOURCE_ONLY_UNBOUND no model or physical execution')
        return
    if any((HERE/name).exists() for name in ('FROZEN.json','INPUT_BINDINGS.json')):
        raise FileExistsError('No blind rebind of a prepared terrain neural stage')
    checked = read(HERE/'SOURCE_ONLY_CHECK.json')
    if not checked['all_passed'] or not all(checked['checks'].values()):
        raise ValueError('Actual source-only checks required')
    core_result = read(TERRAIN/'ACTUAL_PREFLIGHT/RESULT.json')
    if not core_result['all_passed'] or core_result['actual_native_steps'] != 6400 or core_result['actual_Core_constructions'] != 4 or any(not core_result['cases'][case]['all_passed'] for case in CASES):
        raise ValueError('Actual original-vs-derived whole-body contact preflight required')
    binding = read(UPSTREAM/'INPUT_BINDINGS.json')
    if any(binding[key] != value for key,value in EXPECTED.items()):
        raise ValueError('Only actual current92dc/e21d/9969/789f/9e88 identities admitted')
    proof = {}
    for frozen in (read(UPSTREAM/'FROZEN.json'),read(TERRAIN/'PREPARED/FROZEN.json'),checked,core_result):
        for path,digest in frozen.get('source_hashes',{}).items():
            if sha(path) != digest:
                raise ValueError('Existing bound input changed '+path)
            if path in proof and proof[path] != digest:
                raise ValueError('Conflicting ancestry SHA '+path)
            proof[path] = digest
    policy_paths = {'brain':Path(binding['brain_checkpoint']),'odor':Path(binding['odor_checkpoint']),
        'visual':Path(binding['visual_checkpoint']),'motor':MOTOR}
    for key,path in policy_paths.items():
        if sha(path) != EXPECTED[key+'_sha256']:
            raise ValueError('Actual policy bytes changed '+key)
        proof[str(path)] = sha(path)
    paths = [*HERE.glob('*.py'),HERE/'PREREG.json',HERE/'METRIC_CONTRACT.json',HERE/'SOURCE_ONLY_CHECK.json',
        UPSTREAM/'runtime.py',UPSTREAM/'INPUT_BINDINGS.json',UPSTREAM/'FROZEN.json',
        LOOP/'runner.py',LOOP/'audit.py',LOOP/'runtime.py',
        TERRAIN/'PREPARED/FROZEN.json',TERRAIN/'PREPARED/MODEL_LEDGER.json',
        TERRAIN/'ACTUAL_PREFLIGHT/RESULT.json',*(TERRAIN/'ACTUAL_PREFLIGHT').glob('*.json')]
    proof.update({str(path.resolve()):sha(path) for path in paths})
    new = dict(binding)
    new.update(kind=KIND,summary_kind='SOURCE_NECK_ROUGH_TERRAIN_BASELINE_SUMMARY_V1',
        upstream_kind=binding['kind'],upstream_summary_kind=binding['summary_kind'],upstream_root=str(UPSTREAM),
        source_calibration_changed=False,source_neural_condition='V_empty',cases=list(CASES),
        terrain_model_ledger=str(TERRAIN/'PREPARED/MODEL_LEDGER.json'),
        tested_terrain_core_result=str(TERRAIN/'ACTUAL_PREFLIGHT/RESULT.json'),
        observation_version='TERRAIN_RELATIVE_PAD_CLEARANCE_238_V1',
        actor_input_semantics_changed_on_terrain=True,new_physics_steps=0,functional_pass_claim=False)
    save(HERE/'INPUT_BINDINGS.json',new)
    proof[str(HERE/'INPUT_BINDINGS.json')] = sha(HERE/'INPUT_BINDINGS.json')
    save(HERE/'FROZEN.json',dict(kind='BOUND_UNCHANGED_FOURNN_AND_ACTUAL_TESTED_TERRAIN_CORE_V1',
        source_hashes=proof,model_loads=0,Core_constructions=0,new_physics_steps=0))
    print('BOUND_EXISTING4NN_AND_ACTUAL_TERRAIN_CORE_0MODEL_0PHYSICS')


if __name__ == '__main__':
    main()
