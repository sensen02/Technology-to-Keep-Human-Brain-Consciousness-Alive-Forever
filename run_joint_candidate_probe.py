"""S127/P1/S2-S4: one exact S126 candidate, full-body 3ms closed-loop probe.

Import is inert. Untouched S115 is privately imported; only its WEIGHTS source
and serialization/passive S121 capture hooks change. No checkpoint promotion.
Parent must freeze/review this file before invoking main ONCE.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT/'run_extended_initialization_probe.py'
RECORDER = ROOT/'run_constraint_row_recording.py'
S121 = Path('/home/sensen/Desktop/engine修复验收/s121-constraint-row-recording/20261024T072500Z')
S126 = Path('/home/sensen/Desktop/engine修复验收/s126-fixed-contraction/20261024T081500Z')
CANDIDATE = S126/'CANDIDATE_WEIGHTS.npz'
FORMAL = Path('/home/sensen/Desktop/engine修复验收/s119-training-constraints/20261024T070600Z/TRAINING_CONSTRAINTS.json')
OUTPUT = Path('/home/sensen/Desktop/engine修复验收/s127-joint-candidate-closed-loop/20261024T082500Z')
ARM = 'init80ms'
EXPECTED = dict(model_load=1, forward=121, step=120, neural_ticks=166)
FROZEN = {
    str(ORIGINAL): '6385a0842dfb8042f4c97cebcb71a2a68f1bd435ef5fc980d0b852bea7d13eea',
    str(RECORDER): '55afefe38079eba5e00e2bd37c2b249cc21a052cee9bb3079b93a82c2631d4ee',
    str(FORMAL): '582a44503fc8dd5db23fc093b90af5c16eb5d928c4bd7a43eee4d6cf1f62855d',
    str(CANDIDATE): '01508028135c118d5c37dc1842da06164c64e2701932185bea8e4de38b1de49d',
    str(S126/'SUMMARY.json'): '9c9fa07be3db87446c67768f8fdd80bb132eaf44c65988890c89c2d3fef23a7a',
    str(S126/'PARENT_INDEPENDENT_CHECK.json'): '48f309c866615fdd19fff7bb0b87a8753842ae73ef194e3205d7fb4bc0df5656',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def check_hashes(expected):
    for name, digest in expected.items():
        if sha(name) != digest:
            raise ValueError('frozen input changed: '+str(name))


def load_private(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def validate_sources():
    """Read-only validation of immutable parents and exact full candidate bytes."""
    check_hashes(FROZEN)
    inputs = {}
    for base in (S121, S126, FORMAL.parent):
        chain = json.loads((base/'INPUT_HASHES.json').read_text())
        if 'before' in chain:
            assert chain['before'] == chain['after']
            chain = chain['before']
        check_hashes(chain)
        for name, digest in chain.items():
            if name in inputs and inputs[name] != digest:
                raise ValueError('conflicting historical identity: '+name)
            inputs[name] = digest
    complete = json.loads((S121/'COMPLETE.json').read_text())
    result = json.loads((S121/ARM/'RESULT.json').read_text())
    check_hashes(complete['source_chain_hashes'])
    check_hashes(result['hashes'])
    assert result['calls'] == EXPECTED
    assert sha(S121/ARM/'TRAJECTORY.npz') == result['trajectory_sha256']
    assert inputs[str(ORIGINAL)] == FROZEN[str(ORIGINAL)]
    assert inputs[str(RECORDER)] == FROZEN[str(RECORDER)]
    check_hashes(json.loads((S126/'OUTPUT_HASHES.json').read_text()))
    report = json.loads((S126/'SUMMARY.json').read_text())
    assert report['checks'] and all(v is True for v in report['checks'].values())
    assert report['offline_candidate_saved'] is True and report['all_inputs_unchanged'] is True
    independent = json.loads((S126/'PARENT_INDEPENDENT_CHECK.json').read_text())
    assert independent['passed'] is True
    check_hashes(independent['input_sha256'])
    with np.load(CANDIDATE, allow_pickle=False) as z:
        weights = z['weights'].copy()
    with np.load(S121/ARM/'TRAJECTORY.npz', allow_pickle=False) as z:
        baseline = z['weights'].copy()
    assert weights.shape == (66, 5) and np.isfinite(weights).all()
    assert np.all((weights >= 0) & (weights <= 4))
    allowed = np.zeros((66, 5), dtype=bool); allowed[[3, 15, 38], :] = True
    assert exact(weights[~allowed], baseline[~allowed])
    assert exact(weights[36], baseline[36]) and exact(weights == 0, baseline == 0)
    assert not exact(weights, baseline)
    with np.load(S126/'LOCAL_CHAIN.npz', allow_pickle=False) as z:
        assert exact(weights, z['reference_weights'] + z['delta'])
    extras = [Path(__file__).resolve(), ORIGINAL, RECORDER, FORMAL, *map(Path, inputs)]
    for base in (S121, S126, FORMAL.parent):
        extras.extend(sorted(p for p in base.rglob('*') if p.is_file()))
    commitments = {str(p): sha(p) for p in dict.fromkeys(extras)}
    check_hashes(FROZEN)
    return commitments, weights, baseline


def configure(original, validated=None):
    commitments, candidate, baseline = validate_sources() if validated is None else validated
    check_hashes(commitments)
    assert original.PRE == 160 and original.TICKS == 6
    assert original.TIMESTEPS == {ARM: .000025} and original.SUBSTEPS == {ARM: 20}
    assert original.EXPECTED == EXPECTED and original.EXPECTED_PER_ARM == {ARM: EXPECTED}
    old_source = original.WEIGHTS
    with np.load(old_source, allow_pickle=False) as z:
        assert exact(z['weights'], baseline)
    assert str(old_source) in commitments
    # S115 reads this global once as z['weights'], before translator creation and PRE160.
    # Its S113 reference_inputs and hashes remain unaltered, including original W.
    original.WEIGHTS = CANDIDATE
    local_weights = original.local_weights
    def exact_loaded_weights(weights):
        # Called by untouched main immediately after its np.load, before mkdir/PRE.
        assert exact(weights, candidate)
        arms = local_weights(weights)
        assert list(arms) == [ARM] and exact(arms[ARM], candidate)
        return arms
    original.local_weights = exact_loaded_weights
    original.OUTPUT_PARENT = OUTPUT
    recorder = load_private(RECORDER, '_s127_frozen_s121_passive')
    save_json, save_npz = original.atomic_json, original.atomic_npz
    original_capture = original.com_contact_record
    records, metadata = [], {}
    formal = json.loads(FORMAL.read_text())
    intervention = dict(task='S127/P1/S2-S4', candidate=str(CANDIDATE),
        candidate_file_sha256=sha(CANDIDATE), candidate_weights_sha256=hashlib.sha256(candidate.tobytes()).hexdigest(),
        original_weights_source=str(old_source), original_weights_source_sha256=sha(old_source),
        candidate_applied_before_initialization=True, allowed_pools=[3, 15, 38], pool36_unchanged=True,
        nonallowed_weights_byte_identical=True, paired_s121=str(S121), formal_s119=str(FORMAL),
        formal_s119_sha256=sha(FORMAL), training_constraints=formal,
        selection_rule=formal['rules'], constraint_row_schema_version=1,
        constraint_recording='121 original COM capture first then S121 pure copies; no extra physics/neural/state writes',
        single_config=True, additional_parameter_search=False, promotion=False,
        checkpoint_replacement=False, deployment=False, standing_or_walking_claim=False)

    def capture(mj, m, d, floor_ids):
        result = original_capture(mj, m, d, floor_ids)
        records.append(recorder.constraint_record(mj, m, d, len(records)))
        if not metadata:
            metadata.update(recorder.constraint_metadata(mj, m))
        return result

    def json_hook(path, value):
        path = Path(path)
        if path.name == 'INPUT_HASHES.json':
            value = dict(value, **commitments)
        elif path.name == 'PREREG.json':
            assert value['same_weights'] == str(CANDIDATE)
            value = dict(value, **intervention)
            value.update(source_hashes=dict(value['source_hashes'], **commitments),
                         all_weights_identical=False,
                         all_weights_identical_between_arms=True,
                         exact_candidate_full_array_required=True)
        elif path.name == 'COMPLETE.json':
            check_hashes(commitments)
            assert len(records) == 121
            assert {k:value['ledger'][k] for k in EXPECTED} == EXPECTED
            value = dict(value, **intervention)
            value.update(source_chain_hashes=dict(value['source_chain_hashes'], **commitments),
                         constraint_samples=len(records), training_acceptance='PENDING_INDEPENDENT_S119_EVALUATION')
        return save_json(path, value)

    def npz_hook(path, **arrays):
        path = Path(path)
        if 'weights' in arrays:
            assert exact(arrays['weights'], candidate)
        if path.name == 'DIRECTION.npz':
            assert exact(arrays[ARM], candidate)
        if path.name == 'TRAJECTORY.npz':
            assert np.array_equal(arrays['time'], np.asarray([float(r['time']) for r in records]))
            assert exact(arrays['weights'][36], baseline[36])
            save_npz(path.parent/'CONSTRAINT_ROWS.npz', **recorder.pack_records(records))
            save_json(path.parent/'CONSTRAINT_ROW_METADATA.json', metadata)
        return save_npz(path, **arrays)

    original.com_contact_record = capture
    original.atomic_json = json_hook
    original.atomic_npz = npz_hook
    return records, commitments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    out = parser.parse_args().output
    if out != OUTPUT or not out.is_absolute() or out.exists():
        raise ValueError('only exact fixed fresh absolute OUTPUT allowed; no resume')
    validated = validate_sources()
    original = load_private(ORIGINAL, '_s127_untouched_s115')
    configure(original, validated)
    original.main()


if __name__ == '__main__':
    main()
