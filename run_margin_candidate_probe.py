"""S133/P1: exact S132 weights, untouched S115 physics and passive S121 rows.
Import is inert. Parent alone invokes main once after SOURCE READY.
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
S132 = Path('/home/sensen/Desktop/engine修复验收/s132-margin-contact-candidate/20261024T180846Z')
CANDIDATE = S132/'CANDIDATE_WEIGHTS.npz'
FORMAL = Path('/home/sensen/Desktop/engine修复验收/s119-training-constraints/20261024T070600Z/TRAINING_CONSTRAINTS.json')
OUTPUT = Path('/home/sensen/Desktop/engine修复验收/s133-margin-candidate-closed-loop/20261024T181113Z')
ARM = 'init80ms'
EXPECTED = dict(model_load=1, forward=121, step=120, neural_ticks=166)
# Literal identities filled before SOURCE READY; historical sources never edited.
FROZEN = {
    str(ORIGINAL): '6385a0842dfb8042f4c97cebcb71a2a68f1bd435ef5fc980d0b852bea7d13eea',
    str(RECORDER): '55afefe38079eba5e00e2bd37c2b249cc21a052cee9bb3079b93a82c2631d4ee',
    str(FORMAL): '582a44503fc8dd5db23fc093b90af5c16eb5d928c4bd7a43eee4d6cf1f62855d',
    str(CANDIDATE): '7f3154c0de1d238ec126d8d3b5f6e6d62e802dc57ce1b2566fbad1a0fec4b6fb',
    str(S132/'SUMMARY.json'): '0bb5a6d588edcbb87f3ad7bec8b80baeeaad4488aaa2d034eeb3ed6d8bc4977f',
    str(S132/'PARENT_INDEPENDENT_CHECK_FINAL.json'): 'ef9ecb04a6a65f58bc7ae3e9e2e27cbace9c58ffde872af66c59b1137a3f79b4',
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
    check_hashes(FROZEN)
    inputs = {}
    for base in (S121, S132, FORMAL.parent):
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
    check_hashes(json.loads((S132/'OUTPUT_HASHES.json').read_text()))
    report = json.loads((S132/'SUMMARY.json').read_text())
    assert report['offline_candidate_saved'] is True
    assert report['physical_admission'] is False
    independent = json.loads((S132/'PARENT_INDEPENDENT_CHECK_FINAL.json').read_text())
    assert all(independent[k] is True for k in ('passed','evidence_passed','proposal_accepted','certificate_passed','input_output_hashes_rechecked'))
    assert independent['verifier_sha256'] == sha(ROOT/'verify_margin_contact_candidate.py')
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
    with np.load(S132/'OFFLINE_CANDIDATE.npz', allow_pickle=False) as z:
        assert exact(weights, z['proposed_weights'])
        assert exact(baseline, z['reference_weights'])
        assert exact(weights, z['reference_weights'] + z['delta'])
        assert z['predicted_delta_qacc'].shape == (3,48)
    extras = [Path(__file__).resolve(), ORIGINAL, RECORDER, FORMAL, *map(Path, inputs)]
    for base in (S121, S132, FORMAL.parent):
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
    original.WEIGHTS = CANDIDATE
    local_weights = original.local_weights
    def exact_loaded_weights(weights):
        assert exact(weights, candidate)
        arms = local_weights(weights)
        assert list(arms) == [ARM] and exact(arms[ARM], candidate)
        return arms
    original.local_weights = exact_loaded_weights
    original.OUTPUT_PARENT = OUTPUT
    recorder = load_private(RECORDER, '_s133_frozen_s121_passive')
    save_json, save_npz = original.atomic_json, original.atomic_npz
    original_capture = original.com_contact_record
    records, metadata = [], {}
    formal = json.loads(FORMAL.read_text())
    intervention = dict(task='S133/P1/S2-S4', candidate=str(CANDIDATE),
        candidate_file_sha256=sha(CANDIDATE), candidate_weights_sha256=hashlib.sha256(candidate.tobytes()).hexdigest(),
        original_weights_source=str(old_source), original_weights_source_sha256=sha(old_source),
        candidate_applied_before_initialization=True, allowed_pools=[3, 15, 38], pool36_unchanged=True,
        nonallowed_weights_byte_identical=True, paired_s121=str(S121), formal_s119=str(FORMAL),
        formal_s119_sha256=sha(FORMAL), training_constraints=formal,
        selection_rule=formal['rules'], constraint_row_schema_version=1,
        prediction_source=str(S132/'OFFLINE_CANDIDATE.npz'), prediction_scope='3 fixed S121 states only; no full121 or closed-loop prediction',
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
                         all_weights_identical=False, all_weights_identical_between_arms=True,
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
    original = load_private(ORIGINAL, '_s133_untouched_s115')
    configure(original, validated)
    original.main()


if __name__ == '__main__':
    main()
