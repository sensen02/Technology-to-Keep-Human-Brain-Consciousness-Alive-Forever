"""S139/P1: exact independently admitted S138 weights; untouched S115/S121.
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
S138 = Path('/home/sensen/Desktop/engine修复验收/s138-formal-benefit-safety-proposal/20261024T192136Z')
S136 = Path('/home/sensen/Desktop/engine修复验收/s136-safety-trajectory-jacobian/20261024T183631Z')
CANDIDATE = S138/'CANDIDATE_WEIGHTS.npz'
FORMAL = Path('/home/sensen/Desktop/engine修复验收/s119-training-constraints/20261024T070600Z/TRAINING_CONSTRAINTS.json')
OUTPUT = Path('/home/sensen/Desktop/engine修复验收/s139-formal-benefit-candidate-closed-loop/20261024T192550Z')
ARM = 'init80ms'
EXPECTED = dict(model_load=1, forward=121, step=120, neural_ticks=166)
FROZEN = {
    str(ORIGINAL): '6385a0842dfb8042f4c97cebcb71a2a68f1bd435ef5fc980d0b852bea7d13eea',
    str(RECORDER): '55afefe38079eba5e00e2bd37c2b249cc21a052cee9bb3079b93a82c2631d4ee',
    str(FORMAL): '582a44503fc8dd5db23fc093b90af5c16eb5d928c4bd7a43eee4d6cf1f62855d',
    str(CANDIDATE): '6a7a2bcea10de84eaf819424f83e7744a26fb4c1b2ee9ee9c8c0d69628d91d3a',
    str(S138/'PROPOSAL_REPORT.json'): '8ca3cd7ee8bc25e72ec77ebe76094bd7db6f0d4ecc299b1e35ba6667692cbe0a',
    str(S138/'PARENT_FORMAL_BENEFIT_PROPOSAL_INDEPENDENT_CHECK.json'): 'f7d3c1a1ebb4d2b48f87046d591caae2102e3a43d6e48beeefde2fa029fdb76e',
    str(S138/'SOLUTION.npz'): 'f0f25fa0996bd618e51c18e04180896f70c7d132b1740dfd2c51cc5c22479cc4',
    str(S136/'JACOBIAN.npz'): '39cd0861538b327c4b19350c3ac394efbcca75f56f6128a0fa4c2bc29da9cfe3',
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
    canonical = {}
    for base in (S121, S136, S138, FORMAL.parent):
        chain = json.loads((base/'INPUT_HASHES.json').read_text())
        if 'before' in chain:
            assert chain['before'] == chain['after']
            chain = chain['before']
        check_hashes(chain)
        for name, digest in chain.items():
            key = str(Path(name).resolve())
            assert key not in canonical or canonical[key] == digest
            assert name not in inputs or inputs[name] == digest
            canonical[key] = digest
            inputs[name] = digest
    complete = json.loads((S121/'COMPLETE.json').read_text())
    result = json.loads((S121/ARM/'RESULT.json').read_text())
    check_hashes(complete['source_chain_hashes'])
    check_hashes(result['hashes'])
    assert result['calls'] == EXPECTED
    assert sha(S121/ARM/'TRAJECTORY.npz') == result['trajectory_sha256']
    assert canonical[str(ORIGINAL.resolve())] == FROZEN[str(ORIGINAL)]
    assert canonical[str(RECORDER.resolve())] == FROZEN[str(RECORDER)]
    for base in (S136, S138):
        check_hashes(json.loads((base/'OUTPUT_HASHES.json').read_text()))
        assert json.loads((base/'COMPLETE.json').read_text())['status'] == 'COMPLETE'
        assert not (base/'FAILED.json').exists()
    report = json.loads((S138/'PROPOSAL_REPORT.json').read_text())
    assert report['proposal_accepted'] is True and report['physical_admission'] is False
    independent = json.loads((S138/'PARENT_FORMAL_BENEFIT_PROPOSAL_INDEPENDENT_CHECK.json').read_text())
    assert all(independent[k] is True for k in ('passed', 'evidence_passed', 'proposal_admitted', 'candidate_admitted'))
    assert independent['physical_admission'] is False
    assert Path(independent['verifier']).resolve() == ROOT/'verify_formal_benefit_safety_proposal.py'
    assert independent['verifier_sha256'] == sha(independent['verifier'])
    with np.load(CANDIDATE, allow_pickle=False) as z:
        assert z.files == ['weights']
        weights = z['weights'].copy()
    with np.load(S121/ARM/'TRAJECTORY.npz', allow_pickle=False) as z:
        baseline = z['weights'].copy()
    with np.load(S136/'JACOBIAN.npz', allow_pickle=False) as j, np.load(S138/'SOLUTION.npz', allow_pickle=False) as s:
        assert exact(baseline, j['reference_weights'])
        indices = np.asarray([(p,b) for p in (3,15,38) for b in range(5)])
        valid = np.asarray([True,True,True,False,False,True,True,True,True,True,True,True,False,False,True])
        assert exact(indices, j['weight_indices']) and exact(valid, j['valid'])
        allowed = np.zeros((66,5), dtype=bool)
        allowed[indices[:,0], indices[:,1]] = valid
        assert weights.shape == (66,5) and np.isfinite(weights).all()
        assert np.all((weights >= 0) & (weights <= 4))
        assert exact(weights[~allowed], baseline[~allowed])
        assert exact(weights[36], baseline[36]) and exact(weights == 0, baseline == 0)
        assert not exact(weights, baseline)
        assert exact(weights, s['proposed_weights']) and exact(weights-baseline, s['delta'])
        scale = .001/np.sqrt(15)
        actual_z = s['actual_z']
        assert actual_z.shape == (16,) and np.isfinite(actual_z).all()
        assert exact(actual_z[:15], (weights-baseline)[indices[:,0],indices[:,1]]/scale)
        assert np.all(actual_z[:15][~valid] == 0.)
        assert np.linalg.norm(weights-baseline) <= .001+1e-15
        assert np.max(np.abs(weights-baseline)) <= scale+1e-15
        for key in ('signed_joint_position', 'load_ratio'):
            assert exact(j['masked_'+key], np.where(valid, j[key+'_D_half'], 0.))
        qpred = j['reference_signed_joint_position'] + np.einsum('tjk,k->tj', j['masked_signed_joint_position'], scale*actual_z[:15])
        loadpred = j['reference_load_ratio'] + j['masked_load_ratio'] @ (scale*actual_z[:15])
        assert exact(qpred, s['predicted_joint_position']) and exact(loadpred, s['predicted_load_ratio'])
        assert qpred.shape == (121,42) and loadpred.shape == (121,)
    extras = [Path(__file__).resolve(), ORIGINAL, RECORDER, FORMAL, *map(Path, inputs)]
    for base in (S121, S136, S138, FORMAL.parent):
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
    recorder = load_private(RECORDER, '_s139_frozen_s121_passive')
    save_json, save_npz = original.atomic_json, original.atomic_npz
    original_capture = original.com_contact_record
    records, metadata = [], {}
    formal = json.loads(FORMAL.read_text())
    intervention = dict(task='S139/P1/S2-S4', candidate=str(CANDIDATE),
        candidate_file_sha256=sha(CANDIDATE), candidate_weights_sha256=hashlib.sha256(candidate.tobytes()).hexdigest(),
        original_weights_source=str(old_source), original_weights_source_sha256=sha(old_source),
        candidate_applied_before_initialization=True, allowed_pools=[3,15,38], pool36_unchanged=True,
        nonallowed_weights_byte_identical=True, paired_s121=str(S121), formal_s119=str(FORMAL),
        formal_s119_sha256=sha(FORMAL), training_constraints=formal,
        selection_rule=formal['rules'], constraint_row_schema_version=1,
        s138_proposal=str(S138), s138_proposal_independent_check=str(S138/'PARENT_FORMAL_BENEFIT_PROPOSAL_INDEPENDENT_CHECK.json'),
        s136_jacobian=str(S136/'JACOBIAN.npz'), prediction_source=str(S138/'SOLUTION.npz'),
        prediction_scope='full121 all42 joint position and load; S136 masked Jacobian times exact S138 actual_z',
        prediction_is_physical_admission=False, formal_short_window_pass_is_autonomous_success=False,
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
    original = load_private(ORIGINAL, '_s139_untouched_s115')
    configure(original, validated)
    original.main()


if __name__ == '__main__':
    main()
