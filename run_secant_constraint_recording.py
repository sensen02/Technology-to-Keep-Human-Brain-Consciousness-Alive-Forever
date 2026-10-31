"""S122/P1/S2-S4: historical S118 replay with passive S121 constraint copies.

Import is inert. Parent freezes/reviews before main; no extra physics or neural
calls, state/model writes, new tuning, selection or checkpoint replacement.
Run only with --output equal to the fixed, fresh OUTPUT path.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT/'run_secant_weight_probe.py'
RECORDER = ROOT/'run_constraint_row_recording.py'
S118 = Path('/home/sensen/Desktop/engine修复验收/s118-secant-weight-step/20261024T065500Z')
S121 = Path('/home/sensen/Desktop/engine修复验收/s121-constraint-row-recording/20261024T072500Z')
FORMAL = Path('/home/sensen/Desktop/engine修复验收/s119-training-constraints/20261024T070600Z/TRAINING_CONSTRAINTS.json')
OUTPUT = Path('/home/sensen/Desktop/engine修复验收/s122-secant-constraint-pair/20261024T073500Z')
ARM = 'w36b4_secant'
ACTUAL_W = .05655357434207557
WEIGHT_SHA = 'fb09c1cee29d02fc69c2cef54d1137543dd8c73207b77e0346307b8316b092dd'
EXPECTED = dict(model_load=1, forward=121, step=120, neural_ticks=166)
FROZEN_CODE = {
    str(ORIGINAL): 'd7283b15dbac0e6921b833ca9f8a5b5483f325c4b535a4255fd2697d9151ea8b',
    str(RECORDER): '55afefe38079eba5e00e2bd37c2b249cc21a052cee9bb3079b93a82c2631d4ee',
    str(FORMAL): '582a44503fc8dd5db23fc093b90af5c16eb5d928c4bd7a43eee4d6cf1f62855d',
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
            raise ValueError('frozen source/artifact changed: '+str(name))


def load_private(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_sources():
    """Read-only saved-source validation; no imports of physics or output writes."""
    check_hashes(FROZEN_CODE)
    inputs, results = {}, {}
    for base, arm in ((S118, ARM), (S121, 'init80ms')):
        parent_inputs = json.loads((base/'INPUT_HASHES.json').read_text())
        result = json.loads((base/arm/'RESULT.json').read_text())
        complete = json.loads((base/'COMPLETE.json').read_text())
        check_hashes(parent_inputs)
        check_hashes(result['hashes'])
        check_hashes(complete['source_chain_hashes'])
        if sha(base/arm/'TRAJECTORY.npz') != result['trajectory_sha256']:
            raise ValueError('parent trajectory commitment mismatch: '+str(base))
        if result['calls'] != EXPECTED:
            raise ValueError('parent call ledger mismatch: '+str(base))
        inputs.update(parent_inputs)
        results[str(base)] = result
    if inputs[str(ORIGINAL)] != FROZEN_CODE[str(ORIGINAL)]:
        raise ValueError('S118 runner not bound to frozen historical input')
    s121_inputs = json.loads((S121/'INPUT_HASHES.json').read_text())
    if s121_inputs[str(RECORDER)] != FROZEN_CODE[str(RECORDER)]:
        raise ValueError('S121 recorder not bound to historical input')
    for name in ('CONSTRAINT_ROWS.npz', 'CONSTRAINT_ROW_METADATA.json'):
        path = S121/'init80ms'/name
        if str(path) not in results[str(S121)]['hashes']:
            raise ValueError('S121 result does not bind '+str(path))
    weight = json.loads((S118/'WEIGHT_HASHES.json').read_text())
    if weight['new_value'] != ACTUAL_W or weight['new_W_sha256'] != WEIGHT_SHA:
        raise ValueError('historical S118 actual weight identity mismatch')
    with np.load(S118/ARM/'TRAJECTORY.npz', allow_pickle=False) as z:
        actual = np.ascontiguousarray(z['weights'], dtype=np.float64)
        if actual.shape != (66, 5) or actual[36, 4] != ACTUAL_W:
            raise ValueError('historical actual W[36,4] mismatch')
        if hashlib.sha256(actual.tobytes()).hexdigest() != WEIGHT_SHA:
            raise ValueError('historical actual W bytes mismatch')
    extras = [Path(__file__).resolve(), ORIGINAL, RECORDER, FORMAL,
              *sorted(p for p in S118.rglob('*') if p.is_file()),
              *sorted(p for p in S121.rglob('*') if p.is_file()), *map(Path, inputs)]
    commitments = {str(p): sha(p) for p in dict.fromkeys(extras)}
    check_hashes(FROZEN_CODE)
    return commitments


def configure(original, commitments=None):
    """Hooks only: original COM capture first, frozen S121 copies second."""
    commitments = validate_sources() if commitments is None else commitments
    check_hashes(commitments)
    check_hashes(FROZEN_CODE)
    recorder = load_private(RECORDER, '_s122_frozen_s121')
    if (original.ARM != ARM or original.PRE != 160 or original.TICKS != 6
            or original.TIMESTEPS != {ARM: .000025}
            or original.SUBSTEPS != {ARM: 20} or original.EXPECTED != EXPECTED):
        raise ValueError('delegated S118 protocol changed')
    original.OUTPUT_PARENT = OUTPUT
    save_json, save_npz = original.atomic_json, original.atomic_npz
    original_capture = original.com_contact_record
    records, metadata = [], {}
    diagnostic = dict(passive_reproduction_only=True, historical_diagnostic_only=True,
        historical_s118=str(S118), paired_s121=str(S121), formal_s119=str(FORMAL),
        formal_s119_sha256=sha(FORMAL), constraint_row_schema_version=1,
        selection=False, deployment=False, checkpoint_replacement=False,
        new_tuning=False, actual_W36_4=ACTUAL_W, actual_W_sha256=WEIGHT_SHA,
        constraint_recording='121 original-com-first then frozen-S121 pure copies; no extra physics/neural/state writes')

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
            # S118 has retained_S115_reference, not S121's retained_s115 schema.
            if value['new_value'] != ACTUAL_W or value['new_W_sha256'] != WEIGHT_SHA:
                raise ValueError('delegated actual W differs from historical S118')
            value = dict(value, **diagnostic)
            value.update(source_hashes=dict(value['source_hashes'], **commitments),
                selection_rule='NONE: historical S118 diagnostic reproduction only; no W selection or new tuning')
        elif path.name == 'COMPLETE.json':
            check_hashes(commitments)
            if len(records) != 121:
                raise ValueError('expected 121 passive constraint samples')
            value = dict(value, **diagnostic)
            value.update(source_chain_hashes=dict(value['source_chain_hashes'], **commitments),
                constraint_samples=len(records))
        return save_json(path, value)

    def npz_hook(path, **arrays):
        path = Path(path)
        if path.name == 'TRAJECTORY.npz':
            if not np.array_equal(arrays['time'], np.asarray([float(r['time']) for r in records])):
                raise ValueError('constraint and original trajectory clocks differ')
            if arrays['weights'][36, 4] != ACTUAL_W:
                raise ValueError('actual trajectory W differs from historical S118')
            save_npz(path.parent/'CONSTRAINT_ROWS.npz', **recorder.pack_records(records))
            save_json(path.parent/'CONSTRAINT_ROW_METADATA.json', metadata)
        return save_npz(path, **arrays)

    original.com_contact_record = capture
    original.atomic_json = json_hook
    original.atomic_npz = npz_hook
    return records, commitments


def main():
    commitments = validate_sources()
    original = load_private(ORIGINAL, '_s122_untouched_s118')
    configure(original, commitments)
    original.main()


if __name__ == '__main__':
    main()
