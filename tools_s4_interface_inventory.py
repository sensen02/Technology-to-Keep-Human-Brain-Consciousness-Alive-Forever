"""Read-only inventory of frozen P1 interface assumptions; never run archived code."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE = ROOT / 'outputs/experiment_platform/20261013_233237_7be93a1d'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def literal_defaults(source, class_name, names):
    """Read literal dataclass assignments without importing/executing the source."""
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
    if len(classes) != 1:
        raise ValueError('expected exactly one config class')
    values = {}
    for node in classes[0].body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in names:
                values[node.target.id] = ast.literal_eval(node.value)
    if set(values) != set(names):
        raise ValueError('missing literal config defaults')
    return values


def summarize_manifest(manifest, tier):
    if any(not isinstance(r, str) or not r.isdecimal() for r in tier):
        raise ValueError('tier identities must be decimal strings, never floats')
    if len(set(tier)) != len(tier):
        raise ValueError('duplicate tier identity')
    rows = manifest['rows']
    if len(rows) != manifest['n_actuators']:
        raise ValueError('actuator cardinality mismatch')
    declared, present = set(), set()
    for i, row in enumerate(rows):
        if row['actuator_index'] != i:
            raise ValueError('actuator ordering mismatch')
        for root, index in zip(row['tier_root_ids'], row['local_indices'], strict=True):
            if isinstance(root, bool) or not isinstance(root, (str, int)):
                raise ValueError('root identity must be exact')
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(tier):
                raise ValueError('invalid local index')
            if str(root) != tier[index]:
                raise ValueError('motor identity/index mismatch')
        for key in ('pool_root_ids', 'tier_root_ids'):
            if any(isinstance(r, bool) or not isinstance(r, (str, int))
                   or not str(r).isdecimal() for r in row[key]):
                raise ValueError('root identity must be an exact decimal string or integer')
        if not set(map(str, row['tier_root_ids'])).issubset(set(map(str, row['pool_root_ids']))):
            raise ValueError('present motor roots outside declared pool')
        declared.update(map(str, row['pool_root_ids']))
        present.update(map(str, row['tier_root_ids']))
    return dict(status_counts=dict(Counter(r['status'] for r in rows)),
                n_actuators=len(rows), declared_unique_roots=len(declared),
                present_unique_roots=len(present), missing_declared_roots=sorted(declared-present),
                shared_groups=len(manifest['shared_pools']),
                caveat='Coverage is declared identity coverage, not biological recruitment validation')


def inventory(bundle):
    bundle = Path(bundle).resolve()
    result_path = bundle / 'result.json'
    result = json.loads(result_path.read_text())
    archive = bundle / 'source_snapshot.zip'
    if sha(archive) != result['source_snapshot_sha256']:
        raise ValueError('archive digest mismatch')
    frozen = {}
    with zipfile.ZipFile(archive) as z:
        records = json.loads(z.read('manifest.json'))['files']
        mapping = {entry['path']: entry for entry in records}
        if len(mapping) != len(records):
            raise ValueError('duplicate archive source path')
        for relative in ('engine/embodied/sensory_channels.py', 'engine/embodied/motor_output.py',
                         'run_neural_muscle_closure.py', 'run_experiment_platform.py'):
            path = str(ROOT / relative)
            entry = mapping[path]
            raw = z.read(entry['id'])
            if hashlib.sha256(raw).hexdigest() != result['source_hashes'][path]:
                raise ValueError('frozen source digest mismatch')
            frozen[relative] = raw.decode('utf-8')
    sensory = literal_defaults(frozen['engine/embodied/sensory_channels.py'], 'ChannelConfig',
                               ('current_nA_per_unit', 'reference_force', 'reference_angle_rad',
                                'reference_velocity_rad_s', 'unmodelled_channels_emit_zero'))
    motor = literal_defaults(frozen['engine/embodied/motor_output.py'], 'MotorOutputConfig',
                             ('filter_tau_s', 'rate_for_full_control_hz', 'max_control'))
    if motor != result['evidence_schema']['motor_translation']:
        raise ValueError('archived defaults differ from recorded decoder')
    # The archived call site must explicitly use these defaults for this inventory.
    tree = ast.parse(frozen['run_neural_muscle_closure.py'])
    for name in ('ChannelConfig', 'MotorOutputConfig'):
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == name]
        if not calls or any(n.args or n.keywords for n in calls):
            raise ValueError('nondefault configuration requires a new inventory policy')
    platform = ast.parse(frozen['run_experiment_platform.py'])
    omit = [ast.literal_eval(n.value) for n in platform.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'OMIT' for t in n.targets)]
    if len(omit) != 1:
        raise ValueError('missing disabled-channel policy')
    return dict(schema_version=1, scope='S4 preparation only; no new simulation or biological calibration',
                bundle=str(bundle), inputs={str(result_path): sha(result_path), str(archive): sha(archive)},
                source_hashes={str(ROOT / p): result['source_hashes'][str(ROOT / p)] for p in frozen},
                parameter_class='hypothesis (source-attested values, not measured physiology)',
                sensory_defaults=sensory, motor_defaults=motor, disabled_channels=list(omit[0]),
                sensory_policy=result['sensory_policy'], clocks=result['clocks'],
                motor_coverage=summarize_manifest(result['motor_manifest'], result['tier_root_ids']),
                body_branch=result['branch'], biological_mapping_verified=result['biological_mapping_verified'],
                required_data=['paired joint angle/velocity and identified sensory spikes with synchronized timestamps',
                               'identified motor spikes and per-bundle recruitment/activation',
                               'activation, geometry and calibrated force/torque with units',
                               'immutable local metadata generation manifest and individual-root lineage'],
                acceptance_boundary='Hash/config/identity checks only; convergence, standing and biological validity unproven')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    data = inventory(args.bundle)
    # Preserve all prior reports: explicitly refuse overwrite.
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps(data['motor_coverage'], ensure_ascii=False))


if __name__ == '__main__':
    main()
