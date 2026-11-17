"""Stdlib-only exclusive preparation of the single fixed +.55 terrain repair.

No arrays, neural models, Core or physical backend are loaded. Existing binary
artifacts may be streamed only for SHA256 provenance verification. Execute only
with --prepare after all new source files and SOURCE_ONLY_CHECK.json are ready.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import json
import math
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
OLD_NEURAL = HERE.parent / 'terrain_neural_baseline'
OLD_BODY = HERE.parent / 'terrain_body_baseline'
DIAGNOSTIC = HERE.parent / 'terrain_start_area_diagnostic' / 'ACTUAL'
CASE = 'mild_blocks_seed17_start_repair_v1'
OLD_CASE = 'mild_blocks_seed17'
KIND = 'SOURCE_NECK_ROUGH_TERRAIN_START_REPAIR_V1'
SUMMARY_KIND = 'SOURCE_NECK_ROUGH_TERRAIN_START_REPAIR_SUMMARY_V1'
DX = .55
HEIGHTS = [.04, .04, .04, .04, .02, .04]
BLOCKS = ['terrain_block_%02d' % i for i in range(6)]
EXPECTED_IDS = {
    'brain_sha256': '92dc0d243dad42b1df284ab797fa3dcaa13ca2d9ebfbb4cb434cebba7c9fe490',
    'odor_sha256': 'e21dfb68e38760a6c74a871cfa3686cb79d1512b1ca60b5432397a456e473bfc',
    'visual_sha256': '996980806ac107137d6d5f4743de207fb3350e84a8a145dc5f52b8807398bb87',
    'motor_sha256': '789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd',
    'neck_sha256': '9e88a04dd91aa6f9c1c1c68e0562325205e0f36f5cdd9ae41d1dd46c8171dc23',
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def require(flag, message):
    if not flag:
        raise ValueError(message)


def near(a, b):
    return math.isfinite(float(a)) and abs(float(a) - float(b)) <= 1e-12


def passed(value, flag='all_passed'):
    return (value.get(flag) is True and bool(value.get('checks'))
            and all(item is True for item in value['checks'].values()))


def merge(proof, mapping):
    """Verify every ancestry hash now, refusing conflicting paths or digests."""
    for path, digest in mapping.items():
        item = Path(path)
        require(item.is_absolute(), 'Provenance path must be absolute: ' + path)
        require(sha(item) == digest, 'Existing provenance drift: ' + path)
        require(path not in proof or proof[path] == digest, 'Conflicting ancestry: ' + path)
        proof[path] = digest


def bind(proof, path):
    path = str(Path(path).resolve())
    merge(proof, {path: sha(path)})


def bind_artifacts(proof, value):
    """Check saved artifact bytes through metadata only; never deserialize arrays."""
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            merge(proof, {value['path']: value['sha256']})
        for child in value.values():
            bind_artifacts(proof, child)
    elif isinstance(value, list):
        for child in value:
            bind_artifacts(proof, child)


def verify_archives(proof):
    controls = {}
    for case in ('plane_control', 'flat_box_control'):
        folder = OLD_NEURAL / 'trials' / case
        result_path, audit_path = folder / 'RESULT.json', folder / 'VERIFY.json'
        result, audit = read(result_path), read(audit_path)
        require(passed(audit, 'integrity_all_pass'), 'Unqualified archive audit: ' + case)
        require(audit['result_sha256'] == sha(result_path), 'Archive result hash mismatch: ' + case)
        require(result['terrain_case'] == case and result['condition'] == 'V_empty', 'Archive case mismatch')
        require(result['physical_steps'] == result['native']['actual_native_steps'] == 1232000
                and result['native']['post_native_steps'] == 768000, 'Archive full duration required')
        require(result['engineering_pass'] is True and result['functional_stable_body'] is True
                and result['terrain_summary']['behavior_diagnostic_pass'] is True,
                'Archive behavior qualification missing: ' + case)
        merge(proof, result['source_hashes'])
        merge(proof, audit['source_hashes'])
        bind_artifacts(proof, result)
        bind(proof, result_path)
        bind(proof, audit_path)
        controls[case] = dict(result_path=str(result_path), result_sha256=sha(result_path),
            audit_path=str(audit_path), audit_sha256=sha(audit_path), archive_only=True,
            physical_steps=1232000, fresh_rerun=False, integrity_qualified=True)
    jobs_path = OLD_NEURAL / 'JOBS.json'
    jobs = read(jobs_path)
    expected = [('plane_control', 'runner.py', 0), ('plane_control', 'audit.py', 0),
                ('flat_box_control', 'runner.py', 0), ('flat_box_control', 'audit.py', 0),
                (OLD_CASE, 'runner.py', 1)]
    require([(job['case'], job['entry'], job['exit_code']) for job in jobs] == expected,
            'Old five-stage JOBS sequence must remain exactly [0,0,0,0,1]')
    require(all(job['status'] == 'FINISHED' for job in jobs), 'Old JOBS must be terminal')
    bind(proof, jobs_path)
    for job in jobs:
        bind(proof, job['log'])
    folder = OLD_NEURAL / 'trials' / OLD_CASE
    failure_path, partial_path = folder / 'FAILURE.json', folder / 'PARTIAL_NATIVE.json'
    failure, partial, guard = read(failure_path), read(partial_path), read(folder / 'WARM_FLAT_CHECK.json')
    require(failure['error'] == "ValueError('Terrain reached during frozen world-z warm teacher; ANN phase not admitted')",
            'Old failure must remain the actual flat-warm admission refusal')
    require(failure['physical_endpoints'] == 5800 and failure['neural_records'] == 1160,
            'Old mild prefix must remain 5800 ms')
    require(partial['actual_native_steps'] == 464000 and partial['post_native_steps'] == 0,
            'Old mild did not admit a post-warm neural phase')
    require(guard['all_passed'] is False and guard['checked_before_ANN'] is True
            and guard['no_state_reset'] is True, 'Old warm guard evidence changed')
    require(not (folder / 'RESULT.json').exists() and not (folder / 'VERIFY.json').exists(),
            'Old mild partial must not be overwritten as a completed run')
    bind_artifacts(proof, failure)
    for path in (failure_path, partial_path, folder / 'WARM_FLAT_CHECK.json'):
        bind(proof, path)
    return controls, dict(failure_path=str(failure_path), failure_sha256=sha(failure_path),
        partial_native_path=str(partial_path), partial_native_sha256=sha(partial_path),
        jobs_path=str(jobs_path), jobs_sha256=sha(jobs_path), actual_native_steps=464000,
        post_native_steps=0, status='NOT_ADMITTED_NOT_TESTED', old_error_preserved=True)


def diagnostic(proof):
    path = DIAGNOSTIC / 'RESULT.json'
    value = read(path)
    require(passed(value), 'Actual saved-start diagnostic all checks PASS required')
    require(value['source_hashes_before'] == value['source_hashes_after'],
            'Diagnostic before/after source identities differ')
    merge(proof, value['source_hashes_before'])
    for name in ('SOURCE_HASHES_BEFORE.json', 'SOURCE_HASHES_AFTER.json'):
        require(read(DIAGNOSTIC / name) == value['source_hashes_before'], 'Diagnostic source manifest mismatch')
        bind(proof, DIAGNOSTIC / name)
    require(value['NN_forwards'] == value['Core_constructions'] == value['new_physical_steps'] == 0
            and value['scene_modified'] is False, 'Diagnostic must remain saved-data-only')
    plane = value['reports']['plane_control']
    analysis = plane['flat_start_area_analysis_only']
    raw = plane['prefix_0_to_5p8']['sphere_material_extent_max_xyz'][0]
    necessary = analysis['necessary_first_forward_obstacle_x']
    require(near(raw, 1.2492113609914324) and near(analysis['engineering_margin_native'], .05),
            'Frozen raw warm sphere extent or engineering margin changed')
    require(near(necessary, raw + .05)
            and near(analysis['all_warm_sphere_material_xy_max'][0], necessary),
            'Diagnostic necessary bound already contains margin; do not add it twice')
    require(near(analysis['original_first_obstacle_x'], .8)
            and analysis['scene_written_or_moved'] is False, 'Original scene diagnostic mismatch')
    bind(proof, path)
    return dict(result_path=str(path), result_sha256=sha(path),
        raw_warm_material_max_x=raw, engineering_margin_native=.05,
        necessary_first_forward_obstacle_x=necessary,
        saved_ANN_max_sphere_leading_x=analysis['saved_ANN_max_sphere_leading_x'],
        saved_ANN_geometry_is_new_scene_physical_proof=False)


def translated_scene(source_case):
    source_path = Path(source_case['xml'])
    require(sha(source_path) == source_case['xml_sha256'], 'Old source XML mismatch')
    original = ET.parse(source_path).getroot()
    changed = copy.deepcopy(original)
    all_blocks = [geom for geom in changed.iter('geom')
                  if geom.get('name', '').startswith('terrain_block_')]
    require([geom.get('name') for geom in all_blocks] == BLOCKS, 'Exactly six seed17 blocks required')
    world = changed.find('worldbody')
    require(world is not None and all(geom in list(world) for geom in all_blocks),
            'Blocks must remain static direct-world geometries')
    terrain = copy.deepcopy(source_case['terrain'])
    ledger_blocks = [item for item in terrain if item['name'].startswith('terrain_block_')]
    require([item['name'] for item in ledger_blocks] == BLOCKS, 'Ledger block order mismatch')
    require(len(terrain) == 7 and terrain[0] == dict(name='terrain_plane', top_z=0., footprint='infinite'),
            'Unchanged infinite registered plane and six blocks required')
    old_positions, changes = {}, []
    for index, (geom, item) in enumerate(zip(all_blocks, ledger_blocks)):
        old_pos = geom.attrib['pos']
        tokens = old_pos.split()
        require(len(tokens) == 3 and geom.get('type') == 'box' and geom.get('group') == '5',
                'Expected registered axis-aligned box')
        require('quat' not in geom.attrib and 'euler' not in geom.attrib
                and 'axisangle' not in geom.attrib and 'xyaxes' not in geom.attrib
                and 'zaxis' not in geom.attrib, 'Block orientation must remain identity')
        x, y, z = map(float, tokens)
        sx, sy, sz = map(float, geom.attrib['size'].split())
        lo, hi, ymin, ymax = item['footprint']
        require(near(item['top_z'], HEIGHTS[index]) and near(z + sz, HEIGHTS[index])
                and near(z - sz, 0) and near(sx * 2, .25) and near(sy, 5.) and near(y, 0.)
                and near(lo, .8 + .38 * index) and near(hi - lo, .25)
                and near(ymin, -5.) and near(ymax, 5.) and near(x - sx, lo) and near(x + sx, hi),
                'Original height, size, spacing, position or ledger changed')
        old_positions[item['name']] = old_pos
        geom.set('pos', format(x + DX, '.17g') + ' ' + tokens[1] + ' ' + tokens[2])
        item['footprint'] = [lo + DX, hi + DX, ymin, ymax]
        translated_x = float(geom.get('pos').split()[0])
        require(near(translated_x - x, DX)
                and near(item['footprint'][0] - lo, DX) and near(item['footprint'][1] - hi, DX)
                and near(translated_x - sx, item['footprint'][0])
                and near(translated_x + sx, item['footprint'][1]), 'Fixed translation proof failed')
        changes.append(dict(name=item['name'], old_pos=old_pos, new_pos=geom.get('pos'),
                            old_footprint=[lo, hi, ymin, ymax], new_footprint=item['footprint'],
                            top_z=item['top_z'], translation_x_native=DX))
    restored = copy.deepcopy(changed)
    for geom in restored.iter('geom'):
        if geom.get('name') in old_positions:
            geom.set('pos', old_positions[geom.get('name')])
    require(ET.tostring(restored) == ET.tostring(original),
            'Unexpected XML change beyond the six block pos attributes')
    restored_terrain = copy.deepcopy(terrain)
    for item, before in zip([item for item in restored_terrain if item['name'].startswith('terrain_block_')],
                            [item for item in source_case['terrain'] if item['name'].startswith('terrain_block_')]):
        item['footprint'] = copy.deepcopy(before['footprint'])
    require(restored_terrain == source_case['terrain'], 'Unexpected ledger mutation')
    return ET.tostring(changed, encoding='unicode') + '\n', terrain, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if not args.prepare:
        print('SOURCE_ONLY no preparation, arrays, models or physics')
        return
    require(not any((HERE / name).exists() for name in ('PREPARED', 'INPUT_BINDINGS.json', 'FROZEN.json')),
            'Exclusive preparation only; no blind overwrite or retry')
    proof = {}
    checked = read(HERE / 'SOURCE_ONLY_CHECK.json')
    require(passed(checked), 'New actual SOURCE_ONLY_CHECK all checks PASS required')
    require(bool(checked.get('source_hashes')), 'New source-only source hashes required')
    merge(proof, checked['source_hashes'])
    for path in (OLD_NEURAL / 'FROZEN.json', OLD_BODY / 'PREPARED/FROZEN.json'):
        frozen = read(path)
        require(bool(frozen['source_hashes']), 'Complete old frozen ancestry required')
        merge(proof, frozen['source_hashes'])
        bind(proof, path)
    old_preflight_path = OLD_NEURAL / 'PREFLIGHT.json'
    old_preflight = read(old_preflight_path)
    require(passed(old_preflight) and len(old_preflight['checks']) == 20
            and old_preflight['state_total'] == 336 and old_preflight['Core_constructions'] == 0
            and old_preflight['new_physics_steps'] == 0
            and old_preflight['sourcefreeze_sha256'] == sha(OLD_NEURAL / 'FROZEN.json'),
            'Actual old four-NN zero-Core preflight identity required; not a fresh scene test')
    bind(proof, old_preflight_path)
    controls, old_partial = verify_archives(proof)
    diag = diagnostic(proof)
    old_binding_path = OLD_NEURAL / 'INPUT_BINDINGS.json'
    old_binding = read(old_binding_path)
    require(all(old_binding[key] == value for key, value in EXPECTED_IDS.items()), 'Policy/neck IDs changed')
    bind(proof, old_binding_path)
    ledger_path = OLD_BODY / 'PREPARED/MODEL_LEDGER.json'
    old_ledger = read(ledger_path)
    source_case = old_ledger['cases'][OLD_CASE]
    bind(proof, ledger_path)
    xml_text, terrain, changes = translated_scene(source_case)
    first_edge = terrain[1]['footprint'][0]
    require(near(first_edge, 1.35) and first_edge > diag['necessary_first_forward_obstacle_x'],
            'Fixed first edge must be 1.35 and exceed the diagnostic necessary bound')
    # Freeze every assembled new Python source only after source-only evidence exists.
    for path in [*sorted(HERE.glob('*.py')), HERE / 'PREREG.json', HERE / 'SOURCE_ONLY_CHECK.json']:
        bind(proof, path)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    prepared = HERE / 'PREPARED'
    prepared.mkdir(exist_ok=False)
    scene_dir = prepared / timestamp
    scene_dir.mkdir(exist_ok=False)
    xml_path = scene_dir / (CASE + '.xml')
    with xml_path.open('x') as stream:
        stream.write(xml_text)
    case = copy.deepcopy(source_case)
    case.update(xml=str(xml_path), xml_sha256=sha(xml_path), terrain=terrain,
        scene_version=CASE, prepared_utc=timestamp, source_case=OLD_CASE,
        source_xml=source_case['xml'], source_xml_sha256=source_case['xml_sha256'],
        translation_world_x_native=DX, first_forward_edge_native=first_edge,
        change_reason='Only the fixed initial flat start area is extended; all six old blocks translated +.55 world X')
    ledger = copy.deepcopy(old_ledger)
    ledger.update(kind='FULL_BODY_TERRAIN_START_REPAIR_DERIVATIVE_SOURCE_ONLY_V1', cases={CASE: case},
        scene_version=CASE, prepared_utc=timestamp, parent_ledger=str(ledger_path),
        parent_ledger_sha256=sha(ledger_path), only_six_block_world_x_changed=True,
        new_physics_steps=0, functional_pass_claim=False)
    save(prepared / 'MODEL_LEDGER.json', ledger)
    scene_proof = dict(kind='EXACT_SIX_BLOCK_X_TRANSLATION_PROOF_V1', all_passed=True,
        source_xml=source_case['xml'], source_xml_sha256=source_case['xml_sha256'],
        xml=str(xml_path), xml_sha256=sha(xml_path), prepared_utc=timestamp,
        translation_world_x_native=DX, changes=changes, first_edge=first_edge,
        diagnostic=diag, original_xml_restored_serialization_exact=True,
        original_ledger_restored_exact=True, unchanged_height_sequence=HEIGHTS,
        unchanged_width_native=.25, unchanged_y_extent=[-5., 5.],
        all_other_XML_attributes_text_and_elements_unchanged=True,
        whole_body_initial_root_pose_friction_and_physics_unchanged=True,
        geometric_necessary_bound_only=True, new_scene_physical_pass_claim=False,
        archived_controls=controls, old_partial=old_partial,
        reused_fourNN_preflight=dict(path=str(old_preflight_path), sha256=sha(old_preflight_path),
            state_total=336, checks=20, archive_only=True, fresh_fourNN_test=False),
        new_Core_constructions=0, new_NN_forwards=0, new_physical_steps=0)
    save(prepared / 'SCENE_TRANSLATION_PROOF.json', scene_proof)
    binding = copy.deepcopy(old_binding)
    binding.update(kind=KIND, summary_kind=SUMMARY_KIND,
        upstream_kind=old_binding['kind'], upstream_summary_kind=old_binding['summary_kind'],
        upstream_root=str(OLD_NEURAL), upstream_bindings=str(old_binding_path),
        upstream_bindings_sha256=sha(old_binding_path), source_calibration_changed=False,
        cases=[CASE], scene_version=CASE, prepared_utc=timestamp,
        terrain_model_ledger=str(prepared / 'MODEL_LEDGER.json'),
        tested_terrain_core_result=str(HERE / 'ACTUAL_PREFLIGHT/RESULT.json'),
        new_scene_preflight_pending=True, translation_world_x_native=DX,
        archive_controls=controls, old_partial=old_partial,
        reused_fourNN_preflight=scene_proof['reused_fourNN_preflight'],
        new_fourNN_preflight_claim=False, new_physics_steps=0, functional_pass_claim=False)
    save(HERE / 'INPUT_BINDINGS.json', binding)
    for path in (xml_path, prepared / 'MODEL_LEDGER.json', prepared / 'SCENE_TRANSLATION_PROOF.json',
                 HERE / 'INPUT_BINDINGS.json'):
        bind(proof, path)
    save(prepared / 'FROZEN.json', dict(kind='TERRAIN_START_REPAIR_PREPARED_SOURCE_ONLY_V1',
        source_hashes=dict(proof), cases=[CASE], prepared_utc=timestamp,
        Core_constructions=0, NN_forwards=0, new_physics_steps=0))
    bind(proof, prepared / 'FROZEN.json')
    save(HERE / 'FROZEN.json', dict(kind='BOUND_UNCHANGED_FOURNN_AND_FIXED_START_REPAIR_V1',
        source_hashes=proof, cases=[CASE], prepared_utc=timestamp,
        old_controls_archive_only=True, reused_fourNN_preflight=True,
        new_scene_preflight_pending=True, Core_constructions=0, NN_forwards=0,
        new_physics_steps=0, functional_pass_claim=False))
    print(json.dumps(dict(prepared=str(prepared), xml=str(xml_path), cases=[CASE],
        first_edge=first_edge, dx=DX, new_physical_steps=0, new_NN_forwards=0)))


if __name__ == '__main__':
    main()
