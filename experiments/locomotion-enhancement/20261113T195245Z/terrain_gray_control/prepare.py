"""Stdlib-only preparation of the fixed six-block brown-to-gray visual control.

Only --prepare writes exclusively new artifacts. No backend, scientific module,
checkpoint, model or array payload is imported/deserialized. Existing binary
artifacts are streamed solely for SHA256 provenance checks. Brown behavior is
an integrity-qualified reference even though its slip/rough gates failed.
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
OLD_ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z')
OLD_BROWN = OLD_ROOT / 'terrain_start_recovery_v1'
OLD_NEURAL = OLD_ROOT / 'terrain_neural_baseline'
OLD_BODY = OLD_ROOT / 'terrain_body_baseline'
OLD_CASE = 'mild_blocks_seed17_start_repair_v1'
CASE = 'mild_blocks_seed17_gray_visual_control_v1'
KIND = 'SOURCE_NECK_ROUGH_TERRAIN_GRAY_VISUAL_CONTROL_V1'
SUMMARY_KIND = 'SOURCE_NECK_ROUGH_TERRAIN_GRAY_VISUAL_CONTROL_SUMMARY_V1'
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


def passed(value, key='all_passed'):
    return (value.get(key) is True and bool(value.get('checks'))
            and all(flag is True for flag in value['checks'].values()))


def merge(proof, mapping):
    for path, digest in mapping.items():
        require(Path(path).is_absolute(), 'Absolute provenance path required: ' + path)
        require(sha(path) == digest, 'Existing frozen/artifact source drift: ' + path)
        require(path not in proof or proof[path] == digest, 'Conflicting ancestry digest: ' + path)
        proof[path] = digest


def bind(proof, path):
    path = str(Path(path).resolve())
    merge(proof, {path: sha(path)})


def bind_artifacts(proof, value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            merge(proof, {value['path']: value['sha256']})
        for child in value.values():
            bind_artifacts(proof, child)
    elif isinstance(value, list):
        for child in value:
            bind_artifacts(proof, child)


def brown_reference(proof):
    folder = OLD_BROWN / 'trials' / OLD_CASE
    result_path, audit_path = folder / 'RESULT.json', folder / 'VERIFY.json'
    result, audit = read(result_path), read(audit_path)
    require(passed(audit, 'integrity_all_pass') and audit['result_sha256'] == sha(result_path),
            'Actual brown result/fresh-audit integrity qualification required')
    require(result['kind'] == 'SOURCE_NECK_ROUGH_TERRAIN_START_REPAIR_V1'
            and result['terrain_case'] == OLD_CASE and result['condition'] == 'V_empty',
            'Brown reference identity/case mismatch')
    native = result['native']
    require(result['physical_steps'] == native['actual_native_steps'] == 1232000
            and native['post_native_steps'] == 768000, 'Complete brown 1+4.8+9.6s reference required')
    require(native['post_nonfoot_normal_peak_native'] == native['post_nonfoot_peak_native'] == 0
            and native['extra_force_max'] == 0 and native['warm_nonzero_height_contact_count'] == 0,
            'Brown strict native force/warm provenance changed')
    require(result['autonomous_cycles']['all_legs_completed'] is True, 'Brown six-leg cycles required')
    counts = native['post_native_contacts_by_terrain']
    require(sum(count for name, count in counts.items() if name.startswith('terrain_block_')) > 0,
            'Brown reference requires actual post native block contacts')
    warm_path = folder / 'WARM_FLAT_CHECK.json'
    warm = read(warm_path)
    require(passed(warm) and warm['checked_before_ANN'] is True and warm['no_state_reset'] is True,
            'Actual brown warm-flat admission proof required')
    for value in (result, audit):
        merge(proof, value['source_hashes'])
    bind_artifacts(proof, result)
    for path in (result_path, audit_path, warm_path, folder / 'IDENTITY.json',
                 folder / 'INITIAL_STATE.json', folder / 'INITIAL_TERRAIN.json'):
        bind(proof, path)
    jobs_path = OLD_BROWN / 'JOBS.json'
    jobs = read(jobs_path)
    require([(job['mode'], job['case'], job['exit_code'], job['status']) for job in jobs]
            == [('trial', OLD_CASE, 0, 'FINISHED'), ('audit', OLD_CASE, 0, 'FINISHED')],
            'Brown single full-trial/fresh-audit execution records required')
    bind(proof, jobs_path)
    for job in jobs:
        bind(proof, job['log'])
    # Preserve scientific failures verbatim; no gate requires engineering/rough PASS.
    return dict(result_path=str(result_path), result_sha256=sha(result_path),
        audit_path=str(audit_path), audit_sha256=sha(audit_path), integrity_all_pass=True,
        audit_checks_count=len(audit['checks']), all_audit_checks_passed=True,
        physical_steps=result['physical_steps'], post_native_steps=native['post_native_steps'],
        engineering_pass=result['engineering_pass'], functional_stable_body=result['functional_stable_body'],
        terrain_relative_functional_stable_body=result['terrain_relative_functional_stable_body'],
        rough_terrain_pass=result['terrain_summary']['rough_terrain_pass'],
        legacy_slip_ratio_max=result['autonomous']['slip_ratio_max'],
        relative_slip_ratio_max=result['terrain_summary']['relative_five']['slip_ratio_max'],
        legacy_gates=copy.deepcopy(result['autonomous']['gates']),
        relative_gates=copy.deepcopy(result['terrain_summary']['relative_five']['gates']),
        all_six_leg_cycles_completed=result['autonomous_cycles']['all_legs_completed'],
        post_native_contacts_by_terrain=copy.deepcopy(counts), native=copy.deepcopy(native),
        actual_xml=result['actual_xml'], actual_xml_sha256=result['actual_xml_sha256'],
        source_checkpoint_ids={key: result[key] for key in (
            'visual_checkpoint_sha256', 'odor_checkpoint_sha256', 'brain_checkpoint_sha256',
            'checkpoint_sha256', 'neck_source_binding_sha256')},
        archive_only=True, repeated_physical_steps=0, behavior_reclassified=False)


def archive_controls(proof, old_binding):
    controls = copy.deepcopy(old_binding['archive_controls'])
    require(set(controls) == {'plane_control', 'flat_box_control'}, 'Both old controls required')
    for name, item in controls.items():
        merge(proof, {item['result_path']: item['result_sha256'], item['audit_path']: item['audit_sha256']})
        result, audit = read(item['result_path']), read(item['audit_path'])
        require(passed(audit, 'integrity_all_pass') and audit['result_sha256'] == item['result_sha256'],
                'Control archive integrity changed: ' + name)
        require(result['terrain_case'] == name and result['physical_steps'] == 1232000
                and result['native']['post_native_steps'] == 768000, 'Control archive duration changed')
        item.update(archive_only=True, fresh_rerun=False, repeated_physical_steps=0)
    return controls


def gray_scene(source_case):
    source_path = Path(source_case['xml'])
    require(sha(source_path) == source_case['xml_sha256'], 'Actual brown XML SHA mismatch')
    original = ET.parse(source_path).getroot()
    require(not list(original.iter('include')), 'PENDING material audit: XML includes are not admitted')
    # Conservative guard: the inspected source has no material in any geom default.
    # Do not silently add or recolor a material if future inherited material appears.
    for default in original.iter('default'):
        require(not any('material' in geom.attrib for geom in default.iter('geom')),
                'PENDING explicit inherited-material audit; no default geom material admitted')
    changed = copy.deepcopy(original)
    world = changed.find('worldbody')
    blocks = [geom for geom in changed.iter('geom') if geom.get('name', '').startswith('terrain_block_')]
    require([geom.get('name') for geom in blocks] == BLOCKS and world is not None
            and all(geom in list(world) for geom in blocks), 'Exactly six direct-world brown blocks required')
    require([item['name'] for item in source_case['terrain'] if item['name'].startswith('terrain_block_')] == BLOCKS,
            'Brown ledger block list changed')
    old_rgba, changes = {}, []
    for geom in blocks:
        require(geom.get('type') == 'box' and geom.get('group') == '5', 'Registered box geometry required')
        require('material' not in geom.attrib, 'PENDING explicit material audit; block material not admitted')
        text = geom.attrib['rgba']
        tokens = text.split()
        require(len(tokens) == 4, 'Explicit block RGBA required')
        rgba = list(map(float, tokens))
        require(all(math.isfinite(value) and 0 <= value <= 1 for value in rgba), 'Invalid block RGBA')
        require(rgba == [.45, .40, .35, 1.], 'Only the actual fixed brown source RGB is admitted')
        mean = math.fsum(rgba[:3]) / 3.
        gray = format(mean, '.17g')
        old_rgba[geom.get('name')] = text
        geom.set('rgba', ' '.join([gray, gray, gray, tokens[3]]))
        changes.append(dict(name=geom.get('name'), old_rgba_string=text,
            new_rgba_string=geom.get('rgba'), old_rgba=rgba,
            new_rgba=[mean, mean, mean, rgba[3]], rgb_arithmetic_mean=mean,
            alpha_unchanged=True, explicit_material_present=False,
            inherited_geom_material_present=False))
    restored = copy.deepcopy(changed)
    for geom in restored.iter('geom'):
        if geom.get('name') in old_rgba:
            geom.set('rgba', old_rgba[geom.get('name')])
    original_canonical = ET.tostring(original)
    require(ET.tostring(restored) == original_canonical,
            'Unexpected XML mutation beyond six explicit block RGBA RGB channels')
    require(ET.tostring(changed) != original_canonical, 'Gray control must actually alter block RGB')
    return ET.tostring(changed, encoding='unicode') + '\n', changes, hashlib.sha256(original_canonical).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if not args.prepare:
        print('SOURCE_ONLY no model, array, neural, render or physical execution')
        return
    require(not any((HERE / name).exists() for name in ('PREPARED', 'INPUT_BINDINGS.json', 'FROZEN.json')),
            'Exclusive preparation only; never overwrite or blindly retry')
    proof = {}
    checked = read(HERE / 'SOURCE_ONLY_CHECK.json')
    require(passed(checked) and bool(checked.get('source_hashes')), 'New source-only checks and hashes required')
    merge(proof, checked['source_hashes'])
    for path in (OLD_BODY / 'PREPARED/FROZEN.json', OLD_NEURAL / 'FROZEN.json',
                 OLD_BROWN / 'PREPARED/FROZEN.json', OLD_BROWN / 'FROZEN.json'):
        merge(proof, read(path)['source_hashes'])
        bind(proof, path)
    old_preflight_path = OLD_NEURAL / 'PREFLIGHT.json'
    old_preflight = read(old_preflight_path)
    require(passed(old_preflight) and len(old_preflight['checks']) == 20
            and old_preflight['state_total'] == 336 and old_preflight['Core_constructions'] == 0
            and old_preflight['new_physics_steps'] == 0
            and old_preflight['sourcefreeze_sha256'] == sha(OLD_NEURAL / 'FROZEN.json'),
            'Actual old zero-Core20-check336 four-NN proof required; reuse only')
    bind(proof, old_preflight_path)
    binding_path = OLD_BROWN / 'INPUT_BINDINGS.json'
    old_binding = read(binding_path)
    require(all(old_binding[key] == digest for key, digest in EXPECTED_IDS.items()), 'No policy or neck ID change admitted')
    bind(proof, binding_path)
    brown = brown_reference(proof)
    controls = archive_controls(proof, old_binding)
    ledger_path = OLD_BROWN / 'PREPARED/MODEL_LEDGER.json'
    source_ledger = read(ledger_path)
    source_case = source_ledger['cases'][OLD_CASE]
    require(source_case['xml'] == brown['actual_xml']
            and source_case['xml_sha256'] == brown['actual_xml_sha256'], 'Brown actual XML not tied to ledger')
    bind(proof, ledger_path)
    xml_text, changes, canonical_sha = gray_scene(source_case)
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
    case.update(xml=str(xml_path), xml_sha256=sha(xml_path), source_case=OLD_CASE,
        source_xml=source_case['xml'], source_xml_sha256=source_case['xml_sha256'],
        scene_version=CASE, prepared_utc=timestamp, color_only=True,
        translation_world_x_native=0., inherited_start_repair_translation_native=.55,
        change_reason='Only six brown block RGB channels replaced by their arithmetic mean gray; alpha unchanged')
    require(case['terrain'] == source_case['terrain'], 'No terrain ledger position/height changes admitted')
    ledger = copy.deepcopy(source_ledger)
    ledger.update(kind='FULL_BODY_TERRAIN_GRAY_VISUAL_CONTROL_SOURCE_ONLY_V1', cases={CASE: case},
        scene_version=CASE, prepared_utc=timestamp, parent_ledger=str(ledger_path),
        parent_ledger_sha256=sha(ledger_path), only_six_block_world_x_changed=False,
        only_six_block_rgb_changed=True, color_only=True, translation_world_x_native=0.,
        new_physics_steps=0, functional_pass_claim=False)
    save(prepared / 'MODEL_LEDGER.json', ledger)
    color_proof = dict(kind='EXACT_SIX_BLOCK_RGB_MEAN_GRAY_PROOF_V1', all_passed=True,
        scene_version=CASE, prepared_utc=timestamp, color_only=True,
        source_xml=source_case['xml'], source_xml_sha256=source_case['xml_sha256'],
        source_parsed_canonical_sha256=canonical_sha, xml=str(xml_path), xml_sha256=sha(xml_path),
        changes=changes, arithmetic='math.fsum(original RGB)/3; .17g for each gray channel; original alpha token',
        original_RGBA_strings_restored_serialization_exact=True,
        terrain_ledger_exactly_unchanged=True, geometry_pose_height_friction_mass_density_unchanged=True,
        lighting_floor_materials_body71_colors_and_physics_unchanged=True,
        blocks_have_no_explicit_or_inherited_geom_material=True, material_added_or_recolored=False,
        translation_world_x_native=0., first_forward_edge_native=source_case['first_forward_edge_native'],
        brown_reference=brown, archived_controls=controls,
        reused_fourNN_preflight=dict(path=str(old_preflight_path), sha256=sha(old_preflight_path),
            state_total=336, checks=20, archive_only=True, fresh_fourNN_test=False),
        compiled_array_equality_tested=False, new_scene_preflight_pending=True,
        Core_constructions=0, NN_forwards=0, new_physics_steps=0, gray_behavior_pass_claim=False)
    save(prepared / 'GRAY_COLOR_PROOF.json', color_proof)
    binding = copy.deepcopy(old_binding)
    binding.update(kind=KIND, summary_kind=SUMMARY_KIND,
        upstream_kind=old_binding['kind'], upstream_summary_kind=old_binding['summary_kind'],
        upstream_root=str(OLD_BROWN), upstream_bindings=str(binding_path), upstream_bindings_sha256=sha(binding_path),
        cases=[CASE], source_calibration_changed=False, source_neural_condition='V_empty',
        scene_version=CASE, prepared_utc=timestamp, color_only=True, translation_world_x_native=0.,
        inherited_start_repair_translation_native=.55,
        terrain_model_ledger=str(prepared / 'MODEL_LEDGER.json'),
        tested_terrain_core_result=str(HERE / 'ACTUAL_PREFLIGHT/RESULT.json'),
        new_scene_preflight_pending=True, brown_reference=brown, archive_controls=controls,
        reused_fourNN_preflight=color_proof['reused_fourNN_preflight'],
        actual_gray_terrain_RGB_required=True, visual_actor_forced_zero=False,
        new_fourNN_preflight_claim=False, new_physics_steps=0, functional_pass_claim=False)
    save(HERE / 'INPUT_BINDINGS.json', binding)
    for path in (xml_path, prepared / 'MODEL_LEDGER.json', prepared / 'GRAY_COLOR_PROOF.json',
                 HERE / 'INPUT_BINDINGS.json'):
        bind(proof, path)
    save(prepared / 'FROZEN.json', dict(kind='TERRAIN_GRAY_CONTROL_PREPARED_SOURCE_ONLY_V1',
        source_hashes=dict(proof), cases=[CASE], prepared_utc=timestamp,
        color_only=True, Core_constructions=0, NN_forwards=0, new_physics_steps=0))
    bind(proof, prepared / 'FROZEN.json')
    save(HERE / 'FROZEN.json', dict(kind='BOUND_UNCHANGED_FOURNN_AND_GRAY_VISUAL_CONTROL_V1',
        source_hashes=proof, cases=[CASE], prepared_utc=timestamp, color_only=True,
        brown_and_controls_archive_only=True, new_scene_preflight_pending=True,
        Core_constructions=0, NN_forwards=0, new_physics_steps=0, functional_pass_claim=False))
    print(json.dumps(dict(prepared=str(prepared), xml=str(xml_path), cases=[CASE],
        color_only=True, translation_world_x_native=0., new_physics_steps=0, NN_forwards=0)))


if __name__ == '__main__':
    main()
