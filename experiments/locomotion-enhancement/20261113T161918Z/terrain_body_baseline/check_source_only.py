"""Stdlib AST/JSON/source identity checks only. No project modules imported."""
from pathlib import Path
import ast
import hashlib
import json

HERE = Path(__file__).resolve().parent
PROJECT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = PROJECT/'experiments/banc-walking-recovery/20261110T152714Z'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    result = {}
    required = ('prepare.py','terrain_metrics.py','terrain_core.py','preflight.py','check_source_only.py')
    for name in required:
        path = HERE/name
        tree = ast.parse(path.read_text(),filename=str(path))
        compile(tree,str(path),'exec')
        imports = [alias.name for node in tree.body if isinstance(node,ast.Import) for alias in node.names]
        imports += [node.module for node in tree.body if isinstance(node,ast.ImportFrom)]
        result[name+'_stdlib_or_local_import_only'] = not bool(set(imports)&{'numpy','mujoco','torch','physics','student_wave','flygym'})
    tree = ast.parse((BASE/'sources/incremental_actuator.py').read_text())
    target = ast.dump(ast.parse('np.any(model.geom_type == mj.mjtGeom.mjGEOM_PLANE)',mode='eval').body)
    result['exact_one_plane_validator_predicate'] = sum(ast.dump(node)==target for node in ast.walk(tree)) == 1
    prereg = json.loads((HERE/'PREREG.json').read_text())
    result['only_minimal_parent_preflight_ready'] = prereg['full_three_case_neural_runner_ready'] is False
    result['native_budget_6400'] = prereg['bounded_parent_preflight']['native_steps_total'] == 6400
    result['all_required_cases'] = list(prereg['terrain_cases']) == ['plane_control','flat_box_control','mild_blocks_seed17']
    binding_path = PROJECT/'experiments/neck-real-structure/20261113T064313Z/sensory_body_repair/INPUT_BINDINGS.json'
    binding = json.loads(binding_path.read_text())
    expected = prereg['existing_neural_pipeline_preserved_for_next_stage']
    for key in ('brain_sha256','odor_sha256','visual_sha256','motor_sha256','neck_sha256'):
        result['existing_'+key+'_bound'] = binding[key] == expected[key]
    xml = Path(prereg['source_xml'])
    result['original_xml_sha_unchanged'] = digest(xml) == prereg['source_xml_sha256']
    source = {str(HERE/name):digest(HERE/name) for name in (*required,'PREREG.json')}
    source[str(binding_path)] = digest(binding_path)
    report = dict(kind='TERRAIN_BODY_SOURCE_ONLY_CHECK_V1',checks=result,all_passed=all(result.values()),
                  source_hashes=source,Core_constructions=0,scientific_module_imports=0,physical_steps=0,
                  full_neural_runner_ready=False)
    with (HERE/'SOURCE_ONLY_CHECK.json').open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(report))
    if not report['all_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
