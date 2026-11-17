"""Stdlib-only source/namespace/scene contract checks; zero scientific imports.

Loads stdlib-only wrapper namespaces to verify function-global bindings without
calling Core factory, actual_identity, model loader or a policy function.
"""
from pathlib import Path
import ast
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    required = ('runtime.py','prepare.py','preflight.py','run.py','orchestrate.py','check_source_only.py')
    checks = {}
    for name in required:
        path = HERE/name
        tree = ast.parse(path.read_text(),filename=str(path))
        compile(tree,str(path),'exec')
        imports = [alias.name for node in tree.body if isinstance(node,ast.Import) for alias in node.names]
        imports += [node.module for node in tree.body if isinstance(node,ast.ImportFrom)]
        checks[name+'_stdlib_only_module_scope'] = not bool(set(imports)&{'numpy','torch','mujoco','physics','student_wave','flygym'})
    import runtime
    functions = runtime.neural_namespace()
    checks['new_neural_context_artifact_root'] = functions['context'].__globals__['HERE']==HERE
    checks['new_neural_context_terrain_root'] = functions['context'].__globals__['TERRAIN_ROOT']==HERE
    checks['new_neural_case_kind'] = functions['CASES']==(runtime.CASE,) and functions['KIND']==runtime.KIND
    prepared,metric,core = functions['terrain_modules']()
    checks['new_core_type_global_ROOT'] = core.core_type.__globals__['HERE']==HERE
    checks['new_core_prepared_verifier'] = core.core_type.__globals__['verify_prepared'] is runtime.verify_prepared
    checks['prepared_proxy_newROOT'] = prepared.HERE==HERE
    checks['original_Core_XML_request_preserved'] = str(core.core_type.__globals__['XML']).endswith('/experiments/banc-triad-pad/20261110T010000Z/MODEL_TRIAD_x200.xml')
    # Defaults, closures and function code are preserved by clone_functions.
    old = runtime.load('_scene_sourcecheck_old_neural',runtime.OLD_NEURAL/'runtime.py')
    checks['context_code_identical'] = functions['context'].__code__ is not None and functions['context'].__code__.co_code==old.context.__code__.co_code
    checks['upstream_loader_function_still_exact_oldcode'] = functions['upstream'].__code__.co_code==old.upstream.__code__.co_code and functions['UPSTREAM_ROOT']==old.UPSTREAM_ROOT
    namespace = functions['context'].__globals__
    checks['no_original_module_global_mutation'] = old.HERE==runtime.OLD_NEURAL and old.CASES==('plane_control','flat_box_control','mild_blocks_seed17') and old.KIND=='SOURCE_NECK_ROUGH_TERRAIN_BASELINE_V1'
    prereg = json.loads((HERE/'PREREG.json').read_text())
    checks['single_fixed_case'] = prereg['cases']==[runtime.CASE]
    checks['fixed_dx_and_firstfront'] = prereg['fixed_scene_change']['world_x_translation_native']==.55 and prereg['fixed_scene_change']['new_first_forward_edge_native']==1.35
    checks['one_short_and_one_full_budget'] = prereg['fresh_scene_preflight']['Core_constructions']==1 and prereg['fresh_scene_preflight']['native_steps']==1600 and prereg['physical_schedule']['new_trial_count']==1 and prereg['physical_schedule']['native_steps_each']==1232000
    checks['no_threshold_change'] = prereg['measurement_and_gates']['threshold_changes']==0
    binding = json.loads((runtime.OLD_NEURAL/'INPUT_BINDINGS.json').read_text())
    for key,value in prereg['existing_policy_ids'].items():
        checks['old_policy_'+key+'_unchanged'] = binding[key]==value
    diagnostic = json.loads(runtime.DIAGNOSTIC.read_text())
    checks['actual_diagnostic_bound_is_below_fixed_scene'] = diagnostic['all_passed'] and all(diagnostic['checks'].values()) and diagnostic['reports']['plane_control']['flat_start_area_analysis_only']['necessary_first_forward_obstacle_x']<1.35
    checks['all_no_scientific_imports'] = not any(name in sys.modules for name in ('numpy','torch','mujoco','physics','student_wave','flygym'))
    old_proof = json.loads((runtime.OLD_NEURAL/'FROZEN.json').read_text())['source_hashes']
    paths = [HERE/name for name in required]+[HERE/'PREREG.json',runtime.OLD_NEURAL/'runtime.py',runtime.OLD_BODY/'terrain_core.py',runtime.OLD_NEURAL/'metrics.py']
    for path in (runtime.OLD_NEURAL/'runtime.py',runtime.OLD_BODY/'terrain_core.py',runtime.OLD_NEURAL/'metrics.py'):
        checks['executed_source_unchanged:'+str(path)] = old_proof[str(path)]==sha(path)
    value = dict(kind=runtime.KIND,scope='source-only namespace validation',all_passed=all(checks.values()),checks=checks,
        source_hashes={str(path):sha(path) for path in paths},Core_constructions=0,NN_forwards=0,new_physics_steps=0,
        array_loads=0,source_calibration_changed=False,scene_XML_created=False)
    with (HERE/'SOURCE_ONLY_CHECK.json').open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(value))
    if not value['all_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
