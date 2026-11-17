"""Stdlib metadata/hash/complete AST and recursive globals proof; never arrays/render."""
from pathlib import Path
import ast, builtins, dis, hashlib, importlib.util, json, sys, types

HERE = Path(__file__).resolve().parent


def globals_used(code):
    names = {i.argval for i in dis.get_instructions(code)
             if i.opname in ('LOAD_GLOBAL', 'LOAD_NAME')}
    for item in code.co_consts:
        if isinstance(item, types.CodeType):
            names.update(globals_used(item))
    return names


def main():
    checks = {}
    def check(name, value):
        checks[name] = bool(value)
        if not checks[name]:
            raise ValueError('Caption CODECHECK: ' + name)
    spec = importlib.util.spec_from_file_location('_caption_v2_check_render', HERE / 'render.py')
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    old = wrapper.original
    for name in ('render.py', 'check_codeonly.py'):
        tree = ast.parse((HERE / name).read_text())
        compile(tree, str(HERE / name), 'exec')
        check(name + '_stdlib_top_level', not any(
            isinstance(n, (ast.Import, ast.ImportFrom)) and any(
                a.name.split('.')[0] in ('numpy', 'torch', 'mujoco', 'physics') for a in n.names)
            for n in tree.body))
    check('exactly_three_expression_replacements', sorted(r[0] for r in wrapper.REPLACEMENTS) == ['bottom', 'footer', 'header'])
    check('entire_original_AST_compiled', len(wrapper.COMPILED_TREE.body) == len(ast.parse((wrapper.OLD / 'render.py').read_text()).body))
    check('exactly_two_control_cases', wrapper.CASES == ('plane_control', 'flat_box_control'))
    check('compiled_main_new_HERE', wrapper.main.__globals__['HERE'] == HERE)
    check('compiled_main_new_load', wrapper.main.__globals__['load'] is wrapper.load)
    check('original_load_and_own_sources_old_HERE', old.load.__globals__['HERE'] == wrapper.OLD and old.own_sources.__globals__['HERE'] == wrapper.OLD)
    check('original_three_case_admission_retained', len(old.CASES) == 3 and old.admission.__globals__['CASES'] == old.CASES)
    check('explicit_old_stdlib_dependencies', all(n in wrapper.main.__globals__ for n in ('hashlib', 'json', 'argparse', 'os', 'sys', 'subprocess')))
    records = {}
    functions = [('compiled_main', wrapper.main), ('new_load', wrapper.load), ('own_ledger', wrapper.own_ledger), ('pinned_sources', wrapper.pinned_sources)]
    functions += [('old_' + n, f) for n, f in vars(old).items()
                  if isinstance(f, types.FunctionType)]
    for name, function in functions:
        names = globals_used(function.__code__)
        missing = sorted(n for n in names if n not in function.__globals__ and not hasattr(builtins, n))
        records[name] = dict(global_names=sorted(names), unresolved=missing)
        check(name + '_complete_nested_globals_resolved', not missing)
    pinned = wrapper.pinned_sources()
    check('old_source_pins_match', bool(pinned))
    # Keep original whole admission, including all artifact and old check/source hashes.
    ledger = old.admission()
    check('original_full_admission_passed', bool(ledger))
    critical = [wrapper.OLD / 'render.py', wrapper.OLD / 'contracts.py',
                HERE.parent / 'terrain_neural_baseline/runtime.py',
                HERE.parent / 'terrain_start_recovery_v1/runtime.py',
                HERE.parent / 'terrain_body_baseline/terrain_core.py']
    check('actual_rendercore_sources_in_ledger', all(str(p) in ledger for p in critical))
    # Snapshot new inputs (CODECHECK is exclusively created below, included at runtime).
    for name in wrapper.OWN:
        if name != 'CODECHECK.json':
            path = HERE / name
            ledger[str(path)] = old.sha(path)
    summaries = {}
    archived_video_keys = None
    for case in wrapper.CASES:
        folder = old.root(case) / 'trials' / case
        result = old.read(folder / 'RESULT.json')
        audit = old.read(folder / 'VERIFY.json')
        identity = old.read(folder / 'IDENTITY.json')
        check(case + '_fresh_actual_VERIFY', audit['integrity_all_pass'] and all(audit['checks'].values()) and audit['result_sha256'] == old.sha(folder / 'RESULT.json'))
        check(case + '_actual_identity_XML', result['actual_xml'] == identity['xml'] and result['actual_xml_sha256'] == identity['xml_sha256'] == old.sha(identity['xml']))
        check(case + '_Original5_from_actual_engineering_pass', result['engineering_pass'] and result['autonomous']['all_gates'] and all(result['autonomous']['gates'].values()))
        check(case + '_stable_cycles_actual', result['functional_stable_body'] and result['autonomous_cycles']['all_legs_completed'])
        check(case + '_rough_false_preserved', result['terrain_summary']['rough_terrain_pass'] is False)
        # Execute only the replaced text expression against stdlib JSON.
        caption = eval(compile(ast.parse(wrapper.BOTTOM.strip(), mode='eval'), '<caption-only>', 'eval'), {'r': result})
        check(case + '_caption_actualPASS_NA', 'Original5 PASS; Rough task N/A' in caption and 'Body-stable True; sixlegs True' in caption)
        video_folder = wrapper.OLD / ('VIDEO_' + case)
        meta = old.read(video_folder / 'VIDEO.json')
        check(case + '_original_full_render_decode_proof', meta['frames'] == meta['decoded_frames'] == 770 and meta['fps'] == 50 and meta['physical_seconds'] == meta['playback_seconds'] == 15.4 and meta['physics_steps'] == meta['NN_forward_calls'] == meta['integration_attempts'] == 0)
        check(case + '_original_video_sources_current', all(old.sha(p) == h for p, h in meta['source_hashes'].items()))
        keys = sorted(meta)
        if archived_video_keys is None:
            archived_video_keys = keys
        check(case + '_same_original_VIDEO_schema', keys == archived_video_keys)
        for name in ('VIDEO.json', 'FRAME_MAPPING.json', 'FFPROBE.json', 'DECODED_MONTAGE.png', 'ACTUAL_DERIVED_TERRAIN_NEURAL_BODY.mp4'):
            path = video_folder / name
            ledger[str(path)] = old.sha(path)
        mapping = old.read(video_folder / 'FRAME_MAPPING.json')
        check(case + '_original_mapping_770', len(mapping) == 770 and all(x['frame'] == k and x['trace_index'] == k * 20 for k, x in enumerate(mapping)))
        summaries[case] = dict(slip=result['autonomous']['slip_ratio_max'], caption=caption,
                               result_sha256=old.sha(folder / 'RESULT.json'),
                               verify_sha256=old.sha(folder / 'VERIFY.json'),
                               identity_sha256=old.sha(folder / 'IDENTITY.json'),
                               xml_sha256=identity['xml_sha256'])
    check('no_heavy_modules_imported', not any(n in sys.modules for n in ('numpy', 'torch', 'mujoco', 'physics', 'student_wave', 'flygym')))
    check('no_new_actual_render_outputs', not any((HERE / ('VIDEO_' + c)).exists() for c in wrapper.CASES))
    old.unchanged(ledger)
    check('full_source_and_original_outputs_before_after_identical', True)
    report = dict(all_passed=True, checks=checks, source_hashes=ledger,
                  recursive_function_global_analysis=records,
                  original_VIDEO_schema_keys=archived_video_keys, actual_controls=summaries,
                  AST_changes_only=['header lines[3]', 'bottom[1]', 'footer text'],
                  array_reads=0, models_loaded=0, Core_constructions=0, render_calls=0,
                  NN_forward_calls=0, physics_steps=0,
                  CODECHECK_self_hash='Included in full runtime load ledger before/after render; excluded from own recursive digest')
    old.fresh(HERE / 'CODECHECK.json', report)
    print(json.dumps(dict(status='READY', checks=len(checks), source_hashes=len(ledger),
                          compiled_main_globals=len(records['compiled_main']['global_names']),
                          unresolved=[], arrays=0, Core=0, render=0)))


if __name__ == '__main__':
    main()
