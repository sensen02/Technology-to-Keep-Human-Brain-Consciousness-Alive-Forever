"""Exactly two AST display changes; complete archived plotting module compiled once."""
from pathlib import Path
import argparse, ast, builtins, copy, dis, hashlib, json, math, os, resource, signal, sys, time, types

HERE = Path(__file__).resolve().parent
OLD = HERE.parent
PIN = '064b6f7504dce026f7c50fb23f578a9f23067c0e32d6dfc34b93667b8746e66e'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build():
    assert digest(OLD / 'plot.py') == PIN, 'Frozen original plot source'
    tree = ast.parse((OLD / 'plot.py').read_text(), filename=str(OLD / 'plot.py'))
    original = copy.deepcopy(tree)
    height_changes = 0
    table_changes = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == 'figure':
                for keyword in node.keywords:
                    if keyword.arg == 'figsize':
                        assert ast.literal_eval(keyword.value) == (18, 13)
                        keyword.value = ast.copy_location(ast.parse('(18,28)', mode='eval').body, keyword.value)
                        height_changes += 1
            elif node.func.attr == 'table':
                assert not any(k.arg == 'bbox' for k in node.keywords)
                node.keywords.append(ast.keyword(arg='bbox', value=ast.parse('[0,0.03,1,0.90]', mode='eval').body))
                table_changes += 1
    assert height_changes == table_changes == 1, 'Exactly two layout changes'
    reverted = copy.deepcopy(tree)
    for node in ast.walk(reverted):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == 'figure':
                for keyword in node.keywords:
                    if keyword.arg == 'figsize':
                        keyword.value = ast.parse('(18,13)', mode='eval').body
            elif node.func.attr == 'table':
                node.keywords = [k for k in node.keywords if k.arg != 'bbox']
    assert ast.dump(reverted) == ast.dump(original), 'All other AST/data/metrics/admission unchanged'
    # Complete module imports and definitions execute in the same dictionary; no cloned functions.
    namespace = dict(__name__='_layout_recovery_compiled_original', __file__=str(OLD / 'plot.py'),
                     __builtins__=__builtins__, Path=Path, argparse=argparse, ast=ast,
                     builtins=builtins, dis=dis, hashlib=hashlib, json=json, math=math,
                     os=os, resource=resource, signal=signal, sys=sys, time=time, types=types)
    exec(compile(ast.fix_missing_locations(tree), str(OLD / 'plot.py'), 'exec'), namespace)
    namespace['HERE'] = HERE
    assert namespace['main'].__globals__ is namespace and namespace['execute'].__globals__['HERE'] == HERE
    return namespace


NAMESPACE = build()


def check():
    checks = {}
    def require(name, value):
        checks[name] = bool(value)
        assert checks[name], name
    require('complete_original_module_two_layout_changes_only', True)
    require('new_output_HERE_only', NAMESPACE['HERE'] == HERE)
    records = {}
    functions = [(name, function) for name, function in NAMESPACE.items() if isinstance(function, types.FunctionType)]
    functions += [('Ledger.'+name, function) for name, function in vars(NAMESPACE['Ledger']).items() if isinstance(function, types.FunctionType)]
    for name, function in functions:
        used = NAMESPACE['globals_used'](function.__code__)
        missing = sorted(n for n in used if n not in function.__globals__ and not hasattr(builtins, n))
        records[name] = dict(global_names=sorted(used), unresolved=missing)
        require(name+'_complete_nested_globals', not missing)
    ledger = NAMESPACE['Ledger']()
    for name in ('plot.py', 'PREREG.json', 'CODECHECK.json', 'PLOT_VERIFY.json', 'TRAINING_FIT_COMPARISON.png', 'REPORT.zh-CN.md'):
        ledger.bind(OLD/name)
    oldproof = json.loads((OLD/'CODECHECK.json').read_text())
    actual = json.loads((OLD/'PLOT_VERIFY.json').read_text())
    for source in (oldproof['source_hashes'], actual['source_hashes_before']):
        for path, expected in source.items():
            ledger.bind(path, expected)
    require('archived_actual_plot_allchecks', actual['all_integrity_checks_passed'] and actual['source_hashes_before'] == actual['source_hashes_after'])
    require('all12_actual_body_rows_present', len(actual['body_acceptance']) == 3 and all(len(rows) == 4 for rows in actual['body_acceptance']))
    require('all12_actual_bodyFAIL_preserved', all(row['task_pass'] is False for rows in actual['body_acceptance'] for row in rows))
    prereg = json.loads((HERE/'PREREG.json').read_text())
    require('same_candidates_original_prereg', prereg['candidates'] == json.loads((OLD/'PREREG.json').read_text())['candidates'])
    require('actual_thirdreview_required', actual['body_acceptance'][2][0]['review'] == prereg['third_body_review'])
    ledger.bind(prereg['third_body_review'])
    for name in ('wrapper.py','PREREG.json'):
        ledger.bind(HERE/name)
    require('no_new_outputs', not any((HERE/name).exists() for name in NAMESPACE['OUTPUTS']))
    require('no_numeric_or_plot_imports', not any(n in sys.modules for n in ('numpy','matplotlib','torch','mujoco','physics')))
    ledger.after()
    NAMESPACE['fresh'](HERE/'CODECHECK.json', dict(all_passed=True, checks=checks, source_hashes=ledger.hashes,
        recursive_global_analysis=records, layout_changes=['figsize height13to28','table bbox[0,.03,1,.90]'],
        original_outputs_preserved=True, parent_actual_layout_review='PENDING',
        numeric_imports=0, plot_calls=0, NN_forward_calls=0, physics_steps=0))
    print(json.dumps(dict(status='READY_ONLYCODE',checks=len(checks),sources=len(ledger.hashes),unresolved=[],plot=0,NN=0,physics=0)))


if __name__ == '__main__':
    if sys.argv[1:] == ['--check-code-only']:
        check()
    else:
        # Existing parent CLI/admission/data/table/report behavior is compiled unchanged.
        if '--execute-parent-after-calibration' in sys.argv:
            prereg = json.loads((HERE/'PREREG.json').read_text())
            assert '--third-body-review' in sys.argv, 'Actual third body review required'
            assert sys.argv[sys.argv.index('--third-body-review')+1] == prereg['third_body_review'], 'Exact actual third body review path'
        NAMESPACE['main']()
