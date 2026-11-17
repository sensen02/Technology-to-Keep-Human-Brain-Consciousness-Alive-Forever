"""Single-layer, exactly three caption-expression changes to archived renderer."""
from pathlib import Path
import argparse, ast, copy, hashlib, importlib.util, json, os, subprocess, sys

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'terrain_motion_evidence'
CASES = ('plane_control', 'flat_box_control')
OWN = ('PREREG.json', 'render.py', 'check_codeonly.py', 'README.zh-CN.md', 'CODECHECK.json')
HEADER = 'Flat surface z=0 / no rough obstacles in this control / SI NOT_ESTABLISHED'
FOOTER = 'Flat control only; separate rough trial contacts block00/01. No rough behavior claim for this control.'
BOTTOM = '''f'Body-stable {r["functional_stable_body"]}; sixlegs {r["autonomous_cycles"]["all_legs_completed"]}; Original5 ' + ('PASS' if r['engineering_pass'] else 'FAIL') + '; Rough task N/A' '''

# This module is stdlib-only until the parent explicitly executes main.
_spec = importlib.util.spec_from_file_location('_caption_v2_original_contracts', OLD / 'contracts.py')
original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(original)


def pinned_sources():
    prereg = original.read(HERE / 'PREREG.json')
    hashes = {str(OLD / n): h for n, h in prereg['pinned_old_sources'].items()}
    original.unchanged(hashes)
    return hashes


def own_ledger():
    ledger = pinned_sources()
    for name in OWN:
        path = HERE / name
        ledger[str(path)] = original.sha(path)
    # Archive every existing evidence artifact; bytes are hashed, never decoded here.
    for case in CASES:
        for name in ('VIDEO.json', 'FRAME_MAPPING.json', 'FFPROBE.json',
                     'DECODED_MONTAGE.png', 'ACTUAL_DERIVED_TERRAIN_NEURAL_BODY.mp4'):
            path = OLD / ('VIDEO_' + case) / name
            ledger[str(path)] = original.sha(path)
    return ledger


def load(case):
    original.require(case in CASES, 'Only two flat controls')
    proof = original.read(HERE / 'CODECHECK.json')
    original.require(proof['all_passed'] and all(proof['checks'].values()), 'Caption CODECHECK')
    original.unchanged(proof['source_hashes'])
    before = own_ledger()
    saved = original.load(case)  # Entire original admission/arrays/identity checks intact.
    original.unchanged(before)
    original.require(all(k not in saved['source_hashes'] or saved['source_hashes'][k] == v
                         for k, v in before.items()), 'No ledger hash replacement')
    saved['source_hashes'].update(before)
    return saved


def transformed_tree():
    pinned_sources()
    tree = ast.parse((OLD / 'render.py').read_text(), filename=str(OLD / 'render.py'))
    baseline = copy.deepcopy(tree)
    replaced = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == 'lines':
                original.require(isinstance(node.value, ast.List) and len(node.value.elts) == 4, 'Header shape')
                old = node.value.elts[3]
                original.require(isinstance(old, ast.Constant) and old.value == 'Native block heights .02/.04 only; no height exaggeration / SI NOT_ESTABLISHED', 'Exact old header')
                node.value.elts[3] = ast.copy_location(ast.Constant(HEADER), old)
                replaced.append(('header', old.lineno, old.col_offset, copy.deepcopy(old)))
            elif name == 'bottom':
                original.require(isinstance(node.value, ast.List) and len(node.value.elts) == 4, 'Bottom shape')
                old = node.value.elts[1]
                original.require('Rough qualification ' in ast.unparse(old), 'Exact old bottom')
                node.value.elts[1] = ast.copy_location(ast.parse(BOTTOM.strip(), mode='eval').body, old)
                replaced.append(('bottom', old.lineno, old.col_offset, copy.deepcopy(old)))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'text' and len(node.args) > 1:
            old = node.args[1]
            if isinstance(old, ast.Constant) and old.value == 'Rough scene contacts only block00/01 so far; originalguard failure preserved. No wholefield completion claim.':
                node.args[1] = ast.copy_location(ast.Constant(FOOTER), old)
                replaced.append(('footer', old.lineno, old.col_offset, copy.deepcopy(old)))
    original.require(sorted(x[0] for x in replaced) == ['bottom', 'footer', 'header'], 'Exactly three display expressions')
    # Revert only those three nodes and prove full-module structural equality.
    reverted = copy.deepcopy(tree)
    undo = {(line, col): old for _, line, col, old in replaced}
    class Undo(ast.NodeTransformer):
        def visit(self, node):
            key = (getattr(node, 'lineno', None), getattr(node, 'col_offset', None))
            if key in undo:
                return copy.deepcopy(undo[key])
            return super().visit(node)
    reverted = Undo().visit(reverted)
    original.require(ast.dump(reverted) == ast.dump(baseline), 'All noncaption AST unchanged')
    return ast.fix_missing_locations(tree), replaced


def build():
    tree, replacements = transformed_tree()
    # Explicit stdlib dependencies plus every original contracts export.
    namespace = dict(vars(original))
    namespace.update(__name__='_caption_v2_compiled_original', __file__=str(OLD / 'render.py'),
                     argparse=argparse, os=os, sys=sys, json=json, hashlib=hashlib,
                     subprocess=subprocess, __builtins__=__builtins__)
    sentinel = object()
    previous = sys.modules.get('contracts', sentinel)
    sys.modules['contracts'] = original
    try:
        exec(compile(tree, str(OLD / 'render.py'), 'exec'), namespace)
    finally:
        if previous is sentinel:
            del sys.modules['contracts']
        else:
            sys.modules['contracts'] = previous
    namespace.update(HERE=HERE, CASES=CASES, load=load)
    original.require(namespace['main'].__globals__ is namespace, 'Compiled main owns new namespace')
    original.require(original.own_sources.__globals__['HERE'] == OLD, 'Original contracts path intact')
    return namespace['main'], tree, replacements


main, COMPILED_TREE, REPLACEMENTS = build()
if __name__ == '__main__':
    main()
