"""S148/P1: ONE <=250us zero-interface ablation of the exact S146 initial state.

Import and source preparation are inert. Only parent invokes main for actual physics.
Raw genuine neural decoding is retained; only applied 90-channel ctrl is zeroed.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ast
import hashlib
import json
import sys
import types

ROOT = Path(__file__).resolve().parent
S146 = ROOT/'run_secant_pose_physical_probe.py'
S142 = ROOT/'run_free_physical_prewarm_probe.py'
OUTPUT_PARENT = Path('/home/sensen/Desktop/engine修复验收/s148-zero-control-ablation')
S146_REFERENCE = Path('/home/sensen/Desktop/engine修复验收/s146-secant-pose-free-physical/20261024T203645678947Z')
S145_SELECTION = Path('/home/sensen/Desktop/engine修复验收/s145-secant-contact-preload/20261024T203438782961Z')
MAX_STEPS = 10
MAX_DURATION_S = .00025


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_exact(source, replacements):
    changes = []
    for old, new in replacements:
        assert source.count(old) == 1, ('strict source mismatch', old, source.count(old))
        source = source.replace(old, new, 1)
        changes.append(dict(old=old,new=new))
    ast.parse(source)
    return source, changes


def inert_module(path, name, source=None):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_text() if source is None else source,str(path),'exec'),module.__dict__)
    return module


def preflight(selection):
    s146 = inert_module(S146,'_s148_s146')
    assert Path(selection).resolve() == S145_SELECTION.resolve(), 'same independently proven S146 selection only'
    binding = s146.preflight(selection)
    reference_files = [S146_REFERENCE/name for name in (
        'PREREG.json','INPUT_HASHES.json','COMPLETE.json','POSE_INITIALIZATION.npz',
        'SECANT_SELECTION_BINDING.json','PARENT_S146_INDEPENDENT_CHECK.json',
        'init80ms/TRAJECTORY.npz','init80ms/INITIALIZATION.npz',
        'init80ms/NEURAL_COLD.npz','init80ms/NEURAL_COLD_META.json','init80ms/NEURAL_RELEASE_META.json',
        'SOURCE_FREEZE/PRIVATE_RUNNER_SOURCE.py')]
    # Exact reference identities; no model, neural execution or numpy import.
    for path in reference_files:
        assert path.is_file(), path
    proof = json.loads((S146_REFERENCE/'SECANT_SELECTION_BINDING.json').read_text())
    assert float(proof['offset']) == binding['offset']
    binding['hashes'].update({str(p):sha(p) for p in [S146,Path(__file__).resolve(),*reference_files]})
    return binding


def private_source(text, s146, base_transform):
    source, changes = s146.private_source(text,base_transform)
    replacements = [
        ('PRE, TICKS = 160, 20','PRE, TICKS = 160, 1'),
        ("EXPECTED_PER_ARM = {'init80ms': dict(model_load=1, forward=401, step=400, neural_ticks=180)}",
         "EXPECTED_PER_ARM = {'init80ms': dict(model_load=1, forward=11, step=10, neural_ticks=161)}"),
        ('EXPECTED = dict(model_load=1, forward=401, step=400, neural_ticks=180)',
         'EXPECTED = dict(model_load=1, forward=11, step=10, neural_ticks=161)'),
        ('                d.ctrl[:] = u','                d.ctrl[:] = np.zeros_like(u)'),
        ('                for sub in range(substeps):',
         '                for sub in range(min(substeps, 10-(len(rows)-1))):'),
        ('                applied=np.asarray(commands),decoded=np.asarray(commands),current=np.asarray(currents),',
         '                applied=np.zeros_like(np.asarray(commands)),decoded=np.asarray(commands),current=np.asarray(currents),'),
        ("['MAX_DURATION_10MS']","['MAX_DURATION_250US']"),
        ("assert ledger['step'] <= 400 and ledger['model_load'] == 1", "assert ledger['step'] <= 10 and ledger['model_load'] == 1\n            assert ledger['neural_ticks'] == PRE + (ledger['step']+19)//20\n            assert d.time <= .00025+1e-15\n            assert not arrays['ctrl'].any() and not arrays['act'].any()"),
        ('requested_max_duration_ms=10','requested_max_duration_ms=.25'),
        ('physical_duration_ms=3','physical_duration_ms=.25'),
        ('early_window_s=[0, .003]','early_window_s=[0, .00025]'),
        ("early_aggregation='trapezoid actual time divided by .003'","early_aggregation='NOT_USED: bounded ablation, no optimization'"),
        ("clocks='continuous neural/FIR .08s offset; source END at 83ms; physical START at 80ms operator split'",
         "clocks='PRE160 neural-only80ms; one genuine .5ms source tick ending80.5ms; physical0..<=.25ms partial tick'"),
        ("physical_initial='same saved qpos; zero qvel/act/ctrl/external loads; no later pose writes'",
         "physical_initial='S146 identical proven single root-z initialization; zero qvel/act/ctrl/loads; no later pose/act writes', S119_same_condition_candidate=False, zero_control_ablation=True, S146_full_state_reference=s148_reference, S146_reference_hashes=s148_reference_hashes, raw_decoded_preserved=True, applied_interface='all90 ctrl zero; native muscle passive/position/velocity unchanged', sensory_history='S146 identical NEWpose real observation and PRE160; genuine neural rerun, no replay'"),
        ("str(ROOT/'run_secant_pose_physical_probe.py')]:", "str(ROOT/'run_secant_pose_physical_probe.py'), str(ROOT/'run_zero_control_ablation_probe.py')]:"),
    ]
    source, extra = replace_exact(source,replacements)
    return source, changes+extra


def prepare(selection):
    """Filesystem/AST-only preparation. Never imports MuJoCo or execution helpers."""
    binding = preflight(selection)
    s146 = inert_module(S146,'_s148_s146_transform')
    original_s142 = S142.read_text()
    # S142's reporting hook is private too: remove all inherited 10ms metadata.
    patched_s142, wrapper_changes = replace_exact(original_s142,[
        ("protocol='S142 ONE bounded free physical diagnostic; no assistance or activation assignment'", "protocol='S148 ONE <=250us ZERO_CTRL_ABLATION; not candidate or physical prewarm', EXPERIMENT='ZERO_CTRL_ABLATION', candidate=False, S119_same_condition_candidate=False, zero_control_ablation=True"),
        ("physical_duration_ms=10,physical_steps_per_arm={'init80ms':400}", "physical_duration_ms=.25,physical_steps_per_arm={'init80ms':10}"),
        ("physical_samples_per_arm={'init80ms':401},physical_neural_ticks_per_arm=20", "physical_samples_per_arm={'init80ms':11},physical_neural_ticks_per_arm=1"),
        ("clocks='PRE160 neural-only 80ms; body0..<=10ms, source80..<=90ms'", "clocks='PRE160 neural-only80ms; body0..<=.25ms; source80..80.5ms partial-tick operator split'"),
        ('early_window_s=[0,.010]', 'early_window_s=[0,.00025]'),
    ])
    s142 = inert_module(S142,'_s148_private_s142',patched_s142)
    base_transform = s142.private_source
    def transform(text):
        source, changes = private_source(text,s146,base_transform)
        prefix = ('s146_offset = '+repr(binding['offset'])+'\n'
                  's146_selection = '+repr(binding)+'\n'
                  's146_bindings = '+repr(list(binding['hashes']))+'\n'
                  's148_reference = '+repr(str(S146_REFERENCE))+'\n'
                  's148_reference_hashes = '+repr({k:v for k,v in binding['hashes'].items() if str(S146_REFERENCE) in k})+'\n')
        source = prefix+source
        ast.parse(source)
        return source,changes+[dict(scope='private S142 recorder wrapper',**c) for c in wrapper_changes]
    s142.private_source = transform
    original_reasons = s142.stop_reasons
    def reasons(row, qnew, m):
        # Use the saved ORIGINAL root-height value, not subtraction/rebaseline.
        import numpy as np
        from run_native_rk4_tick_consumption import INITIAL
        if not hasattr(reasons,'original'):
            with np.load(INITIAL,allow_pickle=False) as z:
                reasons.original = z['new_qpos'].copy()
        return original_reasons(row,reasons.original,m)
    s142.stop_reasons = reasons
    source, changes = transform(s142.ORIGINAL.read_text())
    assert "d.ctrl[:] = np.zeros_like(u)" in source
    assert "d.ctrl[:] = u" not in source
    assert "PRE, TICKS = 160, 1" in source
    return s142,binding,source,changes,patched_s142


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',default='auto')
    args = p.parse_args()
    out = OUTPUT_PARENT/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') if args.output == 'auto' else Path(args.output)
    if not out.is_absolute() or out.exists():
        raise ValueError('fresh absolute output only; no resume')
    s142,binding,source,changes,wrapper = prepare(args.selection)
    assert binding['hashes'] == {name:sha(name) for name in binding['hashes']}
    # Additional wrapper provenance written before the delegated model load.
    original_main = s142.main
    previous = sys.argv
    try:
        sys.argv = [str(Path(__file__).resolve()),'--output',str(out)]
        original_main()
    finally:
        sys.argv = previous


if __name__ == '__main__':
    main()
