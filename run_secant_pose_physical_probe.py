"""S146/P1 conditional ONE <=10ms free physical secant-pose diagnostic.

Inert import; preparation does not execute models. Requires passed independent
S145 static certificate before any neural/model execution. No old command replay.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ast
import hashlib
import json
import math
import sys
import types

ROOT = Path(__file__).resolve().parent
S142 = ROOT/'run_free_physical_prewarm_probe.py'
OUTPUT_PARENT = Path('/home/sensen/Desktop/engine修复验收/s146-secant-pose-free-physical')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(selection):
    """Filesystem-only admission, deliberately before importing execution code."""
    selection = Path(selection)
    if not selection.is_absolute() or not selection.is_dir():
        raise ValueError('--selection must be an existing absolute S145 root')
    result_path = selection/'RESULT.json'
    proof_path = selection/'STATIC_SECANT_INDEPENDENT_CHECK.json'
    result = json.loads(result_path.read_text())
    proof = json.loads(proof_path.read_text())
    assert proof['passed'] is True, 'independent STATIC_SECANT gate must pass'
    assert proof['dynamic_eligible'] is True
    assert sha(proof['verifier']) == proof['verifier_sha256']
    assert result['dynamic_executed'] is False
    assert result['dynamic_eligible_pending_independent_verifier'] is True
    selected = result['selected']
    assert selected['eligible'] is True
    offset = float(selected['dz'])
    assert math.isfinite(offset) and offset != 0 and offset == float(proof['offset'])
    assert offset == float(result['secant']['zstar']) == float(result['dz'])
    assert selected['reliable'] is True and all(selected['six_gates'])
    assert result['zero_valid'] is True
    # Independently committed current artifacts and input identities are mandatory.
    # No ambiguous recursive metadata interpretation: verifier publishes hashes.
    proof_hashes = proof['hashes']
    assert isinstance(proof_hashes, dict) and proof_hashes
    canonical_proof = {str(Path(k).resolve()): v for k,v in proof_hashes.items()}
    assert canonical_proof[str(result_path.resolve())] == sha(result_path)
    paths = {str(result_path):sha(result_path), str(proof_path):sha(proof_path)}
    for name in ('INPUT_HASHES.json','OUTPUT_HASHES.json','PREREG.json'):
        p = selection/name
        assert p.is_file(), p
        paths[str(p)] = sha(p)
    for manifest in (json.loads((selection/'INPUT_HASHES.json').read_text()),
                     json.loads((selection/'OUTPUT_HASHES.json').read_text()),
                     result['source_hashes'], proof_hashes):
        assert isinstance(manifest,dict) and manifest
        for filename, expected in manifest.items():
            path = Path(filename)
            assert path.is_absolute() and sha(path) == expected, filename
            paths[str(path)] = expected
    assert selected['index'] == 'SECANT'
    artifact = selection/'STATIC_SECANT.npz'
    assert artifact.is_file() and sha(artifact) == selected['snapshot_sha256']
    assert str(artifact.resolve()) in canonical_proof, 'selected artifact not independently bound'
    paths[str(artifact)] = sha(artifact)
    paths.update({str(p):sha(p) for p in (Path(__file__).resolve(),S142,ROOT/'run_extended_initialization_probe.py')})
    return dict(root=str(selection),offset=offset,selected=selected,hashes=paths,
                result=str(result_path),proof=str(proof_path),artifact=str(artifact))


def private_source(text, base_transform):
    """Reuse exact S142 transform then count-check the only S146 changes."""
    source, changes = base_transform(text)
    old = "        q = z['new_qpos'].copy(); sensory_ref = z['sensory_reference_qpos'].copy()"
    new = """        q = z['new_qpos'].copy()
        s146_original_q = q.copy()
        q[2] += s146_offset  # Single INITIAL root-height change, before sole d.qpos write.
        sensory_ref = z['sensory_reference_qpos'].copy()"""
    replacements = [(old,new),
        ("arrays['qpos'][:,2]/q[2] >= .85", "arrays['qpos'][:,2]/s146_original_q[2] >= .85"),
        ("    identities = {str(s): sha(s) for s in sources}",
         "    sources = list(dict.fromkeys([*sources, *[Path(p) for p in s146_bindings]]))\n    identities = {str(s): sha(s) for s in sources}"),
        ("    atomic_json(out/'INPUT_HASHES.json', identities)",
         "    atomic_json(out/'INPUT_HASHES.json', identities)\n    for s146_name in [s146_selection['result'], s146_selection['proof'], s146_selection['artifact'], str(ROOT/'run_secant_pose_physical_probe.py')]:\n        (out/'SOURCE_FREEZE'/('S146_'+Path(s146_name).name)).write_bytes(Path(s146_name).read_bytes())"),
        ("    atomic_npz(out/'DIRECTION.npz', weights=weights, **arms,",
         "    atomic_npz(out/'POSE_INITIALIZATION.npz', original_qpos=s146_original_q, new_qpos=q, sensory_reference_qpos=sensory_ref, offset=np.asarray(s146_offset))\n    atomic_json(out/'SECANT_SELECTION_BINDING.json', s146_selection)\n    atomic_npz(out/'DIRECTION.npz', weights=weights, **arms,"),
        ("                physical_prewarm_accepted=False,behavior_claim='NONE',physical_units='NOT_ESTABLISHED',",
         "                physical_prewarm_accepted=False,behavior_claim='NONE',physical_units='NOT_ESTABLISHED',\n                secant_pose_diagnostic=True,original_height_reference=float(s146_original_q[2]),")]
    for old,new in replacements:
        assert source.count(old) == 1, old
        source = source.replace(old,new,1)
        changes.append(dict(old=old,new=new))
    ast.parse(source)
    return source,changes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',default='auto')
    args = p.parse_args()
    out = (OUTPUT_PARENT/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
           if args.output == 'auto' else Path(args.output))
    if not out.is_absolute() or out.exists(): raise ValueError('fresh absolute output; no resume')
    binding = preflight(args.selection)
    # Run a private S142 MODULE, not its on-disk source; preserve its recorder hooks.
    s142 = types.ModuleType('_s146_private_s142')
    s142.__file__ = str(S142)
    exec(compile(S142.read_text(),str(S142),'exec'),s142.__dict__)
    transform = s142.private_source
    def transformed(text):
        source, changes = private_source(text,transform)
        # Embed filesystem-only constants in compiled source, before main runs.
        prefix = ('s146_offset = '+repr(binding['offset'])+'\n'
                  's146_selection = '+repr(binding)+'\n'
                  's146_bindings = '+repr(list(binding['hashes']))+'\n')
        source = prefix+source
        ast.parse(source)
        return source,changes
    s142.private_source = transformed
    # S142 passes qnew to this hook; explicitly recover original height reference.
    def reasons(row,qnew,m):
        qoriginal = qnew.copy()
        qoriginal[2] -= binding['offset']
        # Avoid subtractive rounding: use the original saved qpos value itself.
        import numpy as np
        from run_native_rk4_tick_consumption import INITIAL
        if not hasattr(reasons,'original'):
            with np.load(INITIAL,allow_pickle=False) as pose:
                reasons.original = pose['new_qpos'].copy()
        return original_reasons(row,reasons.original,m)
    original_reasons = s142.stop_reasons
    s142.stop_reasons = reasons
    # S142 archives private source before model execution. Enrich its existing
    # PREREG write at the same point, without touching old recorder implementation.
    original_source = s142.ORIGINAL.read_text()
    source, _ = transformed(original_source)
    # Count-checked provenance corrections are appended to the compiled PREREG
    # dictionary by wrapping the existing S142 main's json_hook's underlying writer.
    # Source's own prereg is corrected here, then S142 enriches its diagnostics.
    old = "physical_initial='same saved qpos; zero qvel/act/ctrl/external loads; no later pose writes'"
    new = "physical_initial='S146 one root-z offset only; zero qvel/act/ctrl/external loads; no later pose writes', secant_pose_selection=s146_selection, height_gate_reference='ORIGINAL S142 root z, never shifted rebaseline', sensory_history='initial actual observation computed from NEW pose with original sensory reference; PRE160 genuine net(initial_current_newpose); may differ from original neural/FIR history; no replay', S119_same_condition_candidate=False"
    assert source.count(old)==1
    prior_transform = s142.private_source
    def final_transform(text):
        src, changes = prior_transform(text)
        assert src.count(old)==1
        return src.replace(old,new,1), changes+[dict(old=old,new=new)]
    s142.private_source = final_transform
    # Recheck bindings immediately before entry. Actual invocation belongs to parent.
    assert binding['hashes'] == {name:sha(name) for name in binding['hashes']}
    previous = sys.argv
    try:
        sys.argv = [str(Path(__file__).resolve()),'--output',str(out)]
        s142.main()
    finally:
        sys.argv = previous


if __name__ == '__main__':
    main()
