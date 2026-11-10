"""Matched 0.5ms material sampling of original zero-rate-reference PD baseline.
Preserve the prior5ms verification and source arrays; write only a new comparison.
"""
import importlib.util
import json
from pathlib import Path
import numpy as np
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T073258Z/verify_walk.py')
spec=importlib.util.spec_from_file_location('original_pd_verifier',SOURCE)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
FOLDER=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T083316Z/neural/teacher_aligned_4s')
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T084945Z')

if __name__=='__main__':
    arrays=v.load_npz(FOLDER/'COMMANDS.npz');body=v.load_npz(FOLDER/'BODY.npz')
    native=v.load_npz(FOLDER/'NATIVE.npz')['rows'];result=json.loads((FOLDER/'RESULT.json').read_text())
    old=json.loads((FOLDER/'VERIFICATION.json').read_text())
    before={str(p):v.sha(p) for p in [FOLDER/'BODY.npz',FOLDER/'NATIVE.npz',FOLDER/'COMMANDS.npz',FOLDER/'VERIFICATION.json']}
    physics=v.replay(FOLDER,arrays['rows'],arrays,body,native,Path(result['model']),result,1.,30.,40)
    speed=old['gait']['locomotion']['cycle_mean_forward_speed']
    means=np.asarray(physics['material_contact_velocity']['force_weighted_mean_tangential_speed'])
    physics['material_contact_velocity']['slip_to_forward_speed_ratio']=(means/speed).tolist()
    after={p:v.sha(Path(p)) for p in before};assert before==after
    report=v.safe_number(dict(comparison='original zero-D-target finitePD, matched0.5ms material sampling',
        source_folder=str(FOLDER),source_hashes=before,old_verification_retained=True,
        velocity=speed,physics_replay=physics,
        material_pass=bool(np.isfinite(means).all() and (means/speed<=.5).all()),
        slip_bound=.5,source_method=str(SOURCE),source_method_sha256=v.sha(SOURCE),
        script=str(Path(__file__)),script_sha256=v.sha(__file__)))
    output=OUT/'BASELINE_0P5MS.json'
    if output.exists():output.read_text()
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(baseline_speed=speed,material_mean=means.tolist(),ratio=(means/speed).tolist(),
                         replay_pass=physics['passed'],material_pass=report['material_pass'])),flush=True)
