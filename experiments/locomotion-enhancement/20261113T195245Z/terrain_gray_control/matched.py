"""Parent saved-stream brown/gray comparison; no model or neural replay.

The fresh fourNN/source audit is separate. Fixed receptor energy follows the
actual visual source: uint8/255 -> max(R-.5G-.5B,0), before learned conv layers.
"""
from pathlib import Path
import argparse
import json
import sys

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def stats(np,values):
    a = np.asarray(values,dtype=float)
    return dict(shape=list(a.shape),mean=float(a.mean()),rms=float(np.sqrt(np.mean(a*a))),
        min=float(a.min()),max=float(a.max()),max_abs=float(np.max(np.abs(a))))


def analyze(np,runtime,folder):
    result,proof = runtime.read(folder/'RESULT.json'),runtime.read(folder/'VERIFY.json')
    if not proof['integrity_all_pass'] or not all(proof['checks'].values()) or proof['result_sha256']!=runtime.sha(folder/'RESULT.json'):
        raise ValueError('Matched comparison requires completed independent sourceNN audit')
    for name,key in (('neural_decisions.npz','neural_decisions'),('capture_rgb.npz','capture_rgb'),('trace_1ms.npz','trace')):
        if runtime.sha(folder/name)!=result[key]['sha256']:
            raise ValueError('Matched saved artifact mismatch '+name)
    names = ('visual_raw_turn','visual_latent','raw_turn','shadow_decoder_turn','AN_encoded','DN_encoded','AN_held','DN_held','image_capture_id')
    with np.load(folder/'neural_decisions.npz',allow_pickle=False) as archive:
        d = {key:archive[key].copy() for key in names}
    with np.load(folder/'capture_rgb.npz',allow_pickle=False) as archive:
        rgb = archive['rgb'].copy()
    if rgb.shape!=(770,2,32,32,3) or rgb.dtype!=np.uint8 or any(len(value)!=3080 for value in d.values()):
        raise ValueError('Exact770 RGB/3080 source decisions required')
    ids = d['image_capture_id'][1160:].astype(int)
    if np.any(ids<0) or np.any(ids>=770):
        raise ValueError('Invalid actually consumed frame IDs')
    x = rgb.astype(float)/255.
    opponent = np.maximum(x[...,0]-.5*x[...,1]-.5*x[...,2],0.)
    perframe_m2 = np.mean(opponent*opponent,axis=(2,3))
    perframe_mean = np.mean(opponent,axis=(2,3))
    perframe_positive = np.mean(opponent>0,axis=(2,3))
    consumed_m2 = perframe_m2[ids]
    consumed_mean = perframe_mean[ids]
    eye = dict(
        exact_source_receptor='uint8/255; max(R-.5G-.5B,0)',
        post_actor_consumed_pixel_RMS=float(np.sqrt(consumed_m2.mean())),
        per_eye_post_actor_pixel_RMS=np.sqrt(consumed_m2.mean(axis=0)).tolist(),
        post_actor_mean=float(consumed_mean.mean()),
        post_actor_positive_pixel_fraction=float(perframe_positive[ids].mean()),
        left_minus_right_frame_mean=stats(np,consumed_mean[:,0]-consumed_mean[:,1]),
        first_consumed_capture_id=int(ids[0]),last_consumed_capture_id=int(ids[-1]),
        whole_camera_image_includes_fly_floor_and_blocks=True,block_pixel_segmentation_claim=False)
    return dict(result_sha256=runtime.sha(folder/'RESULT.json'),verify_sha256=runtime.sha(folder/'VERIFY.json'),
        identity_file_sha256=runtime.sha(folder/'IDENTITY.json'),canonical_identity_sha256=result['identity_sha256'],
        actual_xml_sha256=result['actual_xml_sha256'],
        post_signals={key:stats(np,value[1160:]) for key,value in d.items() if key!='image_capture_id'},
        source_eye_opponent_energy=eye,autonomous=result['autonomous'],terrain_summary=result['terrain_summary'],
        cycles=result['autonomous_cycles'],native=result['native'],
        independent_full336_replay_qualified=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-read-saved-arrays',action='store_true')
    a = p.parse_args()
    if not a.parent_read_saved_arrays:
        print('SOURCE_ONLY no arrays or NN execution')
        return
    import runtime
    runtime.verify()
    import numpy as np
    gray_folder = HERE/'trials'/runtime.CASE
    paths = [Path(__file__).resolve(),HERE/'PREREG.json',HERE/'FROZEN.json']
    for folder in (gray_folder,runtime.BROWN_FOLDER):
        paths.extend(folder/name for name in ('RESULT.json','VERIFY.json','IDENTITY.json','neural_decisions.npz','capture_rgb.npz','trace_1ms.npz'))
    before = {str(path):runtime.sha(path) for path in paths}
    brown,gray = analyze(np,runtime,runtime.BROWN_FOLDER),analyze(np,runtime,gray_folder)
    after = {str(path):runtime.sha(path) for path in paths}
    if before!=after:
        raise ValueError('Matched input source changed during analysis')
    if any(name in sys.modules for name in ('torch','mujoco','physics','flygym','student_wave')):
        raise ValueError('Matched statistics must not load scientific models/backends')
    value = dict(kind=runtime.KIND,scope='single factor block-world RGB matched saved-stream statistics',
        brown=brown,gray=gray,source_hashes_before=before,source_hashes_after=after,
        comparison=dict(visual_raw_turn_RMS_gray_minus_brown=gray['post_signals']['visual_raw_turn']['rms']-brown['post_signals']['visual_raw_turn']['rms'],
            decoded_turn_RMS_gray_minus_brown=gray['post_signals']['raw_turn']['rms']-brown['post_signals']['raw_turn']['rms'],
            actor_opponent_pixel_RMS_gray_minus_brown=gray['source_eye_opponent_energy']['post_actor_consumed_pixel_RMS']-brown['source_eye_opponent_energy']['post_actor_consumed_pixel_RMS'],
            slip_ratio_gray_minus_brown=gray['autonomous']['slip_ratio_max']-brown['autonomous']['slip_ratio_max']),
        NN_forwards=0,Core_constructions=0,new_physical_steps=0,
        interpretation='These scene/policy color effects do not establish a unique cause of slip, mechanical hooks, general terrain success or new352 deployment. Actual post trajectories can diverge and produce further sensory differences.')
    with (HERE/'MATCHED.json').open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(matched_saved_streams_complete=True,comparison=value['comparison'],new_physical_steps=0)))


if __name__=='__main__':
    main()
