"""One-factor terrain appearance control using frozen336/source-neck pipeline.

Only new static terrain RGBA is compiled. No actor images/features are edited.
The tested brown runtime's namespace builder is cloned with new scene globals.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import types

HERE = Path(__file__).resolve().parent
PROJECT = Path('/run/media/sensen/Data2/cell_wound_prototype')
PARENT = PROJECT/'experiments/locomotion-enhancement/20261113T161918Z'
BROWN = PARENT/'terrain_start_recovery_v1'
BROWN_CASE = 'mild_blocks_seed17_start_repair_v1'
BROWN_FOLDER = BROWN/'trials'/BROWN_CASE
CASE = 'mild_blocks_seed17_gray_visual_control_v1'
KIND = 'SOURCE_NECK_ROUGH_TERRAIN_GRAY_VISUAL_CONTROL_V1'
SUMMARY_KIND = 'SOURCE_NECK_ROUGH_TERRAIN_GRAY_VISUAL_CONTROL_SUMMARY_V1'
sys.dont_write_bytecode = True


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def verify():
    value = read(HERE/'FROZEN.json')
    for path,digest in value['source_hashes'].items():
        if sha(path)!=digest:
            raise ValueError('Gray-control source drift '+path)
    return value


def verify_prepared():
    value = read(HERE/'PREPARED/FROZEN.json')
    for path,digest in value['source_hashes'].items():
        if sha(path)!=digest:
            raise ValueError('Gray-control prepared source drift '+path)
    return value


def color_metadata():
    new = read(HERE/'PREPARED/MODEL_LEDGER.json')['cases'][CASE]
    old = read(BROWN/'PREPARED/MODEL_LEDGER.json')['cases'][BROWN_CASE]
    if new['terrain']!=old['terrain']:
        raise ValueError('Gray control cannot move terrain or change its heights')
    proof = HERE/'PREPARED/GRAY_COLOR_PROOF.json'
    return dict(kind=KIND,scene_version=CASE,color_only=True,world_x_translation=0.,
        source_case=BROWN_CASE,source_xml=old['xml'],source_xml_sha256=old['xml_sha256'],
        new_xml=new['xml'],new_xml_sha256=new['xml_sha256'],
        color_proof=str(proof),color_proof_sha256=sha(proof),
        brown_result=str(BROWN_FOLDER/'RESULT.json'),brown_result_sha256=sha(BROWN_FOLDER/'RESULT.json'),
        brown_verify=str(BROWN_FOLDER/'VERIFY.json'),brown_verify_sha256=sha(BROWN_FOLDER/'VERIFY.json'),
        brown_behavior_pass_required=False,source_calibration_changed=False,
        actual_gray_scene_RGB_to_original_visual_NN=True,actor_RGB_recolored_or_zeroed=False,
        interpretation='One-factor world block RGB experiment, not new motor352 deployment or claw implementation')


def neural_functions():
    brown = load('_gray_control_frozen_brown_runtime',BROWN/'runtime.py')
    # Reuse exactly the tested recovery namespace builder. Its per-scene helper
    # is replaced rather than running its old +.55 layout assertions again.
    namespace = brown.clone_functions(brown,dict(HERE=HERE,CASE=CASE,KIND=KIND,
        scene_metadata=color_metadata,verify=verify,verify_prepared=verify_prepared))
    return namespace['neural_namespace']()


def context(*,audit=False):
    functions = neural_functions()
    namespace,state = functions['context'](CASE,audit=audit)
    namespace.update(KIND=KIND,SUMMARY_KIND=SUMMARY_KIND)
    previous = namespace['save']

    def save(path,value):
        path = Path(path)
        folder = HERE/'trials'/CASE
        if path==folder/'RESULT.json':
            value['visual_color_control'] = color_metadata()
            value['brown_reference_physics_repeated'] = False
            value['new_motor352_or_claw_claim'] = False
        elif path==folder/'VERIFY.json':
            import numpy as np
            checks = value['checks']
            result = read(folder/'RESULT.json')
            ident = read(folder/'IDENTITY.json')
            checks['gray_scene_metadata_exact'] = result['visual_color_control']==color_metadata()
            checks['gray_actual_identity_kind_and_color_source'] = ident['kind']==KIND and ident['scene_repair']==color_metadata()
            # Raw physical warm prefix and motor priming must agree under a
            # color-only environment change. RGB/encoder caches may differ.
            with np.load(folder/'trace_1ms.npz',allow_pickle=False) as current, np.load(BROWN_FOLDER/'trace_1ms.npz',allow_pickle=False) as brown_trace:
                warm = {key:bool(np.array_equal(current[key][:5800],brown_trace[key][:5800])) for key in current.files}
            with np.load(folder/'neural_decisions.npz',allow_pickle=False) as current, np.load(BROWN_FOLDER/'neural_decisions.npz',allow_pickle=False) as brown_decisions:
                priming = {key:bool(np.array_equal(current[key][:1160],brown_decisions[key][:1160])) for key in
                    ('X238','mean','policy_action','actually_applied_action','hidden_before','hidden_after','command')}
                cache_different = int(np.count_nonzero(np.any(current['held_features'][:1160]!=brown_decisions['held_features'][:1160],axis=1)))
            for key,flag in warm.items():
                checks['matched_brown_physical_warm_exact:'+key] = flag
            for key,flag in priming.items():
                checks['matched_brown_motor_priming_exact:'+key] = flag
            value.update(visual_color_control=color_metadata(),physical_warm_exact_by_field=warm,
                motor_priming_exact_by_field=priming,warm_encoder_cache_different_decisions=cache_different,
                all_optical_or_full_neural_warm_byteexact_claim=False)
        return previous(path,value)

    namespace['save'] = save
    return namespace,state
