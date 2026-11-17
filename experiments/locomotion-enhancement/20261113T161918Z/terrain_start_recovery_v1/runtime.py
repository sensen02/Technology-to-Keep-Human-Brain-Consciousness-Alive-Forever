"""Isolated scene-v1 namespace binding over frozen tested body/neural code.

Only the six existing static block world-X positions change by +.55. No frozen
module/global/file is edited. No backend/model/NN is loaded at module import.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD_NEURAL = ROOT/'terrain_neural_baseline'
OLD_BODY = ROOT/'terrain_body_baseline'
DIAGNOSTIC = ROOT/'terrain_start_area_diagnostic/ACTUAL/RESULT.json'
CASE = 'mild_blocks_seed17_start_repair_v1'
KIND = 'SOURCE_NECK_ROUGH_TERRAIN_START_REPAIR_V1'
DX = .55
FIRST_FRONT = 1.35
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


def clone_functions(module,updates):
    """Identical code/defaults/closure, independent globals dictionary."""
    namespace = dict(vars(module))
    namespace.update(updates)
    for name,value in vars(module).items():
        if isinstance(value,types.FunctionType) and value.__module__==module.__name__ and name not in updates:
            cloned = types.FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
            cloned.__kwdefaults__ = None if value.__kwdefaults__ is None else dict(value.__kwdefaults__)
            cloned.__annotations__ = dict(value.__annotations__)
            namespace[name] = cloned
    return namespace


def verify():
    proof = read(HERE/'FROZEN.json')
    for path,digest in proof['source_hashes'].items():
        if sha(path)!=digest:
            raise ValueError('Scene repair source drift '+path)
    return proof


def verify_prepared():
    proof = read(HERE/'PREPARED/FROZEN.json')
    for path,digest in proof['source_hashes'].items():
        if sha(path)!=digest:
            raise ValueError('Scene repair model source drift '+path)
    return proof


def scene_metadata():
    ledger_path = HERE/'PREPARED/MODEL_LEDGER.json'
    ledger = read(ledger_path)['cases'][CASE]
    source_ledger = read(OLD_BODY/'PREPARED/MODEL_LEDGER.json')['cases']['mild_blocks_seed17']
    expected = {item['name']:item for item in source_ledger['terrain']}
    blocks = [item for item in ledger['terrain'] if item['name'].startswith('terrain_block_')]
    if len(blocks)!=6 or abs(min(item['footprint'][0] for item in blocks)-FIRST_FRONT)>1e-12:
        raise ValueError('Only the fixed1.35-front translated6-block scene is admitted')
    for item in blocks:
        old = expected[item['name']]
        if item['top_z']!=old['top_z'] or item['footprint'][2:]!=old['footprint'][2:] or any(abs(item['footprint'][i]-old['footprint'][i]-DX)>1e-12 for i in (0,1)):
            raise ValueError('Block translation proof mismatch')
    controls = {}
    for name in ('plane_control','flat_box_control'):
        folder = OLD_NEURAL/'trials'/name
        controls[name] = dict(result_path=str(folder/'RESULT.json'),result_sha256=sha(folder/'RESULT.json'),
            verify_path=str(folder/'VERIFY.json'),verify_sha256=sha(folder/'VERIFY.json'),replayed_physics=False)
    return dict(kind=KIND,scene_version=CASE,scene_revision=1,case=CASE,world_x_translation=DX,
        first_block_front_world_x=FIRST_FRONT,reason='Only correct flat warm-start coverage; no change chosen from new behavior outcomes',
        source_scene_xml=source_ledger['xml'],source_scene_xml_sha256=source_ledger['xml_sha256'],
        new_scene_xml=ledger['xml'],new_scene_xml_sha256=ledger['xml_sha256'],
        model_ledger=str(ledger_path),model_ledger_sha256=sha(ledger_path),
        measured_warm_diagnostic=str(DIAGNOSTIC),measured_warm_diagnostic_sha256=sha(DIAGNOSTIC),
        archived_controls=controls,old_partial_failure=str(OLD_NEURAL/'trials/mild_blocks_seed17/FAILURE.json'),
        old_partial_failure_sha256=sha(OLD_NEURAL/'trials/mild_blocks_seed17/FAILURE.json'),
        old_jobs=str(OLD_NEURAL/'JOBS.json'),old_jobs_sha256=sha(OLD_NEURAL/'JOBS.json'),
        source_calibration_changed=False,old_case_reclassified_as_pass=False)


def neural_namespace():
    old = load('_scene_v1_frozen_neural_runtime',OLD_NEURAL/'runtime.py')

    def redirected_module(name,path):
        if Path(path).resolve()==HERE/'metrics.py':
            path = OLD_NEURAL/'metrics.py'
        return old.module(name,path)

    def terrain_modules():
        prepared,metrics,core = old.terrain_modules()
        functions = clone_functions(core,dict(HERE=HERE,verify_prepared=verify_prepared,sha=sha))
        actual_identity = functions['actual_identity']

        def identity(core_instance):
            value = actual_identity(core_instance)
            value.update(model_identity_format=value['kind'],kind=KIND,scene_repair=scene_metadata())
            return value

        functions['actual_identity'] = identity
        # The prepared namespace is explicit too; do not return a misleading old
        # HERE while the Core closure compiles the new timestamped XML.
        prepared_proxy = types.SimpleNamespace(**dict(vars(prepared),HERE=HERE,
            CASES=(CASE,),verify_prepared=verify_prepared))
        return prepared_proxy,metrics,types.SimpleNamespace(**functions)

    return clone_functions(old,dict(HERE=HERE,TERRAIN_ROOT=HERE,KIND=KIND,CASES=(CASE,),
        module=redirected_module,terrain_modules=terrain_modules,verify=verify))


def context(*,audit=False):
    functions = neural_namespace()
    namespace,state = functions['context'](CASE,audit=audit)
    namespace.update(KIND=KIND,SUMMARY_KIND='SOURCE_NECK_ROUGH_TERRAIN_START_REPAIR_SUMMARY_V1')
    previous_save = namespace['save']

    def save(path,value):
        path = Path(path)
        folder = HERE/'trials'/CASE
        if path==folder/'RESULT.json':
            value['scene_repair'] = scene_metadata()
            value['archived_control_physics_repeated'] = False
            value['original_stopped_case_status'] = 'NOT_ADMITTED_BEFORE_ANN; preserved, not promoted to PASS'
        elif path==folder/'VERIFY.json':
            result = read(folder/'RESULT.json')
            value['checks']['fixed_scene_repair_metadata_exact'] = result['scene_repair']==scene_metadata()
            ident = read(folder/'IDENTITY.json')
            value['checks']['identity_scene_repair_metadata_exact'] = ident['kind']==KIND and ident['scene_repair']==scene_metadata()
            value['scene_repair'] = scene_metadata()
        return previous_save(path,value)

    namespace['save'] = save
    return namespace,state
