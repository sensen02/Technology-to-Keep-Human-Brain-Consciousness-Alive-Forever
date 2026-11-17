"""New immutable manifest from already-qualified actual12/8 input metadata.
No old sources.bindings() call requiring a nonexistent final600 qualification.
Only stdlib/JSON/NPY headers/byte hashes are used until parent numerical entry.
"""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import struct
import sys
import zipfile
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
ORIGINAL=BASE/'target_feedback_calibration'
RECOVERY=BASE/'target_feedback_calibration_env_recovery_v1'
DIAGNOSTIC=BASE/'conditional_parity_diagnostic'
PARENT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/local_mean_corrective_calibration')
ACCUMULATOR=PARENT.parent/'accumulator_replay_diagnostic'
PARENT_SHA='fcbd818f9d6caf3431d6c545356ecd03c09369aab9eeae29363a7ccf128ce49b'
RESUME_SHA='b562fcc3e39c2aa56381a87b17f1e80124f203fa1a73e3cdf8888778de119f14'
FILES=('staticcheck.py','adapter.py','prepare.py','model.py','common.py','sources.py','data.py','scan.py','learning.py','utils.py','preflight.py','train.py','verify_saved.py','evaluate.py','audit_eval.py','run.py','PREREG.json')


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
def verify(hashes):
    for path,digest in hashes.items():
        if sha(path)!=digest:raise ValueError('Immutable source/input drift: '+path)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value

def _header(path,key):
    with zipfile.ZipFile(path) as archive,archive.open(key+'.npy') as stream:
        if stream.read(6)!=b'\x93NUMPY':raise ValueError('NPY magic')
        version=tuple(stream.read(2))
        if version not in ((1,0),(2,0),(3,0)):raise ValueError('NPY version')
        width=2 if version==(1,0) else 4;size=struct.unpack('<H' if width==2 else '<I',stream.read(width))[0]
        if size>65536:raise ValueError('NPY header too large')
        value=ast.literal_eval(stream.read(size).decode('utf8' if version==(3,0) else 'latin1'))
    if value['fortran_order'] or 'O' in value['descr']:raise ValueError('Row-major non-object source arrays required')
    return dict(shape=list(value['shape']),descr=value['descr'])


def bindings():
    previous_path=RECOVERY/'INPUT_BINDINGS.json';previous=read(previous_path)
    if len(previous['entries'])!=12 or len(previous['dev_entries'])!=8 or sum(previous['target_selector'])!=9:raise ValueError('Already qualified actual12/8/9 source manifest required')
    # Preserve source identities and all old failures; a failed saved verification
    # is evidence, never silently promoted into a required PASS.
    verify(previous['source_hashes']);verify(previous['data_hashes'])
    checkpoint=RECOVERY/'TRAIN/checkpoint_000300.pt'
    if sha(checkpoint)!=RESUME_SHA:raise ValueError('Actual step300 checkpoint SHA required')
    diagnostic=read(DIAGNOSTIC/'RESULT.json')
    if diagnostic['status']!='COMPLETED' or diagnostic['checkpoint_sha256_before']!=RESUME_SHA or diagnostic['checkpoint_sha256_after']!=RESUME_SHA or not diagnostic['all_direct_components_pass'] or not diagnostic['weights_unchanged'] or not diagnostic['source_hashes_unchanged'] or not diagnostic['rng_unchanged_entire_diagnostic']:raise ValueError('Actual readonly exact-component diagnostic evidence required')
    if {row['id'] for row in diagnostic['cases']}!={'stop','old_walk_control'} or not all(row['direct_components_actual_oldmean_exact'] and row['direct_components_pass'] for row in diagnostic['cases']):raise ValueError('Actual diagnostic two-case exact component proof required')
    storage=module('_orderrepair_own_resume_storage',ROOT/'common.py')
    qualifier=getattr(storage,'resume_qualification_hashes',None)
    if not callable(qualifier):raise ValueError('Public common.resume_qualification_hashes() required')
    resume_proofs=qualifier()
    data_hashes={**previous['data_hashes'],**previous['source_hashes'],**resume_proofs,str(previous_path):sha(previous_path),str(checkpoint):RESUME_SHA}
    paths=[DIAGNOSTIC/n for n in ('PREREG.json','diagnose.py','exact_components_scan.py','REPORT.zh-CN.md','RESULT.json')]
    paths += [RECOVERY/'TRAIN/VERIFY_000300.json',RECOVERY/'JOBS.json',ORIGINAL/'INPUT_BINDINGS.json']
    if diagnostic.get('arrays'):
        artifact=Path(diagnostic['arrays']['path'])
        if sha(artifact)!=diagnostic['arrays']['sha256']:raise ValueError('Actual diagnostic array artifact SHA drift')
        paths.append(artifact)
    for name in ('FAILURE.json','TRAIN_FIXED600_CONDITIONAL_FEEDBACK.log'):
        path=RECOVERY/name
        if path.exists():paths.append(path)
    entries=previous['entries'];headers=[]
    for entry in entries:
        h=_header(entry['path'],'X238')
        if h['shape'][1:]!=[238] or entry['length']!=1920:raise ValueError('Unchanged full source sequence header')
        headers.append(dict(id=entry['id'],X238=h))
    data_hashes.update({str(path):sha(path) for path in paths});verify(data_hashes)
    scientific=[ROOT/name for name in FILES]+[ORIGINAL/name for name in ('data.py','scan.py','learning.py')]+[DIAGNOSTIC/'exact_components_scan.py']
    source_hashes={str(path):sha(path) for path in scientific}
    original_old_dir=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning')
    for path in (PARENT/'model.py',PARENT/'common.py',RECOVERY/'model.py',RECOVERY/'common.py',RECOVERY/'adapter.py',original_old_dir/'model.py',original_old_dir/'model_io.py'):
        source_hashes[str(path)]=sha(path)
    return dict(previous,kind='TARGET_FEEDBACK_EXACT_FROZEN_COMPONENT_ORDER_REPAIR_V2',source_hashes=source_hashes,data_hashes=data_hashes,source_manifest_ancestor=str(previous_path),source_manifest_ancestor_sha256=sha(previous_path),resume_checkpoint=str(checkpoint),resume_checkpoint_sha256=RESUME_SHA,frozen_old_component_cache=dict(keys=['basehead','right','left'],head_batch=1,formula='((basehead-gain*predoffset)+weightedRIGHT)+weightedLEFT',weighted_expression_order='negative*RIGHTbound*tanh(RIGHT_head); positive*LEFTbound*tanh(LEFT_head)',old_nontarget_consumed_by_scan=False),actual_source_headers=headers,arrays_loaded=False,targets_changed=False,loss_masks_dtype_budget_changed=False)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-source-metadata',action='store_true');args=parser.parse_args()
    if not args.prepare_source_metadata:parser.error('Explicit metadata preparation required')
    save(ROOT/'INPUT_BINDINGS.json',bindings());print('ORDER_REPAIR_ACTUAL_SOURCE_METADATA_PREPARED',flush=True)
