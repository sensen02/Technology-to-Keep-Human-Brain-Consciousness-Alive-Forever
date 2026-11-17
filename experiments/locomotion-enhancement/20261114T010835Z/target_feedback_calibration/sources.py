"""Own immutable12TRAIN/8DEV/9realizedtarget manifest, stdlib only."""
from pathlib import Path
import importlib.util
import sys
import hashlib
import json

ROOT=Path(__file__).resolve().parent
PARENT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/local_mean_corrective_calibration')
ACCUMULATOR=PARENT.parent/'accumulator_replay_diagnostic'
PARENT_SHA='fcbd818f9d6caf3431d6c545356ecd03c09369aab9eeae29363a7ccf128ce49b'
FILES=('PREREG.json','model.py','common.py','sources.py','data.py','scan.py','learning.py','preflight.py','train.py','verify_saved.py','evaluate.py','audit_eval.py','run.py','utils.py')


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(p,v):
    with Path(p).open('x') as stream:json.dump(v,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
def verify(hashes):
    for path,digest in hashes.items():
        if sha(path)!=digest:raise ValueError('Frozeninput drift: '+path)
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m


def bindings():
    original=module('_targetfeedback_parent_actual12_sources',PARENT/'sources.py').bindings()
    storage=module('_targetfeedback_own_strict_storage',ROOT/'common.py');proofs=storage.parent_qualification_hashes()
    am=module('_targetfeedback_executed_accumulator_metadata',ACCUMULATOR/'metadata.py');normalizers=am.mapping()
    entries=[];paths=[]
    for previous in original['entries']:
        entry=dict(previous);folder=Path(entry['folder']);result=read(folder/'RESULT.json');trace=folder/'trace_1ms.npz';identity=folder/'IDENTITY.json'
        if result['trace']['path']!=str(trace) or sha(trace)!=result['trace']['sha256']:raise ValueError('Actualtrace cumulative targetbinding')
        if am.header(trace,'target')['shape']!=[15400,42] or am.header(trace,'qpos')['shape']!=[15400,49]:raise ValueError('Complete12actualtarget/qpos histories required')
        ident=am.identity_contract(read(identity))
        if ident['compiled_arrays']['jnt_range']['shape']!=[43,2] or ident['compiled_arrays']['key_qpos']['shape']!=[1,49]:raise ValueError('Actualnormalizer compiledshapes')
        entry.update(trace_path=str(trace),identity_path=str(identity),initial_state_path=str(folder/'INITIAL_STATE.json'),target_selector=0 if entry.get('collection') and entry['teacher'] else 1,target_reference_semantics='actualmixedtarget initial/decomposition only; no cumulative supervision' if entry.get('collection') and entry['teacher'] else 'realizedactualappliedtarget cumulative reference')
        entries.append(entry);paths.extend((trace,identity,folder/'INITIAL_STATE.json'))
    if len(entries)!=12 or sum(e['target_selector'] for e in entries)!=9:raise ValueError('Exactly12history/9realizedtarget selectors')
    data_hashes={**original['data_hashes'],**original['source_hashes'],**proofs,str(PARENT/'TRAIN/checkpoint_000600.pt'):PARENT_SHA}
    data_hashes.update({str(p):sha(p) for p in paths});verify(data_hashes)
    sourcepaths=[ROOT/n for n in FILES]+[PARENT/n for n in ('model.py','common.py','sources.py','data.py')]+[ACCUMULATOR/'metadata.py',Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_calibration/preflight.py')]
    olddir=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning');sourcepaths.extend((olddir/'model.py',olddir/'model_io.py'))
    return dict(kind='TARGET_FEEDBACK_12MASKED_TRAIN_9REALIZED_TARGET_8UNLABELED_DEV_V1',entries=entries,dev_entries=original['dev_entries'],normalizer_mapping=normalizers,target_selector=[e['target_selector'] for e in entries],source_hashes={str(p):sha(p) for p in sourcepaths},data_hashes=data_hashes,arrays_loaded=False,targets_changed=False,assisted_unmixed_targets_not_matched_to_mixed_cumulative_reference=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--prepare-source-metadata',action='store_true');a=p.parse_args()
    if not a.prepare_source_metadata:p.error('Explicitsourcepreparation required')
    save(ROOT/'INPUT_BINDINGS.json',bindings());print('TARGET_FEEDBACK_IMMUTABLE12_SOURCE_MANIFEST_PREPARED',flush=True)
