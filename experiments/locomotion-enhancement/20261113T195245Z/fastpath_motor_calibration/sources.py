"""Newdatasetmanifest, exact immutable8training inputs plus unlabeled4DEV/proofs."""
from pathlib import Path
import importlib.util
import sys
import hashlib
import json

ROOT=Path(__file__).resolve().parent
OLDROOT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z')
PARENT=OLDROOT/'independent_motor_calibration'
SHA='454581a41034f92e253a2cb31215b5ddd1e09f8795ae54fb89053fd3187d1398'
FILES=('PREREG.json','sources.py','data.py','model.py','common.py','learning.py','adapter.py','preflight.py','train.py','verify_saved.py','evaluate.py','audit_eval.py','run.py')


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(p,v):
    with Path(p).open('x') as stream:json.dump(v,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
def verify(hashes):
    for p,h in hashes.items():
        if sha(p)!=h:raise ValueError('Frozeninputsource drift: '+p)
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def bindings():
    original=module('_fastsources_original_actual8_sources',PARENT/'sources.py').bindings()
    common=module('_fastsources_new_strict_storage',ROOT/'common.py')
    proofs=common.parent_qualification_hashes()
    data_hashes={**original['data_hashes'],**original['source_hashes'],**proofs,str(PARENT/'TRAIN/checkpoint_000600.pt'):SHA}
    dev=[]
    for case in ('walk','stop','left_pivot','right_pivot'):
        folder=OLDROOT/'independent_motor_body_component/trials'/case;r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json')
        if not a['integrity_all_pass'] or not all(a['checks'].values()) or a['result_sha256']!=sha(folder/'RESULT.json') or r['checkpoint_sha256']!=SHA:raise ValueError('Archivedfailedstudent dev actualauditrequired')
        path=folder/'neural_decisions.npz';item=r['neural_decisions']
        if item['path']!=str(path) or sha(path)!=item['sha256']:raise ValueError('DEVactualsavedidentity')
        paths=[folder/'RESULT.json',folder/'VERIFY.json',path]
        data_hashes.update({str(p):sha(p) for p in paths});dev.append(dict(id=case,path=str(path),start=1160,warm_start=200,warm_length=960,length=1920,labels_available=False,training_input=False))
    verify(data_hashes)
    paths=[ROOT/n for n in FILES]+[PARENT/n for n in ('model.py','common.py','data.py','learning.py','sources.py','preflight.py','train.py','verify_saved.py','evaluate.py','audit_eval.py','run.py')]
    # Theoriginal789f strictsource identities remain explicit in newstorage config.
    olddir=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning')
    paths += [olddir/'model.py',olddir/'model_io.py']
    return dict(kind='FASTPATH_SAME8_ACTUALTRAINING_4UNLABELED_DEV_V1',entries=original['entries'],dev_entries=dev,data_hashes=data_hashes,source_hashes={str(p):sha(p) for p in paths},arrays_loaded=False,targets_changed=False,new_dev_teacher_labels=False)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--prepare-source-metadata',action='store_true');a=p.parse_args()
    if not a.prepare_source_metadata:p.error('Metadataflag required')
    save(ROOT/'INPUT_BINDINGS.json',bindings());print('FASTPATH_ACTUAL8_SOURCE_METADATA_BOUND',flush=True)
