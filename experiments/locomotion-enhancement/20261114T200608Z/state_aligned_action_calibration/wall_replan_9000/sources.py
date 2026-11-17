"""Own actual12/8 unchangedmanifest, directinput/proof SHA ledger, stdlib only."""
from pathlib import Path
import ast,hashlib,importlib.util,json,struct,sys,zipfile
ROOT=Path(__file__).resolve().parent
RESUME_PARENT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_order_repair_v4')
PARENT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/local_mean_corrective_calibration')
PARENT_SHA='41fa1e2232f5143fec0a3ebb5d506c819af0ba084b3a05041e829fd1477b906b'
# Originalfeedbackdata takes PARENT fromsources tofindthefcbdactual12cache.
# KeepthatcanonicalAPI; strict41fa resume ancestry hasitsownRESUME_PARENT.
ORIGINAL=RESUME_PARENT.parent/'target_feedback_calibration'
ACCUMULATOR=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/accumulator_replay_diagnostic')
FILES=('PREREG.json','model.py','common.py','sources.py','prepare.py','data.py','scan.py','learning.py','utils.py','preflight.py','train.py','verify_saved.py','evaluate.py','audit_eval.py','run.py','staticcheck.py')

def read(path):return json.loads(Path(path).read_text())
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify(hashes):
 for path,digest in hashes.items():
  if sha(path)!=digest:raise ValueError('Immutableinputdrift:'+path)
def save(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def header(path,key):
 with zipfile.ZipFile(path) as z,z.open(key+'.npy') as f:
  if f.read(6)!=b'\x93NUMPY':raise ValueError('NPYmagic')
  version=tuple(f.read(2))
  if version not in ((1,0),(2,0),(3,0)):raise ValueError('NPYversion')
  width=2 if version==(1,0) else 4;size=struct.unpack('<H' if width==2 else '<I',f.read(width))[0]
  if size>65536:raise ValueError('NPYheadercap')
  value=ast.literal_eval(f.read(size).decode('utf8' if version==(3,0) else 'latin1'))
 if value['fortran_order'] or 'O' in value['descr']:raise ValueError('Rowmajornonobjectarraysrequired')
 return dict(shape=list(value['shape']),descr=value['descr'])

def bindings():
 previous_path=RESUME_PARENT/'INPUT_BINDINGS.json';previous=read(previous_path)
 if len(previous['entries'])!=12 or len(previous['dev_entries'])!=8 or sum(previous['target_selector'])!=9:raise ValueError('Unchangedactual12/8/9 manifestrequired')
 storage=module('_statealigned_own_strict_storage',ROOT/'common.py');proof=storage.parent_qualification_hashes()
 # Keep the full qualifiedparentmanifest bySHA; bind exact current inputs rather
 # than rehashing thousands of archivedvenv files each training save.
 inherited={**previous['source_hashes'],**previous['data_hashes']};data_hashes={**proof,str(previous_path):sha(previous_path),str(storage.PARENT_CHECKPOINT):PARENT_SHA,str(storage.g['OLD_CHECKPOINT']):storage.g['OLD_SHA']}
 paths=[]
 for e in previous['entries']:
  if e['length']!=1920 or e['warm_length']!=960 or header(e['path'],'X238')['shape'][1:]!=[238]:raise ValueError('Originalfullsourcehistoryunchanged')
  if header(e['trace_path'],'target')['shape']!=[15400,42]:raise ValueError('Originalactualtargethistoryrequired')
  folder=Path(e['folder']);paths.extend(Path(e[k]) for k in ('path','trace_path','identity_path','initial_state_path'))
  paths.extend((folder/'RESULT.json',folder/('NN_VERIFY.json' if e.get('collection') else 'VERIFY.json')))
 for e in previous['dev_entries']:
  if e['labels_available'] or e['training_input'] or header(e['path'],'X238')['shape']!=[e['start']+e['length'],238]:raise ValueError('OriginalvariableDEVunlabeledhistoryrequired')
  paths.append(Path(e['path']))
 xml=Path(previous['normalizer_mapping']['xml']);paths.append(xml)
 for path in paths:
  actual=sha(path);declared=inherited.get(str(path))
  if declared is None or actual!=declared:raise ValueError('ActualdirectsourceancestorSHArequired:'+str(path))
  data_hashes[str(path)]=actual
 comparator=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_calibration/preflight.py')
 actualdata=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/local_mean_corrective_calibration/data.py')
 olddir=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning')
 scientific=[ROOT/n for n in FILES]+[RESUME_PARENT/n for n in ('common.py','model.py','scan.py','data.py','learning.py')]+[PARENT/n for n in ('common.py','model.py','data.py','learning.py')]+[ORIGINAL/n for n in ('common.py','model.py','data.py','learning.py')]+[comparator,actualdata,storage.STORAGE_SOURCE,ACCUMULATOR/'metadata.py',olddir/'model.py',olddir/'model_io.py']
 source_hashes={str(p):sha(p) for p in scientific};verify(source_hashes);verify(data_hashes)
 return dict(previous,kind='STATE_ALIGNED_ACTION_ACTUAL12_8DEV_RESUME600_TO900_V1',source_hashes=source_hashes,data_hashes=data_hashes,source_manifest_ancestor=str(previous_path),source_manifest_ancestor_sha256=sha(previous_path),resume_checkpoint=str(storage.PARENT_CHECKPOINT),resume_checkpoint_sha256=PARENT_SHA,arrays_loaded=False,targets_changed=False,loss_masks_dtype_changed=False,qualified_direct_inputs_only=True,ancestor_full_qualified_manifest_preserved_by_sha=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--prepare-source-metadata',action='store_true');a=p.parse_args()
 if not a.prepare_source_metadata:p.error('Explicitstatealignedmetadatarequired')
 save(ROOT/'INPUT_BINDINGS.json',bindings());print('STATE_ALIGNED_ACTUAL12_8DEV_MANIFEST_PREPARED',flush=True)
