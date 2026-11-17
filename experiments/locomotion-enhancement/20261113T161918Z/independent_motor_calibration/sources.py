"""Actual8 source qualification, stdlib/NPY headers only until data.py parentload."""
from pathlib import Path
import ast
import hashlib
import json
import struct
import zipfile

ROOT=Path(__file__).resolve().parent
EXPERIMENT=ROOT.parent
PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype')
MOTOR_DIR=PROJECT/'experiments/food-approach/20261111T161007Z/combined_turning'
MOTOR_SHA='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd'
NNROOT=PROJECT/'experiments/neck-real-structure/20261113T064313Z/sensory_body_repair/trials'
FILES=('PREREG.json','sources.py','data.py','model.py','common.py','learning.py','preflight.py','train.py','verify_saved.py','evaluate.py','audit_eval.py','run.py')


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def save(p,value):
    with Path(p).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')

def verify(values):
    for p,h in values.items():
        if sha(p)!=h:raise ValueError('Source identity drift: '+p)

def npy_header(path,key):
    with zipfile.ZipFile(path) as z,z.open(key+'.npy') as stream:
        if stream.read(6)!=b'\x93NUMPY':raise ValueError('NPY magic')
        version=tuple(stream.read(2))
        if version not in ((1,0),(2,0),(3,0)):raise ValueError('NPY version')
        size=struct.unpack('<H' if version==(1,0) else '<I',stream.read(2 if version==(1,0) else 4))[0]
        if size>65536:raise ValueError('NPY header unexpectedly large')
        value=ast.literal_eval(stream.read(size).decode('utf8' if version==(3,0) else 'latin1'))
    if value['fortran_order'] or 'O' in value['descr']:raise ValueError('Row-major nonobject arrays required')
    return dict(shape=list(value['shape']),descr=value['descr'])

def entries():
    out=[dict(id='stop',folder=EXPERIMENT/'stop_shadow_control/trials/stop',teacher=True,turn=0.,kind='STOP_FIXED_SHADOW_MATCHED_CONTROL_V1',proof=EXPERIMENT/'stop_shadow_control/MATCHED_COMPARISON.json'),dict(id='left_pivot',folder=EXPERIMENT/'pivot_fixed_shadow_v2/trials/left_pivot',teacher=True,turn=1.,kind='FIXED_PRIVATE_SHADOW_PIVOT_TEACHER_V2',proof=EXPERIMENT/'pivot_fixed_shadow_v2/MATCHED_left_pivot.json'),dict(id='right_pivot',folder=EXPERIMENT/'pivot_fixed_shadow_v2/trials/right_pivot',teacher=True,turn=-1.,kind='FIXED_PRIVATE_SHADOW_PIVOT_TEACHER_V2',proof=EXPERIMENT/'pivot_fixed_shadow_v2/MATCHED_right_pivot.json'),dict(id='walk',folder=EXPERIMENT/'terrain_neural_baseline/trials/plane_control',teacher=False,kind='SOURCE_NECK_ROUGH_TERRAIN_BASELINE_V1')]
    out += [dict(id=name,folder=NNROOT/name,teacher=False,kind='SOURCE_NECK_MULTISENSORY_BODY_REPAIR_V1') for name in ('V_left_pulse','V_right_pulse','V_left_to_right','V_right_to_left')]
    return out

def bindings():
    data_hashes={str(MOTOR_DIR/'COMBINED.pt'):MOTOR_SHA}
    if sha(MOTOR_DIR/'COMBINED.pt')!=MOTOR_SHA:raise ValueError('Strict frozen789f source')
    qualified=[]
    for entry in entries():
        folder=entry['folder'];result=read(folder/'RESULT.json');audit=read(folder/'VERIFY.json')
        if result['kind']!=entry['kind'] or not audit['integrity_all_pass'] or not all(audit['checks'].values()) or audit['result_sha256']!=sha(folder/'RESULT.json'):raise ValueError('Actual qualified source result/audit required: '+entry['id'])
        paths=[folder/'RESULT.json',folder/'VERIFY.json']
        if entry['teacher']:
            proof=read(entry['proof'])
            if not proof['integrity_all_pass'] or not all(proof['checks'].values()) or not result['full_duration'] or result['actual_native_steps']!=1232000 or not all(result['candidate_metrics']['gates'].values()):raise ValueError('Fixedshadow actual qualified teacher only')
            paths.append(entry['proof']);artifact=result['teacher_samples'];filename='teacher_samples_5ms.npz';count=2880;warm_start=0;start=960;keys={'X238':[2880,238],'Y42':[2880,42],'time':[2880],'candidate_phase':[2880]}
        else:
            if result['checkpoint_sha256']!=MOTOR_SHA or result['neural_records']!=3080 or not result['functional_stable_body']:raise ValueError('Actual frozen789f stablebody source required')
            if entry['id']=='walk' and not result['autonomous']['all_gates']:raise ValueError('Plane walking regression qualified source required')
            artifact=result['neural_decisions'];filename='neural_decisions.npz';count=3080;warm_start=200;start=1160;keys={'X238':[3080,238],'mean':[3080,42],'policy_action':[3080,42],'actually_applied_action':[3080,42],'hidden_before':[3080,256],'hidden_after':[3080,256],'command':[3080,2],'raw_turn':[3080],'time':[3080]}
        path=folder/filename
        if artifact['path']!=str(path) or artifact['rows']!=count or artifact['sha256']!=sha(path):raise ValueError('Actual sampleartifact SHA/schema')
        headers={key:npy_header(path,key) for key in keys}
        for key,shape in keys.items():
            if headers[key]['shape']!=shape or headers[key]['descr']!=('|b1' if key=='candidate_phase' else '<f8'):raise ValueError('Actual header shape/dtype: '+key)
        paths.append(path)
        for p in paths:data_hashes[str(p)]=sha(p)
        # Read only exact source identities already bound by actual audited trajectories.
        data_hashes.update(result['source_hashes'])
        qualified.append(dict(id=entry['id'],folder=str(folder),path=str(path),teacher=entry['teacher'],turn=entry.get('turn'),warm_start=warm_start,start=start,warm_length=960,length=1920,headers=headers,result_sha256=sha(folder/'RESULT.json'),audit_sha256=sha(folder/'VERIFY.json')))
    verify(data_hashes)
    source_paths=[ROOT/n for n in FILES]+[MOTOR_DIR/'model.py',MOTOR_DIR/'model_io.py']
    return dict(kind='EIGHT_ACTUAL_INDEPENDENT_MOTOR_SOURCES_V1',entries=qualified,data_hashes=data_hashes,source_hashes={str(p):sha(p) for p in source_paths},arrays_loaded=False,no_duplicate_flatbox=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--prepare-source-metadata',action='store_true');a=p.parse_args()
    if not a.prepare_source_metadata:p.error('Metadata preparation flag required')
    save(ROOT/'INPUT_BINDINGS.json',bindings())
    print(json.dumps(dict(bindings=str(ROOT/'INPUT_BINDINGS.json'),arrays_loaded=False)))
