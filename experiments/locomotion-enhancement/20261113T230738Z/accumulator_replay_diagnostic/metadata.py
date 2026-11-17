"""Stdlib-only immutable identities/NPYheaders/XMLphysicalnormalizers, noCore."""
from pathlib import Path
import ast,hashlib,json,struct,zipfile
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
CAL=ROOT.parent/'local_mean_corrective_calibration'
BODY=ROOT.parent/'local_mean_motor_body_component'
XML=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-triad-pad/20261110T010000Z/MODEL_TRIAD_x200.xml')
SHA='fcbd818f9d6caf3431d6c545356ecd03c09369aab9eeae29363a7ccf128ce49b'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def verify(h):
    for p,v in h.items():
        if sha(p)!=v:raise ValueError('Inputidentitydrift: '+p)
def header(p,k):
    with zipfile.ZipFile(p) as z,z.open(k+'.npy') as f:
        if f.read(6)!=b'\x93NUMPY':raise ValueError('NPYmagic')
        v=tuple(f.read(2));n=2 if v==(1,0) else 4
        if v not in ((1,0),(2,0),(3,0)):raise ValueError('NPYversion')
        size=struct.unpack('<H' if n==2 else '<I',f.read(n))[0]
        if size>65536:raise ValueError('NPYheaderoversize')
        a=ast.literal_eval(f.read(size).decode('utf8' if v==(3,0) else 'latin1'))
    if a['fortran_order'] or 'O' in a['descr']:raise ValueError('Rowmajornonobject')
    return dict(shape=list(a['shape']),descr=a['descr'])

def mapping():
    tree=ET.parse(XML).getroot()
    if tree.find('include') is not None or tree.find('compiler').get('angle')!='radian':raise ValueError('Explicitstandalone radianXMLonly')
    joints=[];address=0
    def body(b):
        nonlocal address
        own=[e for e in b if e.tag in ('joint','freejoint')]
        for e in own:
            free=e.tag=='freejoint' or e.get('type')=='free'
            ranges=[0.,0.] if free else [float(x) for x in e.attrib['range'].split()]
            joints.append(dict(name=e.attrib['name'],qpos_address=address,range=ranges,free=free));address+=7 if free else 1
        for e in b:
            if e.tag=='body':body(e)
    for b in tree.find('worldbody'):
        if b.tag=='body':body(b)
    if address!=49 or len(joints)!=43 or sum(j['free'] for j in joints)!=1:raise ValueError('Expectedonefree42hinges')
    byname={j['name']:j for j in joints};act=list(tree.find('actuator'))
    if len(act)!=42 or any(e.tag!='motor' for e in act):raise ValueError('42motororder')
    indices=[byname[a.attrib['joint']]['qpos_address'] for a in act];ranges=[byname[a.attrib['joint']]['range'] for a in act]
    key=tree.find('keyframe/key');q0=[float(x) for x in key.attrib['qpos'].split()]
    if len(q0)!=49 or len(set(indices))!=42:raise ValueError('Uniqueactuatorqposmapping')
    return dict(xml=str(XML),xml_sha256=sha(XML),qpos_indices=indices,q0_42=[q0[i] for i in indices],halfspan42=[max((r[1]-r[0])/2.,.1) for r in ranges],joint_ranges43=[j['range'] for j in joints],key_qpos49=q0,actuator_joint_names=[a.attrib['joint'] for a in act])

def identity_contract(identity):
    if 'files' in identity and 'actuator' in identity:
        if identity['files'].get(str(XML),{}).get('sha256')!=sha(XML):raise ValueError('ActualXMLidentitybinding')
        return identity
    if identity.get('kind')!='ACTUAL_DERIVED_TERRAIN_BODY_IDENTITY_V1' or identity.get('case')!='plane_control':raise ValueError('Unsupportedidentityschema')
    derived=Path(identity['xml'])
    if sha(derived)!=identity['xml_sha256']:raise ValueError('ActualderivedplaneXMLidentity')
    if 'all finite PD methods unchanged' not in identity.get('actuator_adapter',''):raise ValueError('QualifiedunchangedfinitePDadapter required')
    actuator=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/sources/incremental_actuator.py')
    tree=ast.parse(actuator.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='IncrementalActuator');init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__');defaults=dict(zip([a.arg for a in init.args.kwonlyargs],init.args.kw_defaults))
    if ast.literal_eval(defaults['max_rate_rad_s'])!=2.:raise ValueError('Exactsource maxrate2 default changed')
    m=mapping();ranges=[m['joint_ranges43'][1+i-7] for i in m['qpos_indices']]
    return dict(identity,actuator=dict(target_bounds_rad=ranges,max_rate_rad_s=[2.]*42),actuator_contract_provenance='Derivedplaneidentity explicitly declares unchangedfinitePD; terrain_core.py:82-91 passes no actuatorparameters; exactfrozenIncrementalActuator defaultmaxrate2/nativebounds. Compiledjnt_range/key_qpos SHA still required.',files={str(XML):{'sha256':sha(XML)}})


def qualify():
    review=read(CAL/'PARENT_CALIBRATION_REVIEW.json');evaluation=read(CAL/'EVALUATION.json');ev=read(CAL/'EVALUATION_VERIFY.json');bindings=read(CAL/'INPUT_BINDINGS.json')
    if review['sha256']!=SHA or review['updates']!=600 or not review['source_and_data_hashes_rechecked'] or not ev['integrity_all_pass'] or not all(ev['checks'].values()) or ev['evaluation_sha256']!=sha(CAL/'EVALUATION.json') or evaluation['checkpoint_sha256']!=SHA:raise ValueError('Actualfcbd600qualificationrequired')
    jobs=read(BODY/'JOBS.json')
    if len(jobs)!=9 or any(j.get('exit_code')!=0 for j in jobs):raise ValueError('All4bodycasescompletedrequired')
    hashes={};paths=[ROOT/n for n in ('PREREG.json','metadata.py','analysis.py','diagnose.py')]+[XML,CAL/'PARENT_CALIBRATION_REVIEW.json',CAL/'EVALUATION.json',CAL/'EVALUATION_VERIFY.json',CAL/'INPUT_BINDINGS.json',BODY/'JOBS.json',BODY/'SUMMARY.json']
    evalpath=Path(evaluation['arrays']['path'])
    if sha(evalpath)!=evaluation['arrays']['sha256'] or header(evalpath,'new_action')['shape']!=[1920,12,42]:raise ValueError('ActualEVALpredictionartifact')
    paths.append(evalpath)
    for proof in review['storage_proofs']:
        path=Path(proof['path']);v=read(path)
        if sha(path)!=proof['sha256'] or not v['integrity_all_pass'] or not all(v['checks'].values()):raise ValueError('Actualfreshstoragequalification')
        paths.append(path)
    entries=[]
    for i,e in enumerate(bindings['entries']):
        if e.get('collection') and e['teacher']:continue
        folder=Path(e['folder']);r=read(folder/'RESULT.json');v=read(folder/('NN_VERIFY.json' if e.get('collection') else 'VERIFY.json'))
        if not v['integrity_all_pass'] or v['result_sha256']!=sha(folder/'RESULT.json'):raise ValueError('Realizedsourcequalification')
        entries.append(dict(id=e['id'],folder=str(folder),sample_path=e['path'],start=e['start'],eval_index=i,teacher=e['teacher'],collection=bool(e.get('collection')),actual_new_body=False))
    if len(entries)!=9:raise ValueError('Exactly9realizedsupervision sources')
    for condition in ('walk','stop','left_pivot','right_pivot'):
        folder=BODY/'trials'/condition;r=read(folder/'RESULT.json');v=read(folder/'VERIFY.json')
        if not v['integrity_all_pass'] or v['result_sha256']!=sha(folder/'RESULT.json') or r['checkpoint_sha256']!=SHA:raise ValueError('Actualnew4bodyarchive')
        entries.append(dict(id='actual_fcbd_'+condition,folder=str(folder),sample_path=str(folder/'neural_decisions.npz'),start=1160,eval_index=None,teacher=False,collection=False,actual_new_body=True))
    for e in entries:
        folder=Path(e['folder']);r=read(folder/'RESULT.json');identity=identity_contract(read(folder/'IDENTITY.json'))
        if identity['files'].get(str(XML),{}).get('sha256')!=sha(XML):raise ValueError('ActualXMLidentitybinding')
        for field in ('target_bounds_rad','max_rate_rad_s'):
            if field not in identity['actuator']:raise ValueError('Actualactuatornumericfieldmissing: '+field)
        for key in ('trace','teacher_samples' if e['teacher'] and not e['collection'] else 'neural_decisions'):
            item=r[key]
            if sha(item['path'])!=item['sha256']:raise ValueError('ActualinputartifactSHA')
            paths.append(Path(item['path']))
        paths += [folder/n for n in ('RESULT.json','IDENTITY.json','INITIAL_STATE.json')]+[folder/('NN_VERIFY.json' if e['collection'] else 'VERIFY.json')]
        n=header(e['sample_path'],'X238')['shape'][0]
        if n<=e['start']:raise ValueError('Noactualsuffix')
        e['sample_rows']=n;e['suffix_rows']=n-e['start']
    actuator=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/sources/incremental_actuator.py');terrain_core=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_body_baseline/terrain_core.py')
    paths.extend((actuator,terrain_core))
    plane_entry=next(e for e in entries if e['id']=='walk' and not e['actual_new_body']);plane_id=read(Path(plane_entry['folder'])/'IDENTITY.json');paths.append(Path(plane_id['xml']))
    if plane_id['source_hashes'].get(str(actuator))!=sha(actuator) or plane_id['source_hashes'].get(str(terrain_core))!=sha(terrain_core):raise ValueError('Actualplane unchangedactuatormethod sourcebinding')
    for p in paths:hashes[str(p)]=sha(p)
    verify(hashes)
    return dict(entries=entries,evaluation_path=str(evalpath),evaluation=evaluation,mapping=mapping(),source_hashes=hashes)
