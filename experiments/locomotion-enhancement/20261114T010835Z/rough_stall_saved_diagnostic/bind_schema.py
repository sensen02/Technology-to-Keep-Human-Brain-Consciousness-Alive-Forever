"""Stdlib-only metadata preparation: JSON/XML/source and ZIP NPY headers only."""
from pathlib import Path
import ast
import hashlib
import json
import struct
import xml.etree.ElementTree as ET
import zipfile

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PROJECT = BASE.parents[2]
LONG = BASE/'rough_terrain_long_window_38p4/trials/mild_blocks_seed17_start_repair_v1_long38p4'
V2 = BASE/'rough_terrain_long_window/REPAIR_CAPTURE_HORIZON_ONLY_V2/trials/mild_blocks_seed17_start_repair_v1_long19p2'
REFERENCE = PROJECT/'experiments/locomotion-enhancement/20261113T161918Z/pivot_fixed_shadow_v2/trials/left_pivot'
ARCHIVES = ('trace_1ms.npz','geometry_1ms.npz','terrain_1ms.npz','neural_decisions.npz','runtime_geometry.npz')

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def headers(path):
    rows = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as f:
                magic = f.read(8)
                if magic[:6] != b'\x93NUMPY' or magic[6] not in (1,2,3):
                    raise ValueError('NPY format')
                size = struct.unpack('<H' if magic[6] == 1 else '<I', f.read(2 if magic[6] == 1 else 4))[0]
                if size > 65536:
                    raise ValueError('Header too large')
                h = ast.literal_eval(f.read(size).decode('utf8' if magic[6] == 3 else 'latin1').strip())
                rows[name[:-4]] = dict(descr=h['descr'], shape=list(h['shape']), fortran_order=h['fortran_order'])
    return rows

def xml_schema(path):
    tree = ET.parse(path).getroot()
    # Only explicit free + hinge joints are present, so qpos addresses are source-derived.
    order=[]
    def visit_body(body):
        # MuJoCo orders all direct joints before nested bodies, regardless of XML text order.
        order.extend(e for e in body if e.tag in ('freejoint','joint'))
        for child in body.findall('body'):
            visit_body(child)
    for body in tree.find('worldbody').findall('body'):
        visit_body(body)
    if order[0].tag != 'freejoint' or len(order) != 43:
        raise ValueError('Expected sole leading freejoint plus42 hinges')
    if any(e.get('type','hinge') != 'hinge' for e in order[1:]):
        raise ValueError('Nonhinge joint')
    mapping = {e.get('name'):dict(joint_id=k+1,qpos_address=k+7, bounds_rad=list(map(float,e.attrib['range'].split()))) for k,e in enumerate(order[1:])}
    motors = list(tree.find('actuator'))
    if len(motors) != 42 or any(e.tag != 'motor' for e in motors):
        raise ValueError('Expected42 motors')
    actuator = [dict(name=e.get('name'),joint=e.get('joint'),ctrlrange_native=list(map(float,e.get('ctrlrange').split())),**mapping[e.get('joint')]) for e in motors]
    blocks=[]
    for e in tree.find('worldbody').findall('geom'):
        if e.get('name','').startswith('terrain_block_'):
            p=list(map(float,e.get('pos').split()));s=list(map(float,e.get('size').split()))
            blocks.append(dict(name=e.get('name'),front_x=p[0]-s[0],back_x=p[0]+s[0],y_min=p[1]-s[1],y_max=p[1]+s[1],center_z=p[2],halfheight=s[2],top_z=p[2]+s[2]))
    pads=[e for e in tree.iter('geom') if 'Tarsus5_pad' in e.get('name','')]
    if len(pads)!=18 or any(e.get('type')!='sphere' or float(e.get('size'))!=.008 for e in pads):
        raise ValueError('Expected18 radius.008 spheres')
    if abs(blocks[0]['front_x']-1.35)>1e-12 or blocks[0]['center_z']!=.02 or blocks[0]['top_z']!=.04:
        raise ValueError('Original first block geometry')
    roots=tree.find('worldbody').findall('body')
    root_is_thorax=any(e.get('name')=='Thorax' and e.find('freejoint') is order[0] for e in roots) and len(list(tree.iter('freejoint')))==1
    if not root_is_thorax:
        raise ValueError('Source proof requires direct world Thorax freejoint root')
    return dict(native_dt_s=float(tree.find('option').get('timestep')),actuators=actuator,blocks=blocks,foot_spheres=18,adhesion=False,root_point_is_Thorax_frame_origin=True)

def build():
    cases=[]
    ref=read(REFERENCE/'IDENTITY.json'); length=read(REFERENCE/'BODY_LENGTH.json')
    eqnames=('mesh_vert','mesh_pos','mesh_quat','body_pos','body_quat','body_parentid','jnt_range','jnt_qposadr','actuator_trnid')
    source_paths={Path(__file__).resolve(),HERE/'diagnose_saved.py',REFERENCE/'IDENTITY.json',REFERENCE/'BODY_LENGTH.json'}
    for root in (LONG,V2):
        result=read(root/'RESULT.json');identity=read(root/'IDENTITY.json')
        xml=Path(result['actual_xml']);schema=xml_schema(xml)
        qpos_metadata_hash=hashlib.sha256(struct.pack('<43i',0,*range(7,49))).hexdigest()
        if qpos_metadata_hash!=identity['compiled_arrays']['jnt_qposadr']['sha256']:
            raise ValueError('Source-derived joint addresses differ from saved compiled hash')
        by_joint=sorted(schema['actuators'],key=lambda row:row['joint_id'])
        derived_ranges=[0.,0.]+[v for row in by_joint for v in row['bounds_rad']]
        if hashlib.sha256(struct.pack('<86d',*derived_ranges)).hexdigest()!=identity['compiled_arrays']['jnt_range']['sha256']:
            raise ValueError('Source-derived joint ranges differ from saved compiled hash')
        transmissions=[v for row in schema['actuators'] for v in (row['joint_id'],-1)]
        if hashlib.sha256(struct.pack('<84i',*transmissions)).hexdigest()!=identity['compiled_arrays']['actuator_trnid']['sha256']:
            raise ValueError('Source-derived actuator order differs from saved compiled hash')
        geometry_match={n:identity['compiled_arrays'][n]==ref['compiled_arrays'][n] for n in eqnames}
        if not all(geometry_match.values()):
            raise ValueError('Reference body-length geometry identity mismatch')
        source_paths.update(root/n for n in ('RESULT.json','IDENTITY.json','INITIAL_STATE.json','INITIAL_TERRAIN.json','HANDOVER.json'))
        source_paths.add(xml)
        actual_sources={}
        for p,recorded in result['source_hashes'].items():
            p=Path(p)
            if p.suffix in ('.py','.json','.xml'):
                actual_sources[str(p)]=dict(recorded_sha256=recorded,current_sha256=sha(p))
                source_paths.add(p)
        archives={}
        for n in ARCHIVES:
            key={'trace_1ms.npz':'trace','geometry_1ms.npz':'geometry','terrain_1ms.npz':'terrain_trace','neural_decisions.npz':'neural_decisions','runtime_geometry.npz':'runtime_geometry'}[n]
            archives[n]=dict(headers=headers(root/n),recorded_sha256=result[key]['sha256'])
        cases.append(dict(root=str(root),archives=archives,xml=str(xml),xml_schema=schema,source_hashes=actual_sources,geometry_reference_equal=geometry_match,body_length_reference_native=length['length_native'],body_length_meaning='fixed engineering reference seven-segment projected extent from saved JSON, not a newly measured run-specific length',initial_state=read(root/'INITIAL_STATE.json')))
    return dict(kind='ROUGH_STALL_SAVED_DIAGNOSTIC_BINDING_V1',only_header_arrays=True,preaction_index=5799,window_edges_s=[5.8,15.4,25.0,34.6,44.2],body_length_reference=str(REFERENCE/'BODY_LENGTH.json'),cases=cases,actual_sources_sha256={str(p):sha(p) for p in sorted(source_paths)})

if __name__=='__main__':
    with (HERE/'SOURCE_SCHEMA_BINDINGS.json').open('x') as f:
        json.dump(build(),f,indent=2,allow_nan=False);f.write('\n')
