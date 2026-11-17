"""Parent-only source-geometry binding builder. No neural inputs or runtime constructed."""
from pathlib import Path
import argparse,json
from api import sha,require,source_ports
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
PREV=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T214803Z')
DN_IDS=('720575941433155799','720575941566493282');AN_IDS=('720575941351919408',)
def build():
 version=ROOT/'SELECTED_V888_VERSION_MAPPING.json';versions=json.loads(version.read_text());by888={row['root_888']:row for row in versions['rows']};paths={version,Path(versions['source']),HERE/'PREREG.json',HERE/'api.py',HERE/'source_transport.py',HERE/'build_binding.py',ROOT/'body_interface/interface.py',ROOT/'SCOPED_NEURAL_CALIBRATION_PREREG.json'};ports=[]
 for direction,ids in [('DN',DN_IDS),('AN',AN_IDS)]:
  for rid in ids:
   g=ROOT/'dnp09_geometry' if direction=='DN' else PREV/'geometry_downloads'
   mp=g/(rid+('_METADATA.json' if direction=='DN' else '_GEOMETRY_METADATA.json'));gp=g/(rid+('_GEOMETRY_CHECK.json' if direction=='DN' else '_INDEPENDENT_GEOMETRY_CHECK.json'));swc=g/(rid+'_skeleton.swc');rp=g/'DOWNLOAD_RECEIPTS.json';m=json.loads(mp.read_text());check=json.loads(gp.read_text());receipt=json.loads(rp.read_text());v=by888[rid]
   require(m['materialization']==888 and m['root_id']==rid and m['version_identity_evidence']['root_888']==rid,'wrong materialization or root_888')
   require(v['super_class']==('descending' if direction=='DN' else 'ascending') and v['tract']=='cervical_connective','source projection direction/tract required')
   require(v['cell_type']==('DNp09' if direction=='DN' else 'AN08B018'),'expected source cell type')
   require(check['metadata']==m and check['each_plane_intersected'] and check['is_single_tree'] and not check['training_admitted'],'actual geometry review required')
   require(sha(swc)==m['swc_sha256']==check['source_swc_sha256'],'skeleton identity mismatch')
   require(m['planes']['nm_y']==[370000.,484000.] and m['coordinate_units']=='nm' and m['coordinate_space']=='BANC_NATIVE','geometry units/planes required')
   candidates=[x for x in receipt['objects'] if x['path']==str(swc)];require(len(candidates)==1 and candidates[0]['sha256']==sha(swc) and candidates[0]['md5_verified'],'pinned public download required')
   refs={mp,gp,swc,rp,version,Path(versions['source'])};paths.update(refs)
   for ev in m['source_evidence']:
    p=Path(ev['path']);require(sha(p)==ev['sha256'],'metadata evidence source drift');paths.add(p);refs.add(p)
    if p.name=='NECK_POINT_AND_SAMPLE_GEOMETRY_AUDIT.json':
     for q,h in json.loads(p.read_text())['source_hashes'].items():require(sha(q)==h,'underlying source drift');paths.add(Path(q));refs.add(Path(q))
   ports.append(dict(root_id=rid,direction=direction,source_reviewed=True,structural_evidence_status='SOURCE_GEOMETRY_REVIEWED',native_tuning_established=False,source_cell_type=v['cell_type'],canonical_id=v['root_id'],evidence=[dict(path=str(p),sha256=sha(p)) for p in sorted(refs)]))
 result=dict(schema='BODY_SIDE_SOURCE_PORT_BINDING_V1',scope='SCOPED_ENGINEERING_NECK_BODY_INTERFACE',engineering_execution_admission='PARENT_PREREGISTERED_SCOPED_EXECUTION',status='SOURCE_QUALIFIED_DNP09_BOUNDARY_ENGINEERING_ONLY',ports=ports,source_files=[dict(path=str(p),sha256=sha(p)) for p in sorted(paths)],transport_config=dict(dt_s=.001,delay_steps=2,gain=1.,activity_min=-1.,activity_max=1.,evidence='ENGINEERING_ASSUMPTION_NOT_MEASURED_CONDUCTION'),full_neck_admitted=False,full_neck_training_allowed=False,native_motor_or_AN_tuning_verified=False,new_architecture_walking_verified=False)
 require(source_ports(result)=={'DN':list(DN_IDS),'AN':list(AN_IDS)},'source port order mismatch');return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--execute-parent',action='store_true');a=p.parse_args()
 if not a.execute_parent:print('CODE_ONLY no source binding')
 else:
  b=build();out=HERE/'SOURCE_BINDING.json'
  with out.open('x') as f:json.dump(b,f,indent=2);f.write('\n')
  print(json.dumps(dict(path=str(out),sha256=sha(out),source_dimensions={k:len(v) for k,v in source_ports(b).items()},training_updates=0)))
