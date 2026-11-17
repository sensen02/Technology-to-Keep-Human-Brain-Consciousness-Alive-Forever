"""Parent-only small actual neck annotations, identity joins and SWC geometry checks."""
from pathlib import Path
import json,csv,hashlib,math,collections,os
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[2]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def swc(path):
 nodes={}
 for line in path.read_text().splitlines():
  if not line.strip() or line.lstrip().startswith('#'):continue
  t=line.split();assert len(t)==7;key=int(t[0]);xyz=tuple(float(x) for x in t[2:5]);rad=float(t[5]);parent=int(t[6]);assert key not in nodes and all(math.isfinite(x) for x in xyz) and math.isfinite(rad) and rad>=0;nodes[key]=(xyz,parent)
 roots=[key for key,(xyz,parent) in nodes.items() if parent==-1];assert roots
 for key,(xyz,parent) in nodes.items():assert parent==-1 or parent in nodes
 # Iterative path checks avoid recursion limits and validate each node reaches a root.
 done=set(roots)
 for key in nodes:
  trail=set();q=key
  while q not in done:
   assert q not in trail;trail.add(q);q=nodes[q][1];assert q!=-1
  done.update(trail)
 return nodes,roots
def main():
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 import pyarrow.parquet as pq
 import pyarrow.feather as feather
 dl=ROOT/'geometry_downloads';raw={}
 with (ROOT/'structure_audit/RAW_CROSSING_ID_CATALOG.csv').open(newline='') as f:
  for row in csv.DictReader(f):raw[row['Root ID']]=row
 paths=[dl/n for n in ('neck_connective_y92500.parquet','neck_connective_y121000.parquet','720575941352057904_skeleton.swc','720575941351919408_skeleton.swc')]+[PROJECT/'data/banc888/banc_888_meta.feather'];hashes={str(p):sha(p) for p in paths};tables={};points={}
 for name in ('neck_connective_y92500','neck_connective_y121000'):
  t=pq.read_table(dl/(name+'.parquet'));rows=t.to_pylist();valid=[r for r in rows if r['valid'] is True];present=[r for r in valid if r['pt_root_id'] is not None];ids={str(r['pt_root_id']) for r in present};points[name]=collections.defaultdict(list)
  attrs=json.loads(t.schema.metadata.get(b'PANDAS_ATTRS',b'{}'));resolution=attrs.get('dataframe_resolution');planevalues=sorted({r['pt_position'][1] for r in valid});classcounts=collections.Counter(raw[rid]['Super Class'] for rid in ids if rid in raw)
  for r in present:points[name][str(r['pt_root_id'])].append(r)
  tables[name]=dict(rows=len(rows),valid_rows=len(valid),root_resolved_rows=len(present),null_root_rows=len(valid)-len(present),unique_root_ids=len(ids),duplicate_root_ids={k:len(v) for k,v in points[name].items() if len(v)>1},raw_crossing_candidate_match_count=len(ids&raw.keys()),matched_raw_superclasses=dict(classcounts),candidate_ids_not_in_this_plane=sorted(raw.keys()-ids),table_materialization=attrs.get('materialization_version'),voxel_resolution_nm=resolution,actual_y_values=planevalues,source_sha256=hashes[str(dl/(name+'.parquet'))])
 pmap=PROJECT/'data/banc888/banc_888_meta.feather';mt=feather.read_table(pmap,columns=['root_id','root_888','super_class','cell_class','tract','nerve','root_position_nm']);maprows=mt.to_pylist();allpointids=set(points['neck_connective_y92500'])|set(points['neck_connective_y121000']);selected={str(r['root_id']):r for r in maprows if str(r['root_id']) in allpointids or str(r['root_id']) in raw};counts=collections.Counter(selected[rid]['super_class'] for rid in allpointids if rid in selected)
 skeletons=[]
 for rid in ('720575941352057904','720575941351919408'):
  p=dl/(rid+'_skeleton.swc');nodes,roots=swc(p);planes=[]
  for name,ydoc in [('neck_connective_y92500',370000.),('neck_connective_y121000',484000.)]:
   hits=[]
   for key,(xyz,parent) in nodes.items():
    if parent==-1:continue
    other=nodes[parent][0];a=xyz[1]-ydoc;b=other[1]-ydoc
    if a*b<0 or a==0 or b==0:
     if xyz[1]==other[1]:continue
     frac=(ydoc-xyz[1])/(other[1]-xyz[1]);hits.append(tuple(xyz[k]+frac*(other[k]-xyz[k]) for k in range(3)))
   actual=points[name].get(rid,[]);res=tables[name]['voxel_resolution_nm'];converted=[tuple(x*s for x,s in zip(r['pt_position'],res)) for r in actual] if res else []
   distances=[min(math.dist(point,hit) for hit in hits) for point in converted] if hits else []
   planes.append(dict(table=name,plane_y_nm=ydoc,SWC_segment_intersections=len(hits),intersection_xyz_nm=hits,actual_annotation_rows=len(actual),point_xyz_nm=converted,annotation_to_intersection_min_distances_nm=distances,coordinate_conversion_from_table_metadata=res is not None,interpretation='Intersection/point proximity descriptive only; not calibrated electrode or conduction model'))
  skeletons.append(dict(root_id=rid,raw_super_class=raw[rid]['Super Class'],v888_metadata=selected.get(rid),nodes=len(nodes),roots=len(roots),valid_tree_graph=True,coordinate_units='nm BANC per official public product documentation',bbox_nm=[[min(v[0][k] for v in nodes.values()),max(v[0][k] for v in nodes.values())] for k in range(3)],plane_checks=planes,sha256=sha(p)))
 for p,h in hashes.items():assert sha(p)==h
 report=dict(status='REAL_NECK_MARKER_POINT_AND_TWO_ACTUAL_SWC_GEOMETRY_AUDITED',tables=tables,point_ids_joined_to_local_v888=len(allpointids&selected.keys()),point_ids_missing_local_v888=sorted(allpointids-selected.keys()),v888_point_superclasses=dict(counts),actual_two_skeleton_checks=skeletons,source_hashes=hashes,all_sources_unchanged=True,new_training_updates=0,new_physics_steps=0,real_electrode_operations=0,whole_neck_runtime_admitted=False,limitations='Two skeleton samples do not certify all DN/AN. y121000 table may lack root IDs; no fabricated mapping. Structural, engineering dynamical, behavior and electrode acceptance separate.')
 with (ROOT/'NECK_POINT_AND_SAMPLE_GEOMETRY_AUDIT.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps({'tables':{k:{a:b for a,b in v.items() if a in ('rows','valid_rows','root_resolved_rows','null_root_rows','unique_root_ids','raw_crossing_candidate_match_count','table_materialization','voxel_resolution_nm')} for k,v in tables.items()},'v888_point_superclasses':dict(counts),'skeletons':[{'root_id':s['root_id'],'nodes':s['nodes'],'plane_checks':s['plane_checks']} for s in skeletons]},indent=2))
if __name__=='__main__':main()
