"""Stdlib SWC integrity and source-specified plane intersection, not physiology or model admission."""
from pathlib import Path
import argparse,json,math,hashlib
class GeometryError(ValueError):pass
def require(v,msg):
 if not v:raise GeometryError(msg)
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def parse_swc(path):
 nodes={};comments=[]
 with Path(path).open() as f:
  for lineno,line in enumerate(f,1):
   line=line.strip()
   if not line:continue
   if line.startswith('#'):comments.append(line);continue
   parts=line.split();require(len(parts)==7,'SWC row needs exactly7 fields at '+str(lineno))
   try:nid,kind=int(parts[0]),int(parts[1]);x,y,z,radius=map(float,parts[2:6]);parent=int(parts[6])
   except ValueError:raise GeometryError('invalid numeric fields at '+str(lineno))
   require(nid>0 and nid not in nodes,'duplicate/nonpositive node ID');require(all(math.isfinite(v) for v in (x,y,z,radius)),'nonfinite geometry');require(radius>=0,'negative radius');require(parent==-1 or parent>0,'invalid parent ID');require(parent!=nid,'self-parent cycle');nodes[nid]=(kind,(x,y,z),radius,parent)
 require(bool(nodes),'empty SWC')
 roots=[nid for nid,node in nodes.items() if node[3]==-1];require(bool(roots),'no root')
 for nid,node in nodes.items():require(node[3]==-1 or node[3] in nodes,'parent not in SWC')
 resolved=set()
 for nid in nodes:
  chain=set();cur=nid
  while cur!=-1 and cur not in resolved:
   require(cur not in chain,'cycle in skeleton');chain.add(cur);cur=nodes[cur][3]
  resolved.update(chain)
 return nodes,comments,roots

def validate(path,metadata):
 require(metadata.get('schema')=='BANC_V888_SWC_GEOMETRY_CHECK_V1','explicit metadata schema required')
 require(metadata.get('dataset')=='BANC' and metadata.get('materialization')==888,'exact dataset/materialization required')
 rootid=metadata.get('root_id');require(isinstance(rootid,str) and rootid.isdecimal(),'exact decimal root ID string required')
 require(Path(path).name in (rootid+'_skeleton.swc',rootid+'_l2.swc'),'filename/root ID mismatch')
 require(metadata.get('coordinate_space')=='BANC_NATIVE' and metadata.get('coordinate_units')=='nm','SWC must be native BANC nanometres')
 sources=metadata.get('source_evidence',[]);require(bool(sources),'source evidence required')
 for source in sources:
  p=Path(source['path']);require(p.is_absolute() and p.is_file() and sha(p)==source['sha256'],'evidence source hash mismatch')
 require(sha(path)==metadata.get('swc_sha256'),'SWC byte digest mismatch')
 identity=metadata.get('version_identity_evidence',{});require(identity.get('root_888')==rootid and identity.get('verified') is True,'must verify root888 membership separately')
 plane=metadata.get('planes');require(isinstance(plane,dict),'no hardcoded/guessed neck plane')
 resolution=plane.get('voxel_size_nm');require(resolution==[4,4,45],'documented resolution required')
 ys=plane.get('voxel_y');require(ys==[92500,121000],'source-defined two-plane boundary required')
 nm=[float(y*resolution[1]) for y in ys];require(plane.get('nm_y')==nm,'voxel-to-nm conversion mismatch')
 require(plane.get('source_documentation_url') and plane.get('definition')=='CURATED_PROFILE_SEED_PLANES_NOT_SYNAPSES','plane evidence semantics missing')
 nodes,comments,roots=parse_swc(path);crossings={str(y):[] for y in nm}
 for nid,node in nodes.items():
  parent=node[3]
  if parent==-1:continue
  a=node[1];b=nodes[parent][1]
  for y in nm:
   if (a[1]<y<=b[1]) or (b[1]<y<=a[1]):
    t=(y-a[1])/(b[1]-a[1]);point=[a[j]+t*(b[j]-a[j]) for j in range(3)];crossings[str(y)].append(dict(child=nid,parent=parent,point_nm=point))
 bbox=dict(min=[min(v[1][j] for v in nodes.values()) for j in range(3)],max=[max(v[1][j] for v in nodes.values()) for j in range(3)])
 return dict(status='SOURCE_PINNED_GEOMETRY_CHECK_ONLY_NOT_MODEL_ADMISSION',root_id=rootid,nodes=len(nodes),tree_roots=len(roots),is_single_tree=len(roots)==1,coordinate_space='BANC_NATIVE',coordinate_units='nm',bounding_box_nm=bbox,plane_y_nm=nm,plane_intersections=crossings,each_plane_intersected=all(bool(v) for v in crossings.values()),source_swc_sha256=sha(path),metadata=metadata,SWC_compartment_labels_not_independently_verified=True,projection_direction_not_inferred_from_label2=True,neck_local_synapses_inferred=False,physiology_verified=False,electrode_mapping_verified=False,training_admitted=False,neural_forward_calls=0,physics_steps=0)

def main():
 p=argparse.ArgumentParser();p.add_argument('--swc',required=True,type=Path);p.add_argument('--metadata',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 try:
  result=validate(a.swc,json.loads(a.metadata.read_text()))
  with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
  print(json.dumps({k:result[k] for k in ('status','root_id','nodes','tree_roots','each_plane_intersected','training_admitted')}));return 0
 except (GeometryError,KeyError,ValueError,OSError) as e:print(json.dumps({'status':'GEOMETRY_CHECK_REJECTED','reason':str(e),'training_admitted':False}));return 1
if __name__=='__main__':raise SystemExit(main())
