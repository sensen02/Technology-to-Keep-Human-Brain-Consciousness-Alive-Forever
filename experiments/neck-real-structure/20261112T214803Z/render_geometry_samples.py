"""Render actual two reconstructed SWC samples, never activity or simulated body motion."""
from pathlib import Path
import json,hashlib,os
ROOT=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.collections import LineCollection
 audit=json.loads((ROOT/'NECK_POINT_AND_SAMPLE_GEOMETRY_AUDIT.json').read_text());fig,ax=plt.subplots(1,2,figsize=(14,8));colors=['#2166ac','#b2182b'];paths=[ROOT/'NECK_POINT_AND_SAMPLE_GEOMETRY_AUDIT.json',ROOT/'GEOMETRY_RENDER_PREREG.json',Path(__file__)];hashes={str(p):sha(p) for p in paths};slab=[370000.,484000.]
 for sample,color in zip(audit['actual_two_skeleton_checks'],colors):
  rid=sample['root_id'];p=ROOT/'geometry_downloads'/(rid+'_skeleton.swc');assert sha(p)==sample['sha256'];hashes[str(p)]=sha(p);nodes={}
  for line in p.read_text().splitlines():
   if line.strip() and not line.lstrip().startswith('#'):
    v=line.split();nodes[int(v[0])]=([float(x) for x in v[2:5]],int(v[6]))
  whole=[];neck=[]
  for xyz,parent in nodes.values():
   if parent==-1:continue
   other=nodes[parent][0];whole.append([(xyz[0]/1000,xyz[1]/1000),(other[0]/1000,other[1]/1000)])
   dy=other[1]-xyz[1]
   if dy==0:
    if slab[0]<=xyz[1]<=slab[1]:neck.append([(xyz[0]/1000,xyz[2]/1000),(other[0]/1000,other[2]/1000)])
   else:
    u,v=sorted(((slab[0]-xyz[1])/dy,(slab[1]-xyz[1])/dy));u=max(0,u);v=min(1,v)
    if u<=v:neck.append([((xyz[0]+t*(other[0]-xyz[0]))/1000,(xyz[2]+t*(other[2]-xyz[2]))/1000) for t in (u,v)])
  label=sample['raw_super_class'].upper()+' '+rid
  ax[0].add_collection(LineCollection(whole,colors=color,linewidths=.45,alpha=.8,label=label));ax[1].add_collection(LineCollection(neck,colors=color,linewidths=.7,alpha=.8,label=label))
  for plane in sample['plane_checks'][:1]:
   for pt in plane['point_xyz_nm']:ax[0].scatter([pt[0]/1000],[pt[1]/1000],color=color,s=40,marker='o',edgecolors='black');ax[1].scatter([pt[0]/1000],[pt[2]/1000],color=color,s=55,marker='o',edgecolors='black')
 for y in slab:ax[0].axhline(y/1000,color='gray',ls='--',lw=1)
 for a in ax:a.autoscale();a.set_aspect('equal',adjustable='datalim');a.grid(alpha=.2);a.legend(fontsize=8,loc='best');a.set_xlabel('BANC X (micrometres, nm / 1000)')
 ax[0].set_ylabel('BANC Y (micrometres)');ax[1].set_ylabel('BANC Z (micrometres)');ax[0].set_title('Actual whole-neuron XY projections; dashed: curated neck planes');ax[1].set_title('Actual geometry in Y=370..484 micrometres slab; circles: anterior markers');fig.suptitle('SOURCE-PINNED REAL BANC v888 AN + DN MORPHOLOGY SAMPLES\nNot whole-neck reconstruction, activity, electrode placement or a trained model',fontsize=13);fig.tight_layout(rect=(0,.04,1,.91));fig.text(.02,.015,'Direction from v888 annotation, not SWC label. Both real trees cross both planes; posterior marker Root IDs are unresolved.',fontsize=10)
 path=ROOT/'REAL_NECK_TWO_FIBRE_GEOMETRY.png';fig.savefig(path,dpi=160);plt.close(fig)
 for p,h in hashes.items():assert sha(p)==h
 with (ROOT/'GEOMETRY_RENDER_REVIEW.json').open('x') as f:json.dump(dict(image=str(path),sha256=sha(path),source_hashes=hashes,actual_SWC_geometry_only=True,new_training_updates=0,new_physics_steps=0,neuronal_activity_rendered=False,parent_actual_image_review='PENDING'),f,indent=2);f.write('\n')
 print(str(path))
if __name__=='__main__':main()
