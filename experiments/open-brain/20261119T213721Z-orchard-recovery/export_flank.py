"""Extract an intact upward facing skin facet and its neighborhood from the same Blender apple."""
import bpy
import hashlib
import json
import numpy as np
from pathlib import Path
import sys
d=Path(__file__).resolve().parent
sys.path.insert(0,str(d.parent/'20261119T194948Z-orchard-odor-transfer'))
# Original Blender helper takes its output folder from the post -- argument at import.
from export_crop import geometry,local_coordinates
apple,=[o for o in bpy.context.scene.objects if o.get('fruit_id')=='ground-003']
v,f=geometry(apple);tri=v[f];norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
norm/=np.linalg.norm(norm,axis=1)[:,None]
center=tri.mean(1);height=center[:,2]
valid=np.flatnonzero((height>np.median(height))&(norm[:,2]>.71)&(norm[:,2]<.93))
i=int(valid[np.argmin(abs(norm[valid,2]-.84))])
point=center[i]; uphill=-norm[i,:2]/np.linalg.norm(norm[i,:2])
axes=np.array([[*uphill,0.],[-uphill[1],uphill[0],0.],[0.,0.,1.]])
origin=point-np.r_[1.76*uphill,0.]/100.
ts=local_coordinates(tri,origin,axes)
lo,hi=ts.min(1),ts.max(1)
keep=(hi[:,0]>.25)&(lo[:,0]<3.15)&(hi[:,1]>-1.2)&(lo[:,1]<1.2)&(hi[:,2]>-1.25)&(lo[:,2]<1.25)
soil,=[o for o in bpy.context.scene.objects if o.get('orchard_class')=='soil']
sv,sf=geometry(soil);soiltri=local_coordinates(sv[sf],origin,axes)
lo,hi=soiltri.min(1),soiltri.max(1)
soiltri=soiltri[(hi[:,0]>-.5)&(lo[:,0]<3.2)&(hi[:,1]>-1.5)&(lo[:,1]<1.5)]
np.savez_compressed(d/'FLANK.npz',skin=ts[keep],soil=soiltri,goal=local_coordinates(point,origin,axes),indices=np.flatnonzero(keep))
(d/'FLANK.json').write_text(json.dumps(dict(fruit=apple['fruit_id'],name=apple.name,
    face=i,world_m=point.tolist(),normal=norm[i].tolist(),source=bpy.data.filepath,
    source_sha256=hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
    origin_world_m=origin.tolist(),axes=axes.tolist(),skin=int(keep.sum()),soil=len(soiltri)),indent=2)+'\n')
