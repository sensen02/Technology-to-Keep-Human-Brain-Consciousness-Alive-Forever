"""Independent scalar raw CSV -> neutral NMF2 NPZ audit; never imports acquisition tool."""
import csv, json, math, hashlib, sys
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]);rows=list(csv.DictReader((p/'annotated_3d_walking_kinematics.csv').open()));z=np.load(p/'NMF2_CONTINUOUS.npz',allow_pickle=False)
legs=['LF','LM','LH','RF','RM','RH'];markers=['ThC','CTr','FTi','TiTa','Claw'];mx=0.;ma=0.;mb=0.
for f,r in enumerate(rows):
 assert int(r['frame_idx'])==int(z['source_frame'][f]) and int(r['video_id'])==int(z['trial_id'][f])
 assert abs(z['time_relative_s'][f]-(int(r['frame_idx'])-29)/120)<1e-15
 for l,leg in enumerate(legs):
  a=[[float(r[f'{leg}-{m}_{c}']) for c in 'xyz'] for m in markers]
  for j in range(5):
   for c in range(3):
    mx=max(mx,abs(a[j][c]-z['xyz_source'][f,l,j,c]));assert bool(z['coordinate_valid'][f,l,j,c])==math.isfinite(a[j][c])
  for j in range(4):
   b=math.sqrt(sum((a[j+1][c]-a[j][c])**2 for c in range(3)));mb=max(mb,abs(b-z['bone_lengths_source'][f,l,j]))
  for j in range(3):
   u=[a[j][c]-a[j+1][c] for c in range(3)];v=[a[j+2][c]-a[j+1][c] for c in range(3)]
   co=sum(x*y for x,y in zip(u,v))/math.sqrt(sum(x*x for x in u)*sum(x*x for x in v));ang=math.acos(max(-1,min(1,co)));ma=max(ma,abs(ang-z['geometric_interior_angle_rad'][f,l,j]))
assert mx==0 and mb<1e-12 and ma<1e-12
assert np.array_equal(z['marker_valid'],np.isfinite(z['xyz_source']).all(-1))
assert np.array_equal(z['angle_valid'],np.isfinite(z['geometric_interior_angle_rad']))
assert [sum(z['split']==k) for k in range(3)]==[22,7,8]
q=json.loads((p/'NMF2_QUALITY.json').read_text());assert q['missing_coordinates']==0 and q['span_s']==.3
xyz=z['xyz_source'];assert all(not np.array_equal(xyz[k:],xyz[:-k]) for k in range(1,19))
assert all(not np.array_equal(xyz[:,i],xyz[:,i+3]) for i in range(3))
# Explicit reflection check in retained source frame, with centroid removal.
mirror=[]
for l in range(3):
 a=xyz[:,l]-xyz[:,l].mean((0,1));b=xyz[:,l+3]-xyz[:,l+3].mean((0,1))
 mirror.append(min(float(np.max(abs(a-b*np.array(s)))) for s in [(1,1,1),(1,-1,1),(-1,1,1),(1,1,-1)]))
assert min(mirror)>0
result=dict(passed=True,frames=len(rows),scalar_xyz_max_error=mx,scalar_bone_max_error=mb,scalar_angle_max_error=ma,reflected_right_left_maxerror_min_over_axis_signs=mirror,raw_sha256=hashlib.sha256((p/'annotated_3d_walking_kinematics.csv').read_bytes()).hexdigest(),npz_sha256=hashlib.sha256((p/'NMF2_CONTINUOUS.npz').read_bytes()).hexdigest(),simulation=0,training=0)
(p/'NMF2_INDEPENDENT_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
