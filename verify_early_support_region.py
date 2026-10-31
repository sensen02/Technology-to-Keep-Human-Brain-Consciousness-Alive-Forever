"""Independent saved-JSON verification: no model or forward calls."""
import json
import numpy as np
from analyze_early_support_region import O,FORCE_TOL,GEOM_TOL,LEGS
from run_one_second_motor_pair import dump,sha

def main():
 rows=json.loads((O/'SUPPORT_REGION_TIME_SERIES.json').read_text());meta=json.loads((O/'PROJECTION_AND_CLASSIFICATION.json').read_text());basis=np.array(meta['basis']);assert np.allclose(basis@basis.T,np.eye(2),atol=1e-14)
 checked=0;maxarea=0.
 for r in rows:
  for key,bearing in [('geometric_region',False),('bearing_region',True)]:
   points=np.array([co['position_projected'] for co in r['contacts'] if co['leg']!='NONFOOT' and (not bearing or co['compressive_normal_load']>FORCE_TOL)]).reshape(-1,2);h=np.array(r[key]['vertices']).reshape(-1,2);com=np.array(r['com_projected']);rr=r[key]
   if rr['dimension']<2:assert rr['signed_boundary_distance'] is None and rr['inside_2d'] is None and rr['area']==0;continue
   # Independently check edge halfplanes, polygon shoelace and closest segments.
   v=np.roll(h,-1,axis=0)-h;length=np.linalg.norm(v,axis=1);delta=com-h;sd=(v[:,0]*delta[:,1]-v[:,1]*delta[:,0])/length
   inside=bool(np.all(sd>=-GEOM_TOL));assert inside==rr['inside_2d']
   alpha=np.clip((delta*v).sum(1)/(v*v).sum(1),0,1);distance=float(np.linalg.norm(delta-alpha[:,None]*v,axis=1).min());expected=distance if inside else -distance;assert abs(expected-rr['signed_boundary_distance'])<1e-12
   area=abs(np.dot(h[:,0],np.roll(h[:,1],-1))-np.dot(h[:,1],np.roll(h[:,0],-1)))/2;assert abs(area-rr['area'])<1e-10;maxarea=max(maxarea,abs(area-rr['area']))
   for point in points:
    d=point-h;s=(v[:,0]*d[:,1]-v[:,1]*d[:,0])/length;assert np.all(s>=-1e-9)
   checked+=1
  for leg in LEGS+['NONFOOT']:
   co=[x for x in r['contacts'] if x['leg']==leg];assert r['perleg'][leg]['geometric_contact_count']==len(co);assert r['perleg'][leg]['bearing_contact_count']==sum(x['compressive_normal_load']>FORCE_TOL for x in co)
 assert len(rows)==3201 and all(abs(r['time_s']-i*.0000125)<1e-10 for i,r in enumerate(rows))
 dump(O/'INDEPENDENT_GEOMETRY_VERIFICATION.json',dict(exit_code=0,states=len(rows),nondegenerate_hulls_checked=checked,max_area_difference=maxarea,classification_and_bearing_counts_exact=True,excluded_nonfoot=True,model_load=0,mj_forward=0,mj_step=0,neural_step=0));(O/'verify_early_support_region.py').write_bytes(__import__('pathlib').Path(__file__).read_bytes());print('verified',len(rows),'states',checked,'2D hulls')
if __name__=='__main__':main()
