"""Compare extracted OpenSim to installed MJCF ordered points, no simulator."""
from pathlib import Path
import json,hashlib
O=Path('/home/sensen/Desktop/engine修复验收/s3-promotor-source-objects/20261027T224000Z')
def main():
 os=json.loads((O/'OPENSIM_OBJECTS.json').read_text());mj=json.loads((O/'MJCF_TENDON_ACTUATOR_PATHS.json').read_text());checks=[]
 for r in os[:3]:
  dest=next(d for d in mj if d['name']==r['name']);a=r['pathpoints'];b=dest['pathpoints'];names=[r['name']+'_'+q['name'] for q in a];got=[q['site'] for q in b];parents=[q['parent_frame'].removeprefix('/bodyset/') for q in a];hosts=[q['host_body'] for q in b];errors=[abs(float(x)-float(y)) for p,q in zip(a,b) for x,y in zip(p['location'].split(),q['local_position'].split())];checks.append(dict(name=r['name'],point_count_equal=len(a)==len(b),ordered_point_site_names_equal=names==got,ordered_parent_bodies_equal=parents==hosts,max_abs_local_position_rounding_difference=max(errors),coordinate_tolerance=5e-5,within_tolerance=max(errors)<=5e-5))
 passed=all(c['point_count_equal'] and c['ordered_point_site_names_equal'] and c['ordered_parent_bodies_equal'] and c['within_tolerance'] for c in checks);p=Path(__file__);out=dict(checks=checks,passed=passed,interpretation='numerical correspondence only; not proof of exact converter execution/version or anatomy',script=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),exit_code=0 if passed else 1,model=0,forward=0,neural=0,body=0)
 (O/'OPENSIM_TO_INSTALLED_CHECK.json').write_text(json.dumps(out,indent=2)+'\n');(O/p.name).write_bytes(p.read_bytes());print(json.dumps(out));assert passed
if __name__=='__main__':main()
