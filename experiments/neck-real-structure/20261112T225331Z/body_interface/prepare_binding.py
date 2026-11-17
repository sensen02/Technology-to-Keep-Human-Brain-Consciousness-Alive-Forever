"""Explicit parent source-binding conversion, no numeric model import or execution."""
from pathlib import Path
import importlib.util,json
from interface import sha,source_ports,require
HERE=Path(__file__).resolve().parent
PREREG=HERE.parent/'SCOPED_NEURAL_CALIBRATION_PREREG.json'
def build_binding():
 pr=json.loads(PREREG.read_text());old=Path(pr['parent_structural_binding']);require(sha(old)==pr['parent_structural_binding_sha256'],'parent structural digest mismatch')
 spec=importlib.util.spec_from_file_location('_body_interface_real_neck_boundary',old.parent/'boundary.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);data=json.loads(old.read_text());q=module.checked_sources(data)
 require(q['source_qualified_two_fibres'] and not q['synthetic_fixture'],'actual scoped structure needed')
 ports=[]
 for fibre in data['fibres']:
  refs=[old,Path(fibre['metadata_path']),Path(fibre['independent_check_path']),Path(fibre['swc_path']),old.parent/'boundary.py',PREREG]
  refs.extend(Path(x['path']) for x in data['files']);refs=sorted(set(refs))
  ports.append(dict(root_id=fibre['root_id'],direction=fibre['direction'],source_reviewed=True,structural_evidence_status='SOURCE_GEOMETRY_REVIEWED',evidence=[dict(path=str(p),sha256=sha(p)) for p in refs]))
 result=dict(schema='BODY_SIDE_SOURCE_PORT_BINDING_V1',scope='SCOPED_ENGINEERING_NECK_BODY_INTERFACE',engineering_execution_admission='PARENT_PREREGISTERED_SCOPED_EXECUTION',parent_neck_binding=str(old),parent_neck_binding_sha256=sha(old),ports=ports,full_neck_admitted=False,native_motor_or_AN_tuning_verified=False)
 source_ports(result);return result
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--execute-parent',action='store_true');args=p.parse_args()
 if not args.execute_parent:print('CODE_ONLY no source binding created')
 else:
  result=build_binding();out=HERE/'SOURCE_PORT_BINDING.json'
  with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
  print(json.dumps(dict(path=str(out),sha256=sha(out),dimensions={d:len(v) for d,v in source_ports(result).items()},training_updates=0)))
