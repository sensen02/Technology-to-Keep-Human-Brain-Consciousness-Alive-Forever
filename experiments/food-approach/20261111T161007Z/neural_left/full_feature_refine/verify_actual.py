"""Independent full saved-array audit with strict new feature-refinement format."""
from pathlib import Path
import argparse,json
import evaluate_motor as ev
HERE=ev.HERE
PARENT_AUDITOR=ev.io.PARENT_DIR/'verify_actual.py'
PRIMARY=ev.io.ROOT/'20261110T235830Z/evaluation/verify_actual.py'

def adapted_parent_source():
 source=PARENT_AUDITOR.read_text()
 for old,new in [
  ("payload['format_version']=='strict_LEFT_HEAD_V1' and payload['curve'][-1]['step']==4000","payload['format_version']=='LEFT_FEATURE_REFINE_V1' and payload['curve'][-1]['step']==2000"),
  ("and payload['identity']['encoder_frozen'] and payload['identity']['gru_frozen']","and not payload['identity']['encoder_frozen'] and not payload['identity']['gru_frozen']")]:source=ev.replace_once(source,old,new)
 return source

def build():
 namespace={'__file__':str(PARENT_AUDITOR),'__name__':'_left_feature_independent_actual_audit'}
 exec(compile(adapted_parent_source(),str(PARENT_AUDITOR)+'::FEATURE_FORMAT2000','exec'),namespace)
 return namespace['build']()

def main():
 p=argparse.ArgumentParser();p.add_argument('--condition',choices=list(ev.COMMANDS),required=True);p.add_argument('--parent-resource-released',action='store_true');args=p.parse_args()
 if not args.parent_resource_released:raise ValueError('parent exclusiveheavy release required')
 out=HERE/('VERIFY_'+args.condition+'.json');run=HERE/('EVAL_'+args.condition)
 if out.exists():raise FileExistsError(out)
 try:
  result=build()(run);result['source_hashes'][str(PRIMARY)]=ev.sha256(PRIMARY);result['source_hashes'][str(PARENT_AUDITOR)]=ev.sha256(PARENT_AUDITOR);result['result_sha256']=ev.sha256(run/'RESULT.json')
 except BaseException as exc:
  out.write_text(json.dumps(dict(integrity_all_pass=False,error=repr(exc),physics_integration_steps=0),indent=2)+'\n');raise
 out.write_text(json.dumps(ev.jsonable(result),indent=2,allow_nan=False)+'\n');print(json.dumps(dict(integrity_all_pass=result['integrity_all_pass'],engineering_pass=result['engineering_pass'],checks=len(result['checks']),out=str(out))),flush=True)
if __name__=='__main__':main()
