"""Independent saved actual3bias checkpoint/trace audit, no Core/optimization."""
from pathlib import Path
import argparse,json
import evaluate_motor as ev
HERE=ev.ROOT
PARENT_ADAPTER=HERE.parent/'verify_actual.py'
PRIMARY=HERE.parent.parent.parent/'20261110T235830Z/evaluation/verify_actual.py'
def build():
 src=PARENT_ADAPTER.read_text();src=ev.replace_once(src,"PRIMARY=HERE.parent.parent/'20261110T235830Z/evaluation/verify_actual.py'","PRIMARY=Path("+repr(str(PRIMARY))+")")
 src=ev.replace_once(src,"payload['format_version']=='RIGHT_ONLY_V1' and payload['curve'][-1]['step']==2000","payload['format_version']=='RIGHT_THREE_BIAS_ACTUAL_V1' and payload['training_updates']==0 and payload['optimizer'] is None")
 src=ev.replace_once(src,"rollout_ast=ast.parse((HERE.parent.parent/'20261110T235830Z/evaluation/evaluate_motor.py').read_text())","rollout_ast=ast.parse(Path("+repr(str(ev.PRIMARY))+").read_text())")
 namespace={'__file__':str(PARENT_ADAPTER),'__name__':'_RIGHT3bias_inherited_audit'};exec(compile(src,str(PARENT_ADAPTER)+'::RIGHT3bias_separate_format','exec'),namespace);return namespace['build']()
def main():
 a=argparse.ArgumentParser();a.add_argument('--candidate',choices=['c00','c01','c02','c03'],required=True);a.add_argument('--condition',choices=list(ev.COMMANDS),required=True);args=a.parse_args();case=HERE/args.candidate;out=case/('VERIFY_'+args.condition+'.json')
 if out.exists():raise ValueError('no overwrite independent audit')
 r=build()(case/('EVAL_'+args.condition));r['source_hashes'][str(PARENT_ADAPTER)]=ev.sha256(PARENT_ADAPTER);r['source_hashes'][str(PRIMARY)]=ev.sha256(PRIMARY);out.write_text(json.dumps(ev.jsonable(r),indent=2,allow_nan=False)+'\n');print(json.dumps(dict(candidate=args.candidate,condition=args.condition,integrity_all_pass=r['integrity_all_pass'],engineering_pass=r['engineering_pass'],checks=len(r['checks']))),flush=True)
if __name__=='__main__':main()
