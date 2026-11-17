"""Parent later runs stdlib admission/freeze, rejects incomplete orfailedB before Core."""
from contracts import *
import argparse,ast

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');args=p.parse_args();require(args.parent_resource_released,'CODE_ONLY until parent release')
 require(not (ROOT/'FROZEN.json').exists() and not (ROOT/'INPUT_BINDINGS_C.json').exists(),'nooverwrite Cexecutionfreeze')
 hashes=admission();bfreeze=read(B/'FROZEN.json')
 for path,h in bfreeze['source_hashes'].items():require(sha(path)==h,'Binputdrift '+path);hashes[path]=h
 paths=[ROOT/n for n in ('PREREG_PROPOSAL.json','contracts.py','spatial.py','runtime.py','runner.py','audit.py','summary.py','prepare.py','preflight.py','orchestrate.py','check_codeonly.py')]
 paths += [B/n for n in ('runtime.py','bindings.py','runner.py','audit.py','FROZEN.json','INPUT_BINDINGS.json','PREFLIGHT.json')]
 screen=AB/'SPATIAL_BASELINE_SCREEN.zh-CN.md';paths.append(screen)
 for path in paths:require(path.is_file(),'requiredCinput '+str(path));hashes[str(path)]=sha(path)
 binding=dict(kind='SPATIAL_C6_DEVELOPMENT_V1',motor_sha256=MOTOR_SHA,behavior_sha256=SENSOR_SHA,B_allfunctionalpass=True,B_summary=str(B/'B_FUNCTIONAL_SUMMARY.json'),B_summary_sha256=sha(B/'B_FUNCTIONAL_SUMMARY.json'),proposal_sha256=sha(ROOT/'PREREG_PROPOSAL.json'),layout_count=6,unique_new_physics_planned=6,no_reuse=True,independent_replicates_claimed=False)
 with (ROOT/'INPUT_BINDINGS_C.json').open('x') as f:json.dump(binding,f,indent=2);f.write('\n')
 hashes[str(ROOT/'INPUT_BINDINGS_C.json')]=sha(ROOT/'INPUT_BINDINGS_C.json')
 with (ROOT/'FROZEN.json').open('x') as f:json.dump(dict(kind='SPATIAL_C6_FROZEN_AFTER_B_ADMISSION_BEFORE_CPHYSICS',source_hashes=hashes,Core_constructions=0,model_loads=0,physics_steps=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(B_admitted=True,frozenfiles=len(hashes),Core_constructions=0,checkpoint_binding=binding)))
if __name__=='__main__':main()
