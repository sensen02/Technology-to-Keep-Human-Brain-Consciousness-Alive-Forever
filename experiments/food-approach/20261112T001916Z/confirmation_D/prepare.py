"""Stdlib D27admission/freeze, never admits beforeC6+baseline-onlyscreen."""
from contracts import *
import argparse

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');a=p.parse_args();require(a.parent_resource_released,'parentDscopeandresource release');require(not (ROOT/'FROZEN.json').exists() and not (ROOT/'INPUT_BINDINGS_D.json').exists(),'nooverwriteDfreeze')
 hashes=admission();cf=read(C/'FROZEN.json')
 for path,h in cf['source_hashes'].items():require(sha(path)==h,'Ccapturedinputdrift');hashes[path]=h
 paths=[ROOT/n for n in ('PREREG_PROPOSAL.json','contracts.py','runtime.py','runner.py','audit.py','summary.py','prepare.py','preflight.py','orchestrate.py','check_codeonly.py')]+[C/n for n in ('runtime.py','spatial.py','contracts.py','FROZEN.json','INPUT_BINDINGS_C.json','PREFLIGHT.json')]
 for p in paths:require(p.is_file(),'requiredDfile '+str(p));hashes[str(p)]=sha(p)
 binding=dict(kind='CONFIRMATION_D27_SENSORY_CAUSAL_V1',motor_sha256=MOTOR_SHA,behavior_sha256=SENSOR_SHA,C_allintegritystable=True,C_summary_sha256=sha(C/'C_DEVELOPMENT_SUMMARY.json'),proposal_sha256=sha(ROOT/'PREREG_PROPOSAL.json'),uniquephysicsmax=27,replicates_perlayoutcontrol=1,training=False,frozen_before_any_D_actual=True)
 with (ROOT/'INPUT_BINDINGS_D.json').open('x') as f:json.dump(binding,f,indent=2);f.write('\n')
 hashes[str(ROOT/'INPUT_BINDINGS_D.json')]=sha(ROOT/'INPUT_BINDINGS_D.json')
 with (ROOT/'FROZEN.json').open('x') as f:json.dump(dict(kind='D27_FROZEN_AFTER_C_ADMISSION',source_hashes=hashes,Core_constructions=0,physics_steps=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(D_admitted=True,files=len(hashes),Core_constructions=0)))
if __name__=='__main__':main()
