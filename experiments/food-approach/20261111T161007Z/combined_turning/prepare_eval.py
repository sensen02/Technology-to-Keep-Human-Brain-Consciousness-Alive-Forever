"""Future parent-released zero-Core source/state256 rollout and audit readiness."""
import argparse,json
import torch
from model_io import *
import evaluate_motor as ev
import verify_actual

def main():
 a=argparse.ArgumentParser();a.add_argument('--parent-resource-released',action='store_true');args=a.parse_args()
 if not args.parent_resource_released:raise ValueError('parent composition resource release required')
 torch.set_num_threads(1);prep=json.loads((HERE/'PREPARATION.json').read_text());saved=json.loads((HERE/'VERIFY_SAVED.json').read_text());checks=dict(preparation=prep['all_passed'] and all(prep['checks'].values()),saved=saved['all_passed'] and all(saved['checks'].values()),same_composed_SHA=prep['checkpoint_sha256']==saved['sha256']==sha256(HERE/'COMBINED.pt'))
 verify_hashes(prep['source_hashes']);verify_hashes(prep['component_source_hashes']);verify_hashes(prep['stream_hashes']);model,payload=load_checked(HERE/'COMBINED.pt',sha256(HERE/'COMBINED.pt'));checks['all256_identity']=model.identity()['state_dims']==[128,64,64] and model.identity()['state_total']==256;checks['no_training_fiction']=payload['training_performed'] is False and payload['optimizer'] is None and 'training_updates' not in payload
 checks['explicit256_evaluator_builds']=callable(ev.build());checks['explicit256_saved_audit_builds']=callable(verify_actual.build());paths=ev.source_paths(include_readiness=False);checks['all_sources_exist_preCore']=all(path.is_absolute() and path.is_file() for path in paths);checks['AMENDMENT03_bound']=str(HERE/'AMENDMENT_03_HONEST_DEMONSTRATION_COMPONENTS.json') in payload['source_hashes'] and str(HERE/'AMENDMENT_03_HONEST_DEMONSTRATION_COMPONENTS.json') in model.bindings['composition_scope_hashes']
 result=dict(all_passed=all(checks.values()),checks=checks,source_hashes={str(p):sha256(p) for p in paths},checkpoint_sha256=sha256(HERE/'COMBINED.pt'),state_layout=[128,64,64],state_total=256,Core_constructions=0,physics_steps=0,training_performed=False,actual_composed_qualification='NOT_EVALUATED')
 out=HERE/'READINESS.json'
 if out.exists():raise ValueError('no overwrite composition readiness')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
 if not result['all_passed']:raise ValueError('composed actual preparation failed')
if __name__=='__main__':main()
