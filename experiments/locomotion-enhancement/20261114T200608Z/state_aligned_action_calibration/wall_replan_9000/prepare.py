"""Stdlib qualification reuse: actual96checks + exactscience + pretraining9000wall."""
from pathlib import Path
import argparse,importlib.util,sys
from equivalence import ROOT,ORIGINAL,verify_equivalence
PYTHON=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python')

def plan():
 import sources
 equivalence=verify_equivalence();old= sources.read(ORIGINAL/'PREFLIGHT.json');binding=sources.read(ORIGINAL/'INPUT_BINDINGS.json');prereg=sources.read(ROOT/'PREREG.json')
 if old['all_passed'] is not False or old['timing_status']!='BLOCKED_BEFORE_TRAINING_TIMING' or len(old['checks'])!=96 or not all(v is True for v in old['checks'].values()):raise ValueError('Actualoriginal96correctnesspass_timingonlyblocked_required')
 if old['source_hashes']!=binding['source_hashes'] or old['data_hashes']!=binding['data_hashes']:raise ValueError('Actualnumericpreflight_source_manifest_mismatch')
 sources.verify(old['source_hashes']);sources.verify(old['data_hashes'])
 if old['training_updates']!=0 or old['new_training_updates']!=0 or old['physical_steps']!=0 or not old['integrator13_numeric_replayed']:raise ValueError('Originalbeforetraining_no_newupdatesproof')
 if old['full1920_loss_error']!=0 or old['gradient_error']>1e-10 or old['Adam_error']>1e-10 or old['parameter_error']>1e-10:raise ValueError('Originalfullhistory_numericprecisionproof')
 if len(old['migration20_TF12_conditional12'])!=12 or any(v!=0 for row in old['migration20_TF12_conditional12'] for v in row['conditional_errors']):raise ValueError('Actualconditional12sixoutputs_exact0_required')
 if not all(row['passed'] for row in old['FD']):raise ValueError('OriginalregisteredFDgate_required')
 if old['preflight_RSS_limit_MiB']!=3072 or old['training_RSS_limit_MiB']!=1800 or old['peak_RSS_MiB']>3072 or old['wall_limit_s']!=7200:raise ValueError('RSSunchanged_originalwall7200_required')
 estimate=old['predicted300total_withverification_s'];recomputed=300*old['actual_two_branch_micro4x3_update_s']*1.3+old['three_fresh_save_allowance_s']+old['evaluation_allowance_s']
 if abs(estimate-recomputed)>1e-8 or not 7200<estimate<=9000 or prereg['resources']['wall_s']!=9000:raise ValueError('Measuredcompleteestimatefits_new9000_only')
 logs=list(ORIGINAL.glob('*.log'))
 if not any('Statealignedtimingblockedbeforetraining' in p.read_text() for p in logs):raise ValueError('Actualoriginaltimingfailurelog_required')
 evidence={str(p):sources.sha(p) for p in [ORIGINAL/'PREFLIGHT.json',ORIGINAL/'INPUT_BINDINGS.json',ORIGINAL/'PREREG.json']+logs if p.is_file()}
 if (ORIGINAL/'FAILURE.json').is_file():evidence[str(ORIGINAL/'FAILURE.json')]=sources.sha(ORIGINAL/'FAILURE.json')
 allnew=tuple(sources.FILES)+('equivalence.py',)
 newsource={**binding['source_hashes'],**evidence,**{str(ROOT/n):sources.sha(ROOT/n) for n in allnew}}
 newdata={**binding['data_hashes'],**evidence}
 amended=dict(binding,source_hashes=newsource,data_hashes=newdata,resource_wall_replan=dict(old_wall_s=7200,new_wall_s=9000,new_training_updates=300,total_updates=900,no_numeric_rerun=True,science_equivalence=equivalence))
 reference=dict(path=str(ORIGINAL/'PREFLIGHT.json'),sha256=sources.sha(ORIGINAL/'PREFLIGHT.json'),original_all_passed=False,original_timing_status=old['timing_status'],actual_correctness_checks=96)
 ready=dict(old,all_passed=True,timing_status='READY_STATE_ALIGNED600_TO900',wall_limit_s=9000,source_hashes=newsource,data_hashes=newdata,reused_actual_numeric_preflight=reference,resource_replan_pretraining=True,corrected_readiness_against9000=True,qualification_basis='REUSE_ACTUAL96_CORRECTNESS_AND_EXACT_SCIENCE_SOURCE_EQUIVALENCE_AND_PRETRAINING9000_WALL',numeric_metrics_origin=reference,original_all_passed_false_preserved=True,new_numeric_preflight_runs=0,new_NN_forwards=0,new_optimizer_probe_updates=0,science_equivalence=equivalence)
 sources.verify(newsource);sources.verify(newdata)
 return amended,ready,dict(kind='PRETRAINING_MEASURED_WALL_REPLAN_7200_TO9000',old_numeric_preflight=reference,old_estimate_s=estimate,new_wall_s=9000,science_equivalence=equivalence,NN_forwards=0,model_loads=0,Core_constructions=0,physical_steps=0)

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare-wall-replan9000',action='store_true');a=p.parse_args()
 if not a.prepare_wall_replan9000:p.error('Explicitbeforetrainingresource_replan_required')
 if Path(sys.executable).absolute()!=PYTHON.absolute() or importlib.util.find_spec('torch') is None:raise ValueError('Usequalified125712Python_noTorchimport_inmetadata')
 import sources
 for n in ('INPUT_BINDINGS.json','PREFLIGHT.json','RESOURCE_REPLAN.json'):
  if (ROOT/n).exists():raise FileExistsError(ROOT/n)
 b,pre,receipt=plan();sources.save(ROOT/'INPUT_BINDINGS.json',b);sources.save(ROOT/'PREFLIGHT.json',pre);sources.save(ROOT/'RESOURCE_REPLAN.json',receipt);print('WALL_REPLAN9000_READY_REUSING_ACTUAL96_NO_NN_PREFLIGHT_RERUN',flush=True)
if __name__=='__main__':main()
