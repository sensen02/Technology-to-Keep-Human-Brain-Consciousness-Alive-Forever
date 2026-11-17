"""STDonly sourceprepare for actual600->900 statealignedcalibration."""
from pathlib import Path
import argparse,importlib.util,sys
ROOT=Path(__file__).resolve().parent
PYTHON=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python')
def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare-state-aligned600-to900',action='store_true');a=p.parse_args()
 if not a.prepare_state_aligned600_to900:p.error('Explicitstatealigned600to900metadatarequired')
 if Path(sys.executable).absolute()!=PYTHON.absolute() or importlib.util.find_spec('torch') is None:raise ValueError('Correctqualified125712Torchinterpreterrequired; noTorchimporthere')
 import sources
 target=ROOT/'INPUT_BINDINGS.json'
 if target.exists():raise FileExistsError(target)
 b=sources.bindings();b['resume_lineage']=dict(kind='RESUME600_STATE_ALIGNED_ACTION_TO900',start_step=600,total_final_step=900,new_updates=300,save_steps=[600,750,900],fresh_optimizer=False,reseed=False,reanchor=False,initial_anchor='actual600payload.config.initial_anchor remainsoriginalfcbd',trained_model_format='INDEPENDENT_MOTOR_STATE_ALIGNED_ACTION_V1',qualified_python=str(PYTHON),training_RSS_MiB=1800,reference_preflight_RSS_MiB=3072,wall_s=7200)
 sources.save(target,b);print('STATE_ALIGNED_600_TO900_ACTUAL_SOURCE_METADATA_PREPARED',flush=True)
if __name__=='__main__':main()
