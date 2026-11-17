"""One pre-execution reference-preflight budget amendment; no originalrewrite."""
from pathlib import Path
import argparse,importlib.util,sys
ROOT=Path(__file__).resolve().parent
V3=ROOT.parent/'target_feedback_order_repair_v3'
PYTHON=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python')
def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare-resume300-order-repair',action='store_true');a=p.parse_args()
 if not a.prepare_resume300_order_repair:p.error('Explicitv4metadataflagrequired')
 if Path(sys.executable).absolute()!=PYTHON.absolute() or importlib.util.find_spec('torch') is None:raise ValueError('Correct125712Torchenvironmentrequired')
 import sources
 target=ROOT/'INPUT_BINDINGS.json'
 if target.exists():raise FileExistsError(target)
 if (V3/'PREFLIGHT.json').exists():raise ValueError('Priorv3hasunexpectedserializedpreflight;reviewbeforeamendment')
 logs=list(V3.glob('*.log'))
 if not any('RSS' in x.read_text() and '1800' in x.read_text() for x in logs):raise ValueError('Actualv3RSS1800entryfailurelogrequired')
 b=sources.bindings();evidence={str(x):sources.sha(x) for x in list(V3.glob('*.py'))+[V3/'PREREG.json',V3/'INPUT_BINDINGS.json']+logs if x.is_file()}
 if (V3/'FAILURE.json').is_file():evidence[str(V3/'FAILURE.json')]=sources.sha(V3/'FAILURE.json')
 b['source_hashes'].update(evidence);b['data_hashes'].update(evidence)
 b['resource_amendment']=dict(kind='PREFLIGHT_REFERENCE_ONLY_RSS3072_TRAINING1800',reference_preflight_RSS_MiB=3072,training_RSS_MiB=1800,wall_s=5400,prior_v3_serialized_PREFLIGHT_present=False)
 sources.save(target,b);print('V4_REFERENCE_PREFLIGHT_RESOURCE_SCOPE_BOUND',flush=True)
if __name__=='__main__':main()
