"""Only planned effective receptor onset origin and physical-trial terminology differ."""
from pathlib import Path
from bindings_v2 import verify_all
ROOT=Path(__file__).resolve().parent
def build_source():
 source=(ROOT/'summarize_B.py').read_text()
 old="""   mask=window(.6,2.6)&(first*turn>0)&(np.mean(z,axis=1)>=.05)&(first*(z[:,0]-z[:,1])>=.001)
   metrics['first_correct_response_elapsed_s']=float(e[np.flatnonzero(mask)[0]]) if np.any(mask) else None;flags['response_within2s']=bool(np.any(mask))"""
 new="""   qualified=(np.mean(z,axis=1)>=.05)&(first*(z[:,0]-z[:,1])>=.001)
   origin=float(e[np.flatnonzero(qualified)[0]]) if np.any(qualified) else None
   mask=np.zeros(len(e),dtype=bool) if origin is None else qualified&(e>=origin-1e-8)&(e<origin+2.-1e-8)&(first*turn>0)
   response=float(e[np.flatnonzero(mask)[0]]) if np.any(mask) else None
   metrics['effective_receptor_onset_elapsed_s']=origin;metrics['first_correct_response_elapsed_s']=response;metrics['first_response_latency_from_effective_input_s']=None if response is None else response-origin;flags['response_within2s']=bool(np.any(mask))"""
 if source.count(old)!=1:raise ValueError('frozen summary adaptation mismatch')
 source=source.replace(old,new)
 old='independent_trials=6,correlated_times_not_independent_replicates=True'
 if source.count(old)!=1:raise ValueError('summary terminology adaptation mismatch')
 return source.replace(old,'unique_physical_trials=6,replicates_per_condition=1,correlated_times_not_independent_replicates=True')
if __name__=='__main__':
 ns={'__name__':'_stimulus_summary_effective_onset_v2','__file__':str(Path(__file__).resolve())};exec(compile(build_source(),str(Path(__file__).resolve())+'::frozen_summary_minimal_adaptation','exec'),ns);ns['verify_frozen']=verify_all;ns['main']()
