from pathlib import Path
import json,sys,numpy as np
from derive_required_motor_tier import derive_tier,P
import run_neural_muscle_closure as c
from engine.neck_cut_data import load_banc
from run_one_second_motor_pair import dump,sha
from run_required_motor_tier import O

def main():
 pre=json.loads((O/'PREREGISTRATION.json').read_text());required=pre['required'];cfg=c.TierConfig(background_mean_nA=0,background_sd_nA=0);base=c.build_neural_tier(cfg);default=derive_tier((),base);assert default is base
 for bad in ([required[0],required[0]],[1],required):
  try:derive_tier(bad,base);raise AssertionError('must reject')
  except ValueError:pass
 a=load_banc(str(P/cfg.data_path));lookup={int(x):i for i,x in enumerate(a['root_ids'])};rows=[];old=set(map(int,base['tier']['global_index']))
 for root in required:
  g=lookup.get(root);edge_rows=[]
  if g is not None:
   mask=((a['post']==g)&np.isin(a['pre'],list(old)))|((a['pre']==g)&np.isin(a['post'],list(old)))
   edge_rows=[dict(cache_row=int(k),pre_root=int(a['root_ids'][a['pre'][k]]),post_root=int(a['root_ids'][a['post'][k]]),syn=int(a['syn'][k]),nt_pair=str(a['nt_pair'][k])) for k in np.flatnonzero(mask)]
  rows.append(dict(root=root,cache_present=g is not None,cache_global_index=g,edges_to_old_tier=edge_rows,added=False,reason='full six-root configuration rejected; no partial experiment substituted'))
 dump(O/'OFFLINE_SOURCE_CONNECTIONS.json',rows)
 dump(O/'TESTS.json',dict(default_identity_noop=True,default_rebuild_exact='first attempt rebuilt twice and asserted root/local and CSR indptr/indices/data before failure',duplicate_rejected=True,missing_rejected=True,actual_six_root_request_rejected=True,extended_graph_regression='NOT_RUN: missing cache IDs',network_initialization='ConductanceNetwork constant arrays; no random node parameter draws',body_runs=0))
 dump(O/'FINAL_EXECUTION.json',dict(command='OPENBLAS_NUM_THREADS=1 '+sys.executable+' '+str(Path(__file__)),exit_code=0,original_attempt_exit_code=1,original_failure='two exact required roots absent runtime cache',body_steps=0,neural_steps=0,forward=0,model_load=0,static_tier_build_total=3,source_lookup_extra_loads='first failed derives plus two offline inspection invocations',decision='STOP_NO_SIX_NODE_SOURCED_CONFIGURATION'))
 for p in (Path(__file__),P/'derive_required_motor_tier.py',P/'run_required_motor_tier.py'):(O/p.name).write_bytes(p.read_bytes())
 print('offline verified; no body run')
if __name__=='__main__':main()
