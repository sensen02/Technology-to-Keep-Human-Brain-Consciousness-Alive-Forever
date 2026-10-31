"""Independently verify saved pre-neural phase ranking. No simulation imports."""
from pathlib import Path
import argparse,json
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory');args=ap.parse_args();p=Path(args.directory);rows=json.loads((p/'CANDIDATES.json').read_text());selection=json.loads((p/'SELECTION.json').read_text());valid=[]
 for r in rows:
  eligible=min(r['complete_air_bouts_ge10ms'])>=3 and r['feet_contacting']>=3 and r['nonfoot_count']==0
  assert r['eligible']==eligible
  score=r['native_predicted_10ms_downward_root_displacement']/.1+r['native_predicted_10ms_angular_excursion']/.1+abs(r['native_foot_load_ratio']-1)*.05+r['assist_norm']*.0001
  assert abs(score-r['score'])<1e-12
  if eligible:valid.append(r)
 valid.sort(key=lambda r:(r['score'],-r['target_time_s']));assert valid[0]['target_time_s']==selection['selected']['target_time_s']
 alternative=next(r for r in valid if r['target_time_s']!=valid[0]['target_time_s'])
 out=dict(passed=True,candidates=len(rows),eligible=len(valid),selected_time_s=valid[0]['target_time_s'],best_distinct_time_s=alternative['target_time_s'],best_distinct_score=alternative['score'],baseline_score=valid[0]['score'],scope='Confirms scoring/eligibility only; instantaneous acceleration surrogate is not validated as a predictor of neural support.',new_physics_steps=0,new_neural_steps=0)
 target=p/'PARENT_RANKING_VERIFICATION.json';assert not target.exists();target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
