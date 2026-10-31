"""One prospectively selected alternate physical handover phase; unchanged neural controller."""
from pathlib import Path
import json,hashlib
import run_physical_walk_prewarm as w
R=Path('/home/sensen/Desktop/engine修复验收/s5-handover-phase-selection/20261028T064000Z')
O=Path('/home/sensen/Desktop/engine修复验收/s5-selected-phase-neural-takeover/20261028T065000Z')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ranking=json.loads((R/'RANKING.json').read_text());distinct=[r for r in ranking if abs(r['target_time_s']-1.)>1e-6];chosen=distinct[0];assert chosen['target_time_s']==.915 and min(chosen['complete_air_bouts_ge10ms'])>=3
 p=O.parent/(O.name+'_PREDECLARED.json');p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();w.dump(p,dict(scope='One actual alternate-phase test, selected before neural outcomes. Highest-ranked distinct phase, NOT a better predicted score than already-tested1s.',chosen_phase_s=.915,mechanical_score=chosen['score'],baseline_best_phase_s=1.,selection_reason='Best distinct mechanical surrogate rank, and lower initial root angular speed than1s. Test whether posture/phase changes actual collapse; no claim surrogate predicts improvement.',walking_min_complete_ge10ms_flights=chosen['complete_air_bouts_ge10ms'],baseline='061 1ms command/default5msfilter100Hz decoder',neural_duration_s=.25,command_dt_s=.001,neural_dt_s=.0005,body_dt_s=.0000125,additional_stimulus_or_background=False,post_neural_auxiliary=False,source_hashes={str(x):sha(x) for x in [Path(__file__),Path(w.__file__),R/'RANKING.json',R/'SELECTION.json']}))
 w.run(O,.915,100.,50.,.0000125,neural_duration=.25,neural_command_dt_s=.001,neural_motor_config=None)
 (O/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
if __name__=='__main__':main()
