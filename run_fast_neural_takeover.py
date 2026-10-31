"""P3 bounded command-clock comparison: actual unchanged 1s warmup then 250ms neural-only."""
from pathlib import Path
import json,hashlib,ast
import run_physical_walk_prewarm as w
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-fast-neural-takeover/20261028T061000Z')
OLD=Path('/home/sensen/Desktop/engine修复验收/s5-physical-warmup/20261028T058000Z/run_physical_walk_prewarm.py')
def main():
 assert not OUT.exists()
 # Configuration declaration is written before any model loading or physical/neural step.
 OUT.parent.mkdir(parents=True,exist_ok=True)
 prereg=OUT.parent/'20261028T061000Z_PREDECLARED.json';assert not prereg.exists()
 old=OLD.read_bytes();new=Path(w.__file__).read_bytes();old_ast=ast.parse(old);new_ast=ast.parse(new)
 facts=dict(objective='Only command/sensory hold clock 5ms -> 1ms; no new controller or neural stimulation',old_runner=str(OLD),old_sha256=hashlib.sha256(old).hexdigest(),new_runner=str(Path(w.__file__)),new_sha256=hashlib.sha256(new).hexdigest(),wrapper=str(Path(__file__)),wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parameterization='run gains optional neural_command_dt_s=.005 unchanged default; API receives this argument',warmup_seconds=1.,neural_seconds=.25,body_dt_s=.0000125,neural_dt_s=.0005,command_dt_s=.001,motor_filter_tau_s=.005,motor_rate_full_scale_hz=100.,motor_gain=1.,seed=0,baseline='058 same physical model, same warmup, compare first250ms of its500ms neural phase',forbidden_changes=['network background','extra sensory stimulus','activation preload','support ramp','auxiliary force after handover','model geometry/mass/contact/muscle gain change'],screens=['exact warmup and handover-state identity against058','post-neural qfrc/xfrc zero','command origin k-1','first nonfoot latency vs baseline15ms','motor onset and command onset latencies','last200ms foot-load and nonfoot fraction'],scope='One bounded timing trial, not parameter sweep; shorter neural window declared and comparisons restricted to matched first250ms')
 prereg.write_text(json.dumps(facts,indent=2)+'\n')
 w.run(OUT,1.,100.,50.,.0000125,neural_duration=.25,neural_command_dt_s=.001)
 (OUT/'CONFIG_COMPARISON_PREREGISTRATION.json').write_text(prereg.read_text())
 (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
if __name__=='__main__':main()
