"""Prospective load-transfer test: fade algorithm assist to zero during live neural prewarm."""
from pathlib import Path
import json,hashlib
import run_physical_walk_prewarm as w
P=Path(__file__).parent
O=Path('/home/sensen/Desktop/engine修复验收/s5-assist-fade-neural/20261028T087000Z')
def main():
 pre=O.parent/(O.name+'_PREDECLARED.json');pre.parent.mkdir(parents=True,exist_ok=True);assert not pre.exists() and not O.exists()
 sources=[Path(__file__),P/'run_physical_walk_prewarm.py',P/'engine/embodied/continuous_handover.py',P/'engine/neural_cond.py',P/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json']
 pre.write_text(json.dumps(dict(scope='Load-transfer test, not autonomy. Algorithm joint assist fades linearly to zero across the final200ms while the same live network drives muscles; neural-only afterwards. No invented mappings, no body parameter change.',neural_prewarm_s=.2,assistance_fade_s=.2,neural_only_s=.25,handover_s=1.2,preregistered_success='body sustains foot support in the last100ms of the neural-only phase',source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2)+'\n')
 w.run(O,1.,100.,50.,.0000125,neural_duration=.25,neural_command_dt_s=.001,warmup_settle_s=.15,warmup_hold_s=.05,neural_prewarm_s=.2,assistance_fade_s=.2)
if __name__=='__main__':main()
