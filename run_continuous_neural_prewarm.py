"""Declared real closed-loop neural prewarm; no synthetic muscle targets."""
from pathlib import Path
import json,hashlib
import run_physical_walk_prewarm as w
P=Path(__file__).parent
O=Path('/home/sensen/Desktop/engine修复验收/s5-continuous-neural-prewarm/20261028T086000Z')
def main():
 pre=O.parent/(O.name+'_PREDECLARED.json');pre.parent.mkdir(parents=True,exist_ok=True);assert not pre.exists() and not O.exists()
 sources=[Path(__file__),Path(w.__file__),P/'engine/embodied/continuous_handover.py',P/'engine/neural_cond.py',P/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json']
 pre.write_text(json.dumps(dict(scope='Fix cold neural takeover only. Real sensory feedback and previous-completed-window muscle commands during last200ms of algorithm-assisted warmup; same network/decoder/pending transferred without reset. No invented mappings or physical parameter changes.',neural_prewarm_s=.2,neural_only_s=.25,handover_s=1.2,body_dt_s=.0000125,neural_dt_s=.0005,command_dt_s=.001,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2)+'\n')
 w.run(O,1.,100.,50.,.0000125,neural_duration=.25,neural_command_dt_s=.001,warmup_settle_s=.15,warmup_hold_s=.05,neural_prewarm_s=.2)
if __name__=='__main__':main()
