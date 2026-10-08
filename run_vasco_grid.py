"""Parameter grid for P1, one config per process, JSON out.  Not a result."""
import sys, json, time, os
sys.path.insert(0,'/run/media/sensen/Data2/cell_wound_prototype')
import numpy as np
from vasculogenesis import Vasculogenesis

if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    v = Vasculogenesis(L=cfg["L"], seed=cfg.get("seed",0), n_ec=cfg["n_ec"],
                       n_fibroblast=cfg.get("n_fib",20),
                       contact_inhibited=cfg.get("ci",True),
                       hypoxia_coupling=cfg.get("hyp",True),
                       mode=cfg.get("mode","de_novo"),
                       params={k:val for k,val in cfg.items()
                               if k in ("lambda_chem_tip","lambda_chem_stalk","lambda_length",
                                        "elongation_target","temperature","vegf_decay",
                                        "vegf_secretion_ec","lambda_volume","delta_tip_threshold",
                                        "vegf_hypoxic_amplification")})
    t0=time.time()
    v.run(cfg["mcs"], record_every=cfg["mcs"]//8, verbose=True)
    h = v.history[-1]
    lin = v.lineage_check()
    h["tag"]=cfg.get("tag","")
    h["wall_s"]=time.time()-t0
    h["lineage_ok"]=lin["ok"]
    print("RESULT " + json.dumps(h))
