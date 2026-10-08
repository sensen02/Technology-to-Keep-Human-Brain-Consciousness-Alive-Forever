"""Quick check that the epidermis actually stratifies (not a result)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epidermis import Epidermis
e = Epidermis(Lx=180, Ly=180, seed=0, initial_layers=2)
print("start cells:", len(e.m.alive), "occ:", e.m.occupancy(), flush=True)
e.run(2000, record_every=200, verbose=True)
h = e.history[-1]
print(json.dumps({k: h[k] for k in ('mcs','n_cells','thickness_um','layer_thickness_um',
                                    'o2_at_base','o2_at_surface','o2_min',
                                    'height_stage_correlation','n_divisions_total',
                                    'n_shed_total')}, indent=2))
print("lineage:", e.lineage_check())
