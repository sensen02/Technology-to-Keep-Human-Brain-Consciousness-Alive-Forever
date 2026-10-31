"""Saved-only S170 edge-ordinal contrasts; not tracked physical contact identity."""
import argparse,csv
from pathlib import Path
pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output
rows=list(csv.DictReader((p/'CONTACT_EDGES_GAP_FRICTION.csv').open()));best={tuple(r[k] for k in ['geom1','geom2','pair_occurrence','edge']):r for r in rows if r['state']=='BEST'};out=[]
for r in rows:
 if r['state']=='BEST':continue
 key=tuple(r[k] for k in ['geom1','geom2','pair_occurrence','edge']);b=best.get(key)
 out.append(dict(reference_state=r['state'],geom1=r['geom1'],geom2=r['geom2'],pair_occurrence=r['pair_occurrence'],edge=r['edge'],leg=r['leg'],reference_force=r['edge_force'],best_force=None if b is None else b['edge_force'],force_difference=None if b is None else float(b['edge_force'])-float(r['edge_force']),reference_gap=r['distance'],best_gap=None if b is None else b['distance'],reference_pyramid_margin=r['pyramid_margin'],best_pyramid_margin=None if b is None else b['pyramid_margin'],qualification='same geom-pair occurrence/local edge ordinal only; frames/positions differ, NOT tracked physical force change'))
with (p/'CONTACT_EDGE_CONTRAST.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print('Saved',len(out),'qualified edge-ordinal contrasts')
