"""Resolve BANC FeCO subtype by exact CSV subclass, never name similarity.

Does not authorize a receptor current, infer directional tuning, or equate IDs
across specimens. Absent cache/tier roots are explicit missing records.
"""
import csv
import gzip
from pathlib import Path
import numpy as np


def resolve_feco(csv_path, built):
    roots=np.asarray(built['data']['root_ids'])
    by_root={str(int(root)):i for i,root in enumerate(roots)}
    if len(by_root)!=len(roots):raise ValueError('duplicate cached root IDs')
    groups={s:[] for s in ('claw','hook','club')}
    seen=set()
    with gzip.open(Path(csv_path),'rt',newline='') as f:
        for row in csv.DictReader(f):
            if row.get('Class')!='chordotonal_organ_neuron':continue
            for subtype in groups:
                if row.get('Sub Class')!='front_leg_'+subtype+'_chordotonal_organ_neuron':continue
                root=row['Root ID']
                if not root.isdecimal() or root in seen:raise ValueError('invalid or duplicate subtype root')
                seen.add(root)
                gi=by_root.get(root)
                li=None if gi is None else int(built['local_of_global'][gi])
                if li is not None and li<0:li=None
                nerve=row.get('Nerve','')
                if gi is not None:
                    if str(built['data']['ann_primary_type'][gi])!=row.get('Primary Cell Type'):
                        raise ValueError('CSV/cache primary type disagreement for '+root)
                    cached_nerve=str(built['data']['ann_nerve'][gi])
                    if nerve and cached_nerve!=nerve:raise ValueError('CSV/cache nerve disagreement')
                    nerve=cached_nerve
                expected=('left_prothoracic_leg_nerve','right_prothoracic_leg_nerve')
                if nerve not in expected:raise ValueError('FeCO subtype outside confirmed front-leg nerves')
                groups[subtype].append({'root_id':root,'global_index':gi,'local_index':li,
                    'subtype':subtype,'nerve':nerve,'soma_side':row.get('Soma side'),
                    'missing_reason':'absent_cache' if gi is None else 'absent_tier' if li is None else None,
                    'direction':None,'current_authorized':False})
    if any(not rows for rows in groups.values()):raise ValueError('missing subtype annotation')
    return groups
