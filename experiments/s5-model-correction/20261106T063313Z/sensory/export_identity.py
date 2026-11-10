"""Read-only metadata joins; only the requested new output file is written."""
from pathlib import Path
import hashlib
import json
import numpy as np

R = Path('/run/media/sensen/Data2/cell_wound_prototype')
D = Path('/home/sensen/Desktop/engine修复验收')
META = R / 'data/banc888/banc_888_meta.feather'
IDENT = D / 's5-banc888-physical-handover/20261101T220018Z/IDENTITIES.npz'
LOOKUP = D / 's3-banc888-runtime-admission/20261101T220018Z/SENSORY_LOOKUP_SNAPSHOT888.json'
OUT = D / 's5-model-correction/20261106T063313Z/sensory/IDENTITY.json'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
CHANNELS = ('claw_position','hook_direction','hairplate_angle','campaniform_strain','club_vibration','bristle_touch','multidendritic_noxious')
FIELDS = ('banc_888_id','root_id','side','cell_sub_class','cell_type','body_part_sensory','peripheral_target_type','nerve','cell_function_detailed')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(meta, roots, lookup):
    roots = list(map(str, roots)); local = {r:i for i,r in enumerate(roots)}
    if len(local) != len(roots) or meta.banc_888_id.duplicated().any():
        raise ValueError('ambiguous identity')
    indexed = meta.set_index('banc_888_id', drop=False)
    if any(r not in indexed.index for r in roots):
        raise ValueError('incomplete banc888 join')
    groups = {leg:{ch:[] for ch in CHANNELS} for leg in LEGS}
    for root, pair in lookup['leg_channel_by_root'].items():
        if root in local:
            groups[pair[0]][pair[1]].append(local[root])
    records = []; blocks = {}; quarantine = []
    for leg in LEGS:
        for i in groups[leg]['hairplate_angle']:
            row = indexed.loc[roots[i]]; rec = {key:str(row[key]) for key in FIELDS}
            rec.update(local_index=i, leg=leg, block=None, organ=None, conflict=False, status='LOCATION_UNRESOLVED')
            side = 'left' if leg.startswith('l') else 'right'
            if rec['side'] != side or rec['cell_sub_class'] != ('front_leg' if leg.endswith('f') else 'middle_leg' if leg.endswith('m') else 'hind_leg')+'_hair_plate_neuron':
                raise ValueError('lookup disagrees with native annotation')
            parts = set(rec['body_part_sensory'].split(',')); targets = set(rec['peripheral_target_type'].split(','))
            organ = next((s for s in ('CoHP8','CoHP3','CoHP4') if s in targets), None)
            expected_nerve = {'CoHP8':'ventral_prothoracic_nerve','CoHP3':'dorsal_prothoracic_nerve','CoHP4':'prothoracic_accessory_nerve'}
            coxa = (leg.endswith('f') and parts == {'coxa','front_leg'} and organ is not None
                    and targets == {organ,'hair_plate'} and rec['nerve'] == side+'_'+expected_nerve[organ])
            # A type label cannot override contradictory peripheral location/nerve evidence.
            conflict = rec['cell_type'].startswith('CoHP') and (not coxa or rec['cell_type'] != organ)
            rec['conflict'] = bool(conflict)
            if conflict:
                rec['status'] = 'QUARANTINED_TYPE_LOCATION_CONFLICT'; quarantine.append(rec['banc_888_id'])
            elif coxa:
                rec.update(block='coxa', organ=organ, status='CONSENSUS_LOCATION_NOT_TUNING')
            elif (leg.endswith('f') and parts == {'front_leg','trochanter'} and targets == {'hair_plate'}
                    and rec['nerve'] == side+'_prothoracic_leg_nerve'):
                rec.update(block='trochanter', status='CONSENSUS_LOCATION_NOT_TUNING')
            if rec['block']:
                blocks.setdefault((leg,rec['block']),[]).append((i,rec['banc_888_id']))
            records.append(rec)
    block_rows = [dict(leg=leg, block=block, local_indices=[i for i,r in ids], banc_888_ids=[r for i,r in ids])
                  for (leg,block),ids in sorted(blocks.items())]
    return dict(join_key='banc_888_id', n_neurons=len(roots), roots=roots, channel_groups=groups,
                counts={leg:{ch:len(ids) for ch,ids in vals.items()} for leg,vals in groups.items()},
                generic_root_mismatch_count=sum(str(indexed.loc[r,'root_id']) != r for r in roots),
                hairplate_records=records, consensus_blocks=block_rows, quarantine_ids=sorted(quarantine),
                consensus_count=sum(len(row['local_indices']) for row in block_rows),
                unknown_or_quarantined_count=sum(row['block'] is None for row in records),
                scope='29 location-consensus hairplates; angular amplitude proxy only, no inferred polarity or per-axis preference',
                importance='核心必需')


def main():
    import pandas as pd
    sources = {str(p):sha(p) for p in (META,IDENT,LOOKUP,Path(__file__))}
    meta = pd.read_feather(META).fillna('')
    with np.load(IDENT) as z:
        roots = z['root_ids']
    out = build(meta, roots, json.loads(LOOKUP.read_text()))
    assert out['consensus_count']==29 and len(out['quarantine_ids'])==9
    assert out['generic_root_mismatch_count']==439 and len(out['hairplate_records'])==214
    assert {r['leg']+'/'+r['block']:len(r['local_indices']) for r in out['consensus_blocks']} == {'lf/coxa':8,'rf/coxa':4,'lf/trochanter':8,'rf/trochanter':9}
    assert all(sha(p)==h for p,h in sources.items())
    out['source_sha256']=sources
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open('x') as f:
        json.dump(out,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps(dict(output=str(OUT),consensus_count=29,quarantined=9,unknown_or_quarantined=185)))


if __name__ == '__main__':
    main()
