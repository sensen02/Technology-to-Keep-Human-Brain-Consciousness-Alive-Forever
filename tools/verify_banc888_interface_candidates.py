"""Independent annotation-row and unchanged-runtime checks for root888 candidates."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path.append(str(REPO/'venv/lib/python3.12/site-packages'))
import pandas as pd
O=Path('/home/sensen/Desktop/engine修复验收/s4-banc888-interface-candidates/20261101T212622Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    result=json.loads((O/'RESULT.json').read_text());d=pd.read_feather(REPO/'data/banc888/banc_888_meta.feather')
    indexed=d.assign(root_888=d['root_888'].astype('int64')).set_index('root_888')
    assert indexed.index.is_unique
    expected={};subtypes={'claw_chordotonal_organ_neuron':'claw_position','hook_chordotonal_organ_neuron':'hook_direction','hair_plate_neuron':'hairplate_angle','campaniform_sensillum_neuron':'campaniform_strain','club_chordotonal_organ_neuron':'club_vibration','bristle_neuron':'bristle_touch','multidendritic_neuron':'multidendritic_noxious'}
    for row in d[['root_888','side','cell_sub_class']].itertuples(index=False,name=None):
        root,side,label=row;label=str(label);legtype=label.split('_leg')[0];letter={'front':'f','middle':'m','hind':'h'}.get(legtype)
        if letter is None or side not in ['left','right']:continue
        candidates=[v for text,v in subtypes.items() if text in label]
        if not candidates:continue
        assert len(candidates)==1;expected[str(int(root))]=[side[0]+letter,candidates[0]]
    sensory=json.loads((O/'sensory_channel_lookup_root888.candidate.json').read_text())
    assert sensory['leg_channel_by_root']==expected and sensory['source']['root_id_column']=='root_888'
    old=json.loads((REPO/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json').read_text());new=json.loads((O/'mn_pools_root888.candidate.json').read_text())
    assert old['measured']['unresolved']==new['measured']['unresolved'] and len(new['measured']['unresolved'])==16
    assert len(new['measured']['resolved'])==74
    changed=0
    for a,b in zip(old['measured']['resolved'],new['measured']['resolved']):
        assert {k:v for k,v in a.items() if k!='pool_root_ids'}=={k:v for k,v in b.items() if k!='pool_root_ids'}
        rows=indexed.loc[b['pool_root_ids']]
        assert set(map(int,rows['root_id']))==set(a['pool_root_ids'])
        assert set(rows['flow'])=={'efferent'} and set(rows['side'])=={b['side']} and set(rows['peripheral_target_type'])=={b['muscle_target']}
        body={'F':'front_leg','M':'middle_leg','H':'hind_leg'}[b['leg'][1]];assert set(rows['body_part_effector'])=={body}
        changed+=a['pool_root_ids']!=b['pool_root_ids']
    assert new['status']=='CANDIDATE_NOT_RUNTIME' and new['runtime_allowed'] is False
    for f,h in result['source_sha256'].items():assert sha(Path(f))==h
    verification=dict(passed=True,all_sensory_annotation_assignments_reconstructed=len(expected),motor_rows=74,unresolved_unchanged=16,changed_motor_actuator_rows=changed,source_files_unchanged=True,all_root888_unique=True,calls=dict(model_load=0,network_build=0,neural_step=0,body_step=0,training=0),limits='Candidate identity validation only, not numerical neural replay or deployment certification.')
    (O/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n');print(json.dumps(verification,indent=2))
if __name__=='__main__':main()
