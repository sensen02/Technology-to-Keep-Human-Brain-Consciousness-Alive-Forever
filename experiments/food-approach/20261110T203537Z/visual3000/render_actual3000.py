"""Adapt frozen generic renderer in memory; old source remains byte-for-byte intact."""
from pathlib import Path
import hashlib,json
import sys

HERE=Path(__file__).resolve().parent
REFERENCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/visual/render_motor.py')
VERIFY=HERE.parent/'continuation3000/RECOVERY_VERIFY_ACTUAL_speed085.json'

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

reference_hash=digest(REFERENCE)
assert reference_hash=='985ebb1c85fd4a8ad31c539f86a719748595253534f6e1cd690aa4d357ca7ce6'
source=REFERENCE.read_text()
def replace(old,new):
    global source
    assert source.count(old)==1,old
    source=source.replace(old,new)
replace("paths += [Path(__file__).resolve(),HERE/'PREREG.json']", "paths += [Path(__file__).resolve(),HERE/'PREREG.json',REFERENCE,VERIFY,run/'runtime_geometry.npz']")
replace("result['neural_decisions_sha256']", "result['neural_decisions']['sha256']")
replace("assert result['odor_or_food_success'] is False", """assert result['food_success'] is False
    audit=json.loads(VERIFY.read_text())
    assert audit['integrity_all_pass'] and all(audit['checks'].values())
    assert audit['run']==str(run) and audit['checkpoint_sha256']==result['checkpoint_sha256']
    assert digest(result['checkpoint'])==result['checkpoint_sha256']
    assert result['engineering_pass'] is False and audit['engineering_pass'] is False
    assert hashes[str(run/'runtime_geometry.npz')]==result['runtime_geometry']['sha256']
    for ledger in (result['source_hashes'],audit['source_hashes']):
        for path,expected in ledger.items():assert digest(path)==expected
    assert audit['checks']['all2880_actual_checkpoint_actions_bitexact']
    assert audit['checks']['all2880_actual_checkpoint_means_bitexact']
    assert audit['checks']['all2880_actual_checkpoint_hidden_bitexact']""")
replace("yaw=np.unwrap(d['yaw_after_each_control_ms'])", "yaw=np.unwrap(d['yaw_after_each_1ms'])")
replace("assert len(yaw)==14400", "assert len(yaw)==15400 and d['hidden_before'].shape==(2880,128)")
replace("yaw[-1]-yaw[4799]", "yaw[-1]-yaw[5799]")
replace("status='ENGINEERING GATES PASS' if auto['all_gates'] else 'ENGINEERING GATES FAIL'", "status='ACTUAL FAIL / COLLAPSE / NO FOOD'")
replace("if i>=1000:assert abs(np.arctan2(np.sin(actual_yaw-yaw[i-1000]),np.cos(actual_yaw-yaw[i-1000])))<1e-10", "assert abs(np.arctan2(np.sin(actual_yaw-yaw[i]),np.cos(actual_yaw-yaw[i])))<1e-10")
replace("'PAD z / actual load/W'", "'5 gates: '+ ' '.join(key+':'+('P' if value else 'F') for key,value in auto['gates'].items())")
replace("'PURE CONDITIONAL ANN'", "'PURE NEURAL / speed085'")
replace("'rendered {frame+1}/770'", "'rendered {frame+1}/770'")
namespace=dict(__name__='adapted_renderer',__file__=str(Path(__file__).resolve()),REFERENCE=REFERENCE,VERIFY=VERIFY,digest=digest)
exec(compile(source,str(Path(__file__).resolve())+' [frozen generic adapted]','exec'),namespace)
if __name__=='__main__':
    namespace['main']()
    assert digest(REFERENCE)==reference_hash
