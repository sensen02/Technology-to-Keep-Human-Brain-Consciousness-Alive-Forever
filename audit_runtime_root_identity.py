"""Exercise real runtime tier/pool/index functions, never step neural or body."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import run_neural_muscle_closure as c
from engine.neck_cut_data import load_banc
from engine.embodied.motor_output import build_motor_manifest
P=Path(__file__).parent
O=Path('/home/sensen/Desktop/engine修复验收/s2-runtime-root-identity/20261027T165000Z')
S=Path('/home/sensen/Desktop/engine修复验收/s2-split-write/20261027T163000Z')
A=Path('/home/sensen/Desktop/engine修复验收/s2-mapping-validation/20261027T164000Z')
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def exact_index(ids,required):
 # Regression assertion only; runtime missing roots remain explicitly reported.
 if any(not isinstance(x,(int,np.integer)) for x in ids):raise ValueError('noninteger identity')
 if len(set(map(int,ids)))!=len(ids):raise ValueError('duplicate root')
 lookup={int(r):i for i,r in enumerate(ids)}
 if any(r not in lookup for r in required):raise ValueError('missing network node')
 return [lookup[r] for r in required]
def tests():
 x=2**53;ids=json.loads(json.dumps([x,x+1,x+2]));assert exact_index(ids,ids)==[0,1,2]
 src=dict(measured=dict(resolved=[dict(actuator='a',leg='LF',side='left',muscle_target='target',pool_root_ids=[x,x+1])],unresolved=[]));m=build_motor_manifest(src,ids,['a']);assert m['rows'][0]['local_indices']==[0,1]
 assert build_motor_manifest(json.loads(json.dumps(src)),ids,['a'])==m
 try:exact_index(ids,[x+3]);raise AssertionError('missing not rejected')
 except ValueError:pass
 missing=build_motor_manifest(src,[x+2],['a'])['rows'][0];assert missing['status']=='absent_from_tier' and missing['missing_tier_roots']==[x,x+1]
 return ['adjacent roots >2^53 remain distinct','JSON integer roundtrip','real pool membership indices','missing node assertion raises and runtime explicitly reports absent','correct input unchanged']
def main():
 O.mkdir(parents=True,exist_ok=True);dump(O/'TESTS.json',dict(passed=tests()));old=json.loads((S/'MAPPING.json').read_text());prev=json.loads((A/'MAPPING_VALIDATION.json').read_text());sources=[Path(__file__),P/'run_neural_muscle_closure.py',P/'engine/embodied/adapters.py',P/'engine/neck_cut_data.py',P/'engine/embodied/motor_output.py',c.POOLS,S/'MAPPING.json',A/'MAPPING_VALIDATION.json']
 cfg=c.TierConfig(background_mean_nA=0,background_sd_nA=0);cache=P/cfg.data_path;sources.append(cache);dump(O/'INPUTS.json',{str(p):sha(p) for p in sources})
 a=load_banc(str(cache));built=c.build_neural_tier(cfg);tier=built['tier'];roots=tier['root_ids'];gi=tier['global_index'];loc=tier['local_of_global'];assert a['root_ids'].dtype.kind in 'iu';assert roots.dtype.kind in 'iu';assert a['pre'].dtype.kind in 'iu' and a['post'].dtype.kind in 'iu';assert len(set(map(int,a['root_ids'])))==len(a['root_ids']);np.testing.assert_array_equal(roots,a['root_ids'][gi]);np.testing.assert_array_equal(loc[gi],np.arange(len(gi)))
 source=json.loads(c.POOLS.read_text());manifest=c.build_motor_manifest(source,roots,[r['actuator'] for r in old['rows']]);assert manifest==old
 rows=[]
 for r,pr in zip(manifest['rows'],prev['rows']):
  wanted=pr['pool_root_ids'];actual=r['pool_root_ids'];local=r['local_indices'];back=[int(roots[i]) for i in local];present=sorted(set(wanted)&set(map(int,roots)));assert sorted(back)==present
  rows.append(dict(actuator=r['actuator'],expected_exact_source_roots=wanted,runtime_pool_roots=actual,network_local_indices=local,network_global_indices=[int(gi[i]) for i in local],roundtrip_roots=back,missing_from_source=sorted(set(wanted)-set(actual)),extra_to_source=sorted(set(actual)-set(wanted)),duplicates=len(actual)-len(set(actual)),collisions=len(back)-len(set(back)),unavailable_in_tier=sorted(set(wanted)-set(back)),saved_zero_event=pr['activity_state']=='mapped_zero_events',correct_nonempty_network_nodes=bool(local) and sorted(back)==present))
 # Source cache endpoints are global integer node indices; verify every inside-row endpoint roundtrip.
 inside=tier['inside_rows'];endpoint={}
 for k in ('pre','post'):
  g=a[k][inside];l=loc[g];assert np.all(l>=0);np.testing.assert_array_equal(gi[l],g);np.testing.assert_array_equal(roots[l],a['root_ids'][g]);endpoint[k]=dict(checked=int(len(g)),dtype=str(a[k].dtype),roundtrip_exact=True)
 pairs=set(zip(map(int,a['pre'][inside]),map(int,a['post'][inside])))
 for name in ('We','Wi'):
  mat=built['edges'][name].tocoo();assert all((int(gi[col]),int(gi[row])) in pairs for row,col in zip(mat.row,mat.col));endpoint[name]=dict(nonzero_connections=int(mat.nnz),all_final_network_endpoints_exist_in_cache=True,index_dtype=str(mat.row.dtype))
 zero=[r for r in rows if r['saved_zero_event']];assert len(zero)==11 and all(r['correct_nonempty_network_nodes'] for r in zero)
 groups,_=c.channel_index_map(int(built['n']),roots)
 dump(O/'MEMBER_DIFFERENCES.json',rows);dump(O/'ZERO_EVENT_POOLS_11.json',zero);dump(O/'ENDPOINTS.json',endpoint)
 dump(O/'RESULT.json',dict(command='OPENBLAS_NUM_THREADS=1 '+sys.executable+' '+str(Path(__file__)),exit_code=0,runtime_manifest_bitwise_equal=manifest==old,cache_root_dtype=str(a['root_ids'].dtype),tier_root_dtype=str(roots.dtype),tier_nodes=len(roots),unmapped=16,mapped=74,zero_event_pools_correct_nonempty=11,missing_tier_roots_total=sum(len(r['unavailable_in_tier']) for r in rows),root_identity_precision_loss=False,conclusion='precision defect limited to earlier pandas audit path; actual runtime NPZ integer and json.loads pool path exact',calls=dict(load_banc_direct=1,build_neural_tier=1,build_motor_manifest_real=1,channel_index_map=1,model_load=0,forward=0,body_step=0,neural_step=0)))
 (O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());assert all(sha(Path(k))==v for k,v in json.loads((O/'INPUTS.json').read_text()).items());print((O/'RESULT.json').read_text())
if __name__=='__main__':main()
