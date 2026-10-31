from pathlib import Path
import json,sys,numpy as np,mujoco
from derive_required_motor_tier import derive_tier,P
import run_neural_muscle_closure as c
from run_quarter_second_pair import Observer,OUT as HISTORY
from run_one_second_motor_pair import INITIAL,sha,dump
from engine.neck_cut_data import load_banc
from engine.neural_cond import ConductanceNetwork,ConductanceParams
O=Path('/home/sensen/Desktop/engine修复验收/s2-required-motor-tier/20261027T170000Z')
D=Path('/home/sensen/Desktop/engine修复验收/s2-runtime-root-identity/20261027T165000Z/MEMBER_DIFFERENCES.json')
def main():
 O.mkdir(parents=True,exist_ok=False);diff=json.loads(D.read_text());required=sorted({root for r in diff for root in r['unavailable_in_tier']});assert len(required)==6
 sources=[Path(__file__),P/'derive_required_motor_tier.py',P/'engine/neural_cond.py',P/'engine/embodied/adapters.py',P/'engine/neck_cut_data.py',P/'run_neural_muscle_closure.py',c.POOLS,INITIAL,D,Path(c.cp.CANONICAL_XML),HISTORY/'MANIFEST_SHA256.json']
 dump(O/'PREREGISTRATION.json',dict(required=required,source_hashes={str(p):sha(p) for p in sources},factor='append six exact roots, no neighbor expansion; preserve old indices and graph; only real internal cache edges',clocks=[.0000125,.0005,.005,.005],duration=.25,seed=0,comparison='new6 and affected9 spikes/commands, original11 zero pools, height/root pose/first nonfoot/last50ms contacts and warnings',gate='default exact and sourced connectivity required before one normal run',no_auxiliary=True))
 cfg=c.TierConfig(background_mean_nA=0,background_sd_nA=0);base=c.build_neural_tier(cfg);default=derive_tier((),base);assert default is base;rebuilt=c.build_neural_tier(cfg)
 np.testing.assert_array_equal(default['tier']['root_ids'],rebuilt['tier']['root_ids']);np.testing.assert_array_equal(default['tier']['local_of_global'],rebuilt['tier']['local_of_global'])
 for key in ('We','Wi'):
  x=default['edges'][key];y=rebuilt['edges'][key]
  for field in ('indptr','indices','data'):np.testing.assert_array_equal(getattr(x,field),getattr(y,field))
 tests=['default nodes, indices, edges and weights exact']
 for bad in ([required[0],required[0]],[1]):
  try:derive_tier(bad,base);raise AssertionError('invalid accepted')
  except ValueError:tests.append('duplicate/missing rejected')
 ext=derive_tier(required,base);n=base['n'];assert ext['n']==n+6;np.testing.assert_array_equal(ext['tier']['root_ids'][:n],base['tier']['root_ids']);a=load_banc(str(P/cfg.data_path));gi=ext['tier']['global_index'];pairs=set(zip(map(int,a['pre']),map(int,a['post'])));edge_diff={};node=[];source_rows=set()
 for key in ('We','Wi'):
  x=ext['edges'][key].tocoo();selected=(x.row>=n)|(x.col>=n);rr=x.row[selected];cc=x.col[selected];ww=x.data[selected];assert all((int(gi[col]),int(gi[row])) in pairs for row,col in zip(rr,cc));edge_diff[key]=[dict(pre_root=int(ext['tier']['root_ids'][col]),post_root=int(ext['tier']['root_ids'][row]),pre_local=int(col),post_local=int(row),weight=float(w)) for row,col,w in zip(rr,cc,ww)]
 W=ext['edges']['We']+ext['edges']['Wi'];groups,_=c.channel_index_map(ext['n'],ext['tier']['root_ids']);sensory_indices={int(v) for leg in groups.values() for ch in leg.values() for v in ch}
 for idx in range(n,n+6):
  g=int(gi[idx]);incoming=np.flatnonzero((a['post']==g)&ext['tier']['inside_rows']);outgoing=np.flatnonzero((a['pre']==g)&ext['tier']['inside_rows']);source_rows.update(map(int,incoming));source_rows.update(map(int,outgoing));node.append(dict(root=int(ext['tier']['root_ids'][idx]),local=idx,cache_global=g,cache_in_rows=incoming.tolist(),cache_out_rows=outgoing.tolist(),modelled_in=int(W[idx,:].nnz),modelled_out=int(W[:,idx].nnz),direct_sensory_current=idx in sensory_indices,sensory_presynaptic_nodes=sorted(set(map(int,W[idx,:].indices))&sensory_indices)))
 dump(O/'NEW_NODES.json',node);dump(O/'NEW_MODELLED_EDGES.json',edge_diff);dump(O/'CACHE_EDGE_ROWS.json',[dict(row=j,pre=int(a['root_ids'][a['pre'][j]]),post=int(a['root_ids'][a['post'][j]]),syn=int(a['syn'][j]),nt_pair=str(a['nt_pair'][j])) for j in sorted(source_rows)])
 # Network constructor is deterministic constant initialization, no random per-node parameters.
 oldnet=ConductanceNetwork(n,ConductanceParams(),dt_ms=.5);newnet=ConductanceNetwork(n+6,ConductanceParams(),dt_ms=.5)
 for field in ('v','ge','gi','adaptation','refractory_until_ms','spike_count'):np.testing.assert_array_equal(getattr(oldnet,field),getattr(newnet,field)[:n])
 dump(O/'DEFAULT_REGRESSION.json',dict(passed=True,tests=tests,old_node_count=n,new_node_count=n+6,old_graph_preserved=True,old_network_initial_state_exact=True,random_initialization='none: constant constructor and no noise in episode'))
 manifest=c.build_motor_manifest(json.loads(c.POOLS.read_text()),ext['tier']['root_ids'],[r['actuator'] for r in json.loads((HISTORY.parent.parent/'s2-split-write/20261027T163000Z/MAPPING.json').read_text())['rows']])
 dump(O/'MAPPING.json',manifest);dump(O/'AFFECTED_9.json',[r for r in manifest['rows'] if any(x['actuator']==r['actuator'] and x['unavailable_in_tier'] for x in diff)])
 if not any(x['modelled_in'] for x in node):dump(O/'OFFLINE_STOP.json',dict(reason='no modelled incoming edges to added nodes; no body run'));return
 m,q,gate,joints,geomleg=c.prepare_body(np.load(INITIAL)['qpos'],body_dt_s=.0000125);c.require_six_foot_initial_contacts(m,q,geomleg);np.random.seed(0);obs=Observer(m,geomleg);d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);obs(m,d);ev={}
 result,tr,cmd,sp=c.episode(m,q,joints,geomleg,ext,manifest,groups,'normal',.25,stride=1,full_detail_intervals=0,push=None,external_touch=False,evidence=ev,body_observer=obs,body_dt_s=.0000125)
 obs.save(O/'BODY.npz');dump(O/'CONTACTS.json',obs.contacts);dump(O/'RECORDS.json',result);np.savez_compressed(O/'INTERVALS.npz',commands=cmd,spikes=sp,trajectory=tr,**{k:v for k,v in ev.items() if isinstance(v,np.ndarray)})
 dump(O/'EXECUTION.json',dict(command='OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl '+sys.executable+' '+str(Path(__file__)),exit_code=0,body_steps=20000,neural_steps=500,step_forwards=20000,model_load=1,static_tier_build=2,additional_initialization_forwards='prepare/contact gate plus observer and episode initial',default_regression=True))
 for p in (Path(__file__),P/'derive_required_motor_tier.py'):(O/p.name).write_bytes(p.read_bytes())
 assert all(sha(Path(k))==v for k,v in json.loads((O/'PREREGISTRATION.json').read_text())['source_hashes'].items());print('extended250ms exit0',flush=True)
if __name__=='__main__':main()
