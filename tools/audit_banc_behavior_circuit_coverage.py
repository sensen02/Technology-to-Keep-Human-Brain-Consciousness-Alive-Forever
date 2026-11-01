"""Read-only BANC annotation/archived-runtime structural audit; never builds or steps a network."""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy import sparse
ROOT = Path(__file__).resolve().parents[1]
REPLAY = Path('/home/sensen/Desktop/engine修复验收/s4-neural-boundary-replay/20261101T185129Z')
NATIVE = Path('/home/sensen/Desktop/engine修复验收/s5-muscle-native-recording/20261101T191552Z')
HISTORY = Path('/home/sensen/Desktop/engine修复验收/s5-detailed-handover-measurement/20261028T121000Z')
FIELDS = ['root_id','root_888','root_region','region','super_class','cell_class','cell_sub_class','cell_type','nerve','side','cell_function','cell_function_detailed','neurotransmitter_verified']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def dump(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()),encoding='utf-8')
def table(p,fields,rows):
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def bfs(a,seeds,allowed=None):
    n=a.shape[0]; parent=np.full(n,-1,np.int64); q=collections.deque()
    for v in np.flatnonzero(seeds): parent[v]=v;q.append(int(v))
    while q:
        u=q.popleft()
        for v in a.indices[a.indptr[u]:a.indptr[u+1]]:
            if parent[v]<0 and (allowed is None or allowed[v]): parent[v]=u;q.append(int(v))
    return parent

def path(parent,end):
    r=[int(end)]
    while parent[r[-1]]!=r[-1]:r.append(int(parent[r[-1]]))
    return r[::-1]
def groups(d):
    eq=lambda c,v: np.asarray(d[c].isin(v))
    return {
      'central_complex_class':eq('cell_class',['central_complex_intrinsic_neuron','central_complex_input_neuron','central_complex_output_neuron']),
      'CX_root_region_candidate':eq('root_region',['ITO_midbrain_PB','ITO_midbrain_EB','ITO_midbrain_FB','ITO_midbrain_NO']),
      'visual_machinery_superclass':eq('super_class',['visual_projection','visual_centrifugal','optic_lobe_intrinsic']),
      'visual_function_annotated':np.asarray(d.cell_function.str.startswith('visual_')),
      'photoreceptor_class':eq('cell_class',['photoreceptor_neuron']),
      'descending':eq('super_class',['descending']),
      'ascending':eq('super_class',['ascending']),
      'sensory_ascending':eq('super_class',['sensory_ascending']),
      'sensory_descending':eq('super_class',['sensory_descending']),
      'VNC_region':eq('region',['ventral_nerve_cord']),
      'VNC_intrinsic':eq('super_class',['ventral_nerve_cord_intrinsic']),
      'motor':eq('super_class',['motor']),
      'leg_motor_class':eq('cell_class',['leg_motor_neuron']),
      'brain_sensory':eq('super_class',['sensory']) & eq('region',['optic_lobe','central_brain']),
      'brain_region':eq('region',['optic_lobe','central_brain']),
    }
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--arrow-site',type=Path,default=ROOT/'venv/lib/python3.12/site-packages');args=ap.parse_args();out=args.out
    assert out.is_dir() and not (out/'RESULT.json').exists()
    # Existing Arrow shared libraries only; explicit interpreter remains venv_body, no installation.
    sys.path.append(str(args.arrow_site))
    import pandas as pd, pyarrow
    feather=ROOT/'data/banc888/banc_888_meta.feather';cache=ROOT/'data/flywire/banc_connectome.npz'
    d=pd.read_feather(feather).fillna('');d=d[FIELDS].astype(str)
    assert not d.root_888.duplicated().any()
    raw=np.load(cache,allow_pickle=False);allroots=raw['root_ids'];ids=np.load(REPLAY/'IDENTITIES.npz');roots=ids['root_ids'];gi=ids['global_index'];assert np.array_equal(allroots[gi],roots)
    lookup=d.set_index('root_888',drop=False);full=lookup.loc[allroots.astype(str)].reset_index(drop=True);tier=lookup.loc[roots.astype(str)].reset_index(drop=True)
    identity=json.loads((REPLAY/'IDENTITY.json').read_text());assert identity['tier_fingerprint']=='5ab78f550d682e2849749587cb6f7551' and identity['weight_multiplier']==.2
    hashes={str(p):sha(p) for p in [feather,cache,REPLAY/'IDENTITY.json',REPLAY/'IDENTITIES.npz',REPLAY/'ACTUAL_We.npz',REPLAY/'ACTUAL_Wi.npz',REPLAY/'ROUTE_AUDIT.json',NATIVE/'NEURAL_IDENTITY.json',NATIVE/'IDENTITIES.npz',NATIVE/'ACTUAL_We.npz',NATIVE/'ACTUAL_Wi.npz',HISTORY/'MAPPING.json',HISTORY/'NEURAL.npz',ROOT/'outputs/neural_rebuild/sensory_channel_lookup.json',Path(__file__)]}
    sources=[ROOT/p for p in ['engine/embodied/adapters.py','engine/embodied/continuous_handover.py','engine/embodied/sensory_channels.py','engine/embodied/motor_output.py','engine/neural_cond.py','engine/neck_cut_data.py','run_neural_muscle_closure.py','run_physical_walk_prewarm.py','tools_export_sensory_channel_lookup.py']]
    hashes.update({str(p):sha(p) for p in sources})
    for p,h in identity['input_sha256'].items():
        if p in hashes: assert hashes[p]==h,(p,'historical source drift')
    ni=np.load(NATIVE/'IDENTITIES.npz');assert np.array_equal(ni['root_ids'],roots)
    We=sparse.load_npz(REPLAY/'ACTUAL_We.npz');Wi=sparse.load_npz(REPLAY/'ACTUAL_Wi.npz')
    for name,m in [('We',We),('Wi',Wi)]:
        other=sparse.load_npz(NATIVE/('ACTUAL_'+name+'.npz'));assert (m!=other).nnz==0
    pre=raw['pre'];post=raw['post'];syn=raw['syn'];local=np.full(len(allroots),-1,np.int64);local[gi]=np.arange(len(gi));inside=(local[pre]>=0)&(local[post]>=0)
    # Independently assemble the expected sparse arrays directly from saved source rows;
    # no selection, build_neural_tier, network construction or neural step occurs.
    lab=raw['ann_nt_verified'][pre]
    for labels,m in [(['acetylcholine'],We),(['gaba','glutamate'],Wi)]:
        take=inside & np.isin(lab,labels);expected=sparse.csr_matrix((np.minimum(syn[take].astype(float),8)*.0005,(local[post[take]],local[pre[take]])),shape=m.shape)*.2
        assert (expected!=m).nnz==0
    fg=groups(full);tg=groups(tier);dg=groups(d)
    coverage=[]
    for k in dg:coverage.append(dict(group=k,feather_all=int(dg[k].sum()),cache_all=int(fg[k].sum()),actual_tier=int(tg[k].sum()),evidence='candidate location only' if 'candidate' in k else 'literal annotation'))
    table(out/'GROUP_COVERAGE.csv',list(coverage[0]),coverage)
    hist=[]
    for col in FIELDS[2:]:
        vc=[collections.Counter(x[col]) for x in [d,full,tier]]
        for value in sorted(set().union(*vc)):hist.append(dict(field=col,label=value,feather_all=vc[0][value],cache_all=vc[1][value],actual_tier=vc[2][value]))
    table(out/'ANNOTATION_COUNTS.csv',list(hist[0]),hist)
    tier.insert(0,'local_index',np.arange(len(roots)));tier.insert(1,'global_index',gi)
    tier.to_csv(out/'TIER_ANNOTATIONS.csv',index=False)
    full.to_csv(out/'CACHE_ANNOTATIONS.csv',index=False)
    channels=json.loads((ROOT/'outputs/neural_rebuild/sensory_channel_lookup.json').read_text());assert channels['source']['sha256']==sha(feather)
    curr=np.load(HISTORY/'NEURAL.npz')['sensory_current_nA'];driven=np.any(curr!=0,axis=0);first=np.any(curr[:15]!=0,axis=0)
    channel_rows=[];enabled={'claw_position','hook_direction','hairplate_angle','campaniform_strain','bristle_touch'}
    for i,r in enumerate(roots):
        info=channels['leg_channel_by_root'].get(str(r))
        if info:channel_rows.append(dict(local_index=i,root_888=int(r),leg=info[0],channel=info[1],modelled=info[1] in enabled,observed_nonzero=bool(driven[i]),first15_nonzero=bool(first[i])))
    known=np.array([str(r) in channels['leg_channel_by_root'] for r in roots]);assert not np.any(driven&~known)
    assert not any(row['observed_nonzero'] and not row['modelled'] for row in channel_rows)
    table(out/'SENSORY_TARGETS.csv',list(channel_rows[0]),channel_rows)
    mapping=json.loads((HISTORY/'MAPPING.json').read_text());bindings=[];sets=set();mapped=0
    for row in mapping['rows']:
        ix=row['local_indices'];assert [int(roots[i]) for i in ix]==row['tier_root_ids']
        if ix:mapped+=1;sets.add(tuple(ix))
        bindings.append(dict(actuator_index=row['actuator_index'],actuator=row['actuator'],leg=row['leg'],muscle_target=row['muscle_target'],local_indices=';'.join(map(str,ix)),root_888=';'.join(map(str,row['tier_root_ids'])),status=row['status']))
    assert len(bindings)==90 and mapped==74 and len(sets)==66
    table(out/'MUSCLE_BINDINGS.csv',list(bindings[0]),bindings)
    # Adjacencies use rows=presynaptic, cols=postsynaptic; raw rows are unsigned synapse evidence.
    A=sparse.csr_matrix((np.ones(len(pre),bool),(pre,post)),shape=(len(allroots),len(allroots)))
    At=A[gi][:,gi];S=(We+Wi).astype(bool).T.tocsr()
    result_routes={};witness=[]
    for graph_name,a,g,rts in [('cache_raw',A,fg,allroots),('tier_raw',At,tg,roots),('tier_actual_signed_mask',S,tg,roots)]:
        results={}
        for src_name in ['brain_sensory','visual_machinery_superclass','central_complex_class']+(['observed_current'] if graph_name!='cache_raw' else []):
            seeds=driven if src_name=='observed_current' else g[src_name]
            brain_allowed=g['brain_region']|g['descending']|seeds
            p1=bfs(a,seeds,brain_allowed);dns=(p1>=0)&g['descending']
            p2=bfs(a,dns,g['VNC_region']|g['descending']|g['motor']|g['VNC_intrinsic']);vncs=(p2>=0)&g['VNC_intrinsic']
            p3=bfs(a,vncs,g['VNC_region']|g['motor']|g['VNC_intrinsic']);mn=(p3>=0)&g['motor']
            output=np.zeros(len(rts),bool)
            if graph_name!='cache_raw':output[ids['motor_indices']]=True
            results[src_name]=dict(source_count=int(seeds.sum()),reached_DN=int(dns.sum()),reached_VNC_intrinsic=int(vncs.sum()),reached_MN=int(mn.sum()),reached_actual_readout_MN=int((mn&output).sum()) if graph_name!='cache_raw' else None)
            for end in np.flatnonzero(mn):
                last=path(p3,end);mid=path(p2,last[0]);firstpath=path(p1,mid[0]);chain=firstpath+mid[1:]+last[1:]
                assert all(a[u,v] for u,v in zip(chain,chain[1:]))
                witness.append(dict(graph=graph_name,source_group=src_name,end_root_888=int(rts[end]),DN_root_888=int(rts[mid[0]]),VNC_root_888=int(rts[last[0]]),hops=len(chain)-1,root_path=';'.join(str(rts[i]) for i in chain)))
        # Unrestricted sensory reach is separate from the staged brain route.
        if graph_name!='cache_raw':
            reach=bfs(a,driven)>=0;results['unrestricted_observed_current']=dict(source_count=int(driven.sum()),reached_actual_readout_MN=int(reach[ids['motor_indices']].sum()))
        result_routes[graph_name]=results
    table(out/'PATH_WITNESSES.csv',list(witness[0]),witness)
    old=json.loads((REPLAY/'ROUTE_AUDIT.json').read_text());degree=np.diff(S.indptr);assert int((first&(degree==0)).sum())==old['driven_no_modelled_output']
    result=dict(scope='S3/S4 training preparation; static data only',rows=dict(feather=len(d),cache=len(full),tier=len(tier)),join_key='root_888 (not mutable root_id)',root_id_match=dict(cache=len(set(allroots.astype(str)) & set(d.root_id)),tier=len(set(roots.astype(str)) & set(d.root_id))),identity=dict(fingerprint=identity['tier_fingerprint'],weight_multiplier=.2,matrix_nnz=dict(We=We.nnz,Wi=Wi.nnz),actual_weights_exact_from_raw=True,native_identity_exact=True,source_hashes_match_replay=True),coverage=coverage,routes=result_routes,sensory=dict(lookup_tier=len(channel_rows),modelled_tier=sum(x['modelled'] for x in channel_rows),observed_nonzero=int(driven.sum()),first15_nonzero=int(first.sum()),channels=sorted(enabled),visual=False,maze_wall=False,heading=False),mapping=dict(actuators=90,mapped=74,unmapped=16,shared_bindings=66,readout_MN=len(ids['motor_indices'])),prior_reused=dict(file=str(REPLAY/'ROUTE_AUDIT.json'),driven_no_modelled_output=old['driven_no_modelled_output'],unknown_label_with_raw_edges=old['empty_label_no_output_with_raw_in_tier_edges']),calls=dict(tier_build=0,network_instances=0,neural_steps=0,body_steps=0,training=0),environment=dict(executable=sys.executable,arrow_site=str(args.arrow_site),pandas=pd.__version__,pyarrow=pyarrow.__version__),limitations=['Boolean topology only; ignores signs, weights, delays, state and dynamic feasibility.','Each route explicitly visits a DN, VNC intrinsic, then MN. Brain segment stays in brain/DN/source masks; later segments stay in VNC/DN/MN masks. Within segments arbitrary number of edges allowed; this is not functional turning-circuit identification.','CX root region is a candidate location label, not proof of arborization or turning function.','Feather includes non-neuronal and disconnected rows; denominators are kept separate.'])
    dump(out/'RESULT.json',result);dump(out/'INPUT_SOURCE_SHA256.json',hashes)
    dump(out/'SCHEMA.json',dict(fields=FIELDS,join='root_888',candidate_policy='CX root-region rows are candidate labels only; no steering cell identity inferred.',matrix_orientation='source rows to destination columns in reachability; saved We/Wi are destination rows/source columns'))
    assert hashes=={p:sha(p) for p in hashes}
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
