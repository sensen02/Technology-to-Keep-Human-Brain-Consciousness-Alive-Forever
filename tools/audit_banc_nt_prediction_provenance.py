"""Source audit only: verified NT labels vs official ML predictions.
No sign fill, new connectivity, neural/physical evaluation or runtime changes.
"""
from pathlib import Path
import sys,json,csv,hashlib,base64,urllib.request,argparse,io
import numpy as np
from scipy.sparse import load_npz
R=Path(__file__).resolve().parents[1];sys.path.append(str(R/'venv/lib/python3.12/site-packages'))
import pandas as pd
O=Path('/home/sensen/Desktop/engine修复验收/s3-banc888-nt-prediction-provenance/20261101T230101Z')
F=R/'data/banc888/banc_888_meta.feather';CACHE=R/'data/flywire/banc_connectome.npz'
B=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
OBJECT='compiled_data/banc_888/banc_888_neurotransmitter_prediction_v2.csv'
URL='https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/'+OBJECT+'?generation=1778713090033523'
PURE=('acetylcholine','gaba','glutamate');CUTS=(.5,.8,.9,.95)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()
def put_text(p,text):
    encoded=text.encode('utf8')
    if p.exists():assert p.read_bytes()==encoded,'Existing audit data differs: '+str(p)
    else:p.write_bytes(encoded)
def dump(p,v):put_text(p,json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def counts(x):
    labels,n=np.unique(np.asarray(x,dtype=str),return_counts=True);return {str(k):int(v) for k,v in zip(labels,n)}
def main():
    O.mkdir(parents=True,exist_ok=True)
    parser=argparse.ArgumentParser();parser.add_argument('--resume-schema',action='store_true');args=parser.parse_args()
    assert not (O/'RESULT.json').exists(), 'Never overwrite completed metrics'
    assert not (O/'PREREGISTRATION.json').exists() or args.resume_schema, 'Use explicit schema continuation after a stopped attempt'
    sources=[Path(__file__),F,CACHE,B/'IDENTITIES.npz',B/'ACTUAL_We.npz',B/'ACTUAL_Wi.npz',B/'RUN/NEURAL.npz',B/'RUN/MAPPING.json',R/'engine/neck_cut_data.py']
    hashes={str(p):sha(p) for p in sources}
    if not (O/'PREREGISTRATION.json').exists():
        dump(O/'PREREGISTRATION.json',dict(scope='S3 source provenance/census only; no repeated all-empty fill experiments',source_sha256=hashes,official_prediction=dict(url=URL,bytes=21107592,md5_base64='TrvR1uBdQZKtDG2ydzmo4w==',version='per-neuron v2 synapse-classifier aggregation, BANC888'),question='Do already available real prediction columns provide evidence different from historical hardcoded cholinergic fill, particularly for currently driven zero-output cells?',cuts=list(CUTS),limits='Prediction label is not measured receptor sign. Score cuts are descriptive, not validated error probabilities. Existing labels never overridden, default graph/model untouched.',comparisons='Check official CSV against local metadata by explicit root888; count exact verified-cache differences and pure-rule differences separately, and confidence/label coverage among actual driven nodes.'))
    else:
        prior=json.loads((O/'PREREGISTRATION.json').read_text())['source_sha256']
        for name,h in prior.items():
            if name!=str(Path(__file__)):assert sha(Path(name))==h
    schema_url='https://raw.githubusercontent.com/htem/BANC-project/e31a2e26b9937dca72e5ca1c1960df6454d76114/manuscript/print/dataverse/documentation/banc_888_neurotransmitter_prediction_v2.md'
    if (O/'OFFICIAL_SCHEMA.md').exists():schema=(O/'OFFICIAL_SCHEMA.md').read_bytes()
    else:
        with urllib.request.urlopen(urllib.request.Request(schema_url,headers={'User-Agent':'BANC-source-audit/1.0'}),timeout=60) as f:schema=f.read()
        (O/'OFFICIAL_SCHEMA.md').write_bytes(schema)
    assert 'Root ID of the presynaptic neuron at v888' in schema.decode()
    addendum=O/('SCHEMA_AND_DUPLICATE_ADDENDUM_'+sha(Path(__file__))[:12]+'.json');assert not addendum.exists()
    dump(addendum,dict(before_analysis=True,source_sha256=hashes,schema_url=schema_url,schema_sha256=sha(O/'OFFICIAL_SCHEMA.md'),identity='Official documented CSV root_id is v888, synonymous with banc_888_id; join ONLY against local root_888, not local current root_id',duplicate_projection='For duplicated root IDs require all8NT counts,prediction,score,total count exactly equal across rows; collapse identical NT payload only. Exclude every conflicting root entirely. Preserve original CSV/duplicate census; no keep-first over conflicting data.',additional_support_threshold='In addition to score cuts report count>=20 as in official usage; descriptive only. Use official individual predictions, never cell-type consensus fallback.',resume='Retain prior stopped source/preregistration and reuse byte-verified downloaded CSV, no duplicate download.'))
    hashes[str(O/'OFFICIAL_SCHEMA.md')]=sha(O/'OFFICIAL_SCHEMA.md');hashes[str(addendum)]=sha(addendum)
    path=O/'OFFICIAL_banc_888_neurotransmitter_prediction_v2.csv';md=hashlib.md5();n=0
    if path.exists():
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1048576),b''):md.update(block);n+=len(block)
    else:
        with urllib.request.urlopen(urllib.request.Request(URL,headers={'User-Agent':'BANC-source-audit/1.0'}),timeout=60) as f,path.open('xb') as out:
            for block in iter(lambda:f.read(1048576),b''):out.write(block);md.update(block);n+=len(block)
    assert n==21107592 and base64.b64encode(md.digest()).decode()=='TrvR1uBdQZKtDG2ydzmo4w=='
    official=pd.read_csv(path,dtype={'root_id':str,'supervoxel_id':str});print('OFFICIAL_COLUMNS',list(official.columns),flush=True)
    df=pd.read_feather(F);assert df['root_888'].is_unique;df.index=df['root_888'].astype('int64')
    data=np.load(CACHE);roots=data['root_ids'];meta=df.loc[roots]
    verified=meta['neurotransmitter_verified'].fillna('').astype(str).to_numpy();pred=meta['neurotransmitter_predicted'].fillna('').astype(str).to_numpy();score=meta['neurotransmitter_score'].to_numpy(float)
    assert 'root_id' in official.columns and 'at v888' in schema.decode()
    payload=['acetylcholine','dopamine','gaba','glutamate','histamine','octopamine','serotonin','tyramine','neurotransmitter_predicted','neurotransmitter_score','count']
    grouped=official.groupby('root_id',sort=False);unique_values=grouped[payload].nunique(dropna=False)
    conflicting=unique_values.index[(unique_values>1).any(axis=1)]
    multiplicity=grouped.size();duplicates=multiplicity[multiplicity>1]
    duplicate_report=dict(raw_rows=len(official),unique_root_ids=int(official['root_id'].nunique()),duplicate_root_ids=len(duplicates),duplicate_extra_rows=int((duplicates-1).sum()),conflicting_NT_payload_root_ids=len(conflicting),excluded_conflicting_roots=list(map(str,conflicting)),projection='Identical NT payload per root only, retaining raw source; all conflicting roots excluded')
    dump(O/'DUPLICATE_PROJECTION.json',duplicate_report)
    put_text(O/'DUPLICATE_ROOTS.csv',pd.DataFrame({'root_id':duplicates.index,'rows':duplicates.to_numpy(),'NT_payload_conflict':duplicates.index.isin(conflicting)}).to_csv(index=False))
    official=official.loc[~official['root_id'].isin(conflicting)].drop_duplicates(subset=['root_id'])
    official.index=official['root_id'].astype('int64');assert official.index.is_unique
    assert 'neurotransmitter_predicted' in official.columns and 'neurotransmitter_score' in official.columns
    shared=df.index.intersection(official.index);a=df.loc[shared];b=official.loc[shared]
    eq=(a['neurotransmitter_predicted'].fillna('').astype(str).to_numpy()==b['neurotransmitter_predicted'].fillna('').astype(str).to_numpy())
    delta=abs(a['neurotransmitter_score'].to_numpy(float)-b['neurotransmitter_score'].to_numpy(float));finite=np.isfinite(delta)
    compare=dict(local_rows=len(df),official_rows=len(official),shared_root888=len(shared),predicted_label_matches=int(eq.sum()),predicted_label_differences=int((~eq).sum()),score_finite_pairs=int(finite.sum()),score_max_abs_difference=float(delta[finite].max()),score_pairs_differ_gt_point000051=int((delta[finite]>.000051).sum()))
    aligned=official.reindex(roots)
    official_pred=aligned['neurotransmitter_predicted'].fillna('').astype(str).to_numpy();official_score=aligned['neurotransmitter_score'].to_numpy(float);official_count=aligned['count'].fillna(0).to_numpy(float)
    compare['cache_roots_with_official_row']=int(np.isin(roots,official.index.to_numpy()).sum())
    ids=np.load(B/'IDENTITIES.npz');globalidx=ids['global_index'];tier=ids['root_ids'];assert np.array_equal(roots[globalidx],tier)
    We=load_npz(B/'ACTUAL_We.npz');Wi=load_npz(B/'ACTUAL_Wi.npz');outdegree=np.asarray(We.getnnz(axis=0)+Wi.getnnz(axis=0)).ravel()
    drive=np.any(np.load(B/'RUN/NEURAL.npz')['sensory_current_nA']!=0,axis=0)
    mapping=json.loads((B/'RUN/MAPPING.json').read_text());motor=np.zeros(len(tier),bool);motor[list({i for row in mapping['rows'] for i in row['local_indices']})]=True
    inside=np.zeros(len(roots),bool);inside[globalidx]=True;pre=data['pre'];post=data['post'];syn=data['syn'];rowinside=inside[pre]&inside[post]
    global_to_local=np.full(len(roots),-1);global_to_local[globalidx]=np.arange(len(tier));rawdegree=np.bincount(global_to_local[pre[rowinside]],minlength=len(tier))
    stored=data['ann_nt_verified'].astype(str);s=stored[globalidx];pv=official_pred[globalidx];sc=official_score[globalidx];support=official_count[globalidx];dead=drive&(outdegree==0)&~motor
    constraints=[]
    for cut in CUTS:
        for support_min in (0,20):
            eligible=dead&(s=='')&np.isin(pv,PURE)&np.isfinite(sc)&(sc>=cut)&(support>=support_min)
            global_eligible=np.zeros(len(roots),bool);global_eligible[globalidx[eligible]]=True;rows=rowinside&global_eligible[pre]
            constraints.append(dict(score_cut=cut,support_min=support_min,nodes=int(eligible.sum()),predicted_labels=counts(pv[eligible]),raw_in_tier_rows=int(rows.sum()),raw_in_tier_synapse_sum=float(syn[rows].sum()),existing_signed_labels_overridden=0))
    with io.StringIO(newline='') as f:
        writer=csv.writer(f);writer.writerow(['root_888','tier_index','cache_verified','local_verified','official_predicted','official_score','official_synapse_count','local_predicted','modeled_outdegree','raw_tier_outdegree','motor_readout','modeled_zero_output'])
        for i in np.flatnonzero(drive):writer.writerow([int(tier[i]),int(i),str(s[i]),str(verified[globalidx[i]]),str(pv[i]),float(sc[i]) if np.isfinite(sc[i]) else '',int(support[i]),str(pred[globalidx[i]]),int(outdegree[i]),int(rawdegree[i]),int(motor[i]),int(dead[i])])
        put_text(O/'DRIVEN_RECIPIENTS.csv',f.getvalue())
    pureboth=np.isin(verified,PURE)&np.isin(official_pred,PURE);high=pureboth&np.isfinite(official_score)&(official_score>=.9)&(official_count>=20)
    result=dict(
        source_sha256=hashes,
        official_download=dict(url=URL,bytes=n,md5=md.hexdigest(),sha256=sha(path)),
        official_vs_local=compare,duplicate_projection=duplicate_report,
        cache_vs_local_verified=dict(
            exact_differences=int((stored!=verified).sum()),
            modeled_pure_class_rule_differences=int((np.isin(stored,PURE)!=np.isin(verified,PURE)).sum())),
        cache_verified_labels=counts(stored),local_prediction_labels_cache=counts(pred),
        verified_vs_predicted_overlap=dict(
            pure_both=int(pureboth.sum()),pure_agreements=int((verified[pureboth]==official_pred[pureboth]).sum()),
            score_ge_point9_support20_pure_both=int(high.sum()),score_ge_point9_support20_agreements=int((verified[high]==official_pred[high]).sum())),
        actual_tier=dict(
            nodes=len(tier),modeled_nonzero_input_nodes=int(drive.sum()),
            modeled_zero_outdegree_driven_nonreadout=int(dead.sum()),
            zero_outdegree_driven_readout_overlap=int((drive&(outdegree==0)&motor).sum()),
            dead_with_raw_outgoing=int((dead&(rawdegree>0)).sum())),
        prediction_candidates=constraints,
        source_unchanged=all(sha(Path(name))==h for name,h in hashes.items()),
        calls=dict(model_load=0,tier_build=0,neural_step=0,body_step=0,sign_fill=0,training=0),
        interpretation='Evidence availability only. Existing all_empty implementation assigns constant ACh; it does not read this prediction table. This audit does not create or admit a new E/I graph. Any future graph must preserve known labels, cite predictor/version/confidence and receptor assumptions, and be validated as a distinct model.')
    assert result['source_unchanged'];dump(O/'RESULT.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','cache_verified_labels','local_prediction_labels_cache')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
