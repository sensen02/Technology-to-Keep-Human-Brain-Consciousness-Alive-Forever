"""Independent scalar CSV and per-node output audit; no new model/graph."""
from pathlib import Path
import sys,csv,json,math,hashlib,collections
import numpy as np
from scipy.sparse import load_npz
R=Path(__file__).resolve().parents[1];sys.path.append(str(R/'venv/lib/python3.12/site-packages'))
import pandas as pd
O=Path('/home/sensen/Desktop/engine修复验收/s3-banc888-nt-prediction-provenance/20261101T230101Z')
B=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
PURE=('acetylcholine','gaba','glutamate');NTS=('acetylcholine','dopamine','gaba','glutamate','histamine','octopamine','serotonin','tyramine')
def cls(s):return {'acetylcholine':'EXC','gaba':'INH','glutamate':'INH'}.get(s,'EXCLUDED')
def main():
    result=json.loads((O/'RESULT.json').read_text());official={};multiplicity=collections.Counter();conflicts=set();total=0;count_sums_checked=0
    def missing(s):return s in ('','NA','NaN','nan','NULL')
    def integer(s):
        if missing(s):return None
        value=float(s);assert math.isfinite(value) and value.is_integer();return int(value)
    with (O/'OFFICIAL_banc_888_neurotransmitter_prediction_v2.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            root=int(row['root_id']);payload=tuple(integer(row[k]) for k in NTS)+('' if missing(row['neurotransmitter_predicted']) else row['neurotransmitter_predicted'],None if missing(row['neurotransmitter_score']) else float(row['neurotransmitter_score']),integer(row['count']))
            if all(v is not None for v in payload[:8]) and payload[-1] is not None:
                assert sum(payload[:8])==payload[-1];count_sums_checked+=1
            total+=1;multiplicity[root]+=1
            if root in official and official[root]!=payload:conflicts.add(root)
            else:official[root]=payload
    for root in conflicts:official.pop(root,None)
    assert total==169635 and len(multiplicity)==168432 and len(conflicts)==0
    d=pd.read_feather(R/'data/banc888/banc_888_meta.feather');d.index=d['root_888'].astype('int64');assert d.index.is_unique
    data=np.load(R/'data/flywire/banc_connectome.npz');allroots=data['root_ids'];ids=np.load(B/'IDENTITIES.npz');roots=ids['root_ids'];global_index=ids['global_index'];current=np.load(B/'RUN/NEURAL.npz')['sensory_current_nA']
    We=load_npz(B/'ACTUAL_We.npz').tocsc();Wi=load_npz(B/'ACTUAL_Wi.npz').tocsc();degree=np.diff(We.indptr)+np.diff(Wi.indptr)
    recipients=list(csv.DictReader((O/'DRIVEN_RECIPIENTS.csv').open(newline='')));assert len(recipients)==len(set(np.nonzero(current)[1]))==3455
    mapping=json.loads((B/'RUN/MAPPING.json').read_text());readout={i for row in mapping['rows'] for i in row['local_indices']}
    inside=np.zeros(len(allroots),bool);inside[global_index]=True;pre=data['pre'];post=data['post'];syn=data['syn'];mask=inside[pre]&inside[post]
    rawrows=collections.Counter();rawsyn=collections.defaultdict(float)
    for i,n in zip(pre[mask],syn[mask]):rawrows[int(i)]+=1;rawsyn[int(i)]+=float(n)
    candidates={cut:[] for cut in (.5,.8,.9,.95)};conservative={cut:[] for cut in candidates};local_label_disagreements=[];score_parser_max_difference=0.
    stored=data['ann_nt_verified'].astype(str);local=d.loc[allroots,'neurotransmitter_verified'].fillna('').astype(str).to_numpy();affected_tier=0
    with (O/'VERIFIED_LABEL_DIFFERENCES.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['root_888','cache_label','local_label','cache_class','local_class','class_changed'])
        for root,a,b in zip(allroots,stored,local):
            if a!=b:writer.writerow([int(root),a,b,cls(a),cls(b),int(cls(a)!=cls(b))])
    for row in recipients:
        i=int(row['tier_index']);g=int(global_index[i]);root=int(row['root_888']);assert root==int(roots[i])
        assert int(row['modeled_outdegree'])==degree[i] and int(row['raw_tier_outdegree'])==rawrows[g]
        assert (i in readout)==bool(int(row['motor_readout'])) and row['cache_verified']==stored[g] and row['local_verified']==local[g]
        p=official.get(root);label='' if p is None else p[8];score=None if p is None else p[9];support=0 if p is None or p[10] is None else p[10]
        assert row['official_predicted']==label and int(row['official_synapse_count'])==support
        if score is not None:
            error=abs(float(row['official_score'])-score);score_parser_max_difference=max(score_parser_max_difference,error);assert error<=1e-12
        eligible=degree[i]==0 and i not in readout and stored[g]=='' and label in PURE and score is not None and support>=20
        if eligible:
            for cut in candidates:
                if score>=cut:
                    candidates[cut].append(g)
                    if local[g]=='':conservative[cut].append(g)
                    else:local_label_disagreements.append(root)
    for expected in result['prediction_candidates']:
        if expected['support_min']!=20:continue
        selected=candidates[expected['score_cut']]
        assert len(selected)==expected['nodes'] and sum(rawrows[g] for g in selected)==expected['raw_in_tier_rows'] and math.fsum(rawsyn[g] for g in selected)==expected['raw_in_tier_synapse_sum']
    conservative_stats=[]
    for cut,selected in conservative.items():conservative_stats.append(dict(score_cut=cut,support_min=20,cache_and_local_verified_both_empty=True,nodes=len(selected),raw_rows=sum(rawrows[g] for g in selected),raw_synapses=math.fsum(rawsyn[g] for g in selected)))
    c0=np.array([cls(s) for s in stored]);c1=np.array([cls(s) for s in local]);changes=c0!=c1
    for row in recipients:affected_tier+=int(changes[int(global_index[int(row['tier_index'])])])
    report=dict(passed=True,official_rows=total,unique_roots=len(multiplicity),duplicated_root_keys=sum(n>1 for n in multiplicity.values()),duplicate_extra_rows=total-len(multiplicity),conflicting_NT_payload_keys=len(conflicts),all8_counts_sum_to_total_count_checked_rows=count_sums_checked,verified_label_exact_differences=int((stored!=local).sum()),verified_receptor_class_differences=int(changes.sum()),verified_receptor_class_differences_tier=int(changes[global_index].sum()),verified_receptor_class_differences_driven=affected_tier,conservative_both_sources_empty_candidates=conservative_stats,prediction_candidate_local_verified_nonempty_roots=sorted(set(local_label_disagreements)),driven_nodes_checked=len(recipients),score_parser_max_difference=score_parser_max_difference,calls=dict(model_load=0,tier_build=0,neural_step=0,body_step=0,sign_fill=0),limits='Scalar source crosscheck and counts only; classifier probabilities not calibrated sign accuracy. No candidate graph built or runtime data modified.')
    for name,h in result['source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
    (O/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
