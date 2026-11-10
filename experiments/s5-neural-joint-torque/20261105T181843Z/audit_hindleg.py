"""Read-only causal-chain audit of the latest actual P0 trajectory."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,hashlib,csv
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T181843Z')
SRC=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T153657Z')
MAP=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z/RUN/MAPPING.json')
LEGS=('LF','LM','LH','RF','RM','RH')
def js(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v))+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    O.mkdir(parents=True,exist_ok=False);best=js(SRC/'SUMMARY.json')['P0']['best'];run=Path(best['folder'])
    files=[SRC/'SUMMARY.json',SRC/'COVERAGE.json',SRC/'WARM_P0/NEURAL.npz',SRC/'WARM_P0/BODY.npz',SRC/'WARM_P0/NATIVE.npz',MAP,Path(best['matrix_file'])]+[run/n for n in ['NEURAL.npz','BODY.npz','NATIVE.npz']]+[Path(__file__)]
    before={str(p):sha(p) for p in files};dump(O/'AUDIT_PREREGISTRATION.json',dict(question='RH zero load: distinguish silent neurons, filtered-rate persistence, output cancellation/saturation and foot trajectory',source_hashes=before,windows=dict(precollapse=[0,.06],full=[0,.2]),importance='核心必需',units='NOT_ESTABLISHED'))
    cov=js(SRC/'COVERAGE.json');pools=cov['pools'];rows=js(MAP)['rows'];legs={tuple(r['local_indices']):r['leg'] for r in rows if r['local_indices']}
    S=np.array(js(best['matrix_file'])['matrix']);limits=np.array([j['limit'] for j in cov['joints']]);n=np.load(run/'NEURAL.npz');body=np.load(run/'BODY.npz');native=np.load(run/'NATIVE.npz')['rows'];wn=np.load(SRC/'WARM_P0/NEURAL.npz');wb=np.load(SRC/'WARM_P0/BODY.npz')
    out=[];poolrows=[]
    for leg in LEGS:
        cols=[i for i,p in enumerate(pools) if legs[tuple(p)]==leg];cells=sorted(set(i for c in cols for i in pools[c]));jidx=[i for i,j in enumerate(cov['joints']) if j['joint'].startswith('joint_'+leg)]
        counts=n['spikes'][:,cells].sum(0);first=n['spikes'][:60,cells].sum(0)
        group_rates=n['source_rates'][:,cols];applied=n['applied'][:,jidx];li=LEGS.index(leg)
        raw=limits[jidx][None,:]/100*(n['source_rates']@S[jidx].T)
        feet=body['feet'][:,li];force=native[:,1+li]
        out.append(dict(leg=leg,cells=cells,cell_count=len(cells),silent_cells=[cells[i] for i in np.flatnonzero(counts==0)],spikes_full=int(counts.sum()),spikes_first60ms=int(first.sum()),active_cells=int(np.count_nonzero(counts)),pool_count=len(cols),silent_pools=[cols[i] for i in np.flatnonzero(np.max(n['rates'][:,cols],axis=0)<1e-9)],mean_pool_rate_hz=float(group_rates.mean()),normalized_mean_abs_command=float(np.mean(abs(applied)/limits[jidx])),saturation_fraction=float(np.mean(abs(applied)>=limits[jidx])),max_raw_normalized=float(np.max(abs(raw)/limits[jidx])),foot_load_mean=float(force.mean()/24.446354755878005),foot_load_first60ms=float(force[:4801].mean()/24.446354755878005),contact_duty=float((force>1e-9).mean()),foot_origin_z_min=float(feet[:,2].min()),foot_origin_z_at_release=float(feet[0,2]),foot_origin_z_at_60ms=float(feet[120,2]),foot_origin_displacement_60ms=feet[120]-feet[0],warm_foot_z_min=float(wb['feet'][wb['time']>=1.1,li,2].min())))
        for p in cols:
            poolrows.append(dict(leg=leg,pool=p,cells=pools[p],spikes=int(n['spikes'][:,pools[p]].sum()),spikes_first60ms=int(n['spikes'][:60,pools[p]].sum()),warm_last200ms_spikes=int(wn['spikes'][wn['time']>=1.1-1e-9][:,pools[p]].sum()),rate_mean_hz=float(n['rates'][:,p].mean()),rate_at_release=float(n['source_rates'][0,p]),joints=[cov['joints'][j]['joint'] for j in np.flatnonzero(S[:,p])]))
    rhcols=[i for i,p in enumerate(pools) if legs[tuple(p)]=='RH'];jidx=[i for i,j in enumerate(cov['joints']) if j['joint'].startswith('joint_RH')]
    jointrows=[]
    for j in jidx:
        contributions=n['source_rates']*S[j][None,:]*limits[j]/100;positive=np.maximum(contributions,0).sum(1);negative=np.minimum(contributions,0).sum(1)
        jointrows.append(dict(joint=cov['joints'][j]['joint'],limit=float(limits[j]),positive_drive_mean=float(positive.mean()),negative_drive_mean=float(negative.mean()),applied_mean=float(n['applied'][:,j].mean()),applied_min=float(n['applied'][:,j].min()),applied_max=float(n['applied'][:,j].max()),saturation_fraction=float(np.mean(abs(n['applied'][:,j])>=limits[j])),nonzero_source_pools=int(np.count_nonzero(S[j])),active_positive_pools=[int(p) for p in np.flatnonzero((S[j]>0)&(n['rates'].max(0)>1e-9))],active_negative_pools=[int(p) for p in np.flatnonzero((S[j]<0)&(n['rates'].max(0)>1e-9))]))
    dump(O/'CHAIN_AUDIT.json',dict(per_leg=out,per_pool=poolrows,RH_joints=jointrows,baseline=best,interpretation_limits='Foot origin height is not sole-ground clearance. No force inference from height alone. Silent at this trajectory is not intrinsically silent.',units='NOT_ESTABLISHED'))
    with (O/'PER_POOL.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(poolrows[0]));writer.writeheader();writer.writerows(poolrows)
    after={str(p):sha(p) for p in files};assert before==after;dump(O/'AUDIT_INPUT_HASHES_AFTER.json',after)
    print(json.dumps(out[-1],ensure_ascii=False),flush=True)
    print(json.dumps(jointrows,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
