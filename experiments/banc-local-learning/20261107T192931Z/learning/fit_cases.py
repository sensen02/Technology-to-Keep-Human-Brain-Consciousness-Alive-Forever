"""Fixed preregistered readout fits from saved traces; no neural/physical runs."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import traceback
import numpy as np
from scipy.optimize import lsq_linear
from runtime import canonical_identity

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
OLD=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z/readout')
PHYSICAL=Path('/home/sensen/Desktop/engine修复验收/flybody-retarget/20261107T182940Z/teacherderived_slow4_physics')
CASES={'oldinstant':[0],'oldhistory':[0,1,4,10],'newinstant':[0],'newhistory':[0,1,4,10]}


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def dump(p,x):
    Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')

def causal_features(rates,lags):
    blocks=[]
    for lag in lags:
        block=np.zeros_like(rates,dtype=float)
        if lag==0:block[:]=rates
        else:block[lag:]=rates[:-lag]
        blocks.append(block/100.)
    return np.concatenate(blocks,axis=1)


def fit(case):
    out=HERE/'cases'/case
    if out.exists():raise FileExistsError('Preserve completed or failed case: '+str(out))
    out.mkdir(parents=True)
    start=time.monotonic()
    try:
        prereg=json.loads((HERE/'PREREG.json').read_text())
        assert prereg['primary_fixed_before_results'].startswith('newhistory')
        oldtrace=load(OLD/'TRAIN_TRACE.npz');oldmodel=load(OLD/'MODEL.npz')
        oldresult=json.loads((OLD/'RESULT.json').read_text())
        for name,expected in oldresult['files'].items():assert sha(OLD/name)==expected
        rates_path=OLD/'TRAIN_TRACE.npz' if case.startswith('old') else BASE/'dynamics/SOURCE_TRACE.npz'
        source=oldtrace if case.startswith('old') else load(rates_path)
        rates=np.asarray(source['mn_rates_hz'],float)
        assert rates.shape==(1817,323) and np.isfinite(rates).all() and np.all(rates>=0)
        assert np.array_equal(source['time_s'],oldtrace['time_s'])
        roots,local,legs,names=canonical_identity()
        for key,value in [('mn_root_ids',roots),('mn_local',local),('mn_leg',legs),('joint_names',names)]:assert np.array_equal(oldmodel[key],value)
        commands=load(PHYSICAL/'COMMANDS.npz')['rows']
        labels=np.stack([commands[k:k+5,1:43].mean(0) for k in range(0,1817*5,5)])
        assert np.array_equal(labels,oldtrace['target_rate'])
        times=oldtrace['time_s'];train=(times>=.2-1e-9)&(times<6.-1e-9);temporal=times>=6.-1e-9
        assert train.sum()==1160 and temporal.sum()==617
        assert np.array_equal(train,oldtrace['train_mask']) and np.array_equal(temporal,oldtrace['temporal_mask'])
        lags=CASES[case]
        features=causal_features(rates,lags)
        featurelegs=np.tile(legs,len(lags))
        weights=np.zeros((len(lags)*323,42));fitrecords=[]
        if case=='oldinstant':weights[:]=oldmodel['weights']
        for j,name in enumerate(names):
            idx=np.flatnonzero(featurelegs==name.removeprefix('joint_')[:2].lower())
            x=features[train][:,idx]
            y=np.arctanh(np.clip(labels[train,j]/30.,-.999,.999))
            if case!='oldinstant':
                design=np.vstack([x,np.eye(len(idx))]);target=np.r_[y,np.zeros(len(idx))]
                solve=lsq_linear(design,target,bounds=(-1.,1.),tol=1e-9,max_iter=100)
                if not solve.success:raise RuntimeError('Constrained fit unsuccessful for '+str(name)+': '+solve.message)
                weights[idx,j]=solve.x
                solveinfo=dict(status=int(solve.status),message=solve.message,iterations=int(solve.nit),reported_optimality=float(solve.optimality))
            else:solveinfo=dict(status='CACHED_NO_FIT',iterations=0)
            beta=weights[idx,j];res=x@beta-y;gradient=x.T@res+beta
            projected=float(np.max(np.abs(beta-np.clip(beta-gradient,-1,1))))
            fitrecords.append(dict(joint=str(name),eligible_parameters=len(idx),objective=.5*float(res@res+beta@beta),projected_KKT_residual=projected,**solveinfo))
        predict=30*np.tanh(features@weights)
        assert np.isfinite(predict).all() and np.all(np.abs(weights)<=1)
        assert max(item['projected_KKT_residual'] for item in fitrecords)<=1e-6
        input_paths=[HERE/'PREREG.json',HERE/'runtime.py',Path(__file__),OLD/'MODEL.npz',OLD/'TRAIN_TRACE.npz',OLD/'RESULT.json',PHYSICAL/'COMMANDS.npz']
        if case.startswith('new'):input_paths.extend([rates_path,BASE/'dynamics/module.py'])
        hashes={str(p):sha(p) for p in dict.fromkeys(input_paths)}
        np.savez_compressed(out/'MODEL.npz',weights=weights,mn_root_ids=roots,mn_local=local,mn_leg=legs,joint_names=names,lags_bins=np.array(lags,dtype=np.int64),feature_hz_scale=100.,rate_cap=30.,bias=np.zeros(42),bin_ms=5.,case=case,primary=case=='newhistory',ridge_lambda=1.,source_hashes_json=json.dumps(hashes),feature_order='lagmajor_then323canonicalMN')
        np.savez_compressed(out/'PREDICTIONS.npz',time_s=times,target_rate=labels,predicted_rate=predict,train_mask=train,temporal_mask=temporal)
        rms=lambda a:float(np.sqrt(np.mean(a*a)))
        metrics=dict(train_RMS_rad_s=rms(predict[train]-labels[train]),temporal_RMS_rad_s=rms(predict[temporal]-labels[temporal]),zero_train_RMS_rad_s=rms(labels[train]),zero_temporal_RMS_rad_s=rms(labels[temporal]))
        result=dict(status='FIXED_READOUT_CASE_FITTED_NOT_ACCEPTANCE',case=case,primary=case=='newhistory',lags_bins=lags,features=features.shape[1],samples=len(rates),train_samples=int(train.sum()),temporal_samples=int(temporal.sum()),fit_solves=0 if case=='oldinstant' else 42,eligible_weight_count=sum(item['eligible_parameters'] for item in fitrecords),fit_records=fitrecords,metrics=metrics,per_joint_temporal_RMS_rad_s=np.sqrt(np.mean((predict[temporal]-labels[temporal])**2,axis=0)),temporal_better_than_zero=metrics['temporal_RMS_rad_s']<metrics['zero_temporal_RMS_rad_s'],source_hashes=hashes,files={name:sha(out/name) for name in ('MODEL.npz','PREDICTIONS.npz')},new_neural_steps=0,new_physical_steps=0,source_quality='failed slow4 reference; sameepisode temporal only',wall_s=time.monotonic()-start,runtime=dict(executable=sys.executable,numpy=np.__version__))
        dump(out/'RESULT.json',result)
        print(json.dumps(dict(case=case,fit_solves=result['fit_solves'],metrics=metrics,max_KKT=max(x['projected_KKT_residual'] for x in fitrecords),wall_s=result['wall_s'])),flush=True)
    except Exception:
        dump(out/'FAILED.json',dict(case=case,error=traceback.format_exc(),wall_s=time.monotonic()-start,new_neural_steps=0,new_physical_steps=0))
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['old','new'],required=True)
    phase=parser.parse_args().phase
    for case in CASES:
        if case.startswith(phase):fit(case)

if __name__=='__main__':main()
