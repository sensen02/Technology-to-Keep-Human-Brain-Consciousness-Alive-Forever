"""Offline predictive baselines for BANC fly73; never a motor controller.
No SNN, connectome, physics, output decoder or runtime configuration changes.
"""
from pathlib import Path
import json,hashlib,csv,time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
S=Path('/home/sensen/Desktop/engine修复验收/s4-banc-individual-behavior/20261101T211209Z')
O=Path('/home/sensen/Desktop/engine修复验收/s4-banc-causal-baselines/20261101T212622Z')
LAMS=(.01,.1,1.,10.,100.)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item(),allow_nan=False)+'\n')
def metrics(y,p):
    p=np.clip(p,1e-12,1-1e-12);return dict(n=int(len(y)),NLL=float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),Brier=float(np.mean((p-y)**2)),accuracy=float(np.mean((p>=.5)==y)),predicted_right_mean=float(p.mean()),observed_right_mean=float(y.mean()))
def main():
    O.mkdir(parents=True,exist_ok=False);inputs=[Path(__file__),S/'TURN_EVENTS.csv',S/'AUDIT.json'];hashes={str(p):sha(p) for p in inputs}
    dump(O/'PREREGISTRATION.json',dict(source_sha256=hashes,scope='S4 statistical training/evaluation benchmarks only, not SNN or brain simulation',features=['intercept','previous right1','previous right2','previous right3','incoming arm1','incoming arm2','log previous completed interturn interval'],causality='Features at event i use only previous outcomes i-1..i-3, previous event times i-1/i-2 and known source arm; never target outcome, destination arm, current completed interval or future coordinates. Require3prior events.',models='Empirical training-only constant, Laplace1-smoothed one-step Markov, L2 logistic with unpenalized intercept',lambda_grid=list(LAMS),selection='Smallest validationNLL then larger lambda as deterministic tie-break; test metrics computed only after selection. All fit standardization only from train.',split='Reuse sealed .6/.2/.2 chronological split and official-bias label mask, further require3past choices. Teacher-forced observed history at evaluation; NOT autonomous generative or neural control.',uncertainty='Paired heldout NLL differences vs constant,1000 circular moving-block bootstraps, block16events,seed7311; fixed block size descriptive sensitivity, not evidence582events IID or independent animals.',limits='Only single-fly observational prediction. Aggregate test statistics previously audited. Strong correlation can reflect maze/task geometry, label semantics, movement dynamics or latent state, not a neural mechanism. No controller deployment, no SNN training.'))
    events=list(csv.DictReader((S/'TURN_EVENTS.csv').open()));n=len(events)
    yall=np.array([int(e['right']) for e in events]);times=np.array([float(e['time_float64_s']) for e in events]);splits=np.array([int(e['split']) for e in events]);include=np.array([e['included_in_upstream_rBias']=='1' for e in events]);arms=np.array([int(e['from_arm']) for e in events]);to=np.array([int(e['to_arm']) for e in events]);assert np.array_equal(arms[1:],to[:-1])
    idx=np.arange(3,n);assert np.all(include[idx]);y=yall[idx].astype(float);part=splits[idx]
    X=np.column_stack([np.ones(len(idx)),2*yall[idx-1]-1,2*yall[idx-2]-1,2*yall[idx-3]-1,(arms[idx]==1).astype(float),(arms[idx]==2).astype(float),np.log1p(times[idx-1]-times[idx-2])])
    train=part==0;valid=part==1;test=part==2
    mean=float(X[train,-1].mean());std=float(X[train,-1].std());assert std>0;X[:,-1]=(X[:,-1]-mean)/std
    p0=float(y[train].mean());constant=np.full(len(y),p0);markov=np.zeros(len(y));last=yall[idx-1]
    for previous in (0,1):
        selected=train&(last==previous);p=(y[selected].sum()+1)/(selected.sum()+2);markov[last==previous]=p
    fits=[];predictions={};weights={};started=time.time()
    for lam in LAMS:
        def fg(w):
            eta=X[train]@w;loss=float(np.mean(np.logaddexp(0,eta)-y[train]*eta)+lam*.5*np.dot(w[1:],w[1:])/train.sum())
            grad=X[train].T@(expit(eta)-y[train])/train.sum();grad[1:]+=lam*w[1:]/train.sum();return loss,grad
        w0=np.zeros(X.shape[1]);w0[0]=np.log(p0/(1-p0));fit=minimize(fg,w0,jac=True,method='L-BFGS-B',options=dict(maxiter=500,ftol=1e-12,gtol=1e-9));assert fit.success,fit.message
        w=fit.x;p=expit(X@w);predictions[str(lam)]=p;weights[str(lam)]=w
        fits.append(dict(lam=lam,optimizer_success=bool(fit.success),iterations=int(fit.nit),train=metrics(y[train],p[train]),validation=metrics(y[valid],p[valid]),max_abs_gradient=float(abs(fg(w)[1]).max())))
    best=min(fits,key=lambda r:(r['validation']['NLL'],-r['lam']));key=str(best['lam']);selected=predictions[key]
    summary={}
    for label,p in [('train_constant',constant),('previous_turn_markov',markov),('selected_logistic',selected)]:summary[label]={name:metrics(y[mask],p[mask]) for name,mask in [('train',train),('validation',valid),('test',test)]}
    p=np.clip(selected[test],1e-12,1-1e-12);q=constant[test];yy=y[test]
    difference=-(yy*np.log(p)+(1-yy)*np.log1p(-p))+(yy*np.log(q)+(1-yy)*np.log1p(-q))
    rng=np.random.default_rng(7311);nb=len(difference);boot=[]
    for _ in range(1000):
        starts=rng.integers(0,nb,size=(nb+15)//16);ix=np.concatenate([(s+np.arange(16))%nb for s in starts])[:nb];boot.append(float(difference[ix].mean()))
    np.savez_compressed(O/'FEATURES_AND_PREDICTIONS.npz',event_indices=idx,source_feature_last_event=idx-1,source_feature_last_time=times[idx-1],target_time=times[idx],X=X,y=y,split=part,constant=constant,markov=markov,selected_logistic=selected,selected_weights=weights[key],all_weights=np.stack([weights[str(l)] for l in LAMS]),all_predictions=np.stack([predictions[str(l)] for l in LAMS]),lambdas=LAMS,previous_interval_mean=mean,previous_interval_sd=std,bootstrap_means=boot)
    assert np.all(times[idx-1]<times[idx]);assert hashes=={str(p):sha(p) for p in inputs}
    dump(O/'RESULT.json',dict(source_sha256=hashes,eligible_events=len(idx),split_counts=dict(train=int(train.sum()),validation=int(valid.sum()),test=int(test.sum())),train_constant=p0,lambda_fits=fits,selected_lambda=best['lam'],metrics=summary,paired_test_NLL_difference_logistic_minus_constant=float(difference.mean()),moving_block_bootstrap_95_interval=np.quantile(boot,[.025,.975]),elapsed_seconds=time.time()-started,model_load=0,neural_step=0,physics_step=0,SNN_training=False,interpretation='Teacher-forced statistical predictor with past observed behavior; cannot generate gait, infer unique neuron dynamics or establish body support. Not a runtime policy.'))
    print(json.dumps(dict(metrics=summary,selected_lambda=best['lam'],difference=float(difference.mean()),interval=np.quantile(boot,[.025,.975]).tolist()),indent=2))
if __name__=='__main__':main()
