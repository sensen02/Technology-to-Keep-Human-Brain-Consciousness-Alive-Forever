"""Preregistered offline information diagnostic: 72 fixed ridge fits, zero simulation steps.

For each of 6 legs x 4 representations x 3 targets: ridge lambda=1 on train-only
standardized features with an unpenalized intercept (diagnostic model only), and
balanced class weights for classification. Train = generations 0,1 (24 episodes,
4800 bins); test = generations 2,3 (24 episodes, 4800 bins). No tuning, no test
selection, every comparison retained.

Recoverability from correlated saved data is not causal neural representation,
perception, or movement success. Nothing here is fitted for deployment.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
OLD=ROOT/'experiments/banc-local-learning/20261107T185910Z'
WEIGHT=24.446354755878005
STANCE_T=.05*WEIGHT
AIR_T=.005*WEIGHT
KNEE_SIGN_T=1e-3
LAGS=(0,1,4,10)
LAM=1.0
STD_FLOOR=1e-8
REPRESENTATIONS=('encoded4','MNinstant','MNhistory','rawphysical')
TASKS=('stance_air','knee_velocity_sign','signed_knee_velocity')


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else (v.tolist() if isinstance(v,np.ndarray) else str(v)))+'\n')


def leg_indices(d,leg):
    names=list(d['joint_names']);mn_leg=d['mn_leg']
    joints=[i for i,n in enumerate(names) if n.startswith('joint_'+leg)]
    knees=[i for i in joints if names[i]=='joint_'+leg+'Tibia_pitch']
    mn=np.flatnonzero(mn_leg==leg.lower())
    assert len(joints)==7 and len(knees)==1 and len(mn)>0,(leg,joints,knees,len(mn))
    return joints,knees[0],mn


def build_history(mn,episode,lag):
    """Lag-major history, zero-prefilled separately inside each episode, no future wrap."""
    out=np.zeros_like(mn)
    for episode_id in np.unique(episode):
        rows=np.flatnonzero(episode==episode_id)
        block=mn[rows]
        if lag==0:out[rows]=block
        else:
            out[rows[lag:]]=block[:-lag]
    return out


def labels(d,leg,foot_load_index,knee_index):
    load=d['physical'][:,84+foot_load_index];velocity=d['physical'][:,42+knee_index]
    stance=np.where(load>STANCE_T,1.0,np.where(load<AIR_T,-1.0,np.nan))
    sign=np.where(velocity>KNEE_SIGN_T,1.0,np.where(velocity<-KNEE_SIGN_T,-1.0,np.nan))
    return dict(stance_air=stance,knee_velocity_sign=sign,signed_knee_velocity=velocity)


def features(d,representation,leg_index,joints,knee_index,mn_index):
    if representation=='encoded4':return d['encoded'][:,leg_index,:].astype(float)
    if representation=='MNinstant':return d['mn_rates'][:,mn_index]/100.
    if representation=='MNhistory':return np.concatenate([build_history(d['mn_rates'][:,mn_index],d['episode_id'],lag)/100. for lag in LAGS],axis=1)
    if representation=='rawphysical':
        return np.concatenate([d['physical'][:,joints],d['physical'][:,42+np.array(joints)],d['physical'][:,84+leg_index][:,None]],axis=1)
    raise ValueError(representation)


def standardize(x_fit,x_parts):
    mean=x_fit.mean(axis=0);std=x_fit.std(axis=0)
    std=np.where(std<STD_FLOOR,STD_FLOOR,std)
    return mean,std,[(x-mean)/std for x in x_parts]


def fit_ridge(x,y,weights):
    """Weighted ridge with an UNPENALIZED intercept (diagnostic model only)."""
    total=weights.sum()
    xbar=(weights[:,None]*x).sum(axis=0)/total;ybar=(weights*y).sum()/total
    xc=x-xbar;yc=y-ybar
    lhs=xc.T@(weights[:,None]*xc)+LAM*np.eye(x.shape[1])
    rhs=xc.T@(weights*yc)
    beta=np.linalg.solve(lhs,rhs)
    return dict(beta=beta,intercept=float(ybar-float(xbar@beta)),lhs=lhs,rhs=rhs,xbar=xbar,ybar=float(ybar))


def class_counts(y,index):
    v=y[index];valid=~np.isnan(v)
    pos=int(np.sum(v[valid]>0));neg=int(np.sum(v[valid]<0))
    return valid,pos,neg


def balanced_accuracy(y_true,y_pred,valid):
    yt=y_true[valid];yp=y_pred[valid]
    recalls={}
    for cls in (1.0,-1.0):
        m=yt==cls
        if not np.any(m):return None,recalls
        recalls[cls]=float(np.mean(yp[m]==cls))
    return float(np.mean(list(recalls.values()))),recalls


def confusion(y_true,y_pred,valid):
    yt=y_true[valid];yp=y_pred[valid]
    return {str(int(a)):{str(int(b)):int(np.sum((yt==a)&(yp==b))) for b in (1,-1)} for a in (1,-1)}


def main():
    if (HERE/'FIT_RESULTS.json').exists():raise FileExistsError('Preserve prior diagnostic: '+str(HERE/'FIT_RESULTS.json'))
    data=load(HERE/'SAVED_DATA.npz');audit=json.loads((HERE/'ENCODER_AUDIT.json').read_text())
    source_hashes={str(p):sha(p) for p in (HERE/'PREREG.json',HERE/'audit_encoder.py',Path(__file__),HERE/'ENCODER_AUDIT.json',HERE/'SAVED_DATA.npz',OLD/'encoder/encoder.py')}
    generation=data['generation'];episode=data['episode_id'];mn=data['mn_rates']
    train_index=np.flatnonzero(generation<=1);test_index=np.flatnonzero(generation>=2)
    assert len(train_index)==4800 and len(test_index)==4800
    assert len(np.unique(episode[train_index]))==24 and len(np.unique(episode[test_index]))==24
    source_physical=data['oldphysical'];source_mn=data['oldmn']
    source_episode=np.zeros(len(source_physical),dtype=int)
    fits=[];arrays={};fit_count=0
    for leg_index,leg in enumerate(data['foot_order']):
        joints,knee_index,mn_index=leg_indices(data,leg)
        lab=labels(data,leg,leg_index,knee_index)
        source_lab=labels({'physical':source_physical},leg,leg_index,knee_index)
        for representation in REPRESENTATIONS:
            x=features(data,representation,leg_index,joints,knee_index,mn_index)
            if representation=='MNhistory':xsource=np.concatenate([build_history(source_mn[:,mn_index],source_episode,lag)/100. for lag in LAGS],axis=1)
            elif representation=='MNinstant':xsource=source_mn[:,mn_index]/100.
            elif representation=='encoded4':xsource=data['oldencoded'][:,leg_index,:].astype(float)
            else:xsource=np.concatenate([source_physical[:,joints],source_physical[:,42+np.array(joints)],source_physical[:,84+leg_index][:,None]],axis=1)
            mean,std,(xtrain_raw,xtest_raw,xsource_raw)=standardize(x[train_index],[x[train_index],x[test_index],xsource])
            for task in TASKS:
                y=lab[task];ysource=source_lab[task]
                classification=task!='signed_knee_velocity'
                train_valid,tr_pos,tr_neg=(class_counts(y,train_index) if classification else (np.ones(len(train_index),bool),0,0))
                test_valid,te_pos,te_neg=(class_counts(y,test_index) if classification else (np.ones(len(test_index),bool),0,0))
                src_valid=None
                src_pos=src_neg=None
                if classification:
                    src_valid,src_pos,src_neg=class_counts(ysource,np.arange(len(ysource)))
                record=dict(fit_id=f'{leg}_{representation}_{task}',leg=leg,leg_index=leg_index,representation=representation,task=task,
                            classification=classification,feature_dimension=int(x.shape[1]),lambda_ridge=LAM,
                            train_episodes=int(len(np.unique(episode[train_index]))),test_episodes=int(len(np.unique(episode[test_index]))),
                            train_bins=int(len(train_index)),test_bins=int(len(test_index)),
                            train_labelled=int(np.sum(train_valid)),test_labelled=int(np.sum(test_valid)),
                            train_excluded=int(len(train_index)-np.sum(train_valid)),test_excluded=int(len(test_index)-np.sum(test_valid)),
                            train_class_counts={'+1':tr_pos,'-1':tr_neg},test_class_counts={'+1':te_pos,'-1':te_neg},
                            old_source_bins=int(len(source_physical)),old_source_labelled=int(np.sum(src_valid)) if classification else int(len(source_physical)),
                            old_source_class_counts={'+1':src_pos,'-1':src_neg} if classification else None,
                            old_source_excluded=int(len(source_physical)-np.sum(src_valid)) if classification else 0,
                            tautological_note=('campaniform strain contains foot load by construction, so this target is a tautological availability/compression check, not learning' if task=='stance_air' and representation=='encoded4' else None),
                            deployable=bool(representation!='rawphysical'),
                            fitting_status='FITTED',metrics={})
                if classification and (tr_pos==0 or tr_neg==0):
                    record['fitting_status']='NOT_ESTIMABLE';record['not_estimable_reason']='train missing class'
                    fits.append(record);continue
                if classification:
                    # Intermediate-threshold rows are excluded from fitting: zero weight, finite placeholder target.
                    labelled=int(np.sum(train_valid));counts_train=np.array([tr_pos,tr_neg],float)
                    weights=np.zeros(len(train_index))
                    weights[train_valid]=np.where(y[train_index][train_valid]>0,labelled/(2.*counts_train[0]),labelled/(2.*counts_train[1]))
                    y_fit=np.where(train_valid,y[train_index],0.)
                    record['balance_weight_multipliers']={'+1':float(labelled/(2.*counts_train[0])),'-1':float(labelled/(2.*counts_train[1]))}
                else:
                    weights=np.ones(len(train_index));y_fit=y[train_index]
                model=fit_ridge(xtrain_raw,y_fit,weights)
                fit_count+=1
                yhat_train=model['intercept']+xtrain_raw@model['beta']
                yhat_test=model['intercept']+xtest_raw@model['beta']
                yhat_source=model['intercept']+xsource_raw@model['beta']
                arrays[record['fit_id']+'/beta']=model['beta'];arrays[record['fit_id']+'/intercept']=np.array(model['intercept'])
                arrays[record['fit_id']+'/mean']=mean;arrays[record['fit_id']+'/std']=std
                arrays[record['fit_id']+'/yhat_train']=yhat_train;arrays[record['fit_id']+'/yhat_test']=yhat_test
                arrays[record['fit_id']+'/yhat_source']=yhat_source
                arrays[record['fit_id']+'/lhs']=model['lhs'];arrays[record['fit_id']+'/rhs']=model['rhs']
                if classification:
                    pred_train=np.where(yhat_train>0,1.0,-1.0);pred_test=np.where(yhat_test>0,1.0,-1.0);pred_source=np.where(yhat_source>0,1.0,-1.0)
                    arrays[record['fit_id']+'/pred_source']=pred_source
                    ba_train,rec_train=balanced_accuracy(y[train_index],pred_train,train_valid)
                    ba_test,rec_test=balanced_accuracy(y[test_index],pred_test,test_valid)
                    ba_source,rec_source=(balanced_accuracy(ysource,pred_source,src_valid) if np.any(src_valid) else (None,{}))
                    record['metrics']=dict(balanced_accuracy_train=ba_train,balanced_accuracy_test=ba_test,
                                           balanced_accuracy_old_source_transfer=ba_source,
                                           per_class_recall_train={str(int(k)):v for k,v in rec_train.items()},
                                           per_class_recall_test={str(int(k)):v for k,v in rec_test.items()},
                                           per_class_recall_old_source_transfer={str(int(k)):v for k,v in rec_source.items()},
                                           confusion_train=confusion(y[train_index],pred_train,train_valid),
                                           confusion_test=confusion(y[test_index],pred_test,test_valid),
                                           confusion_old_source_transfer=(confusion(ysource,pred_source,src_valid) if np.any(src_valid) else None),
                                           accuracy_train=float(np.mean(pred_train[train_valid]==y[train_index][train_valid])),
                                           accuracy_test=float(np.mean(pred_test[test_valid]==y[test_index][test_valid])),
                                           accuracy_old_source_transfer=(float(np.mean(pred_source[src_valid]==ysource[src_valid])) if np.any(src_valid) else None),
                                           balanced_accuracy_undefined_reason=('test missing class: balanced accuracy undefined' if (te_pos==0 or te_neg==0) else None),
                                           source_transfer_note='old failed slow4 physical source with the new adapted-core MN rate source; external diagnostic only, no refit, not independent data and not a success label')
                else:
                    target_train=y[train_index];target_test=y[test_index]
                    mse_train=float(np.mean((yhat_train-target_train)**2));mse_test=float(np.mean((yhat_test-target_test)**2))
                    baseline=float(target_train.mean())
                    def r2(truth,pred):
                        var=float(np.var(truth))
                        return None if var==0 else float(1-np.sum((truth-pred)**2)/np.sum((truth-baseline)**2))
                    record['metrics']=dict(mse_train=mse_train,mse_test=mse_test,train_mean_baseline_mse_test=float(np.mean((target_test-baseline)**2)),
                                           train_mean_baseline_mse_train=float(np.mean((target_train-baseline)**2)),
                                           r2_train=r2(target_train,yhat_train),r2_test=r2(target_test,yhat_test),
                                           target_train_variance=float(np.var(target_train)),target_test_variance=float(np.var(target_test)),
                                           train_target_mean=baseline,
                                           mse_old_source_transfer=float(np.mean((yhat_source-ysource)**2)),
                                           old_source_target_mean=float(ysource.mean()),
                                           old_source_target_variance=float(np.var(ysource)),
                                           source_transfer_note='external diagnostic only, no refit')
                fits.append(record)
    assert fit_count<=72,fit_count
    np.savez_compressed(HERE/'FIT_MODELS.npz',**arrays)
    summary=dict(status='PASS_PREREGISTERED_72_FIT_DIAGNOSTIC',source_hashes=source_hashes,
                 fit_count=fit_count,maximum_allowed_fits=72,not_estimable=sum(1 for f in fits if f['fitting_status']!='FITTED'),
                 representations=list(REPRESENTATIONS),tasks=list(TASKS),lags_bins=list(LAGS),lambda_ridge=LAM,
                 train_generations=[0,1],test_generations=[2,3],train_episodes=24,test_episodes=24,
                 train_bins=4800,test_bins=4800,old_failed_source_bins=int(len(source_physical)),
                 label_thresholds=dict(stance_above_native=STANCE_T,air_below_native=AIR_T,knee_sign_rad_s=KNEE_SIGN_T,
                                       body_weight_native=WEIGHT,intermediate_excluded=True),
                 standardization='train-only per-feature mean/std over all 4800 train bins of this leg (no label filtering), std floor 1e-8',
                 intercept='unpenalized, diagnostic model only, never an actor bias',
                 class_weight_rule='N/(2*n_class) from train counts, N = number of labelled training rows; intermediate-threshold rows carry zero weight and are excluded from fitting as well as scoring',
                 excluded_trials=[0,13,26,39,52],trials_used=[1+13*g+c for g in range(4) for c in range(12)],
                 data_artifact_reads=2,physics_steps=0,neural_steps=0,training_updates=0,
                 fits=fits,
                 limits='Coefficient fits on raw scalar channel means are not cell tuning. The encoded representation is a deliberately coarse uncalibrated proxy; its information loss does not make source sensory biology false. rawphysical is privileged ground truth and never deployable. Train/test share one canonical checkpoint and deterministic policy variants, so all samples are correlated; results are recoverability from correlated saved data, not generalization across animals, not causal neural representation, perception, or movement success.')
    dump(HERE/'FIT_RESULTS.json',summary)
    print(json.dumps(dict(fit_count=fit_count,not_estimable=summary['not_estimable'],
        stance_air_encoded4_test_ba={f['leg']:f['metrics'].get('balanced_accuracy_test') for f in fits if f['task']=='stance_air' and f['representation']=='encoded4'},
        stance_air_rawphysical_test_ba={f['leg']:f['metrics'].get('balanced_accuracy_test') for f in fits if f['task']=='stance_air' and f['representation']=='rawphysical'},
        knee_sign_test_ba={f"{f['leg']}/{f['representation']}":f['metrics'].get('balanced_accuracy_test') for f in fits if f['task']=='knee_velocity_sign'}),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
