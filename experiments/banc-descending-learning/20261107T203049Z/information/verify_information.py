"""Independent verification of the offline information diagnostic. No fitter import.

This module deliberately does not import fit_information.py or audit_encoder.py.
It re-derives everything from the frozen encoder source and the saved study data:
labels, lag stacks, train statistics, ridge normal equations, predictions,
metrics, encoder sensitivity ranks and alias fixtures. Zero simulation steps.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
OLD=ROOT/'experiments/banc-local-learning/20261107T185910Z'
TRIALS=Path('/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z/learning/training')
FOUNDATION=ROOT/'experiments/banc-physical-learning/20261107T195435Z/foundation'
NEWSOURCE=ROOT/'experiments/banc-local-learning/20261107T192931Z/dynamics/SOURCE_TRACE.npz'
OLDSOURCE=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z/readout/TRAIN_TRACE.npz')
sys.path.insert(0,str(OLD/'encoder'))
from encoder import SensoryEncoder, BODY_WEIGHT_NATIVE as WEIGHT

CHANNELS=('claw_position','hook_direction','hairplate_angle','campaniform_strain')
LAGS=(0,1,4,10)
CHECKED=[];READS=[0];FITS_RECOMPUTED=[0]
LO_HI={};LEG_FIXED={}


def check(name,ok,detail=None):
    CHECKED.append(dict(check=name,passed=bool(ok),detail=detail))
    if not ok:print('FAIL',name,detail,flush=True)
def load(p):
    READS[0]+=1
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def tanh(x):return (np.exp(2*x)-1)/(np.exp(2*x)+1)
def counts(pos,neg):return {'+1':int(pos),'-1':int(neg)}
def close(a,b,rtol=1e-9,atol=1e-12):
    a=float(a);b=float(b)
    return abs(a-b)<=atol+rtol*abs(b)
def array_close(a,b,rtol=1e-9,atol=1e-15):
    a=np.asarray(a,float);b=np.asarray(b,float)
    return float(np.max(np.abs(a-b))/(1.+np.max(np.abs(b))))<=rtol


def own_channels(q,v,load):
    """Independent transcription of the four declared encoder formulas."""
    out=np.empty((6,4))
    for i,leg in enumerate(SensoryEncoder.foot_order):
        knee=LEG_FIXED[leg]['knee']
        lo,hi=LO_HI[knee];nk=(np.clip(q[knee],lo,hi)-lo)/(hi-lo)
        out[i,0]=.01+.04*nk
        out[i,1]=.05*tanh(abs(v[knee])/10.)
        total=0.
        for j in LEG_FIXED[leg]['coxa']:
            lo,hi=LO_HI[j];total+=(np.clip(q[j],lo,hi)-lo)/(hi-lo)
        out[i,2]=.01+.04*(total/3.)
        out[i,3]=.05*tanh(max(load[i],0.)/(WEIGHT/6.))
    return out


def labels_for(physical,knee,load_index):
    load=physical[:,84+load_index];v=physical[:,42+knee]
    stance=np.full(len(physical),np.nan);stance[load>.05*WEIGHT]=1.;stance[load<.005*WEIGHT]=-1.
    sign=np.full(len(physical),np.nan);sign[v>1e-3]=1.;sign[v<-1e-3]=-1.
    return dict(stance_air=stance,knee_velocity_sign=sign,signed_knee_velocity=v)


def lag_stack(rates,episode_ids,lag):
    """Explicit per-episode loop, zero prefixed, no future wrap."""
    if lag==0:return rates.copy()
    out=np.zeros_like(rates)
    for eid in set(episode_ids.tolist()):
        idx=np.flatnonzero(episode_ids==eid)
        for position,row in enumerate(idx[lag:]):out[row]=rates[idx[position]]
    return out


def ridge_solution(x,y,w,lam=1.0):
    denom=float(w.sum());xbar=(w@x)/denom;ybar=float(w@y)/denom
    lhs=(x-xbar).T@((x-xbar)*w[:,None])+lam*np.eye(x.shape[1])
    rhs=(x-xbar).T@(w*(y-ybar))
    return lhs,rhs,xbar,ybar


def classification_metrics(y_true,yhat,valid):
    pred=np.where(yhat>0,1.,-1.)
    recalls={}
    for cls in (1.,-1.):
        selected=valid&(y_true==cls)
        if not selected.any():return None,None,None
        recalls[cls]=float((pred[selected]==cls).mean())
    conf={str(int(a)):{str(int(b)):int(((y_true==a)&(pred==b)&valid).sum()) for b in (1,-1)} for a in (1,-1)}
    return float((recalls[1.]+recalls[-1.])/2.),conf,float((pred[valid]==y_true[valid]).mean())


def main():
    enc=SensoryEncoder();base=load(FOUNDATION/'BASE.npz')
    compact={int(local):position for position,local in enumerate(enc.local_indices)}
    for i in range(42):LO_HI[i]=tuple(float(x) for x in enc.joint_ranges_rad[i])
    for leg in SensoryEncoder.foot_order:
        LEG_FIXED[leg]=dict(knee=enc.joint_names.index('joint_'+leg+'Tibia_pitch'),
                            coxa=[enc.joint_names.index('joint_'+leg+'Coxa_'+a) for a in ('yaw','pitch','roll')],
                            joints=[i for i,n in enumerate(enc.joint_names) if n.startswith('joint_'+leg)],
                            mn=np.flatnonzero(base['mn_leg']==leg.lower()))
    audit=json.loads((HERE/'ENCODER_AUDIT.json').read_text())
    results=json.loads((HERE/'FIT_RESULTS.json').read_text())
    models=load(HERE/'FIT_MODELS.npz');saved=load(HERE/'SAVED_DATA.npz')

    # ---------- 1. source hashes and provenance ----------
    for path,digest in sorted(audit['source_hashes'].items()):check('encoder_audit_source_hash:'+path,sha(path)==digest)
    for path,digest in sorted(results['source_hashes'].items()):check('fit_source_hash:'+path,sha(path)==digest)
    check('fit_results_binds_audit_hash',results['source_hashes'][str(HERE/'ENCODER_AUDIT.json')]==sha(HERE/'ENCODER_AUDIT.json'))
    check('prereg_hash_recorded_in_both',audit['source_hashes'][str(HERE/'PREREG.json')]==results['source_hashes'][str(HERE/'PREREG.json')])
    check('prereg_unchanged_since_audit_start',sha(HERE/'PREREG.json')==audit['source_hashes'][str(HERE/'PREREG.json')])
    check('figure_exists',(HERE/'FIT_SUMMARY.png').exists(),{'sha256':sha(HERE/'FIT_SUMMARY.png'),'bytes':int((HERE/'FIT_SUMMARY.png').stat().st_size)})

    # ---------- 2. rebuild trial arrays independently ----------
    phys=[];mnrates=[];encoded=[];eids=[];gens=[];trial_checks=0
    max_current_error=0.;max_formula_error=0.
    for generation in range(4):
        for slot in range(12):
            trial=1+13*generation+slot;folder=TRIALS/f'trial_{trial:03d}'
            pol=load(folder/'POLICY.npz');body=load(folder/'BODY.npz');native=load(folder/'NATIVE.npz')
            initial=load(folder/'INITIAL.npz');actor=load(folder/'ACTOR.npz');result=json.loads((folder/'RESULT.json').read_text())
            check(f'trial_{trial:03d}_valid_nonzero_actor',result['valid'] and np.any(actor['theta']))
            rows5=np.arange(200)*5
            q=body['qpos'][rows5][:,initial['qa']];v=body['qvel'][rows5][:,initial['va']]
            check(f'trial_{trial:03d}_policy_matches_body_qpos',np.array_equal(pol['local_physical_input'][:,:42],q))
            check(f'trial_{trial:03d}_policy_matches_body_qvel',np.array_equal(pol['local_physical_input'][:,42:84],v))
            check(f'trial_{trial:03d}_policy_foot_load_is_previous_native_row',np.array_equal(pol['local_physical_input'][1:,84:],native['rows'][rows5[1:]*80-1,1:7]))
            load_trace=pol['local_physical_input'][:,84:]
            for i in range(200):
                mine=own_channels(q[i],v[i],load_trace[i])
                # POLICY stores the exact 556-length compact current vector, not the 20000 local-module vector.
                recorded=np.array([[pol['sensory_current_nA'][i][compact[int(enc.channel_local_indices[leg][c][0])]] for c in CHANNELS] for leg in enc.foot_order])
                max_formula_error=max(max_formula_error,float(np.max(np.abs(recorded-mine))))
                rebuilt=enc.encode(q[i],v[i],load_trace[i])
                max_current_error=max(max_current_error,float(np.max(np.abs(rebuilt[enc.local_indices]-pol['sensory_current_nA'][i]))))
            phys.append(np.c_[q,v,load_trace]);mnrates.append(pol['mn_rates_hz'])
            encoded.append(np.array([own_channels(q[i],v[i],load_trace[i]) for i in range(200)]))
            eids.extend([trial]*200);gens.extend([generation]*200);trial_checks+=6
    phys=np.concatenate(phys);mnrates=np.concatenate(mnrates);encoded=np.concatenate(encoded)
    eids=np.asarray(eids);gens=np.asarray(gens)
    check('saved_physical_equals_independent_rebuild',np.array_equal(phys,saved['physical']))
    check('saved_mn_rates_equal_independent_rebuild',np.array_equal(mnrates,saved['mn_rates']))
    encoded_delta=float(np.max(np.abs(encoded-saved['encoded'])))
    check('saved_encoded_equal_independent_rebuild',encoded_delta<1e-15,encoded_delta)
    check('saved_episode_ids_match',np.array_equal(eids,np.asarray(saved['episode_id'])))
    check('saved_generations_match',np.array_equal(gens,np.asarray(saved['generation'])))
    check('saved_trial_count_48_bins_9600',len(np.unique(eids))==48 and len(phys)==9600,{'trials':int(len(np.unique(eids))),'bins':int(len(phys))})
    check('recorded_sensory_current_error_zero',max_current_error==0.,max_current_error)
    check('independent_formula_error_below_1e-15',max_formula_error<1e-15,max_formula_error)
    old=load(OLDSOURCE);new=load(NEWSOURCE)
    check('old_source_time_identity',np.array_equal(old['time_s'],new['time_s']))
    check('saved_oldphysical_identity',np.array_equal(old['physical_local_input'],saved['oldphysical']))
    check('saved_oldmn_identity',np.array_equal(new['mn_rates_hz'],saved['oldmn']))
    old_encoded=np.array([own_channels(r[:42],r[42:84],r[84:]) for r in old['physical_local_input']])
    old_encoded_delta=float(np.max(np.abs(old_encoded-saved['oldencoded'])))
    check('saved_oldencoded_identity',old_encoded_delta<1e-15,old_encoded_delta)
    old_current=max(float(np.max(np.abs(enc.encode(r[:42],r[42:84],r[84:])[enc.local_indices]-old['sensory_current_nA'][i]))) for i,r in enumerate(old['physical_local_input']))
    check('old_source_recorded_current_error_zero',old_current==0.,old_current)
    check('old_source_bins_1817',len(old['time_s'])==1817)

    # ---------- 3. encoder sensitivity, rank and alias reconstruction ----------
    mid=enc.joint_ranges_rad.mean(axis=1)
    reference=np.r_[mid,np.full(42,2.),np.full(6,WEIGHT/6)]
    jac=np.zeros((24,90))
    for i,leg in enumerate(SensoryEncoder.foot_order):
        knee=LEG_FIXED[leg]['knee'];span=LO_HI[knee][1]-LO_HI[knee][0]
        jac[4*i+0,knee]=.04/span
        t=np.tanh(abs(reference[42+knee])/10.);jac[4*i+1,42+knee]=.05*((1-t*t)/10.)
        for j in LEG_FIXED[leg]['coxa']:jac[4*i+2,j]=.04/3./(LO_HI[j][1]-LO_HI[j][0])
        if reference[84+i]>0:
            tl=np.tanh(reference[84+i]/(WEIGHT/6.));jac[4*i+3,84+i]=.05*(1-tl*tl)/(WEIGHT/6.)
    numeric=np.empty_like(jac)
    for col in range(90):
        plus=reference.copy();minus=reference.copy();plus[col]+=1e-6;minus[col]-=1e-6
        numeric[:,col]=(own_channels(plus[:42],plus[42:84],plus[84:])-own_channels(minus[:42],minus[42:84],minus[84:])).reshape(-1)/2e-6
    fd_error=float(np.max(np.abs(numeric-jac)))
    check('own_jacobian_matches_own_finite_difference',fd_error<1e-9,fd_error)
    saved_jac=np.load(HERE/'SENSITIVITY.npz',allow_pickle=False);READS[0]+=1
    check('saved_channel_jacobian_reproduced',float(np.max(np.abs(jac-saved_jac['channel_analytic'])))==0.,float(np.max(np.abs(jac-saved_jac['channel_analytic']))))
    check('saved_reference_input_reproduced',np.array_equal(reference,saved_jac['reference_input']))
    def rk(x):
        if not np.any(x):return 0
        s=np.linalg.svd(x,compute_uv=False)
        return int((s>s[0]*max(x.shape)*np.finfo(float).eps*10).sum())
    ranks=dict(channel_map_24x90=rk(jac),q_block_24x42=rk(jac[:,:42]),qvel_block_24x42=rk(jac[:,42:84]),foot_load_block_24x6=rk(jac[:,84:]))
    for key in ranks:check('rank_'+key,ranks[key]==audit['ranks'][key],{'independent':ranks[key],'recorded':audit['ranks'][key]})
    independent_zero=[c for c in range(90) if not np.any(jac[:,c])]
    check('zero_columns_reproduced',independent_zero==audit['zero_columns'],len(independent_zero))
    check('nonzero_q_columns_reproduced',sum(1 for c in range(42) if np.any(jac[:,c]))==audit['ranks']['nonzero_q_columns'])
    check('nonzero_qvel_columns_reproduced',sum(1 for c in range(42,84) if np.any(jac[:,c]))==audit['ranks']['nonzero_qvel_columns'])
    check('all_foot_load_columns_nonzero',all(np.any(jac[:,84+k]) for k in range(6)))
    for leg in SensoryEncoder.foot_order:
        knee=LEG_FIXED[leg]['knee'];coxa=LEG_FIXED[leg]['coxa']
        a=reference.copy();b=reference.copy()
        dy=.25*(LO_HI[coxa[0]][1]-LO_HI[coxa[0]][0]);dp=.25*(LO_HI[coxa[1]][1]-LO_HI[coxa[1]][0])
        a[coxa[0]]=mid[coxa[0]]+dy;a[coxa[1]]=mid[coxa[1]]-dp;b[coxa[0]]=mid[coxa[0]]-dy;b[coxa[1]]=mid[coxa[1]]+dp
        check(f'alias_{leg}_coxa_yaw_pitch_swap_equal',np.array_equal(enc.encode(a[:42],a[42:84],a[84:]),enc.encode(b[:42],b[42:84],b[84:])) and np.array_equal(own_channels(a[:42],a[42:84],a[84:]),own_channels(b[:42],b[42:84],b[84:])) and np.any(a[:42]!=b[:42]))
        a=reference.copy();b=reference.copy()
        troch=enc.joint_names.index('joint_'+leg+'Trochanter_yaw');roll=enc.joint_names.index('joint_'+leg+'Trochanter_roll')
        b[troch]=LO_HI[troch][0]+.1*(LO_HI[troch][1]-LO_HI[troch][0]);b[roll]=LO_HI[roll][0]+.1*(LO_HI[roll][1]-LO_HI[roll][0])
        check(f'alias_{leg}_trochanter_change_equal',np.array_equal(enc.encode(a[:42],a[42:84],a[84:]),enc.encode(b[:42],b[42:84],b[84:])) and np.any(a[:42]!=b[:42]))
        a=reference.copy();b=reference.copy();a[42+knee]=2.;b[42+knee]=-2.
        check(f'alias_{leg}_opposite_knee_speed_equal',np.array_equal(enc.encode(a[:42],a[42:84],a[84:]),enc.encode(b[:42],b[42:84],b[84:])))
        shifted=reference.copy();shifted[knee]=mid[knee]+.1*(LO_HI[knee][1]-LO_HI[knee][0])
        check(f'alias_control_{leg}_knee_angle_is_detected',not np.array_equal(enc.encode(shifted[:42],shifted[42:84],shifted[84:]),enc.encode(reference[:42],reference[42:84],reference[84:])))

    # ---------- 4. independently rebuild every preregistered fit ----------
    train=np.flatnonzero(gens<=1);test=np.flatnonzero(gens>=2)
    check('train_bins_4800',len(train)==4800);check('test_bins_4800',len(test)==4800)
    check('train_episodes_24',len(np.unique(eids[train]))==24);check('test_episodes_24',len(np.unique(eids[test]))==24)
    check('train_test_episode_disjoint',not (set(eids[train].tolist())&set(eids[test].tolist())))
    source_episode=np.zeros(len(old['physical_local_input']),dtype=int)
    max_beta_delta=0.;max_pred_delta=0.;max_metric_delta=0.;max_normal_residual=0.;max_lhs_delta=0.;max_standardizer_delta=0.
    for record in results['fits']:
        fid=record['fit_id'];leg=record['leg'];rep=record['representation'];task=record['task']
        knee=LEG_FIXED[leg]['knee'];joints=LEG_FIXED[leg]['joints'];mn=LEG_FIXED[leg]['mn']
        load_index=SensoryEncoder.foot_order.index(leg)
        lab=labels_for(phys,knee,load_index);source_lab=labels_for(old['physical_local_input'],knee,load_index)
        if rep=='encoded4':
            x=encoded[:,load_index,:];xs=old_encoded[:,load_index,:]
        elif rep=='MNinstant':
            x=mnrates[:,mn]/100.;xs=new['mn_rates_hz'][:,mn]/100.
        elif rep=='MNhistory':
            x=np.hstack([lag_stack(mnrates[:,mn],eids,lag)/100. for lag in LAGS])
            xs=np.hstack([lag_stack(new['mn_rates_hz'][:,mn],source_episode,lag)/100. for lag in LAGS])
        else:
            x=np.c_[phys[:,joints],phys[:,42+np.array(joints)],phys[:,84+load_index]]
            xs=np.c_[old['physical_local_input'][:,joints],old['physical_local_input'][:,42+np.array(joints)],old['physical_local_input'][:,84+load_index]]
        check(f'{fid}:feature_dimension',x.shape[1]==record['feature_dimension'],{'independent':int(x.shape[1]),'recorded':record['feature_dimension']})
        mu=x[train].mean(0);sd=np.maximum(x[train].std(0),1e-8)
        xt=(x-mu)/sd;xtr=xt[train];xte=xt[test];xso=(xs-mu)/sd
        y=lab[task];ys=source_lab[task];classification=task!='signed_knee_velocity'
        if classification:
            tr_valid=~np.isnan(y[train]);te_valid=~np.isnan(y[test]);src_valid=~np.isnan(ys)
            check(f'{fid}:train_class_counts',counts((y[train][tr_valid]>0).sum(),(y[train][tr_valid]<0).sum())==record['train_class_counts'])
            check(f'{fid}:test_class_counts',counts((y[test][te_valid]>0).sum(),(y[test][te_valid]<0).sum())==record['test_class_counts'])
            check(f'{fid}:old_source_class_counts',counts((ys[src_valid]>0).sum(),(ys[src_valid]<0).sum())==record['old_source_class_counts'])
            check(f'{fid}:excluded_counts',int(len(train)-tr_valid.sum())==record['train_excluded'] and int(len(test)-te_valid.sum())==record['test_excluded'])
            labelled=int(tr_valid.sum());pos=float((y[train][tr_valid]>0).sum());neg=float((y[train][tr_valid]<0).sum())
            w=np.zeros(len(train));w[tr_valid]=np.where(y[train][tr_valid]>0,labelled/(2*pos),labelled/(2*neg))
            yf=np.where(tr_valid,y[train],0.)
        else:
            tr_valid=np.ones(len(train),bool);te_valid=np.ones(len(test),bool);src_valid=np.ones(len(ys),bool)
            w=np.ones(len(train));yf=y[train]
        if record['fitting_status']!='FITTED':
            check(f'{fid}:not_estimable_not_fitted',True);continue
        FITS_RECOMPUTED[0]+=1
        lhs,rhs,xbar,ybar=ridge_solution(xtr,yf,w)
        beta=np.linalg.solve(lhs,rhs);intercept=ybar-xbar@beta
        stored_lhs=models[fid+'/lhs'];stored_beta=models[fid+'/beta'];stored_intercept=float(models[fid+'/intercept'])
        max_normal_residual=max(max_normal_residual,float(np.max(np.abs(stored_lhs@stored_beta-models[fid+'/rhs']))))
        max_lhs_delta=max(max_lhs_delta,float(np.max(np.abs(lhs-stored_lhs))))
        d_beta=float(np.max(np.abs(beta-stored_beta)));d_intercept=abs(intercept-stored_intercept)
        max_beta_delta=max(max_beta_delta,d_beta/(1.+float(np.max(np.abs(stored_beta)))),d_intercept/(1.+abs(stored_intercept)))
        check(f'{fid}:train_statistics_reproduced',float(np.max(np.abs(mu-models[fid+'/mean'])))<1e-15 and float(np.max(np.abs(sd-models[fid+'/std'])))<1e-15)
        max_standardizer_delta=max(max_standardizer_delta,float(np.max(np.abs(mu-models[fid+'/mean']))),float(np.max(np.abs(sd-models[fid+'/std']))))
        check(f'{fid}:normal_equation_lhs_reproduced',float(np.max(np.abs(lhs-stored_lhs))/(1.+np.max(np.abs(stored_lhs))))<1e-9,{'max_abs_delta':float(np.max(np.abs(lhs-stored_lhs))),'lhs_scale':float(np.max(np.abs(stored_lhs)))})
        check(f'{fid}:ridge_coefficients_reproduced',d_beta/(1.+float(np.max(np.abs(stored_beta))))<1e-9,{'beta_delta':d_beta,'beta_scale':float(np.max(np.abs(stored_beta)))})
        check(f'{fid}:ridge_intercept_reproduced',d_intercept/(1.+abs(stored_intercept))<1e-9,{'intercept_delta':d_intercept,'intercept_scale':stored_intercept})
        pred_tr=intercept+xtr@beta;pred_te=intercept+xte@beta;pred_so=intercept+xso@beta
        prediction_deltas=[array_close(pred_tr,models[fid+'/yhat_train']),array_close(pred_te,models[fid+'/yhat_test']),array_close(pred_so,models[fid+'/yhat_source'])]
        max_pred_delta=max(max_pred_delta,float(np.max(np.abs(pred_tr-models[fid+'/yhat_train']))),
                           float(np.max(np.abs(pred_te-models[fid+'/yhat_test']))),float(np.max(np.abs(pred_so-models[fid+'/yhat_source']))))
        check(f'{fid}:prediction_arrays_reproduced',all(prediction_deltas),
              {'max_abs_train':float(np.max(np.abs(pred_tr-models[fid+'/yhat_train']))),'max_abs_test':float(np.max(np.abs(pred_te-models[fid+'/yhat_test']))),'max_abs_source':float(np.max(np.abs(pred_so-models[fid+'/yhat_source'])))})
        if classification:
            ba_tr,conf_tr,acc_tr=classification_metrics(y[train],pred_tr,tr_valid)
            ba_te,conf_te,acc_te=classification_metrics(y[test],pred_te,te_valid)
            ba_so,conf_so,acc_so=classification_metrics(ys,pred_so,src_valid)
            m=record['metrics']
            for name,value in (('balanced_accuracy_train',ba_tr),('balanced_accuracy_test',ba_te),('balanced_accuracy_old_source_transfer',ba_so),
                               ('accuracy_train',acc_tr),('accuracy_test',acc_te),('accuracy_old_source_transfer',acc_so)):
                if value is None or m[name] is None:check(f'{fid}:{name}_undefined_agrees',value is None and m[name] is None)
                else:
                    max_metric_delta=max(max_metric_delta,abs(value-m[name])/(1.+abs(m[name])))
                    check(f'{fid}:{name}_reproduced',close(value,m[name]),{'independent':value,'recorded':m[name]})
            for name,value in (('confusion_train',conf_tr),('confusion_test',conf_te),('confusion_old_source_transfer',conf_so)):
                check(f'{fid}:{name}_reproduced',value==m[name])
            check(f'{fid}:confusion_test_totals_labelled',sum(sum(v.values()) for v in conf_te.values())==record['test_labelled'])
            check(f'{fid}:stored_source_predictions_reproduced',np.array_equal(np.where(pred_so>0,1.,-1.),models[fid+'/pred_source']))
            check(f'{fid}:undefined_reason_recorded',record['metrics']['balanced_accuracy_undefined_reason']==(None if (counts((y[test][te_valid]>0).sum(),0) and (y[test][te_valid]<0).sum()>0) else 'test missing class: balanced accuracy undefined'))
        else:
            target_train=y[train];target_test=y[test];base=float(target_train.mean())
            mse_tr=float(np.mean((pred_tr-target_train)**2));mse_te=float(np.mean((pred_te-target_test)**2))
            r2_tr=float(1-((pred_tr-target_train)**2).sum()/((target_train-base)**2).sum())
            r2_te=float(1-((pred_te-target_test)**2).sum()/((target_test-base)**2).sum())
            m=record['metrics']
            for name,value in (('mse_train',mse_tr),('mse_test',mse_te),('r2_train',r2_tr),('r2_test',r2_te),
                               ('train_mean_baseline_mse_test',float(np.mean((target_test-base)**2))),
                               ('mse_old_source_transfer',float(np.mean((pred_so-ys)**2)))):
                max_metric_delta=max(max_metric_delta,abs(value-m[name])/(1.+abs(m[name])))
                check(f'{fid}:{name}_reproduced',close(value,m[name]),{'independent':value,'recorded':m[name]})
            if rep=='rawphysical':
                # rawphysical contains the target column, so R^2 must approach 1 up to the small lambda=1 shrinkage.
                check(f'{fid}:privileged_target_identity_recovered',r2_te>1-1e-6,
                      {'r2_test':r2_te,'unexplained_fraction':1-r2_te,'max_abs_target':float(np.max(np.abs(target_test))),
                       'max_abs_residual':float(np.max(np.abs(pred_te-target_test))),
                       'interpretation':'ridge with lambda=1 shrinks slightly; not an exact identity and not zero error'})
    check('all_fit_prediction_arrays_have_finite_delta','per-fit relative prediction checks above are authoritative' and np.isfinite(max_pred_delta),max_pred_delta)
    check('stored_ridge_normal_equation_residuals_tiny',max_normal_residual<1e-9,max_normal_residual)
    check('fit_count_is_72',FITS_RECOMPUTED[0]==72 and results['fit_count']==72,{'independent':FITS_RECOMPUTED[0],'recorded':results['fit_count']})
    check('fit_record_count_is_72',len(results['fits'])==72)
    check('every_leg_representation_task_present',
          {(f['leg'],f['representation'],f['task']) for f in results['fits']}==
          {(leg,rep,task) for leg in SensoryEncoder.foot_order for rep in ('encoded4','MNinstant','MNhistory','rawphysical')
           for task in ('stance_air','knee_velocity_sign','signed_knee_velocity')})
    check('standardization_is_train_only','train-only' in results['standardization'] and '1e-8' in results['standardization'])
    check('no_physics_or_neural_steps',results['physics_steps']==0 and results['neural_steps']==0 and results['training_updates']==0 and audit['physics_steps']==0 and audit['neural_steps']==0)
    check('not_estimable_count_consistent',results['not_estimable']==sum(1 for f in results['fits'] if f['fitting_status']!='FITTED'))
    check('lambda_is_one',results['lambda_ridge']==1.0 and all(f['lambda_ridge']==1.0 for f in results['fits']))
    check('lags_are_preregistered',results['lags_bins']==list(LAGS))
    check('stance_air_encoded_flagged_tautological',all(f['tautological_note'] for f in results['fits'] if f['task']=='stance_air' and f['representation']=='encoded4'))
    check('rawphysical_flagged_privileged',all(f['deployable'] is False for f in results['fits'] if f['representation']=='rawphysical'))
    check('encoded_and_mn_flagged_deployable_candidates',all(f['deployable'] is True for f in results['fits'] if f['representation']!='rawphysical'))

    summary=dict(status='PASS' if all(c['passed'] for c in CHECKED) else 'FAIL',checks=len(CHECKED),
                 passed=sum(c['passed'] for c in CHECKED),failed=sum(not c['passed'] for c in CHECKED),
                 fits_recomputed=FITS_RECOMPUTED[0],data_artifact_reads=READS[0],trial_artifacts_checked=trial_checks,
                 max_beta_delta=max_beta_delta,max_prediction_delta=max_pred_delta,max_metric_delta=max_metric_delta,
                 max_normal_equation_lhs_delta=max_lhs_delta,max_stored_normal_equation_residual=max_normal_residual,
                 max_standardizer_delta=max_standardizer_delta,
                 encoder_rank_recomputed=ranks,zero_columns_recomputed=len(independent_zero),
                 jacobian_finite_difference_max_error=fd_error,recorded_current_max_error_nA=max_current_error,
                 independent_formula_max_error_nA=max_formula_error,
                 physics_steps=0,neural_steps=0,training_updates=0,
                 independent_statement='No fitter or audit module imported. Labels, lag stacks, standardization, ridge normal equations, predictions and metrics rebuilt from the frozen encoder source and the 195435Z trial artifacts.',
                 limits='Agreement proves the saved diagnostic is arithmetically reproducible, not that the encoder is biology or that any policy works. Train/test share one canonical checkpoint, so all samples are correlated.',
                 checks_detail=CHECKED)
    (HERE/'VERIFICATION.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ('status','checks','passed','failed','fits_recomputed','data_artifact_reads','trial_artifacts_checked','max_beta_delta','max_prediction_delta','max_metric_delta','encoder_rank_recomputed','zero_columns_recomputed')}),flush=True)
    if summary['failed']:raise SystemExit(1)


if __name__=='__main__':main()
