"""Independent case/runtime/KKT verification: does not import fit_cases or optimize."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import tempfile
import traceback
import numpy as np
from runtime import Readout,canonical_identity

HERE=Path(__file__).resolve().parent
OLD=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z/readout')
PHYSICAL=Path('/home/sensen/Desktop/engine修复验收/flybody-retarget/20261107T182940Z/teacherderived_slow4_physics')
ALL=('oldinstant','oldhistory','newinstant','newhistory')


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def independent_features(rates,lags):
    # Explicit sample indexing, no rolling, slice-padding or trainer helper reuse.
    result=np.zeros((len(rates),len(lags)*323))
    for row in range(len(rates)):
        for block,lag in enumerate(lags):
            past=row-int(lag)
            if past>=0:result[row,block*323:(block+1)*323]=rates[past]/100.
    return result


class Verify:
    def __init__(self):self.checks=[]
    def check(self,name,ok,**details):
        self.checks.append(dict(name=name,passed=bool(ok),**details))
        if not ok:raise AssertionError(name)
    def same(self,name,a,b,tol=0):
        a,b=np.asarray(a),np.asarray(b)
        self.check(name+'_shape',a.shape==b.shape)
        if a.dtype.kind in 'USO' or b.dtype.kind in 'USO':self.check(name,np.array_equal(a,b));return
        self.check(name+'_finite',np.isfinite(a).all() and np.isfinite(b).all())
        error=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.
        self.check(name,np.array_equal(a,b) if tol==0 else error<=tol,max_abs_error=error,tolerance=tol)


def verify_runtime(v,case,path,rates,features,prediction):
    runtime=Readout(path)
    v.same(case+'_fresh_zero_action',runtime.action(),np.zeros(42))
    max_feature_error=0.;max_action_error=0.
    for i,row in enumerate(rates):
        runtime.observe(row)
        max_feature_error=max(max_feature_error,float(np.max(np.abs(runtime.feature_vector()-features[i]))))
        max_action_error=max(max_action_error,float(np.max(np.abs(runtime.action()-prediction[i]))))
    v.check(case+'_entire_runtime_features_exact',max_feature_error==0,max_abs_error=max_feature_error)
    v.check(case+'_entire_runtime_actions_match',max_action_error<=1e-10,max_abs_error=max_action_error)
    state=runtime.state_dict();replica=Readout(path).load_state_dict(state)
    v.same(case+'_restored_action',runtime.action(),replica.action())
    state['history_rates_hz'][:]=0
    v.check(case+'_snapshot_is_copy',np.any(runtime.history_rates_hz!=0))
    marker=np.arange(1,324,dtype=float)/323.
    steady=Readout(path)
    for _ in range(11):steady.observe(marker)
    v.same(case+'_stationary_lagfeature_equivalence',steady.feature_vector(),np.tile(marker/100.,len(steady.lags_bins)))
    impulse=Readout(path)
    for age in range(12):
        impulse.observe(marker if age==0 else np.zeros(323))
        expected=np.zeros((len(impulse.lags_bins),323))
        for block,lag in enumerate(impulse.lags_bins):
            if age==lag:expected[block]=marker/100.
        v.same(case+'_impulse_age_'+str(age),impulse.feature_vector(),expected.reshape(-1))
    v.same(case+'_history_fully_drained_at55ms',impulse.action(),np.zeros(42))
    if len(steady.lags_bins)>1:
        v.check(case+'_max_history50ms',steady.history_rates_hz.shape==(11,323) and steady.lags_bins.max()*steady.bin_ms==50.)
    for label,bad in [('shape',np.zeros(324)),('nan',np.full(323,np.nan)),('infinity',np.full(323,np.inf)),('negative',-np.ones(323)),('complex',np.ones(323,dtype=complex)),('string',np.full(323,'1'))]:
        try:steady.observe(bad)
        except ValueError:v.check(case+'_reject_'+label,True)
        else:v.check(case+'_reject_'+label,False)
    future=rates.copy();future[1000:]+=123.
    altered=independent_features(future,steady.lags_bins)
    v.same(case+'_future_suffix_cannot_change_prefix',features[:1000],altered[:1000])
    # Temporary invalid metadata fixtures are deleted immediately after checks.
    original=load(path)
    with tempfile.TemporaryDirectory(prefix='identity_test_',dir=HERE) as tmp:
        for kind in ('root','joint','leg','crossleg','bias','lag'):
            mutated={k:a.copy() for k,a in original.items()}
            if kind=='root':mutated['mn_root_ids'][0]+=1
            elif kind=='joint':mutated['joint_names'][[0,1]]=mutated['joint_names'][[1,0]]
            elif kind=='leg':mutated['mn_leg'][0]='xx'
            elif kind=='bias':mutated['bias'][0]=.1
            elif kind=='lag':mutated['lags_bins'][-1]=11
            else:
                wrong=np.flatnonzero(mutated['mn_leg']!='lf')[0];mutated['weights'][wrong,0]=.123
            fixture=Path(tmp)/(kind+'.npz');np.savez_compressed(fixture,**mutated)
            try:Readout(fixture)
            except ValueError:v.check(case+'_reject_model_'+kind,True)
            else:v.check(case+'_reject_model_'+kind,False)


def verify_case(v,case,oldtrace,labels):
    folder=HERE/'cases'/case
    model,pred=load(folder/'MODEL.npz'),load(folder/'PREDICTIONS.npz')
    result=json.loads((folder/'RESULT.json').read_text())
    for path,expected in result['source_hashes'].items():v.check(case+'_hash_'+path,sha(path)==expected)
    for name,expected in result['files'].items():v.check(case+'_output_hash_'+name,sha(folder/name)==expected)
    source=oldtrace if case.startswith('old') else load(HERE.parent/'dynamics/SOURCE_TRACE.npz')
    rates=source['mn_rates_hz']
    v.same(case+'_source_time',source['time_s'],oldtrace['time_s'])
    roots,local,legs,names=canonical_identity()
    for key,value in (('mn_root_ids',roots),('mn_local',local),('mn_leg',legs),('joint_names',names)):v.same(case+'_'+key,model[key],value)
    expected_lags=[0] if case.endswith('instant') else [0,1,4,10]
    v.same(case+'_fixed_lags',model['lags_bins'],expected_lags)
    v.check(case+'_primary_fixed',bool(model['primary'])==(case=='newhistory')==bool(result['primary']))
    v.check(case+'_fit_count',result['fit_solves']==(0 if case=='oldinstant' else 42))
    if case=='oldinstant':v.same('oldinstant_cached_weights_unchanged',model['weights'],load(OLD/'MODEL.npz')['weights'])
    features=independent_features(rates,expected_lags)
    w=model['weights'];featurelegs=np.tile(legs,len(expected_lags));jointlegs=np.array([name.removeprefix('joint_')[:2].lower() for name in names])
    v.check(case+'_weights_finite_bounded',np.isfinite(w).all() and np.all(np.abs(w)<=1))
    v.check(case+'_crossleg_exactzero',np.all(w[featurelegs[:,None]!=jointlegs[None,:]]==0))
    v.same(case+'_bias_zero',model['bias'],np.zeros(42))
    times=oldtrace['time_s'];train=(times>=.2-1e-9)&(times<6.-1e-9);temporal=times>=6.-1e-9
    v.same(case+'_labels_from_actual_commands',pred['target_rate'],labels)
    v.same(case+'_prediction_time',pred['time_s'],times)
    v.same(case+'_trainmask',pred['train_mask'],train)
    v.same(case+'_temporalmask',pred['temporal_mask'],temporal)
    v.check(case+'_1160_617_split',train.sum()==1160 and temporal.sum()==617)
    prediction=30*np.tanh(features@w)
    v.same(case+'_independent_prediction',prediction,pred['predicted_rate'],1e-10)
    rms=lambda x:float(np.sqrt(np.mean(x*x)))
    metrics=dict(train_RMS_rad_s=rms(prediction[train]-labels[train]),temporal_RMS_rad_s=rms(prediction[temporal]-labels[temporal]),zero_train_RMS_rad_s=rms(labels[train]),zero_temporal_RMS_rad_s=rms(labels[temporal]))
    for key,value in metrics.items():v.same(case+'_'+key,value,result['metrics'][key],1e-10)
    v.same(case+'_perjoint_temporal_RMS',np.sqrt(np.mean((prediction[temporal]-labels[temporal])**2,axis=0)),result['per_joint_temporal_RMS_rad_s'],1e-10)
    kkt=[]
    for j,name in enumerate(names):
        selected=np.flatnonzero(featurelegs==jointlegs[j]);x=features[train][:,selected]
        y=np.arctanh(np.clip(labels[train,j]/30.,-.999,.999));beta=w[selected,j]
        residual=x@beta-y;grad=x.T@residual+beta
        projected=float(np.max(np.abs(beta-np.clip(beta-grad,-1,1))))
        lower=beta<=-1+1e-7;upper=beta>=1-1e-7
        raw=float(np.max(np.abs(np.where(lower,np.minimum(grad,0),np.where(upper,np.maximum(grad,0),grad)))))
        cost=.5*float(residual@residual+beta@beta)
        v.check(case+'_KKT_'+str(name),projected<=1e-6,projected_residual=projected,raw_active_set_residual=raw)
        v.same(case+'_objective_'+str(name),cost,result['fit_records'][j]['objective'],1e-10)
        kkt.append(dict(joint=str(name),projected_residual=projected,raw_active_set_residual=raw,objective=cost))
    verify_runtime(v,case,folder/'MODEL.npz',rates,features,prediction)
    return dict(case=case,metrics=metrics,KKT=kkt,max_projected_residual=max(x['projected_residual'] for x in kkt),model_sha256=sha(folder/'MODEL.npz'),model_path=str(folder/'MODEL.npz'),primary=case=='newhistory')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['old','all'],default='all');args=parser.parse_args()
    output=HERE/('OLD_RUNTIME_VERIFICATION.json' if args.phase=='old' else 'INDEPENDENT_VERIFICATION.json')
    if output.exists():raise FileExistsError(str(output))
    v=Verify()
    try:
        prereg=json.loads((HERE/'PREREG.json').read_text());v.check('primary_preregistered_newhistory',prereg['primary_fixed_before_results'].startswith('newhistory'))
        oldtrace=load(OLD/'TRAIN_TRACE.npz');commands=load(PHYSICAL/'COMMANDS.npz')['rows']
        labels=np.stack([commands[row:row+5,1:43].mean(0) for row in range(0,1817*5,5)])
        v.same('oldlabels_actual_five_ms_applied',labels,oldtrace['target_rate'])
        cases=ALL[:2] if args.phase=='old' else ALL
        results=[verify_case(v,case,oldtrace,labels) for case in cases]
        # Syntax only: compile never imports/runs the physical runner.
        compile((HERE/'run_closed_loop.py').read_text(),str(HERE/'run_closed_loop.py'),'exec')
        v.check('physical_runner_syntax_only',True)
        v.check('no_fit_neural_physics_imports',not any(name in sys.modules for name in ('fit_cases','module','mujoco','scipy.optimize')))
        result=dict(status='PASS_OLD_RUNTIME_ONLY' if args.phase=='old' else 'PASS_FIXED_FOUR_CASE_ENGINEERING',checks_passed=len(v.checks),checks=v.checks,cases=results,case_model_sha256={x['case']:x['model_sha256'] for x in results},primary='newhistory',new_neural_steps=0,new_physical_steps=0,new_fit_solves=0,source_quality='failed sameepisode reference; no walking or biology claim',verifier_sha256=sha(__file__),runtime_sha256=sha(HERE/'runtime.py'),prereg_sha256=sha(HERE/'PREREG.json'),runner_sha256=sha(HERE/'run_closed_loop.py'))
        output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(status=result['status'],checks_passed=result['checks_passed'],cases=[dict(case=x['case'],metrics=x['metrics'],max_KKT=x['max_projected_residual']) for x in results])),flush=True)
    except Exception:
        (HERE/('FAILED_VERIFICATION_'+args.phase.upper()+'.json')).write_text(json.dumps(dict(error=traceback.format_exc(),checks=v.checks),indent=2)+'\n')
        raise

if __name__=='__main__':main()
