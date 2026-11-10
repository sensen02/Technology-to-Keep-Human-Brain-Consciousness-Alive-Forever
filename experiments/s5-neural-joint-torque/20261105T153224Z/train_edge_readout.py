"""Independent nonnegative gain on model-supported fixed-sign readout entries.
Teacher torque is used offline only. Online inference uses filtered SNN rates only.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
import sys,importlib.util,time
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
D=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-joint-torque/20261105T152908Z/causal_diagnosis.py')
spec=importlib.util.spec_from_file_location('diagnosis_code',D);dmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(dmod)
b=dmod.b
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T153224Z')
DIAG=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T152908Z')
SOURCE=dmod.SOURCE
b.O=O
js=b.js;dump=b.dump;sha=b.sha

def inputs():return b.sources()+[SOURCE/'FULL_COVERAGE.json',SOURCE/'SUMMARY.json',DIAG/'DIAGNOSIS.json',D,Path(__file__)]
def prereg():
    assert js(DIAG/'DIAGNOSIS.json')['escalation_needed']
    O.mkdir(parents=True,exist_ok=False)
    dump(O/'PREREGISTRATION.json',dict(stage='S5 escalation after failed causal interventions',authorization='User confirmed more aggressive methods if causal diagnosis cannot resolve failure; no external algorithm replacing neurons',
        readout='tau_j=clip(limit_j * sum_p S_jp * c_jp * filtered_rate_p/100, +/-limit_j), 174 independent c>=0; no bias, time, phase, body state or desired angle inputs',
        mapping='same full fixed sign and 174 nonzero entries; no added edges or neurons; limits unchanged',
        teacher='only offline regression on actual algorithm-warmup torque and prior completed interval filtered neural rates; no teacher calls during release',
        alignment='warm neural record at t has rates completed at t+1ms; use previous record rates for torque applied over [t,t+1ms]; teacher target average of ctrl at t+0.5ms and t+1ms',
        fitting=dict(train_time=[.3,1.1],holdout_time=[1.1,1.3],bounds=[0,8],ridge=.001,loss='mean squared torque normalized by joint limits + .001*sum(c^2)',solver='scipy lsq_linear bvls; per-joint independent; empty input remains zero'),
        search=dict(seed=20261105,scales=[.25,.5,1.,2.,4.],generations=2,population=8,sigma=[.35,.20],candidate='positive fitted coefficients times exp(Gaussian perturbation), clipped [0,32]; zero coefficients remain zero',training_evaluations_per_arm=21,selection='same native dynamic support objective as prior; report failed D1-D8'),
        controls='each P0/P2: old-best reproduction plus best repeat, readout cut, sensory cut; total 50 rollouts (25 each)',
        timing=dict(warmup_s=1.3,release_s=.2,body_dt_s=b.DT,neural_dt_s=b.NDT,command_dt_s=b.CDT),
        gate='If all candidates fail dynamic gates, do not claim walking and do not silently raise actuator limits or add external controllers',
        resource='single process, all failures retained',units='NOT_ESTABLISHED',importance='核心必需'))
    dump(O/'INPUT_HASHES_BEFORE.json',{str(p):sha(p) for p in inputs()})
    dump(O/'CHANGE_LEDGER.json',dict(old_file=str(SOURCE/'FULL_COVERAGE.json'),old_sha256=sha(SOURCE/'FULL_COVERAGE.json'),new_file=str(O/'PREREGISTRATION.json'),new_sha256=sha(O/'PREREGISTRATION.json'),reason='Replace factorized 66 pool weights x42 joint gains by independent gains on the same 174 model-supported signed entries, offline teacher initialization then closed-loop search.'))

def fit(t,arm,S):
    n=np.load(O/('WARM_'+arm)/'NEURAL.npz');body=np.load(O/('WARM_'+arm)/'BODY.npz')
    times=n['time'];rates=np.vstack([np.zeros((1,66)),n['rates'][:-1]])/100
    target=[]
    for ti in times:
        ids=[int(np.argmin(abs(body['time']-(ti+shift)))) for shift in [.0005,.001]]
        assert all(abs(body['time'][idx]-(ti+shift))<1e-9 for idx,shift in zip(ids,[.0005,.001]))
        target.append(body['ctrl'][ids].mean(0)/t.limits)
    target=np.array(target);train=(times>=.3-1e-10)&(times<1.1-1e-10);hold=(times>=1.1-1e-10)
    C=np.zeros_like(S);stats=[]
    for j in range(42):
        cols=np.flatnonzero(S[j]);X=rates[:,cols]*S[j,cols];ntrain=int(train.sum())
        design=np.vstack([X[train]/np.sqrt(ntrain),np.sqrt(.001)*np.eye(len(cols))]);y=np.r_[target[train,j]/np.sqrt(ntrain),np.zeros(len(cols))]
        sol=lsq_linear(design,y,bounds=(0,8),method='bvls',tol=1e-10,max_iter=1000);assert sol.success
        C[j,cols]=sol.x
        stats.append(dict(joint=t.m.joint(int(t.jids[j])).name,train_mse=float(np.mean((X[train]@sol.x-target[train,j])**2)),holdout_mse=float(np.mean((X[hold]@sol.x-target[hold,j])**2)),zero_predictor_holdout_mse=float(np.mean(target[hold,j]**2)),solver_status=int(sol.status)))
    folder=O/('FIT_'+arm);folder.mkdir();matrix=S*C
    np.savez_compressed(folder/'FIT_DATA.npz',time=times,previous_completed_rates_hz=rates*100,teacher_normalized=target,train_mask=train,holdout_mask=hold,coefficients=C,matrix=matrix)
    dump(folder/'FIT.json',dict(arm=arm,per_joint=stats,train_mse=float(np.mean([r['train_mse'] for r in stats])),holdout_mse=float(np.mean([r['holdout_mse'] for r in stats])),zero_predictor_holdout_mse=float(np.mean([r['zero_predictor_holdout_mse'] for r in stats])),nonzero_coefficients=int(np.count_nonzero(C)),source='only prior completed SNN rates, no teacher during inference',importance='核心必需'))
    return C

def evaluate(t,arm,matrix,label,sensory_cut=False,cut=False,previous=None):
    # Freeze the exact candidate and its hashes before any physical integration.
    stamp=time.strftime('%Y%m%dT%H%M%S',time.gmtime())+f'{time.time_ns()%1000000000:09d}Z'
    candidate=O/(stamp+'_'+label+'_CONFIG');candidate.mkdir()
    dump(candidate/'READOUT_MATRIX.json',dict(matrix=matrix,sensory_cut=sensory_cut,cut=cut))
    dump(candidate/'PREREGISTRATION.json',dict(parent=str(O/'PREREGISTRATION.json'),old_file=previous,old_sha256=sha(previous) if previous else None,new_file=str(candidate/'READOUT_MATRIX.json'),new_sha256=sha(candidate/'READOUT_MATRIX.json'),reason=label))
    t.S=matrix.copy();t.sensory_cut=sensory_cut
    r=t.evaluate(arm,np.zeros(108),label,cut=cut,previous=str(candidate/'READOUT_MATRIX.json'))
    dump(Path(r['folder'])/'READOUT_IDENTITY.json',dict(file=str(candidate/'READOUT_MATRIX.json'),sha256=sha(candidate/'READOUT_MATRIX.json')))
    r['matrix_file']=str(candidate/'READOUT_MATRIX.json')
    return r

def run():
    before=js(O/'INPUT_HASHES_BEFORE.json');assert before=={str(p):sha(p) for p in inputs()}
    t=dmod.DiagnosticTrial();(O/'COVERAGE.json').rename(O/'DOMINANT_COVERAGE.json');full=js(SOURCE/'FULL_COVERAGE.json');dump(O/'COVERAGE.json',full);S=np.asarray(full['sign_matrix']);mask=S!=0
    previous=js(SOURCE/'SUMMARY.json');summary={};history=[]
    try:
        for arm in ['P0','P2']:
            t.sensory_cut=False;t.warm(arm);zz=np.array(previous[arm]['best_z']);oldmatrix=S*np.exp(zz[:66])[None,:]*np.exp(zz[66:])[:,None]
            ref=evaluate(t,arm,oldmatrix,arm+'_old_best')
            # Algebraic factor reassociation may change rounding, so compare tolerantly.
            old=np.load(Path(previous[arm]['best']['folder'])/'BODY.npz');new=np.load(Path(ref['folder'])/'BODY.npz')
            err=float(np.max(abs(old['qpos']-new['qpos'])));assert err<1e-8 and abs(ref['J']-previous[arm]['best']['J'])<1e-8
            C=fit(t,arm,S);best=None;bestC=None;rng=np.random.default_rng(20261105)
            for scale in [.25,.5,1.,2.,4.]:
                cc=np.clip(C*scale,0,32);r=evaluate(t,arm,S*cc,arm+'_fit_scale_'+str(scale));history.append(r)
                if best is None or r['J']<best['J']:best=r;bestC=cc.copy()
                dump(O/'HISTORY.json',history)
            for gen,sigma in enumerate([.35,.20]):
                center=bestC.copy();parent=best['matrix_file']
                for i in range(8):
                    cc=center.copy();cc[mask]=np.clip(center[mask]*np.exp(sigma*rng.standard_normal(int(mask.sum()))),0,32)
                    r=evaluate(t,arm,S*cc,f'{arm}_g{gen}_i{i}',previous=parent);history.append(r)
                    if r['J']<best['J']:best=r;bestC=cc.copy()
                    dump(O/'HISTORY.json',history)
            repeat=evaluate(t,arm,S*bestC,arm+'_repeat',previous=best['matrix_file']);cut=evaluate(t,arm,S*bestC,arm+'_P1_cut',cut=True,previous=best['matrix_file']);sens=evaluate(t,arm,S*bestC,arm+'_sensory_cut',sensory_cut=True,previous=best['matrix_file'])
            a=np.load(Path(best['folder'])/'BODY.npz');bb=np.load(Path(repeat['folder'])/'BODY.npz');assert best['J']==repeat['J'] and all(np.array_equal(a[k],bb[k]) for k in a.files)
            summary[arm]=dict(old_best_replay=ref,old_best_qpos_error=err,baseline=history[-21],best=best,repeat=repeat,cut=cut,sensory_cut=sens,exact_repeat=True,training_evaluations=21,best_coefficients=bestC,fit=js(O/('FIT_'+arm)/'FIT.json'))
            dump(O/'SUMMARY.json',summary)
        print('ESCALATION COMPLETE 50 rollouts',flush=True)
    finally:
        after={str(p):sha(p) for p in inputs()};dump(O/'INPUT_HASHES_AFTER.json',after);assert before==after
if __name__=='__main__':
    if sys.argv[1]=='prereg':prereg()
    elif sys.argv[1]=='run':run()
