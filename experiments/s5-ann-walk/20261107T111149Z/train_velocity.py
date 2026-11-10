"""One fixed velocity-loss student experiment; offline training only.

Architecture, schedules, perturbations and rollout limits are unchanged. Added
normalized velocity loss is .01*||vpred-vtruth||²; no phase input at deployment.
Teacher remains slip-incomplete; physical student acceptance is a separate test.
"""
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
SOURCE=ROOT/'experiments/s5-ann-walk/20261107T102842Z/run_student.py'
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T111149Z')
OLD=OUT.parent/'20261107T102842Z'
spec=importlib.util.spec_from_file_location('velocity_student_original_runtime',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
n=b.n
n.OUT=OUT
ORIGINAL_TRAIN=inspect.getsource(n.train_model)
patches=[
    ('        loss = torch.zeros(())',
     '        loss = torch.zeros(())\n        position_loss = torch.zeros(())\n        velocity_loss = torch.zeros(())'),
    ('            loss = loss + ((nxt-target[:,h])**2).mean()/horizon',
     """            position_step = ((nxt-target[:,h])**2).mean()/horizon
            # Truth previous state stays unperturbed, while the predicted previous
            # state includes unchanged noisy/self-predicted history. Recovery is
            # therefore penalized honestly rather than silently removing noise.
            truth_previous = torch.tensor(truth[:,K+h-1],dtype=torch.float32)
            vtrue = (target[:,h]-truth_previous)/PDT
            vpred = (nxt-hist[:,-1])/PDT
            velocity_step = ((vpred-vtrue)**2).mean()/horizon
            position_loss = position_loss + position_step
            velocity_loss = velocity_loss + velocity_step
            loss = loss + position_step + .01*velocity_step"""),
    ('item = dict(step=it,horizon=horizon,loss=float(loss.detach()),wall_s=round(time.monotonic()-start,1))',
     'item = dict(step=it,horizon=horizon,loss=float(loss.detach()),position_loss=float(position_loss.detach()),velocity_loss=float(velocity_loss.detach()),weighted_velocity_loss=.01*float(velocity_loss.detach()),wall_s=round(time.monotonic()-start,1))'),
]
EFFECTIVE_TRAIN=ORIGINAL_TRAIN
for old,new in patches:
    assert EFFECTIVE_TRAIN.count(old)==1,('Expected one original train body match',old)
    EFFECTIVE_TRAIN=EFFECTIVE_TRAIN.replace(old,new,1)
EFFECTIVE_SHA256=hashlib.sha256(EFFECTIVE_TRAIN.encode()).hexdigest()
ast.parse(EFFECTIVE_TRAIN)
exec(compile(EFFECTIVE_TRAIN,str(Path(__file__).resolve())+'::effective_train','exec'),n.__dict__)


def source_hashes():
    result={str(Path(__file__).resolve()):n.sha(__file__),str(SOURCE):n.sha(SOURCE),
        str(Path(n.__file__)):n.sha(n.__file__),str(n.GAIT):n.sha(n.GAIT),
        str(OLD/'MODEL.npz'):n.sha(OLD/'MODEL.npz')}
    seen=set()
    def visit(module):
        if id(module) in seen:return
        seen.add(id(module));path=getattr(module,'__file__',None)
        if path and str(path).startswith(str(ROOT)) and '/site-packages/' not in str(path):
            result[str(Path(path))]=n.sha(path)
            for value in vars(module).values():
                if isinstance(value,type(sys)):visit(value)
    visit(b)
    for module in list(sys.modules.values()):
        if module is not None:visit(module)
    return result


def diagnose_previous():
    """Saved actual commanded rates, compared with separately labelled phases."""
    p=n.Plant();ref=n.Reference(p)
    with np.load(OLD/'ann_screen/POLICY.npz') as z:
        targets=z['targets'].copy();actions=z['actions'].copy()
    times=np.arange(len(targets))*n.PDT;mask=times>=1.
    q=targets[mask];rates=actions[mask]
    grid=np.linspace(0,2*np.pi,4096,endpoint=False);reference=ref.at(grid)
    normalized=(q-ref.center)/ref.scale
    normalgrid=(reference-ref.center)/ref.scale
    nearest=[]
    for start in range(0,len(q),32):
        distances=np.mean((normalized[start:start+32,None,:]-normalgrid[None,:,:])**2,axis=2)
        nearest.extend(np.argmin(distances,axis=1))
    nearest=np.asarray(nearest);phase=grid[nearest]
    # Exact derivative of the same periodic PCHIP reference, in rad/s.
    aligned_velocity=ref.spline(phase,1)*(2*np.pi*n.FREQ)
    clockphase=2*np.pi*n.FREQ*(times[mask]-n.RAMP/2)
    clock_velocity=ref.spline(clockphase%(2*np.pi),1)*(2*np.pi*n.FREQ)
    def metrics(truth):
        difference=rates-truth
        return dict(rate_error_RMSE_rad_s=float(np.sqrt(np.mean(difference*difference))),
            normalized_rate_error_RMSE_per_s=float(np.sqrt(np.mean((difference/ref.scale)**2))),
            per_joint_rate_RMSE_rad_s=np.sqrt(np.mean(difference*difference,axis=0)),
            rate_error_abs_max_rad_s=float(abs(difference).max()))
    n.dump(OUT/'PREVIOUS_RATE_DIAGNOSIS.json',dict(stage='Previousstudent saved-rate diagnostic only',
        source=str(OLD/'ann_screen/POLICY.npz'),source_sha256=n.sha(OLD/'ann_screen/POLICY.npz'),
        samples=len(q),phase_aligned=metrics(aligned_velocity),teacher_clock=metrics(clock_velocity),
        normalized_nearest_reference_pose_RMS=float(np.sqrt(np.mean((normalized-normalgrid[nearest])**2))),
        nearest_phase=phase,clock_phase=clockphase%(2*np.pi),physics_steps=0,
        limitations='Nearestposephase is estimated from own integratedtarget, not supplied to deployedANN. Jointcurve can approach itself, so phaseestimate ambiguous and not proofjittercausation. Clockcomparison mixesphase drift. SavedPOLICY.actions are clipped commandedrates, not physicaljointvelocity.'))


def main():
    import mujoco as mj
    def forbidden(*a,**kw):raise AssertionError('No mj_step in velocity-loss training')
    mj.mj_step=forbidden
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PREREGISTRATION.json').exists():raise FileExistsError('Preserve prior training '+str(OUT))
    assert n.FREQ==1. and n.K==40 and np.array_equal(n.LAGS,[0,8,24,32,36,38,39])
    hashes=source_hashes()
    n.dump(OUT/'PREREGISTRATION.json',dict(stage='S5 replacementANN normalizedvelocity-loss experiment only',
        source_hashes=hashes,source_teacher='101114broadlift+gain128; originalteacherorder unchanged',
        teacher_status='AllD/sixsteps/headingpass butmaterialslipratios [.2491,.3544,.7531,.3039,.5216,.7440] fail; noacceptedteacherclaim.',
        architecture='UnchangedMLP294->128tanh->128tanh->42; causal40frameown-targethistory7lags, samenormalization/exportkeys',
        steps=3000,seed=731,frequency_hz=1.,loss='normalizedpositionMSE + .01*normalizedfinite-differencevelocityMSE',
        velocity_truth='(teachertruth[K+h]-teachertruth[K+h-1])/.005',
        velocity_prediction='(actualjoint/rateclipped nextprediction-currentnoisy/selfpredictedhistorylast)/.005',
        weight_interpretation='.01=(.1s)^2 timeweight; noweightsearch',
        unchanged='Adam.001then.0003/.0001at400/650; horizon1/16/64at400/650; batch128/32; originalnoiseevery2of3iterations; globalgradientclip1; samephysicalrate30/jointranges.',
        prior_result='102842student support/heading passed butmaterialslipworse; ratejitter is hypothesis, not measuredcausation.',
        diagnostics='PrevioussavedPOLICYratesvsrefvelocity withlabelledteacherclock andseparatenearestposephaseestimate; no inferredphaseinput atdeployment.',
        evaluation='ExportTorch/NumPyparity<1e-5, unchanged6initialphase20s offline autonomousrollouts; parentownsphysicalscreen/independentreplay.',
        original_brain='NOTRUN, replacementANN only',physics_steps=0,neural_claim='OfflineANNtrainingonly; no physical/neuralwalkingacceptance.',
        effective_train_sha256=EFFECTIVE_SHA256,patches=[dict(old=old,new=new,required_matches=1) for old,new in patches]))
    (OUT/'EFFECTIVE_TRAIN.py').write_text(EFFECTIVE_TRAIN)
    diagnose_previous()
    path=n.train_model(steps=3000,name='MODEL_VELOCITY')
    n.offline(path)
    n.dump(OUT/'STATUS.json',dict(status='TRAINED_AND_OFFLINE_EVALUATED',model=str(path),model_sha256=n.sha(path),
        source_sha256=n.sha(__file__),effective_train_sha256=EFFECTIVE_SHA256,
        source_hashes_unchanged={f:n.sha(f)==h for f,h in hashes.items()},
        physics_steps=0,physical_status='NOT_RUN; parentmustjudgeoriginalsupport/steps/heading/materialslip independently'))


if __name__=='__main__':main()
