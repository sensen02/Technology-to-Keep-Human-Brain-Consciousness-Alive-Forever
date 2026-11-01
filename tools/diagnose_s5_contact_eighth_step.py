"""One additional timestep arm, using the previously validated frozen continuation.
Fresh evidence output only. No analyze-only or overwrite mode.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse, importlib.util, json, sys, platform, subprocess, time, traceback
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
PREV=Path('/home/sensen/Desktop/engine修复验收/s5-contact-resolution/20261101T184041Z')
BASE=Path('/home/sensen/Desktop/engine修复验收/s5-baseline-recording/20261101T183002Z')
DEPENDENCY=ROOT/'tools/diagnose_s5_contact_resolution.py'
spec=importlib.util.spec_from_file_location('contact_resolution_frozen',DEPENDENCY)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
sha,dump=helper.sha,helper.dump

def load(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}

def manifest(out):
    files={str(p):dict(sha256=sha(p),bytes=p.stat().st_size,importance='重要辅助' if p.suffix in ('.log','.md') or 'VERIFY' in p.name else '核心必需') for p in sorted(out.iterdir()) if p.is_file() and p.name not in ('MANIFEST_SHA256.json','RUN.log')}
    for p in (Path(__file__).resolve(),DEPENDENCY,ROOT/'engine/embodied/contact_metrics.py'):
        files[str(p)]=dict(sha256=sha(p),bytes=p.stat().st_size,importance='核心必需')
    dump(out/'MANIFEST_SHA256.json',files)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();out=args.out.resolve()
    if out in (BASE,PREV) or BASE in out.parents or PREV in out.parents:raise ValueError('frozen input output forbidden')
    out.mkdir(parents=True,exist_ok=True)
    if any(p.name not in ('PREREGISTRATION.json','RUN.log') for p in out.iterdir()):raise ValueError('only fresh preregistered directory permitted')
    prereg=json.loads((out/'PREREGISTRATION.json').read_text());started=time.monotonic()
    try:
        assert sha(DEPENDENCY)==prereg['dependency_sha256']
        assert json.loads((PREV/'RECOVERY.json').read_text())['passed']
        dt=float(prereg['dt']); assert dt==.0000125 and prereg['duration_s']==2.0
        ident=json.loads((BASE/'MODEL_IDENTITY.json').read_text())
        with np.load(BASE/'trajectory.npz') as z:
            a={k:z[k][:1].copy() for k in ('qpos','qvel','ctrl','act','xfrc_applied','qfrc_applied','qacc_warmstart','time','filtered_load')}
        m=mujoco.MjModel.from_binary_path(str(BASE/'model.mjb'));m.opt.timestep=dt
        r=helper.Runner(m,a,ident)
        saved=load(PREV/'INITIAL_STATE.npz')
        current=dict(integration_state=helper.state(m,r.d),filtered_load=r.lf,qn=r.qn)
        assert all(np.array_equal(v,saved[k]) for k,v in current.items())
        assert all(getattr(r.d,k).size==0 for k in ('act','history','plugin_state','mocap_pos','mocap_quat','userdata','eq_active'))
        np.savez_compressed(out/'INITIAL_STATE.npz',**current)
        mujoco.mj_saveModel(m,str(out/'model.mjb'),None)
        inputs=[BASE/'trajectory.npz',BASE/'model.mjb',BASE/'MODEL_IDENTITY.json',PREV/'INITIAL_STATE.npz',PREV/'RECOVERY.json',PREV/'trajectory_0p05ms.npz',PREV/'trajectory_0p025ms.npz',PREV/'SUMMARY.json',PREV/'MANIFEST_SHA256.json',DEPENDENCY,ROOT/'engine/embodied/contact_metrics.py']
        dump(out/'ENVIRONMENT.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),python=sys.version,executable=sys.executable,mujoco=mujoco.__version__,numpy=np.__version__,platform=platform.platform(),MUJOCO_GL=os.getenv('MUJOCO_GL'),PYTHONDONTWRITEBYTECODE=os.getenv('PYTHONDONTWRITEBYTECODE'),script_sha256=sha(__file__),inputs={str(p):sha(p) for p in inputs}))
        dump(out/'MODEL_IDENTITY.json',dict(**{k:v for k,v in ident.items() if k!='dt'},dt=dt,source_model=str(BASE/'model.mjb'),source_model_sha256=sha(BASE/'model.mjb'),derived_model_sha256=sha(out/'model.mjb'),only_change='opt.timestep',model_options=str(m.opt),state_spec=int(helper.STATE),initial_sha256=sha(out/'INITIAL_STATE.npz'),initial_bit_equal_previous=True,alpha=float(r.alpha)))
        b=r.run(2.,out/'trajectory.npz')
        summary,episodes=helper.analyze(b);summary['observation_warning_peak']=b['observation_warning_number'].max(axis=0).tolist()
        dump(out/'CONTACT_EPISODES.json',episodes);del episodes
        summaries=json.loads((PREV/'SUMMARY.json').read_text());summaries['0p0125ms']=summary
        dump(out/'SUMMARY.json',summaries)
        c=load(PREV/'trajectory_0p025ms.npz')
        fine=helper.difference(c,b,2)
        coarse=load(PREV/'trajectory_0p05ms.npz')
        # Contraction uses the .05ms shared grid for both differences, no interpolation.
        cd=helper.difference(coarse,c,2)
        fd=helper.difference({k:v[::2] for k,v in c.items()},b,4)
        contraction={k:fd[k]['rms']/cd[k]['rms'] if cd[k]['rms'] else None for k in ('qpos','qvel','root','com','feet','ctrl','filtered_load')}
        dump(out/'TRAJECTORY_DEVIATIONS.json',dict(pair_0p025_vs_0p0125_on_0p025_grid=fine,coarse_difference_on_0p05_grid=cd,fine_difference_on_0p05_grid=fd,fine_over_coarse_rms_on_0p05_grid=contraction,endpoint='states include endpoint, ctrl excludes held endpoint; no interpolation'))
        scaling={}
        for definition in ('abs_force','signed_force','normal_force','candidate','geometry'):
            scaling[definition]={}
            for leg in helper.LEGS:
                prior=summaries['0p025ms']['contacts'][definition][leg];new=summary['contacts'][definition][leg]
                scaling[definition][leg]=dict(onset_rate_ratio=new['onsets_per_s']/prior['onsets_per_s'],mean_complete_duration_ratio=new['mean_complete_duration']/prior['mean_complete_duration'],prior=prior,new=new)
        dump(out/'EVENT_SCALING.json',scaling)
        dump(out/'RUN_STATUS.json',dict(status='ONE_NEW_ARM_COMPLETED',steps=160000,duration_s=2.,dt=dt,elapsed_s=time.monotonic()-started,warnings_zero=True,finite=True,no_further_refinement=True))
        print('completed',json.dumps(summary['paths']), 'elapsed_s',time.monotonic()-started,flush=True)
        return 0
    except Exception:
        dump(out/'FAILURE.json',dict(traceback=traceback.format_exc(),elapsed_s=time.monotonic()-started));raise
    finally:manifest(out)

if __name__=='__main__':sys.exit(main())
