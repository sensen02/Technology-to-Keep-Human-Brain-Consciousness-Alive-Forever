"""S287 independent saved-data verifier. No MuJoCo or network imports/execution.
Causal command timing, actual submitted currents, archive identity and physical
outcomes are checked separately; no support/walking success is inferred.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

REFERENCE=Path('/home/sensen/Desktop/engine修复验收/s286-command-timing-support/20261026T170200Z/causal_carry')
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
RUNNER=ROOT/'run_s287_sensory_causal_ablation.py'
ARMS=('live_sensory','sensory_zero')
LEGS=('lf','lm','lh','rf','rm','rh')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(str(type(v)))
def eq(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype:return False
    return np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b)
def delta(a,b,t):
    d=abs(a.astype(float)-b.astype(float));flat=d.reshape(len(d),-1);idx=np.flatnonzero(np.any(flat!=0,axis=1))
    return dict(max_abs=float(d.max()) if d.size else 0.,changed_values=int(np.count_nonzero(d)),first_diff_index=int(idx[0]) if idx.size else None,first_diff_time_ms=float(t[idx[0]]) if idx.size else None)
def window(tr,foot,nonfoot,angle,lo,hi):
    t=tr['time']*1000;mask=(t>=lo-1e-9)&(t<hi-1e-9)
    if not mask.any():return {'status':'NOT_ESTABLISHED','requested_start_ms':lo,'requested_end_ms':hi}
    return dict(start_ms=lo,end_exclusive_ms=hi,samples=int(mask.sum()),foot_Fz_mean_native=float(foot[mask,:,2].sum(axis=1).mean()),foot_Fz_per_leg_mean_native=foot[mask,:,2].mean(axis=0).tolist(),nonfoot_Fz_mean_native=float(nonfoot[mask,2].mean()),COM_dz_mean_native=float((tr['com'][mask,2]-tr['com'][0,2]).mean()),COM_vz_mean_native=float(tr['com_velocity'][mask,2].mean()),root_dz_mean_native=float((tr['qpos'][mask,2]-tr['qpos'][0,2]).mean()),orientation_mean_rad=float(angle[mask].mean()),orientation_max_rad=float(angle[mask].max()),joint_limit_max_native=float(tr['max_limit'][mask].max()))

def verify_timing(out,arm,a,check,metrics):
    z=npz(out/arm/'COMMAND_TIMING.npz');a['timing']=z
    req={'initial_carry','decoded','source_start_s','source_end_s','valid_from_s','applied','applied_start_s','applied_end_s','applied_valid_from_s','applied_source_index','pending_final'}
    check(arm+' timing complete',req<=set(z));initial=z['initial_carry'];decoded=z['decoded'];applied=z['applied'];starts=np.arange(400)*.001;ends=(np.arange(400)+1)*.001
    check(arm+' timing dimensions',initial.shape==(90,) and decoded.shape==(400,90) and applied.shape==(400,90))
    initial_saved=npz(out/arm/'INITIAL_CARRY.npz')
    check(arm+' initial control explicit and equal',eq(initial,a['tr']['ctrl'][0]) and eq(initial,initial_saved['command']) and np.array_equal(initial,np.zeros(90)))
    check(arm+' initial valid',float(initial_saved['valid_from_s'])==0. and int(initial_saved['source_interval'])==-1)
    check(arm+' neural interval contiguous',np.allclose(z['source_start_s'],starts,rtol=0,atol=1e-12) and np.allclose(z['source_end_s'],ends,rtol=0,atol=1e-12))
    check(arm+' valid_from equals source end',eq(z['valid_from_s'],z['source_end_s']))
    check(arm+' physical intervals',np.allclose(z['applied_start_s'],starts,rtol=0,atol=1e-11) and np.allclose(z['applied_end_s'],ends,rtol=0,atol=1e-11))
    rebuilt=np.concatenate([initial[None,:],decoded[:-1]]);indices=np.arange(400)-1;valid=np.concatenate([[0.],z['valid_from_s'][:-1]])
    check(arm+' causal commands reconstructed',eq(applied,rebuilt))
    check(arm+' causal source identities',np.array_equal(z['applied_source_index'],indices))
    check(arm+' applied validity reconstructed',np.allclose(z['applied_valid_from_s'],valid,rtol=0,atol=1e-12))
    expanded=np.repeat(rebuilt,10,axis=0)
    check(arm+' all4000 actual controls reconstructed',eq(expanded,a['co']['ctrl']) and eq(expanded,a['ne']['ctrl']) and eq(expanded,a['tr']['ctrl'][1:]))
    violations=int(np.count_nonzero(z['applied_start_s']+1e-11<valid));check(arm+' no anticipatory commands',violations==0)
    decjs=js(out/arm/'DECODED_COMMANDS.json');appjs=js(out/arm/'COMMAND_APPLICATIONS.json')
    check(arm+' JSON commands',len(decjs)==400 and len(appjs)==400 and np.array_equal(np.array([r['command'] for r in decjs]),decoded) and np.array_equal(np.array([r['command'] for r in appjs]),applied))
    check(arm+' JSON times and source',np.array_equal(np.array([r['valid_from_s'] for r in decjs]),z['valid_from_s']) and np.array_equal(np.array([r['valid_from_s'] for r in appjs]),z['applied_valid_from_s']) and np.array_equal(np.array([r['source_interval'] for r in appjs]),indices))
    check(arm+' pending final source399 preserved',eq(z['pending_final'],decoded[-1]) and indices[-1]==398 and np.all(z['applied_source_index']<399))
    check(arm+' final not applied',bool(z['pending_at_capture_end'][-1]) and np.count_nonzero(z['pending_at_capture_end'])==1 and np.isnan(z['actual_applied_start_s'][-1]) and np.isnan(z['actual_applied_end_s'][-1]) and decjs[-1]['actual_applied_start_s'] is None and decjs[-1]['actual_applied_end_s'] is None)
    check(arm+' pending plan400to401',abs(z['planned_applied_start_s'][-1]-.4)<1e-12 and abs(z['planned_applied_end_s'][-1]-.401)<1e-12)
    metrics[arm]['timing_violations']=violations

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);out=ap.parse_args(argv).output
    checks=[];metrics={};selfhash=sha(__file__)
    def check(name,value,**extra):checks.append(dict(name=name,passed=bool(value),**extra))
    try:
        pre=js(out/'PREREG.json');check('round S287',pre['round']=='S287');check('arms expected',pre['arms']==list(ARMS))
        for p,h in pre['source_hashes'].items():check('source hash '+p,Path(p).is_file() and sha(p)==h)
        check('source hashes nonempty',bool(pre['source_hashes']));check('actual runner hash preregistered',str(RUNNER) in pre['source_hashes'] and sha(RUNNER)==pre['source_hashes'].get(str(RUNNER)))
        complete=js(out/'COMPLETE.json');ledger=js(out/'CALL_LEDGER.json');expected=dict(model_load=2,tier_build=1,network_init=2,live_step=8000,neural_tick=1600,live_fwd_position=2,live_forward=0,diagnostic_forward=8002)
        check('complete',complete.get('capture_complete') is True and complete.get('source_unchanged') is True);check('no failure marker',not(out/'FAILED.json').exists());check('exact simulation calls',ledger==expected);check('complete ledger',complete.get('calls')==ledger)
        # Static structure checks are independent of logged flags, never exec/import runner.
        source=RUNNER.read_text();tree=ast.parse(source)
        advance=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='advance']
        check('actual advance gets submitted vector',any(len(n.args)>=2 and isinstance(n.args[0],ast.Constant) and n.args[0].value==2 and isinstance(n.args[1],ast.Name) and n.args[1].id in ('submitted','submitted_current') for n in advance))
        wrappers=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='tick']
        wrapper_ok=False
        for fn in wrappers:
            captures=[n for n in ast.walk(fn) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id in ('a','args') and isinstance(n.slice,ast.Constant) and n.slice.value==0]
            forwards=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='step' and any(isinstance(v,ast.Starred) and isinstance(v.value,ast.Name) and v.value.id in ('a','args') for v in n.args)]
            appends=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='actual_step_currents' and n.func.attr=='append' and n.args and isinstance(n.args[0],ast.Name) and n.args[0].id=='actual_drive']
            wrapper_ok |= bool(captures and len(forwards)==1 and appends and max(n.lineno for n in captures+appends)<forwards[0].lineno)
        check('wrapper captures actual positional argument before same-argument original step',wrapper_ok)
        data={}
        for arm in ARMS:
            dest=out/arm;tr=npz(dest/'TRAJECTORY.npz');ne=npz(dest/'NEURAL.npz');co=npz(dest/'CTRL.npz');meta=js(dest/'MODEL_METADATA.json');data[arm]=dict(tr=tr,ne=ne,co=co,meta=meta,params=js(dest/'PARAMETERS.json'));t=tr['time']
            check(arm+' body dimensions',tr['qpos'].shape==(4001,49) and tr['qvel'].shape==(4001,48));check(arm+' control dimensions',co['ctrl'].shape==(4000,90) and ne['ctrl'].shape==(4000,90));check(arm+' neural dimensions',ne['spikes'].shape[0]==800 and ne['rates_hz'].shape==(400,90) and ne['raw_rates_hz'].shape==(400,2,90))
            check(arm+' timestamps',np.allclose(t,np.arange(4001)*.0001,rtol=0,atol=1e-11) and np.array_equal(ne['time_ms'],np.arange(1,801)*.5));check(arm+' actual controls agree',eq(co['ctrl'],ne['ctrl']) and eq(co['ctrl'],tr['ctrl'][1:]));check(arm+' controls bounded',np.all((co['ctrl']>=0)&(co['ctrl']<=1)));check(arm+' names',co['names'].tolist()==meta['names'] and len(meta['names'])==90);check(arm+' no external load',not np.any(tr['external_applied']) and not np.any(tr['qfrc_applied']));check(arm+' finite data',all(np.isfinite(v).all() for z in (tr,ne) for v in z.values() if v.dtype.kind in 'biufc'))
            enc=ne['encoded_current'];sub=ne['submitted_current'];bg=ne['background_current'];actual=ne['actual_step_current'];n=enc.shape[1]
            check(arm+' encoded/submitted shapes',enc.shape==(400,n) and sub.shape==(400,n));check(arm+' actual step input shape',actual.shape==(800,n));check(arm+' background shape',bg.shape in ((n,),(400,n),(800,n)))
            check(arm+' sensory and actual-step timestamps',np.array_equal(ne['sensory_time_ms'],np.arange(400,dtype=float)) and np.array_equal(ne['actual_step_time_ms'],ne['time_ms']))
            bgrepeat=np.broadcast_to(bg,(800,n)) if bg.shape==(n,) else np.repeat(bg,2,axis=0) if bg.shape==(400,n) else bg
            expected_current=bgrepeat+np.repeat(sub,2,axis=0)
            check(arm+' actual net.step argument equals background plus submitted',eq(actual,expected_current));check(arm+' background nonzero',bool(np.any(bg!=0)));check(arm+' encoded nonzero',bool(np.any(enc!=0)))
            if arm=='live_sensory':check('live submitted equals encoded',eq(sub,enc))
            else:check('zero submitted identically zero',not np.any(sub))
            if 'sensory_current' in ne:check(arm+' legacy sensory record is encoded',eq(ne['sensory_current'],enc))
            raw=tr['contact_raw'];offs=tr['contact_offsets'];counts=tr['contact_count'];check(arm+' packed contacts',raw.ndim==2 and raw.shape[1]==21 and offs.shape==(4002,) and np.array_equal(np.diff(offs),counts) and offs[-1]==len(raw));contact=js(dest/'CONTACTS.json');check(arm+' JSON contact agreement',len(contact)==4001 and all(np.array_equal(np.asarray(c,float).reshape(-1,21),raw[int(offs[i]):int(offs[i+1])]) for i,c in enumerate(contact)))
            floor=int(meta['floor']);gl={int(k):v for k,v in meta['geom_leg'].items()};foot=np.zeros((4001,6,3));non=np.zeros((4001,3));pen=np.zeros(4001,int);first=None;margin=None;penetrate=None
            for i in range(4001):
                for c in raw[int(offs[i]):int(offs[i+1])]:
                    g1,g2=int(c[0]),int(c[1])
                    if floor not in (g1,g2):continue
                    other=g2 if g1==floor else g1;force=(1 if g1==floor else -1)*(c[6:15].reshape(3,3).T@c[15:18])
                    if other in gl:foot[i,LEGS.index(gl[other])]+=force
                    else:
                        non[i]+=force
                        if first is None:first=float(t[i]*1000)
                        if c[2]>=0 and margin is None:margin=float(t[i]*1000)
                        if c[2]<0:
                            pen[i]+=1
                            if penetrate is None:penetrate=float(t[i]*1000)
            check(arm+' independent physical foot forces',np.allclose(foot,tr['foot_force_world'],rtol=0,atol=2e-12));check(arm+' independent body penetration',np.array_equal(pen,tr['body_penetrating_contacts']))
            q=tr['qpos'][:,3:7];angle=2*np.arccos(np.clip(abs((q*q[0]).sum(axis=1)),0,1));times=ne['time_ms'];spiket=times[np.any(ne['spikes'],axis=1)];late=t>=.35-1e-12
            metrics[arm]=dict(target_spikes_0_200=int(ne['spikes'][times<=200].sum()),target_spikes_200_400=int(ne['spikes'][times>200].sum()),target_last_spike_ms=float(spiket[-1]) if len(spiket) else None,COM_dz400_native=float(tr['com'][-1,2]-tr['com'][0,2]),root_dz400_native=float(tr['qpos'][-1,2]-tr['qpos'][0,2]),first_nonfoot_contact_including_margin_ms=first,first_nonfoot_margin_dist_ge0_ms=margin,first_nonfoot_penetration_ms=penetrate,orientation_max_rad=float(angle.max()),joint_limit_max_native=float(tr['max_limit'].max()),foot_Fz_350_400_mean_native=float(foot[late,:,2].sum(axis=1).mean()),nonfoot_Fz_350_400_mean_native=float(non[late,2].mean()),common_150_200ms=window(tr,foot,non,angle,150.,200.),own20ms_before_first_nonfoot_contact=window(tr,foot,non,angle,max(0.,first-20),first) if first is not None else {'status':'NOT_ESTABLISHED'},support_walking='NOT_ESTABLISHED',physical_calibration='NOT_ESTABLISHED')
            data[arm].update(foot=foot,nonfoot=non,angle=angle);verify_timing(out,arm,data[arm],check,metrics)
        common_end=min(metrics[a]['first_nonfoot_contact_including_margin_ms'] for a in ARMS)
        for a in ARMS:
            d=data[a];metrics[a]['common_before_either_nonfoot_contact']=window(d['tr'],d['foot'],d['nonfoot'],d['angle'],0.,common_end)
        for key in ('qpos','qvel','ctrl','act'):check('identical initial '+key,eq(data[ARMS[0]]['tr'][key][0],data[ARMS[1]]['tr'][key][0]))
        check('neural parameters identical',data[ARMS[0]]['params']==data[ARMS[1]]['params']);check('background identical',eq(data[ARMS[0]]['ne']['background_current'],data[ARMS[1]]['ne']['background_current']))
        # Gate independent of runner report: all reference fields, including NaNs in timing.
        for filename,slot in [('TRAJECTORY.npz','tr'),('NEURAL.npz','ne'),('CTRL.npz','co'),('COMMAND_TIMING.npz','timing'),('INITIAL_CARRY.npz',None)]:
            ref=npz(REFERENCE/filename);actual=data['live_sensory'][slot] if slot else npz(out/'live_sensory'/filename);check('live contains historical fields '+filename,set(ref)<=set(actual))
            for k,v in ref.items():check('live historical exact '+filename+'/'+k,k in actual and eq(v,actual[k]))
        for filename in ('STATE.csv','CHANNELS.json','CONTACTS.json','PARAMETERS.json','DECODED_COMMANDS.json','COMMAND_APPLICATIONS.json'):
            check('live historical exact '+filename,(REFERENCE/filename).read_bytes()==(out/'live_sensory'/filename).read_bytes())
        left,right=data[ARMS[0]],data[ARMS[1]];metrics['target_paired']={k:delta(left['ne'][k],right['ne'][k],left['ne']['time_ms']) for k in ('spikes','ge','gi','v')};metrics['paired_channels']=[]
        for j in range(90):
            metrics['paired_channels'].append(dict(index=j,actuator=left['meta']['names'][j],raw_rate=delta(left['ne']['raw_rates_hz'][:,:,j].reshape(800,1),right['ne']['raw_rates_hz'][:,:,j].reshape(800,1),left['ne']['time_ms']),decoded_rate=delta(left['ne']['rates_hz'][:,j:j+1],right['ne']['rates_hz'][:,j:j+1],np.arange(1,401)),decoded_command=delta(left['timing']['decoded'][:,j:j+1],right['timing']['decoded'][:,j:j+1],np.arange(1,401)),applied_command=delta(left['co']['ctrl'][:,j:j+1],right['co']['ctrl'][:,j:j+1],np.arange(4000)*.1)))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(4,2,figsize=(14,14))
        for arm,a in data.items():
            tr=a['tr'];tt=tr['time']*1000
            ax[0,0].plot(tt,tr['com'][:,2]-tr['com'][0,2],label=arm);ax[0,1].plot(tt,a['angle'],label=arm);ax[1,0].plot(tt,a['foot'][:,:,2].sum(axis=1),label=arm+' foot');ax[1,0].plot(tt,a['nonfoot'][:,2],ls='--',label=arm+' nonfoot');ax[1,1].plot(a['ne']['time_ms'],a['ne']['spikes'].sum(axis=1),label=arm);ax[2,0].plot(np.arange(400),np.max(abs(a['ne']['submitted_current']),axis=1),label=arm);ax[2,1].plot(a['ne']['time_ms'],np.max(abs(a['ne']['actual_step_current']),axis=1),label=arm)
        ax[3,0].plot(np.arange(4000)*.1,np.max(abs(left['co']['ctrl']-right['co']['ctrl']),axis=1),label='max across90');ax[3,1].plot(np.arange(1,401),np.max(abs(left['timing']['decoded']-right['timing']['decoded']),axis=1),label='max decoded across90')
        titles=['COM dz native','Orientation relative initial rad','Physical Fz native','Target-only spikes','Submitted sensory current nA','Actual net.step input nA','Applied command difference','Decoded command difference']
        for a,title in zip(ax.flat,titles):a.set_title(title);a.set_xlabel('individual clock ms');a.legend(fontsize=7);a.grid(alpha=.3)
        fig.suptitle('S287 sensory causal ablation: saved-data verification, not support/walking acceptance');fig.tight_layout();fig.savefig(out/'OUTCOMES.png',dpi=140);plt.close(fig)
        check('no simulation/network imports',not any(k=='mujoco' or k.startswith('engine.') for k in sys.modules));check('verifier unchanged',sha(__file__)==selfhash)
        result=dict(passed=all(c['passed'] for c in checks),checks=checks,failed=[c['name'] for c in checks if not c['passed']],verifier_path=str(Path(__file__).resolve()),verifier_sha256=selfhash,offline_calls=dict(model_load=0,forward=0,physics_step=0,neural_tick=0),scope='static actual-argument capture and archived identity/scheduling checks; behavior NOT_ESTABLISHED')
        (out/'METRICS.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,default=serial)+'\n');(out/'VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=serial)+'\n');print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=result['failed'])))
        return 0 if result['passed'] else 1
    except Exception as e:print('offline verifier error: '+repr(e),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
