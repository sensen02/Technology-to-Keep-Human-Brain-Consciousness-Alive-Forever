"""S289 offline decoder-range verification. No simulation/neural execution.
Saved-data validation is not sustained-support or walking acceptance.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
REFERENCE=Path('/home/sensen/Desktop/engine修复验收/s288-sensory-gain-support/20261026T172200Z/gain_1')
RUNNER=ROOT/'run_s289_decoder_range_support.py'
ARMS=('decoder_original','decoder_range_x2')
LEGS=('lf','lm','lh','rf','rm','rh')
GAIN_FILE=Path('/home/sensen/Desktop/engine修复验收/s255-static-gain-can-it-sustain/20261025T221308522972Z/ARM_VECTORS.json')

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
    d=np.abs(a.astype(float)-b.astype(float));flat=d.reshape(len(d),-1);ix=np.flatnonzero(np.any(flat!=0,axis=1))
    return dict(max_abs=float(d.max()) if d.size else 0.,changed_values=int(np.count_nonzero(d)),first_diff_index=int(ix[0]) if ix.size else None,first_diff_time_ms=float(t[ix[0]]) if ix.size else None)
def window(tr,foot,non,angle,lo,hi):
    t=tr['time']*1000;mask=(t>=lo-1e-9)&(t<hi-1e-9)
    if hi<=lo or not mask.any():return dict(status='NOT_ESTABLISHED',start_ms=lo,end_exclusive_ms=hi,samples=0)
    return dict(start_ms=lo,end_exclusive_ms=hi,samples=int(mask.sum()),foot_Fz_mean_native=float(foot[mask,:,2].sum(axis=1).mean()),foot_Fz_per_leg_mean_native=foot[mask,:,2].mean(axis=0).tolist(),nonfoot_Fz_mean_native=float(non[mask,2].mean()),COM_z_mean_native=float(tr['com'][mask,2].mean()),COM_dz_mean_native=float((tr['com'][mask,2]-tr['com'][0,2]).mean()),COM_vz_mean_native=float(tr['com_velocity'][mask,2].mean()),root_dz_mean_native=float((tr['qpos'][mask,2]-tr['qpos'][0,2]).mean()),orientation_mean_rad=float(angle[mask].mean()),orientation_max_rad=float(angle[mask].max()),joint_limit_max_native=float(tr['max_limit'][mask].max()))
def dwell(mask):
    mask=np.asarray(mask,bool);edges=np.diff(np.r_[False,mask,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1);lengths=ends-starts
    return dict(total_dwell_ms=float(mask.sum()*.1),longest_contiguous_dwell_ms=float(lengths.max()*.1) if lengths.size else 0.,episodes=int(len(starts)),initial_contiguous_dwell_ms=float(ends[0]*.1) if starts.size and starts[0]==0 else 0.)
def timing(out,arm,a,check):
    z=npz(out/arm/'COMMAND_TIMING.npz');a['timing']=z;initial=z['initial_carry'];decoded=z['decoded'];applied=z['applied'];starts=np.arange(400)*.001;ends=(np.arange(400)+1)*.001
    check(arm+' timing shapes',initial.shape==(90,) and decoded.shape==(400,90) and applied.shape==(400,90))
    saved=npz(out/arm/'INITIAL_CARRY.npz');check(arm+' explicit zero initial carry',eq(initial,saved['command']) and eq(initial,a['tr']['ctrl'][0]) and np.array_equal(initial,np.zeros(90)));check(arm+' initial valid',float(saved['valid_from_s'])==0. and int(saved['source_interval'])==-1)
    check(arm+' source intervals',np.allclose(z['source_start_s'],starts,rtol=0,atol=1e-12) and np.allclose(z['source_end_s'],ends,rtol=0,atol=1e-12));check(arm+' valid from equals end',eq(z['valid_from_s'],z['source_end_s']))
    rebuilt=np.concatenate([initial[None,:],decoded[:-1]]);indices=np.arange(400)-1;valid=np.r_[0.,z['valid_from_s'][:-1]]
    check(arm+' actual causal source and commands',eq(rebuilt,applied) and np.array_equal(z['applied_source_index'],indices));check(arm+' actual interval times',np.allclose(z['applied_start_s'],starts,rtol=0,atol=1e-11) and np.allclose(z['applied_end_s'],ends,rtol=0,atol=1e-11));check(arm+' applied validity',np.allclose(z['applied_valid_from_s'],valid,rtol=0,atol=1e-12))
    expanded=np.repeat(rebuilt,10,axis=0);check(arm+' reconstructed4000 controls',eq(expanded,a['co']['ctrl']) and eq(expanded,a['ne']['ctrl']) and eq(expanded,a['tr']['ctrl'][1:]));viol=int(np.count_nonzero(z['applied_start_s']+1e-11<valid));check(arm+' no timing violations',viol==0)
    dec=js(out/arm/'DECODED_COMMANDS.json');app=js(out/arm/'COMMAND_APPLICATIONS.json');check(arm+' JSON command values',len(dec)==400 and len(app)==400 and np.array_equal(np.array([r['command'] for r in dec]),decoded) and np.array_equal(np.array([r['command'] for r in app]),applied));check(arm+' JSON validity/source',np.array_equal(np.array([r['valid_from_s'] for r in dec]),z['valid_from_s']) and np.array_equal(np.array([r['valid_from_s'] for r in app]),valid) and np.array_equal(np.array([r['source_interval'] for r in app]),indices))
    check(arm+' final399 pending not applied',eq(z['pending_final'],decoded[-1]) and indices[-1]==398 and np.all(z['applied_source_index']<399) and bool(z['pending_at_capture_end'][-1]) and np.count_nonzero(z['pending_at_capture_end'])==1 and np.isnan(z['actual_applied_start_s'][-1]) and np.isnan(z['actual_applied_end_s'][-1]) and dec[-1]['actual_applied_start_s'] is None and dec[-1]['actual_applied_end_s'] is None);check(arm+' final plan400to401',abs(z['planned_applied_start_s'][-1]-.4)<1e-12 and abs(z['planned_applied_end_s'][-1]-.401)<1e-12)
    check(arm+' decoded equals recorded postclip',eq(decoded,a['ne']['decoder_postclip']))
    return viol

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);out=ap.parse_args(argv).output;checks=[];metrics={};selfhash=sha(__file__)
    def check(name,value,**extra):checks.append(dict(name=name,passed=bool(value),**extra))
    try:
        pre=js(out/'PREREG.json');check('round S289',pre['round']=='S289');check('arms expected',pre['arms']==list(ARMS));check('source hashes nonempty',bool(pre['source_hashes']))
        for p,h in pre['source_hashes'].items():check('source hash '+p,Path(p).is_file() and sha(p)==h)
        check('runner preregistered',str(RUNNER) in pre['source_hashes'] and sha(RUNNER)==pre['source_hashes'].get(str(RUNNER)));check('motor gain vector preregistered',str(GAIN_FILE) in pre['source_hashes'] and sha(GAIN_FILE)==pre['source_hashes'].get(str(GAIN_FILE)))
        complete=js(out/'COMPLETE.json');ledger=js(out/'CALL_LEDGER.json');expected=dict(model_load=2,tier_build=1,network_init=2,live_step=8000,neural_tick=1600,live_fwd_position=2,live_forward=0,diagnostic_forward=8002);check('complete',complete.get('capture_complete') is True and complete.get('source_unchanged') is True);check('no failed marker',not(out/'FAILED.json').exists());check('experiment ledger',ledger==expected and complete.get('calls')==ledger)
        tree=ast.parse(RUNNER.read_text());advance=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='advance'];check('actual advance submitted',any(len(n.args)>=2 and isinstance(n.args[0],ast.Constant) and n.args[0].value==2 and isinstance(n.args[1],ast.Name) and n.args[1].id in ('submitted','submitted_current') for n in advance))
        good=False
        for fn in [n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='tick']:
            captures=[n for n in ast.walk(fn) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id in ('a','args') and isinstance(n.slice,ast.Constant) and n.slice.value==0];calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='step' and any(isinstance(v,ast.Starred) and isinstance(v.value,ast.Name) and v.value.id in ('a','args') for v in n.args)];appends=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='actual_step_currents' and n.func.attr=='append' and n.args and isinstance(n.args[0],ast.Name) and n.args[0].id=='actual_drive'];good |= bool(captures and len(calls)==1 and appends and max(n.lineno for n in captures+appends)<calls[0].lineno)
        check('capture actual argument before same original call',good)
        encpath=ROOT/'engine/embodied/sensory_channels.py';enc_tree=ast.parse(encpath.read_text());encode=[n for n in ast.walk(enc_tree) if isinstance(n,ast.FunctionDef) and n.name=='encode'];check('encoder source preregistered',str(encpath) in pre['source_hashes']);check('encoder has no clip/min/max call',len(encode)==1 and not any(isinstance(n,ast.Call) and ((isinstance(n.func,ast.Attribute) and n.func.attr in ('clip','maximum','minimum')) or (isinstance(n.func,ast.Name) and n.func.id in ('min','max'))) for n in ast.walk(encode[0])))
        motorpath=ROOT/'engine/embodied/motor_output.py';motor=motorpath.read_text();check('output clip declared','np.clip(rate, 0, self.config.max_control)' in motor);gains=np.asarray(js(GAIN_FILE)['v1'],float);check('90 fixed motor gains and26 zero masks',gains.shape==(90,) and np.count_nonzero(gains==0)==26)
        data={}
        for arm,denom in zip(ARMS,(450.,900.)):
            dest=out/arm;tr=npz(dest/'TRAJECTORY.npz');ne=npz(dest/'NEURAL.npz');co=npz(dest/'CTRL.npz');meta=js(dest/'MODEL_METADATA.json');a=dict(tr=tr,ne=ne,co=co,meta=meta,params=js(dest/'PARAMETERS.json'));data[arm]=a;t=tr['time'];check(arm+' trajectory shapes',tr['qpos'].shape==(4001,49) and tr['qvel'].shape==(4001,48));check(arm+' control/rate shapes',co['ctrl'].shape==(4000,90) and ne['rates_hz'].shape==(400,90) and ne['raw_rates_hz'].shape==(400,2,90));check(arm+' timestamps',np.allclose(t,np.arange(4001)*.0001,rtol=0,atol=1e-11) and np.array_equal(ne['time_ms'],np.arange(1,801)*.5));check(arm+' control agreements',eq(co['ctrl'],ne['ctrl']) and eq(co['ctrl'],tr['ctrl'][1:]));check(arm+' bounds/names',np.all((co['ctrl']>=0)&(co['ctrl']<=1)) and co['names'].tolist()==meta['names'] and len(meta['names'])==90);check(arm+' no external load',not np.any(tr['external_applied']) and not np.any(tr['qfrc_applied']));check(arm+' finite trajectory/neural',all(np.isfinite(v).all() for z in (tr,ne) for v in z.values() if v.dtype.kind in 'biufc'))
            enc=ne['encoded_current'];sub=ne['submitted_current'];bg=ne['background_current'];actual=ne['actual_step_current'];n=enc.shape[1];check(arm+' input shapes',enc.shape==(400,n) and sub.shape==(400,n) and bg.shape==(n,) and actual.shape==(800,n));check(arm+' real input times',np.array_equal(ne['sensory_time_ms'],np.arange(400,dtype=float)) and np.array_equal(ne['actual_step_time_ms'],ne['time_ms']));check(arm+' signed sensory gain1',eq(sub,enc) and np.any(enc<0) and np.any(enc>0));check(arm+' background nonzero',bool(np.any(bg!=0)));check(arm+' real per-tick input',eq(actual,np.broadcast_to(bg,(800,n))+np.repeat(enc,2,axis=0)));check(arm+' legacy sensory field encoded',eq(ne['sensory_current'],enc))
            check(arm+' decoder added shapes',ne['decoder_preclip'].shape==(400,90) and ne['decoder_postclip'].shape==(400,90) and ne['decoder_motor_gains'].shape==(90,));check(arm+' decoder scalars',ne['decoder_rate_for_full_control_hz'].shape==() and float(ne['decoder_rate_for_full_control_hz'])==denom and ne['filter_tau_s'].shape==() and float(ne['filter_tau_s'])==.005 and ne['decoder_max_control'].shape==() and float(ne['decoder_max_control'])==1.)
            check(arm+' fixed gain vector exact',eq(ne['decoder_motor_gains'],gains))
            decay=math.exp(-.0005/.005);state=np.zeros(90);filtered=[]
            for interval in ne['raw_rates_hz']:
                for sample in interval:state=decay*state+(1-decay)*sample
                filtered.append(state.copy())
            filtered=np.asarray(filtered);preclip=(filtered/denom)*gains;postclip=np.clip(preclip,0.,1.)
            check(arm+' independent per-neural-tick recursive5ms filter',np.allclose(filtered,ne['rates_hz'],rtol=1e-13,atol=1e-11),max_abs=float(abs(filtered-ne['rates_hz']).max()))
            check(arm+' recorded preclip own rates denominator gains',np.allclose(ne['decoder_preclip'],(ne['rates_hz']/denom)*gains,rtol=1e-13,atol=1e-13))
            check(arm+' independent filter to preclip',np.allclose(ne['decoder_preclip'],preclip,rtol=1e-13,atol=1e-12));check(arm+' independent filter to postclip',np.allclose(ne['decoder_postclip'],postclip,rtol=1e-13,atol=1e-12));check(arm+' actual postclip exact own preclip',eq(ne['decoder_postclip'],np.clip(ne['decoder_preclip'],0.,1.)))
            check(arm+'26 decoder zero masks preserved',not np.any(ne['decoder_preclip'][:,gains==0]) and not np.any(ne['decoder_postclip'][:,gains==0]))
            raw=tr['contact_raw'];offs=tr['contact_offsets'];counts=tr['contact_count'];check(arm+' packed contact dimensions',raw.ndim==2 and raw.shape[1]==21 and offs.shape==(4002,) and np.array_equal(np.diff(offs),counts) and offs[-1]==len(raw));contacts=js(dest/'CONTACTS.json');check(arm+' JSON packed contact agreement',len(contacts)==4001 and all(np.array_equal(np.asarray(c,float).reshape(-1,21),raw[int(offs[i]):int(offs[i+1])]) for i,c in enumerate(contacts)));floor=int(meta['floor']);gl={int(k):v for k,v in meta['geom_leg'].items()};foot=np.zeros((4001,6,3));non=np.zeros((4001,3));pen=np.zeros(4001,int);first=None;margin=None;penetrate=None
            for i in range(4001):
                for c in raw[int(offs[i]):int(offs[i+1])]:
                    g1,g2=int(c[0]),int(c[1])
                    if floor not in (g1,g2):continue
                    other=g2 if g1==floor else g1;f=(1 if g1==floor else -1)*(c[6:15].reshape(3,3).T@c[15:18])
                    if other in gl:foot[i,LEGS.index(gl[other])]+=f
                    else:
                        non[i]+=f
                        if first is None:first=float(t[i]*1000)
                        if c[2]>=0 and margin is None:margin=float(t[i]*1000)
                        if c[2]<0:
                            pen[i]+=1
                            if penetrate is None:penetrate=float(t[i]*1000)
            check(arm+' physical foot reconstruction',np.allclose(foot,tr['foot_force_world'],rtol=0,atol=2e-12));check(arm+' body penetration reconstruction',np.array_equal(pen,tr['body_penetrating_contacts']));q=tr['qpos'][:,3:7];angle=2*np.arccos(np.clip(abs((q*q[0]).sum(axis=1)),0,1));times=ne['time_ms'];spiket=times[np.any(ne['spikes'],axis=1)]
            split=min(150.,max(0.,first)) if first is not None else 150.;early=dict(total=window(tr,foot,non,angle,0.,150.),before_first_nonfoot=window(tr,foot,non,angle,0.,split),after_first_nonfoot=window(tr,foot,non,angle,split,150.) if first is not None and first<150 else dict(status='NOT_ESTABLISHED',reason='no nonfoot contact in common0to150ms'))
            own=window(tr,foot,non,angle,max(0.,first-20.),first) if first is not None else dict(status='NOT_ESTABLISHED');own['truncated_at_initial_time']=bool(first is not None and first<20.)
            metrics[arm]=dict(rate_for_full_control_hz=denom,target_spikes_0_200=int(ne['spikes'][times<=200].sum()),target_spikes_200_400=int(ne['spikes'][times>200].sum()),target_last_spike_ms=float(spiket[-1]) if len(spiket) else None,COM_z400_native=float(tr['com'][-1,2]),COM_dz400_native=float(tr['com'][-1,2]-tr['com'][0,2]),COM_vz400_native=float(tr['com_velocity'][-1,2]),root_dz400_native=float(tr['qpos'][-1,2]-tr['qpos'][0,2]),first_nonfoot_contact_including_margin_ms=first,first_nonfoot_margin_dist_ge0_ms=margin,first_nonfoot_penetration_ms=penetrate,orientation_max_rad=float(angle.max()),joint_limit_max_native=float(tr['max_limit'].max()),common_0_150ms=early,own20ms_before_first_nonfoot_contact=own,late350_400ms=window(tr,foot,non,angle,350.,400.0000001),support_walking='NOT_ESTABLISHED',physical_calibration='NOT_ESTABLISHED',decoder_preclip_max=float(ne['decoder_preclip'].max()),decoder_filter_max_abs_error=float(abs(filtered-ne['rates_hz']).max()))
            a.update(foot=foot,nonfoot=non,angle=angle);metrics[arm]['timing_violations']=timing(out,arm,a,check)
            dw=[]
            for j in range(90):
                c=co['ctrl'][:,j];lo=dwell(c==0.);hi=dwell(c==1.);check(arm+f' channel{j} zero-gain mask consistent',gains[j]!=0 or not np.any(c));dw.append(dict(index=j,actuator=meta['names'][j],fixed_motor_gain=float(gains[j]),zero_motor_gain_mask=bool(gains[j]==0),lower_bound_0=lo,upper_bound_1=hi,interpretation='Dwell is descriptive. Initial carry zeros and zero-gain masks are not automatically pathological saturation.'))
            metrics[arm]['command_bound_dwell']=dw
        for k in ('qpos','qvel','ctrl','act'):check('initial identical '+k,eq(data[ARMS[0]]['tr'][k][0],data[ARMS[1]]['tr'][k][0]))
        check('neural parameters identical',data[ARMS[0]]['params']==data[ARMS[1]]['params']);check('background identical',eq(data[ARMS[0]]['ne']['background_current'],data[ARMS[1]]['ne']['background_current']));check('motor gains identical',eq(data[ARMS[0]]['ne']['decoder_motor_gains'],data[ARMS[1]]['ne']['decoder_motor_gains']))
        for filename,slot in [('TRAJECTORY.npz','tr'),('NEURAL.npz','ne'),('CTRL.npz','co'),('COMMAND_TIMING.npz','timing'),('INITIAL_CARRY.npz',None)]:
            ref=npz(REFERENCE/filename);got=data[ARMS[0]][slot] if slot else npz(out/ARMS[0]/filename);check('original historical fields '+filename,set(ref)<=set(got))
            for k,v in ref.items():check('original exact '+filename+'/'+k,k in got and eq(v,got[k]))
        for filename in ('STATE.csv','CHANNELS.json','CONTACTS.json','PARAMETERS.json','DECODED_COMMANDS.json','COMMAND_APPLICATIONS.json'):check('original exact '+filename,(REFERENCE/filename).read_bytes()==(out/ARMS[0]/filename).read_bytes())
        left,right=data[ARMS[0]],data[ARMS[1]];metrics['target_paired']={k:delta(left['ne'][k],right['ne'][k],left['ne']['time_ms']) for k in ('spikes','ge','gi','v')};metrics['paired_channels']=[]
        for j in range(90):metrics['paired_channels'].append(dict(index=j,actuator=left['meta']['names'][j],raw_rate=delta(left['ne']['raw_rates_hz'][:,:,j].reshape(800,1),right['ne']['raw_rates_hz'][:,:,j].reshape(800,1),left['ne']['time_ms']),decoded_rate=delta(left['ne']['rates_hz'][:,j:j+1],right['ne']['rates_hz'][:,j:j+1],np.arange(1,401)),preclip=delta(left['ne']['decoder_preclip'][:,j:j+1],right['ne']['decoder_preclip'][:,j:j+1],np.arange(1,401)),postclip=delta(left['ne']['decoder_postclip'][:,j:j+1],right['ne']['decoder_postclip'][:,j:j+1],np.arange(1,401)),decoded_command=delta(left['timing']['decoded'][:,j:j+1],right['timing']['decoded'][:,j:j+1],np.arange(1,401)),applied_command=delta(left['co']['ctrl'][:,j:j+1],right['co']['ctrl'][:,j:j+1],np.arange(4000)*.1)))
        metrics['comparison_scope']='Each arm reconstructed using its own raw rates and denominator. Candidate is NOT required to equal baseline half after feedback diverges.'
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(4,2,figsize=(14,14))
        for arm,a in data.items():
            tt=a['tr']['time']*1000;ne=a['ne'];ax[0,0].plot(tt,a['tr']['com'][:,2]-a['tr']['com'][0,2],label=arm);ax[0,1].plot(tt,a['angle'],label=arm);ax[1,0].plot(tt,a['foot'][:,:,2].sum(axis=1),label=arm+' feet');ax[1,0].plot(tt,a['nonfoot'][:,2],ls='--',label=arm+' nonfoot');ax[1,1].plot(ne['time_ms'],ne['spikes'].sum(axis=1),label=arm);ax[2,0].plot(np.arange(1,401),np.max(ne['decoder_preclip'],axis=1),label=arm);ax[2,1].plot(np.arange(1,401),np.max(ne['decoder_postclip'],axis=1),label=arm)
            dw=metrics[arm]['command_bound_dwell'];ax[3,1].plot(np.arange(90),[d['upper_bound_1']['total_dwell_ms'] for d in dw],label=arm+' upper1');ax[3,1].plot(np.arange(90),[d['lower_bound_0']['total_dwell_ms'] for d in dw],ls='--',label=arm+' lower0')
        ax[3,0].plot(np.arange(4000)*.1,np.max(abs(left['co']['ctrl']-right['co']['ctrl']),axis=1),label='max across90');titles=['COM dz native','Orientation relative initial rad','Physical Fz native','Target-only spikes','Decoder preclip max across90','Decoder postclip max across90','Applied command difference','Bound dwell ms (not pathological by definition)']
        for aa,title in zip(ax.flat,titles):aa.set_title(title);aa.grid(alpha=.3);aa.legend(fontsize=7);aa.set_xlabel('individual clock ms')
        ax[3,1].set_xlabel('actuator index');fig.suptitle('S289 decoder range pair: offline verification, no support/walking acceptance');fig.tight_layout();fig.savefig(out/'OUTCOMES.png',dpi=140);plt.close(fig)
        check('no simulation/network imports',not any(k=='mujoco' or k.startswith('engine.') for k in sys.modules));check('verifier unchanged',sha(__file__)==selfhash);result=dict(passed=all(c['passed'] for c in checks),checks=checks,failed=[c['name'] for c in checks if not c['passed']],verifier_path=str(Path(__file__).resolve()),verifier_sha256=selfhash,offline_calls=dict(model_load=0,forward=0,physics_step=0,neural_tick=0),scope='saved decoder/current/command/physical checks, not behavior acceptance')
        (out/'METRICS.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,default=serial)+'\n');(out/'VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=serial)+'\n');print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=result['failed'])))
        return 0 if result['passed'] else 1
    except Exception as e:print('offline verifier error: '+repr(e),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
