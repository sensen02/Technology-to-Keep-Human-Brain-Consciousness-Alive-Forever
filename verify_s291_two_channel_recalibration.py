"""S291 independent offline verification; no model, tier or neural execution.
Original-mask identity, two-channel derivation, runtime mapping, own-arm decoding
and physics metrics are separate checks; no support/walking success is inferred.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
REFERENCE=Path('/home/sensen/Desktop/engine修复验收/s289-decoder-range-support/20261026T173400Z/decoder_original')
RUNNER=ROOT/'run_s291_two_channel_recalibration.py'
ARMS=('original_mask','recalibrated_55_62')
GAIN_FILE=Path('/home/sensen/Desktop/engine修复验收/s255-static-gain-can-it-sustain/20261025T221308522972Z/ARM_VECTORS.json')
POOLS=ROOT/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json'
XML=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
LEGS=('lf','lm','lh','rf','rm','rh')
spec=importlib.util.spec_from_file_location('s291_offline_utilities',ROOT/'verify_s289_decoder_range_support.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
sha,js,npz,eq,delta,window,dwell,timing,serial=b.sha,b.js,b.npz,b.eq,b.delta,b.window,b.dwell,b.timing,b.serial

def xml_names(path,stack=()):
    path=Path(path).resolve()
    if path in stack:raise ValueError('XML include cycle')
    root=ET.fromstring(path.read_bytes())
    def visit(node):
        for child in list(node):
            if child.tag=='include':
                sub=parse(path.parent/child.attrib['file'],(*stack,path));i=list(node).index(child);node.remove(child)
                for off,e in enumerate(list(sub)):node.insert(i+off,e)
            else:visit(child)
    def parse(p,s):
        p=Path(p).resolve()
        if p in s:raise ValueError('XML include cycle')
        r=ET.fromstring(p.read_bytes())
        def expand(n):
            for c in list(n):
                if c.tag=='include':
                    rr=parse(p.parent/c.attrib['file'],(*s,p));i=list(n).index(c);n.remove(c)
                    for k,e in enumerate(list(rr)):n.insert(i+k,e)
                else:expand(c)
        expand(r);return r
    visit(root);return [e.attrib['name'] for section in root.findall('./actuator') for e in section]

def verify_mapping(dest,meta,n,check):
    """Runtime roots/manifest/compiled identities must be archived, not inferred."""
    roots_record=npz(dest.parent/'TIER_ROOT_IDS.npz');manifest=js(dest.parent/'MOTOR_MANIFEST.json');compiled=js(dest/'COMPILED_MAPPING.json')
    roots=roots_record['root_ids'];names=meta['names'];declared=xml_names(XML)
    check(dest.name+' selected roots unique dimension',roots.shape==(n,) and roots.dtype.kind in 'iu' and len(set(map(int,roots)))==n)
    check(dest.name+' XML names ordered90',len(names)==90 and len(set(names))==90 and names==declared)
    check(dest.name+' compiled names ordered90',compiled['passed'] is True and compiled['actuator_names']==names and compiled['asset_names']==names and compiled['xml_declaration_names']==declared and compiled['xml_sha256']==sha(XML))
    at=compiled['calls_at_check'];prior=0 if dest.name==ARMS[0] else 1
    check(dest.name+' compiled check before own dynamics',at['model_load']==prior+1 and at['network_init']==prior and at['live_step']==prior*4000 and at['neural_tick']==prior*800 and at['live_fwd_position']==prior)
    rows=manifest['rows'];check(dest.name+' runtime manifest dimensions',manifest['n_neurons']==n and manifest['n_actuators']==90 and len(rows)==90)
    source=js(POOLS)['measured'];resolved={r['actuator']:r for r in source['resolved']};unresolved={r['actuator']:r for r in source['unresolved']};local={int(r):i for i,r in enumerate(roots)}
    check(dest.name+' source pool complete names',set(resolved)|set(unresolved)==set(names) and not(set(resolved)&set(unresolved)))
    for i,(name,row) in enumerate(zip(names,rows)):
        present=sorted(set(map(int,resolved[name]['pool_root_ids']))) if name in resolved else []
        expected=[r for r in present if r in local];ids=[local[r] for r in expected]
        check(dest.name+f' mapping{i} exact source to local',row['actuator']==name and row['actuator_index']==i and row['pool_root_ids']==present and row.get('tier_root_ids',[])==expected and row['local_indices']==ids)
        check(dest.name+f' mapping{i} unique boundaries',len(set(row['local_indices']))==len(row['local_indices']) and all(isinstance(v,int) and not isinstance(v,bool) and 0<=v<n for v in row['local_indices']))
        if i in (55,62):check(dest.name+f' mapping{i} existing nonempty pool',bool(present) and bool(expected) and all(int(roots[j])==r for j,r in zip(ids,expected)))
    return dict(roots=roots,manifest=manifest,compiled=compiled)

def verify_derivation(out,old,check):
    # Exact schema is finalized against the independently created derivation asset.
    derivation=js(out/'DERIVATION.json')
    raise NotImplementedError('Unexecuted full-pair verification disabled: no accepted S291 derivation; refuse guessing sample or masked denominator')

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);out=ap.parse_args(argv).output
    checks=[];metrics={};selfhash=sha(__file__)
    def check(name,value,**extra):checks.append(dict(name=name,passed=bool(value),**extra))
    try:
        pre=js(out/'PREREG.json')
        if (out/'STOP.json').exists():
            stop=js(out/'STOP.json');ledger=js(out/'CALL_LEDGER.json')
            check('stopped S291 preregistered',pre.get('round')=='S291')
            check('stop explicitly no candidate',stop.get('candidate_executed') is False)
            check('all dynamic calls zero',bool(ledger) and all(isinstance(v,int) and v==0 for v in ledger.values()))
            check('stop ledger same',stop.get('calls')==ledger)
            check('no dynamic artifacts',not any((out/arm).exists() for arm in ARMS) and not any(out.rglob('TRAJECTORY.npz')) and not (out/'TIER_ROOT_IDS.npz').exists() and not (out/'MOTOR_MANIFEST.json').exists())
            check('not capture completed',not (out/'COMPLETE.json').exists())
            registered=pre.get('source_hashes',pre.get('input_hashes',{}))
            check('stop sources registered',bool(registered))
            for p,h in registered.items():check('stop source '+p,Path(p).is_file() and sha(p)==h)
            dpath=Path(pre['derivation']);d=js(dpath)
            check('derivation refuses unresolved semantics',d.get('passed') is False and pre.get('derivation_passed') is False)
            check('derivation registered hash',sha(dpath)==pre.get('derivation_sha256'))
            check('derivation copied verbatim content',js(out/'DERIVATION.json')==d)
            check('candidate values not invented',d.get('candidate_gains_authorized') is False and d.get('gain_new') is None and pre.get('new_gains') is None and all(r.get('gain_new') is None for r in d['per_channel']))
            for p,h in d['sources_sha256'].items():check('derivation evidence '+p,Path(p).is_file() and sha(p)==h)
            historical=Path('/home/sensen/Desktop/engine修复验收/s254-per-row-gain/20261025T220602503017Z/CTRL_A0_gate_default.npz')
            with np.load(historical,allow_pickle=False) as z:
                check('historical ctrl only no timestamps or filter state',set(z.files)=={'ctrl','names','time_base'})
                hctrl=z['ctrl'].copy();time_base=str(z['time_base'])
            check('historical shape and ordinal time base only',hctrl.shape==(200,90) and time_base=='per_ms')
            other=Path('/home/sensen/Desktop/engine修复验收/s253-suppression-only/20261025T220043907094Z/CTRL_N0_gate_default.npz')
            check('historical A0 N0 byte identical',sha(historical)==sha(other))
            with np.load('/home/sensen/Desktop/engine修复验收/s244-chirality-of-the-5-of-6-arm/20261025T205824474993Z/CMD_A3_sym_matched.npz',allow_pickle=True) as z:target=z['cmd'].copy();names=z['names'].tolist()
            if target.ndim!=1:raise ValueError('Replay target semantics unexpected; refuse guessing')
            g254=np.asarray(js('/home/sensen/Desktop/engine修复验收/s254-per-row-gain/20261025T220602503017Z/ARM_VECTORS.json')['perrow_apex'],float)
            unit=hctrl[17];rebuilt=np.zeros(90);m=(target>0)&(unit>0);rebuilt[m]=target[m]/unit[m]
            old=np.asarray(js(GAIN_FILE)['v1'],float)
            check('old calibration divides saved postclip not rates',eq(rebuilt,g254) and eq(old,g254*1.236819199))
            current=npz(REFERENCE/'NEURAL.npz');clock=npz(REFERENCE/'COMMAND_TIMING.npz')
            check('current ordinal17 known but not historical match',abs(clock['source_start_s'][17]-.017)<1e-12 and abs(clock['source_end_s'][17]-.018)<1e-12 and abs(clock['valid_from_s'][17]-.018)<1e-12 and clock['applied_source_index'][18]==17)
            check('only target channels discussed',[r['index'] for r in d['per_channel']]==[55,62])
            for row in d['per_channel']:
                i=row['index'];rate=float(current['rates_hz'][17,i]);preclip=rate/450.;postclip=float(np.clip(preclip,0,1))
                check(f'channel{i} current positive not substituting masked control',target[i]>0 and unit[i]==0 and old[i]==0 and rate>0 and not np.any(current['decoder_postclip'][:,i]) and row['actuator']==names[i] and row['historical_postclip_ctrl17']==0)
                check(f'channel{i} recorded ordinal facts',row['target']==float(target[i]) and row['current_ordinal17_filtered_rate_hz']==rate and row['current_ordinal17_unitgain_preclip']==preclip and row['current_ordinal17_unitgain_postclip']==postclip)
            check('missing historical semantics explicitly declared',d['status']=='STOP_HISTORICAL_SAMPLE_SEMANTICS_NOT_ESTABLISHED' and bool(d['unestablished_prerequisites']))
            check('no model/network imports',not any(k=='mujoco' or k.startswith('engine.') for k in sys.modules))
            result=dict(passed=all(x['passed'] for x in checks),mode='verified_prerequisite_stop_not_behavioral_completion',checks=checks,failed=[x for x in checks if not x['passed']],verifier_sha256=selfhash,verifier_path=str(Path(__file__).resolve()),offline_calls=dict(model_load=0,forward=0,physics_step=0,neural_tick=0,tier_build=0))
            (out/'VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=serial)+'\n');print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=result['failed'])));return 0 if result['passed'] else 1
        check('S291 declared arms',pre['round']=='S291' and pre['arms']==list(ARMS))
        check('nonempty registered dependencies',bool(pre['source_hashes']))
        for dependency in (RUNNER,XML,POOLS,GAIN_FILE):check('preregistered '+str(dependency),str(dependency) in pre['source_hashes'])
        for p,h in pre['source_hashes'].items():check('registered input '+p,Path(p).is_file() and sha(p)==h)
        old=np.asarray(js(GAIN_FILE)['v1'],float);new,derivation=verify_derivation(out,old,check)
        changed=np.flatnonzero(new!=old);check('ONLY gain55 and62 changed',changed.tolist()==[55,62] and np.all(old[[55,62]]==0) and np.all(new[[55,62]]>0) and np.all(np.isfinite(new)))
        remain=(old==0);remain[[55,62]]=False;check('remaining24 zero masks',remain.sum()==24 and np.all(new[remain]==0))
        complete=js(out/'COMPLETE.json');ledger=js(out/'CALL_LEDGER.json');expected=dict(model_load=2,tier_build=1,network_init=2,live_step=8000,neural_tick=1600,live_fwd_position=2,live_forward=0,diagnostic_forward=8002)
        check('completed pair calls',ledger==expected and complete.get('capture_complete') is True and complete.get('source_unchanged') is True and complete.get('calls')==ledger);check('no failed marker',not(out/'FAILED.json').exists())
        code=ast.parse(RUNNER.read_text());check('actual submitted argument used',any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='advance' and len(x.args)==2 and isinstance(x.args[1],ast.Name) and x.args[1].id=='submitted' for x in ast.walk(code)))
        capture=False
        for fn in [x for x in ast.walk(code) if isinstance(x,ast.FunctionDef) and x.name=='tick']:
            cap=[x for x in ast.walk(fn) if isinstance(x,ast.Subscript) and isinstance(x.value,ast.Name) and x.value.id=='a' and isinstance(x.slice,ast.Constant) and x.slice.value==0]
            call=[x for x in ast.walk(fn) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='step' and any(isinstance(v,ast.Starred) and isinstance(v.value,ast.Name) and v.value.id=='a' for v in x.args)]
            capture|=bool(cap and len(call)==1 and max(x.lineno for x in cap)<call[0].lineno)
        check('actual current captured before original same-argument step',capture)
        data={}
        for arm,gains in zip(ARMS,(old,new)):
            dest=out/arm;tr=npz(dest/'TRAJECTORY.npz');ne=npz(dest/'NEURAL.npz');co=npz(dest/'CTRL.npz');meta=js(dest/'MODEL_METADATA.json');a=dict(tr=tr,ne=ne,co=co,meta=meta,params=js(dest/'PARAMETERS.json'));data[arm]=a;t=tr['time'];n=ne['encoded_current'].shape[1]
            a.update(verify_mapping(dest,meta,n,check))
            check(arm+' all recorded finite except timing NaN sentinels',all(np.isfinite(v).all() for archive in (tr,ne) for v in archive.values() if v.dtype.kind in 'biufc'))
            check(arm+' all controls bounded',np.all((co['ctrl']>=0)&(co['ctrl']<=1)))
            check(arm+' trajectory shapes',tr['qpos'].shape==(4001,49) and tr['qvel'].shape==(4001,48) and co['ctrl'].shape==(4000,90))
            check(arm+' timestamps',np.allclose(t,np.arange(4001)*.0001,rtol=0,atol=1e-11) and np.array_equal(ne['time_ms'],np.arange(1,801)*.5))
            check(arm+' actual controls agree',eq(co['ctrl'],ne['ctrl']) and eq(co['ctrl'],tr['ctrl'][1:]) and co['names'].tolist()==meta['names'])
            check(arm+' no auxiliary forces',not np.any(tr['external_applied']) and not np.any(tr['qfrc_applied']))
            enc=ne['encoded_current'];sub=ne['submitted_current'];bg=ne['background_current'];actual=ne['actual_step_current']
            check(arm+' input shapes',enc.shape==(400,n) and sub.shape==(400,n) and bg.shape==(n,) and actual.shape==(800,n));check(arm+' sensory gain1 signed',eq(enc,sub) and eq(enc,ne['sensory_current']) and np.any(enc<0) and np.any(enc>0));check(arm+' real actual input',bool(np.any(bg)) and eq(actual,bg[None,:]+np.repeat(enc,2,axis=0)))
            check(arm+' actual gain vector',eq(ne['decoder_motor_gains'],gains));check(arm+'450Hz and5ms fixed',float(ne['decoder_rate_for_full_control_hz'])==450 and float(ne['filter_tau_s'])==.005 and float(ne['decoder_max_control'])==1)
            state=np.zeros(90);filtered=[];decay=math.exp(-.0005/.005)
            for pair in ne['raw_rates_hz']:
                for sample in pair:state=decay*state+(1-decay)*sample
                filtered.append(state.copy())
            filtered=np.asarray(filtered);preclip=(filtered/450.)*gains
            check(arm+' own raw rates recursively filtered',np.allclose(filtered,ne['rates_hz'],rtol=1e-13,atol=1e-11));check(arm+' own gain preclip exact',eq(ne['decoder_preclip'],(ne['rates_hz']/450.)*gains));check(arm+' independently filtered decoder',np.allclose(preclip,ne['decoder_preclip'],rtol=1e-13,atol=1e-12));check(arm+' own clipped output',eq(ne['decoder_postclip'],np.clip(ne['decoder_preclip'],0,1)));check(arm+' remaining24 silent commands',not np.any(ne['decoder_postclip'][:,remain]))
            raw=tr['contact_raw'];offs=tr['contact_offsets'];foot=np.zeros((4001,6,3));non=np.zeros((4001,3));pen=np.zeros(4001,int);first=None;penetrate=None;floor=int(meta['floor']);gl={int(k):v for k,v in meta['geom_leg'].items()}
            check(arm+' contact packing',raw.shape[1]==21 and offs.shape==(4002,) and np.array_equal(np.diff(offs),tr['contact_count']) and offs[-1]==len(raw))
            for i in range(4001):
                for c in raw[int(offs[i]):int(offs[i+1])]:
                    g1,g2=int(c[0]),int(c[1])
                    if floor not in (g1,g2):continue
                    other=g2 if g1==floor else g1;force=(1 if g1==floor else -1)*(c[6:15].reshape(3,3).T@c[15:18])
                    if other in gl:foot[i,LEGS.index(gl[other])]+=force
                    else:
                        non[i]+=force
                        if first is None:first=float(t[i]*1000)
                        if c[2]<0:
                            pen[i]+=1
                            if penetrate is None:penetrate=float(t[i]*1000)
            check(arm+' physical forces independently reconstructed',np.allclose(foot,tr['foot_force_world'],rtol=0,atol=2e-12) and np.array_equal(pen,tr['body_penetrating_contacts']))
            q=tr['qpos'][:,3:7];angle=2*np.arccos(np.clip(abs((q*q[0]).sum(1)),0,1));times=ne['time_ms'];spiket=times[np.any(ne['spikes'],axis=1)];split=min(150.,first) if first is not None else 150.
            early=dict(total=window(tr,foot,non,angle,0,150),before_first_nonfoot=window(tr,foot,non,angle,0,split),after_first_nonfoot=window(tr,foot,non,angle,split,150) if split<150 else dict(status='NOT_ESTABLISHED',reason='no nonfoot contact in early window'))
            own=window(tr,foot,non,angle,max(0,first-20),first) if first is not None else dict(status='NOT_ESTABLISHED')
            metrics[arm]=dict(target_spikes_0_200=int(ne['spikes'][times<=200].sum()),target_spikes_200_400=int(ne['spikes'][times>200].sum()),target_last_spike_ms=float(spiket[-1]) if len(spiket) else None,first_nonfoot_contact_including_margin_ms=first,first_nonfoot_penetration_ms=penetrate,COM_z400_native=float(tr['com'][-1,2]),COM_dz400_native=float(tr['com'][-1,2]-tr['com'][0,2]),COM_vz400_native=float(tr['com_velocity'][-1,2]),orientation_max_rad=float(angle.max()),joint_limit_max_native=float(tr['max_limit'].max()),common_0_150ms=early,own20ms_before_first_nonfoot_contact=own,late350_400ms=window(tr,foot,non,angle,350,400.0000001),support_walking='NOT_ESTABLISHED',physical_calibration='NOT_ESTABLISHED')
            a.update(foot=foot,nonfoot=non,angle=angle);metrics[arm]['timing_violations']=timing(out,arm,a,check)
            metrics[arm]['command_bound_dwell']=[dict(index=j,actuator=meta['names'][j],motor_gain=float(gains[j]),zero_gain=bool(gains[j]==0),lower_bound_0=dwell(co['ctrl'][:,j]==0),upper_bound_1=dwell(co['ctrl'][:,j]==1)) for j in range(90)]
            drives={}
            for j in (55,62):
                drives[str(j)]={}
                for label,(lo,hi) in dict(early_0_150ms=(0,150),own20ms=(max(0,first-20),first) if first is not None else (0,0),late350_400ms=(350,400)).items():
                    mask=(np.arange(4000)*.1>=lo-1e-9)&(np.arange(4000)*.1<hi-1e-9);c=co['ctrl'][mask,j]
                    drives[str(j)][label]=dict(samples=int(mask.sum()),mean=float(c.mean()) if c.size else None,max=float(c.max()) if c.size else None,nonzero_ms=float(np.count_nonzero(c)*.1),high_ge08_ms=float(np.count_nonzero(c>=.8)*.1),longest_high_ge08_ms=dwell(c>=.8)['longest_contiguous_dwell_ms'])
            metrics[arm]['drive_55_62']=drives
        for key in ('qpos','qvel','ctrl','act'):check('same initial '+key,eq(data[ARMS[0]]['tr'][key][0],data[ARMS[1]]['tr'][key][0]))
        check('same background',eq(data[ARMS[0]]['ne']['background_current'],data[ARMS[1]]['ne']['background_current']));check('same neural parameters',data[ARMS[0]]['params']==data[ARMS[1]]['params']);check('same roots and mappings',eq(data[ARMS[0]]['roots'],data[ARMS[1]]['roots']) and data[ARMS[0]]['manifest']==data[ARMS[1]]['manifest'] and data[ARMS[0]]['compiled']['actuator_names']==data[ARMS[1]]['compiled']['actuator_names'])
        for filename,slot in [('TRAJECTORY.npz','tr'),('NEURAL.npz','ne'),('CTRL.npz','co'),('COMMAND_TIMING.npz','timing'),('INITIAL_CARRY.npz',None)]:
            reference=npz(REFERENCE/filename);got=data[ARMS[0]][slot] if slot else npz(out/ARMS[0]/filename)
            for k,v in reference.items():check('original exact '+filename+'/'+k,k in got and eq(v,got[k]))
        for name in ('STATE.csv','CHANNELS.json','CONTACTS.json','PARAMETERS.json','DECODED_COMMANDS.json','COMMAND_APPLICATIONS.json'):check('original exact '+name,(REFERENCE/name).read_bytes()==(out/ARMS[0]/name).read_bytes())
        left,right=data[ARMS[0]],data[ARMS[1]];metrics['target_paired']={k:delta(left['ne'][k],right['ne'][k],left['ne']['time_ms']) for k in ('spikes','ge','gi','v')};metrics['paired_channels']=[]
        for j in range(90):metrics['paired_channels'].append(dict(index=j,actuator=left['meta']['names'][j],raw_rate=delta(left['ne']['raw_rates_hz'][:,:,j].reshape(800,1),right['ne']['raw_rates_hz'][:,:,j].reshape(800,1),left['ne']['time_ms']),filtered_rate=delta(left['ne']['rates_hz'][:,j:j+1],right['ne']['rates_hz'][:,j:j+1],np.arange(1,401)),preclip=delta(left['ne']['decoder_preclip'][:,j:j+1],right['ne']['decoder_preclip'][:,j:j+1],np.arange(1,401)),decoded_command=delta(left['timing']['decoded'][:,j:j+1],right['timing']['decoded'][:,j:j+1],np.arange(1,401)),applied_command=delta(left['co']['ctrl'][:,j:j+1],right['co']['ctrl'][:,j:j+1],np.arange(4000)*.1)))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(3,2,figsize=(14,11))
        for arm,a in data.items():
            tt=a['tr']['time']*1000;ax[0,0].plot(tt,a['tr']['com'][:,2]-a['tr']['com'][0,2],label=arm);ax[0,1].plot(tt,a['angle'],label=arm);ax[1,0].plot(tt,a['foot'][:,:,2].sum(1),label=arm+' feet');ax[1,0].plot(tt,a['nonfoot'][:,2],ls='--',label=arm+' nonfoot');ax[1,1].plot(a['ne']['time_ms'],a['ne']['spikes'].sum(1),label=arm)
            for j in (55,62):ax[2,0].plot(np.arange(4000)*.1,a['co']['ctrl'][:,j],label=arm+' '+str(j))
        ax[2,1].plot(np.arange(4000)*.1,np.max(abs(left['co']['ctrl']-right['co']['ctrl']),axis=1),label='max across90')
        for aa,title in zip(ax.flat,['COM dz native','Orientation rad','Physical Fz native','Target spikes','Applied controls55/62','Applied difference']):aa.set_title(title);aa.legend(fontsize=7);aa.grid(alpha=.3);aa.set_xlabel('individual clock ms')
        fig.suptitle('S291 two-channel recalibration: offline verification, not support/walking acceptance');fig.tight_layout();fig.savefig(out/'OUTCOMES.png',dpi=140);plt.close(fig)
        check('no model/network imports',not any(k=='mujoco' or k.startswith('engine.') for k in sys.modules));check('verifier unchanged',sha(__file__)==selfhash)
        result=dict(passed=all(x['passed'] for x in checks),checks=checks,failed=[x for x in checks if not x['passed']],verifier_sha256=selfhash,verifier_path=str(Path(__file__).resolve()),offline_calls=dict(model_load=0,forward=0,physics_step=0,neural_tick=0,tier_build=0))
        (out/'METRICS.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,default=serial)+'\n');(out/'VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=serial)+'\n');print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=result['failed'])));return 0 if result['passed'] else 1
    except Exception as e:print('S291 offline verifier error '+repr(e),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
