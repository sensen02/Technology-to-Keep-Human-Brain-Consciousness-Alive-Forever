"""S292 independent offline current-sample calibration verifier; no simulator execution.
Uses current S289 ordinal17 PRECLIP unit-gain quantity, not a historical match.
"""
from __future__ import annotations
import argparse
import ast
import importlib.util
import json
import math
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=ROOT/'verify_s289_decoder_range_support.py'
spec=importlib.util.spec_from_file_location('s292_offline_utilities',BASE)
util=importlib.util.module_from_spec(spec);spec.loader.exec_module(util)
sha,js,npz,serial,eq,delta,window,dwell,timing=(getattr(util,k) for k in ('sha','js','npz','serial','eq','delta','window','dwell','timing'))
REFERENCE=Path('/home/sensen/Desktop/engine修复验收/s289-decoder-range-support/20261026T173400Z/decoder_original')
RUNNER=ROOT/'run_s292_current_sample_two_channel_calibration.py'
XML=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
GAIN_FILE=Path('/home/sensen/Desktop/engine修复验收/s255-static-gain-can-it-sustain/20261025T221308522972Z/ARM_VECTORS.json')
TARGET_FILE=Path('/home/sensen/Desktop/engine修复验收/s244-chirality-of-the-5-of-6-arm/20261025T205824474993Z/CMD_A3_sym_matched.npz')
POOLS=ROOT/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json'
ARMS=('original_mask','current_sample_55_62')
LEGS=('lf','lm','lh','rf','rm','rh')
CHANGED=(55,62)

def recursive(raw):
    decay=math.exp(-.0005/.005);state=np.zeros(90);out=[]
    for interval in raw:
        for sample in interval:state=decay*state+(1-decay)*sample
        out.append(state.copy())
    return np.asarray(out)

def expanded_xml(path,stack=()):
    path=Path(path).resolve()
    if path in stack:raise ValueError('XML include cycle')
    root=ET.fromstring(path.read_bytes())
    def visit(node):
        for child in list(node):
            if child.tag=='include':
                sub=expanded_xml(path.parent/child.attrib['file'],(*stack,path));i=list(node).index(child);node.remove(child)
                for j,item in enumerate(list(sub)):node.insert(i+j,item)
            else:visit(child)
    visit(root);return root

def calibration(out,check,names,old):
    source=npz(REFERENCE/'NEURAL.npz');clock=npz(REFERENCE/'COMMAND_TIMING.npz');filtered=recursive(source['raw_rates_hz'])
    check('calibration source recursive own filter',np.allclose(filtered,source['rates_hz'],rtol=1e-13,atol=1e-11),max_abs=float(abs(filtered-source['rates_hz']).max()))
    check('current sample17 source clock',abs(clock['source_start_s'][17]-.017)<1e-12 and abs(clock['source_end_s'][17]-.018)<1e-12 and abs(clock['valid_from_s'][17]-.018)<1e-12)
    check('current sample17 application18to19',clock['applied_source_index'][18]==17 and abs(clock['applied_start_s'][18]-.018)<1e-12 and abs(clock['applied_end_s'][18]-.019)<1e-12)
    check('current sample17 two neural samples17.5and18ms',np.array_equal(source['time_ms'][34:36],np.array([17.5,18.])))
    with np.load(TARGET_FILE,allow_pickle=True) as z:target=z['cmd'].copy();target_names=[str(x) for x in z['names']]
    check('target order90',target.shape==(90,) and target_names==names)
    gains=old.copy();rows=[]
    for i in CHANGED:
        q=float(source['rates_hz'][17,i]/450.);scaled=float(target[i]*1.236819199)
        check(f'channel{i} positive source and target',np.isfinite(q) and q>0 and 0<scaled<1 and old[i]==0)
        gains[i]=scaled/q;rows.append(dict(index=i,actuator=names[i],q=q,scaled_target=scaled,gain_new=float(gains[i])))
    doc=js(out/'CALIBRATION.json')
    check('calibration accepted',doc.get('passed',doc.get('calibration_passed')) is True)
    check('explicit new preclip formula no historical match',doc['formula_status']=='EXPLICIT_NEW_CURRENT_SAMPLE_FORMULA_NOT_OLD_CALIBRATION_REPRODUCTION' and doc['sample_search'] is False and doc['sample']['historical_semantic_match_claim'] is False and doc['sample']['index']==17 and doc['scale']==1.236819199 and doc['full_control_hz']==450.)
    check('calibration complete config vectors',np.array_equal(np.asarray(doc['old_gains']),old) and np.array_equal(np.asarray(doc['new_gains']),gains) and doc['changed_indices']==list(CHANGED))
    for field in ('source_hashes','input_hashes'):
        for p,h in doc.get(field,{}).items():check('calibration source '+p,Path(p).is_file() and sha(p)==h)
    entries=doc.get('per_channel',doc.get('channels'))
    if not isinstance(entries,list):raise ValueError('CALIBRATION per_channel schema missing')
    by={int(r['index']):r for r in entries};check('calibration only55and62',set(by)==set(CHANGED))
    for row in rows:
        saved=by[row['index']]
        check(f"calibration gain{row['index']} independent",math.isclose(float(saved.get('gain_new',saved.get('new_gain'))),row['gain_new'],rel_tol=1e-14,abs_tol=1e-14))
        if 'q' in saved:check(f"calibration q{row['index']}",saved['q']==row['q'])
        check(f"calibration primitive facts{row['index']}",saved['gain_old']==old[row['index']] and saved['filtered_rate_hz']==float(source['rates_hz'][17,row['index']]) and saved['original_target']==float(target[row['index']]) and saved['target']==row['scaled_target'] and saved['scale']==1.236819199 and saved['raw_half_ms_hz']==source['raw_rates_hz'][17,:,row['index']].tolist())
        if 'pool_root_ids' in saved:
            pool={r['actuator']:r for r in js(POOLS)['measured']['resolved']}[row['actuator']]
            check(f"calibration source roots{row['index']}",sorted(map(int,saved['pool_root_ids']))==sorted(set(map(int,pool['pool_root_ids']))))
    check('only two configured gains changed',np.flatnonzero(gains!=old).tolist()==list(CHANGED) and np.count_nonzero(gains==0)==24)
    return gains,rows

def runtime(out,names,check):
    roots=npz(out/'TIER_ROOT_IDS.npz')['root_ids'];man=js(out/'MOTOR_MANIFEST.json');reported=js(out/'RUNTIME_MAPPING_CHECK.json')
    check('actual tier integer roots unique',roots.ndim==1 and roots.dtype.kind in 'iu' and len(set(map(int,roots)))==len(roots))
    check('manifest dimensions',man['n_neurons']==len(roots) and man['n_actuators']==90 and len(man['rows'])==90)
    pooldoc=js(POOLS);source={r['actuator']:r for r in pooldoc['measured']['resolved']};allsource={r['actuator']:r for r in pooldoc['measured']['resolved']+pooldoc['measured']['unresolved']}
    for i,row in enumerate(man['rows']):
        check(f'actual manifest{i} unchanged source annotation',row.get('source_annotation')==allsource[names[i]])
        ids=row['local_indices'];valid=all(not isinstance(k,bool) and isinstance(k,int) and 0<=k<len(roots) for k in ids) and len(set(ids))==len(ids)
        check(f'actual manifest{i} index/name',row['actuator_index']==i and row['actuator']==names[i] and valid)
        if not valid:continue
        projected=[int(roots[k]) for k in ids]
        check(f'actual manifest{i} local root identity',projected==list(map(int,row.get('tier_root_ids',[]))) and set(projected)<=set(map(int,row.get('pool_root_ids',[]))))
        if i in CHANGED:
            expected=sorted(set(map(int,source[names[i]]['pool_root_ids'])))
            check(f'actual target{i} complete exact source pool',bool(expected) and row['status']=='complete' and row['pool_root_ids']==expected and sorted(projected)==expected and len(projected)==len(expected) and all(np.count_nonzero(roots==r)==1 for r in expected))
    check('runtime mapping assertion passed',reported.get('passed') is True)
    for arm in ARMS:
        c=js(out/arm/'COMPILED_MAPPING.json');observed=c.get('actuator_names',c.get('compiled_actuator_names'))
        check(arm+' observed compiled90 order',observed==names and len(observed)==len(set(observed))==90)
        check(arm+' observed compiled provenance',c['passed'] is True and c['xml_sha256']==sha(XML) and c['asset_names']==names and c['xml_declaration_names']==names and c['runtime_mapping_check_passed'] is True and 'mj_id2name' in c['source'])
        counts=c.get('calls_at_check',{})
        expected=dict(model_load=1,tier_build=1,network_init=0,live_step=0,neural_tick=0,live_fwd_position=0,live_forward=0,diagnostic_forward=0) if arm==ARMS[0] else dict(model_load=2,tier_build=1,network_init=1,live_step=4000,neural_tick=800,live_fwd_position=1,live_forward=0,diagnostic_forward=4001)
        check(arm+' compiled check before own forward/network/dynamics',counts==expected)

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);out=ap.parse_args(argv).output
    checks=[];metrics={};selfhash=sha(__file__)
    def check(name,value,**extra):checks.append(dict(name=name,passed=bool(value),**extra))
    try:
        pre=js(out/'PREREG.json');check('S292 round/arms',pre['round']=='S292' and pre['arms']==list(ARMS))
        check('nonempty dependency hashes',bool(pre['source_hashes']))
        for p,h in pre['source_hashes'].items():check('source '+p,Path(p).is_file() and sha(p)==h)
        check('essential sources preregistered',all(str(p) in pre['source_hashes'] for p in (RUNNER,XML,POOLS,GAIN_FILE)))
        with np.load(TARGET_FILE,allow_pickle=True) as z:names=[str(x) for x in z['names']]
        tree=expanded_xml(XML);declared=[e.get('name') for sec in tree.findall('./actuator') for e in sec]
        check('declaration names/order90',len(names)==len(set(names))==90 and declared==names)
        old=np.asarray(js(GAIN_FILE)['v1'],float);check('original26zero',old.shape==(90,) and np.count_nonzero(old==0)==26)
        new,calcrows=calibration(out,check,names,old);metrics['calibration_independent']=calcrows
        runtime(out,names,check)
        complete=js(out/'COMPLETE.json');ledger=js(out/'CALL_LEDGER.json');expected=dict(model_load=2,tier_build=1,network_init=2,live_step=8000,neural_tick=1600,live_fwd_position=2,live_forward=0,diagnostic_forward=8002)
        check('capture complete',complete.get('capture_complete') is True and complete.get('source_unchanged') is True);check('no failure marker',not(out/'FAILED.json').exists() and not(out/'STOP.json').exists());check('actual experiment ledger',ledger==expected and complete.get('calls')==ledger)
        base_source=Path(pre['runtime_base_source']);check('registered runtime base hash',sha(base_source)==pre['runtime_base_source_sha256'] and base_source==ROOT/'run_s291_two_channel_recalibration.py')
        tree=ast.parse(base_source.read_text());advance=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='advance']
        check('advance real submitted current',any(len(n.args)>=2 and isinstance(n.args[0],ast.Constant) and n.args[0].value==2 and isinstance(n.args[1],ast.Name) and n.args[1].id=='submitted' for n in advance))
        good=False
        for fn in [n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='tick']:
            captures=[n for n in ast.walk(fn) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id in ('a','args') and isinstance(n.slice,ast.Constant) and n.slice.value==0]
            calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='step' and any(isinstance(v,ast.Starred) for v in n.args)]
            appends=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='actual_step_currents' and n.func.attr=='append']
            good|=bool(captures and len(calls)==1 and appends and max(n.lineno for n in captures+appends)<calls[0].lineno)
        check('actual drive captured before same original call',good)
        data={}
        for arm,gains in zip(ARMS,(old,new)):
            dest=out/arm;tr=npz(dest/'TRAJECTORY.npz');ne=npz(dest/'NEURAL.npz');co=npz(dest/'CTRL.npz');meta=js(dest/'MODEL_METADATA.json');a=dict(tr=tr,ne=ne,co=co,meta=meta,params=js(dest/'PARAMETERS.json'));data[arm]=a;t=tr['time']
            check(arm+' trajectory/control shapes',tr['qpos'].shape==(4001,49) and tr['qvel'].shape==(4001,48) and co['ctrl'].shape==(4000,90));check(arm+' rates shapes',ne['rates_hz'].shape==(400,90) and ne['raw_rates_hz'].shape==(400,2,90))
            check(arm+' clocks/no overrun',np.allclose(t,np.arange(4001)*.0001,rtol=0,atol=1e-11) and np.array_equal(ne['time_ms'],np.arange(1,801)*.5))
            check(arm+' actual controls and names',eq(co['ctrl'],ne['ctrl']) and eq(co['ctrl'],tr['ctrl'][1:]) and co['names'].tolist()==names and meta['names']==names)
            check(arm+' bounds no external',np.all((co['ctrl']>=0)&(co['ctrl']<=1)) and not np.any(tr['external_applied']) and not np.any(tr['qfrc_applied']))
            check(arm+' finite all saved state',all(np.isfinite(v).all() for z in (tr,ne) for v in z.values() if v.dtype.kind in 'biufc'))
            enc=ne['encoded_current'];sub=ne['submitted_current'];bg=ne['background_current'];actual=ne['actual_step_current'];n=enc.shape[1]
            check(arm+' current shapes',enc.shape==(400,n) and sub.shape==(400,n) and bg.shape==(n,) and actual.shape==(800,n));check(arm+' current clocks',np.array_equal(ne['sensory_time_ms'],np.arange(400,dtype=float)) and np.array_equal(ne['actual_step_time_ms'],ne['time_ms']))
            check(arm+' signed gain1 and retained background',eq(sub,enc) and eq(ne['sensory_current'],enc) and np.any(enc<0) and np.any(enc>0) and np.any(bg!=0));check(arm+' actual per-tick bg plus own encoding',eq(actual,np.broadcast_to(bg,(800,n))+np.repeat(enc,2,axis=0)))
            check(arm+' exact gain configuration',eq(ne['decoder_motor_gains'],gains));check(arm+' fixed450/5ms/max1',float(ne['decoder_rate_for_full_control_hz'])==450. and float(ne['filter_tau_s'])==.005 and float(ne['decoder_max_control'])==1.)
            cfg=js(dest/'DECODER_CONFIG.json');scfg=js(dest/'SENSORY_GAIN_CONFIG.json');check(arm+' explicit decoder config',cfg['filter_tau_s']==.005 and cfg['rate_for_full_control_hz']==450. and cfg['max_control']==1. and np.array_equal(np.asarray(cfg['per_actuator_gain']),gains));check(arm+' explicit fixed sensorygain1',scfg['sensory_gain']==1. and scfg['clipping'] is False and scfg['signed'] is True and scfg['background_unchanged'] is True)
            filtered=recursive(ne['raw_rates_hz']);preclip=(filtered/450.)*gains;postclip=np.clip(preclip,0.,1.)
            check(arm+' independently recursive filter',np.allclose(filtered,ne['rates_hz'],rtol=1e-13,atol=1e-11));check(arm+' own preclip/postclip',np.allclose(ne['decoder_preclip'],preclip,rtol=1e-13,atol=1e-12) and np.allclose(ne['decoder_postclip'],postclip,rtol=1e-13,atol=1e-12));check(arm+' exact own saved preclip equation',eq(ne['decoder_preclip'],(ne['rates_hz']/450.)*gains) and eq(ne['decoder_postclip'],np.clip(ne['decoder_preclip'],0,1)))
            check(arm+' remaining zero masks preserved',not np.any(ne['decoder_preclip'][:,gains==0]) and not np.any(ne['decoder_postclip'][:,gains==0]))
            raw=tr['contact_raw'];offs=tr['contact_offsets'];counts=tr['contact_count'];check(arm+' raw contact packing',raw.ndim==2 and raw.shape[1]==21 and offs.shape==(4002,) and np.array_equal(np.diff(offs),counts) and offs[-1]==len(raw))
            contacts=js(dest/'CONTACTS.json');check(arm+' contact JSON exact',len(contacts)==4001 and all(np.array_equal(np.asarray(c,float).reshape(-1,21),raw[int(offs[i]):int(offs[i+1])]) for i,c in enumerate(contacts)))
            floor=int(meta['floor']);gl={int(k):v for k,v in meta['geom_leg'].items()};foot=np.zeros((4001,6,3));non=np.zeros((4001,3));pen=np.zeros(4001,int);first=None;margin=None;penetrate=None
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
            check(arm+' reconstructed physical forces',np.allclose(foot,tr['foot_force_world'],rtol=0,atol=2e-12) and np.array_equal(pen,tr['body_penetrating_contacts']))
            q=tr['qpos'][:,3:7];angle=2*np.arccos(np.clip(abs((q*q[0]).sum(axis=1)),0,1));times=ne['time_ms'];spiket=times[np.any(ne['spikes'],axis=1)];split=min(150.,max(0.,first)) if first is not None else 150.
            early=dict(total=window(tr,foot,non,angle,0.,150.),before_first_nonfoot=window(tr,foot,non,angle,0.,split),after_first_nonfoot=window(tr,foot,non,angle,split,150.) if first is not None and first<150 else dict(status='NOT_ESTABLISHED',reason='no nonfoot contact in common0to150ms'))
            own=window(tr,foot,non,angle,max(0.,first-20),first) if first is not None else dict(status='NOT_ESTABLISHED');own['truncated_at_initial_time']=bool(first is not None and first<20.)
            metrics[arm]=dict(target_spikes_0_200=int(ne['spikes'][times<=200].sum()),target_spikes_200_400=int(ne['spikes'][times>200].sum()),target_last_spike_ms=float(spiket[-1]) if len(spiket) else None,COM_z400_native=float(tr['com'][-1,2]),COM_dz400_native=float(tr['com'][-1,2]-tr['com'][0,2]),COM_vz400_native=float(tr['com_velocity'][-1,2]),root_dz400_native=float(tr['qpos'][-1,2]-tr['qpos'][0,2]),first_nonfoot_contact_including_margin_ms=first,first_nonfoot_margin_dist_ge0_ms=margin,first_nonfoot_penetration_ms=penetrate,orientation_max_rad=float(angle.max()),joint_limit_max_native=float(tr['max_limit'].max()),common_0_150ms=early,own20ms_before_first_nonfoot_contact=own,late350_400ms=window(tr,foot,non,angle,350.,400.0000001),support_walking='NOT_ESTABLISHED',physical_calibration='NOT_ESTABLISHED')
            a.update(foot=foot,nonfoot=non,angle=angle);metrics[arm]['timing_violations']=timing(out,arm,a,check)
            dw=[]
            for j in range(90):
                c=co['ctrl'][:,j];check(arm+f' zero gain{j}',gains[j]!=0 or not np.any(c));dw.append(dict(index=j,actuator=names[j],fixed_motor_gain=float(gains[j]),zero_motor_gain_mask=bool(gains[j]==0),lower_bound_0=dwell(c==0),upper_bound_1=dwell(c==1),interpretation='Descriptive dwell; initial zeros and declared zero gains are not pathological by definition'))
            metrics[arm]['command_bound_dwell']=dw;drive=[]
            for j in CHANGED:
                c=co['ctrl'][:,j];windows={}
                for label,lo,hi in [('early0_150',0.,150.),('own_precontact20',max(0,first-20),first),('late350_400',350.,400.)] if first is not None else [('early0_150',0.,150.),('late350_400',350.,400.)]:
                    selected=c[(np.arange(4000)*.1>=lo-1e-9)&(np.arange(4000)*.1<hi-1e-9)];windows[label]=dict(nonzero_ms=float(np.count_nonzero(selected)*.1),mean=float(selected.mean()) if selected.size else None,max=float(selected.max()) if selected.size else None,upper=dwell(selected==1))
                drive.append(dict(index=j,actuator=names[j],gain=float(gains[j]),raw_rate_max_hz=float(ne['raw_rates_hz'][:,:,j].max()),decoded_max=float(ne['decoder_postclip'][:,j].max()),applied_nonzero_ms=float(np.count_nonzero(c)*.1),windows=windows))
            metrics[arm]['drive_55_62']=drive
        for key in ('qpos','qvel','ctrl','act'):check('initial same '+key,eq(data[ARMS[0]]['tr'][key][0],data[ARMS[1]]['tr'][key][0]))
        check('same neural parameters/background',data[ARMS[0]]['params']==data[ARMS[1]]['params'] and eq(data[ARMS[0]]['ne']['background_current'],data[ARMS[1]]['ne']['background_current']))
        for filename,slot in [('TRAJECTORY.npz','tr'),('NEURAL.npz','ne'),('CTRL.npz','co'),('COMMAND_TIMING.npz','timing'),('INITIAL_CARRY.npz',None)]:
            ref=npz(REFERENCE/filename);got=data[ARMS[0]][slot] if slot else npz(out/ARMS[0]/filename);check('original historical keys '+filename,set(ref)<=set(got))
            for k,v in ref.items():check('original historical exact '+filename+'/'+k,k in got and eq(v,got[k]))
        for filename in ('STATE.csv','CHANNELS.json','CONTACTS.json','PARAMETERS.json','MODEL_METADATA.json','DECODED_COMMANDS.json','COMMAND_APPLICATIONS.json'):check('original historical bytes '+filename,(REFERENCE/filename).read_bytes()==(out/ARMS[0]/filename).read_bytes())
        for filename in ('CURRENT_PATH_CHECK.json','SENSORY_GAIN_CONFIG.json','DECODER_CONFIG.json'):
            historical=js(REFERENCE/filename);present=js(out/ARMS[0]/filename);historical.pop('arm',None);present.pop('arm',None)
            if filename=='DECODER_CONFIG.json':
                # Runtime metadata describes the same old decoder; generated text differs only in
                # the wording of the gain vector and an optional provenance flag.
                historical.pop('no_direct_halfcommand_claim',None);present.pop('no_direct_halfcommand_claim',None)
                historical.pop('changed_gain_indices',None);present.pop('changed_gain_indices',None)
                historical.pop('derivation_sha256',None);present.pop('derivation_sha256',None)
                historical['formula']=present.get('formula',historical.get('formula')).replace('V1','own gains')
            check('original historical config semantic equality excluding arm label '+filename,historical==present)
        left,right=data[ARMS[0]],data[ARMS[1]];metrics['target_paired']={k:delta(left['ne'][k],right['ne'][k],left['ne']['time_ms']) for k in ('spikes','ge','gi','v')};metrics['paired_channels']=[]
        for j in range(90):metrics['paired_channels'].append(dict(index=j,actuator=names[j],raw_rate=delta(left['ne']['raw_rates_hz'][:,:,j].reshape(800,1),right['ne']['raw_rates_hz'][:,:,j].reshape(800,1),left['ne']['time_ms']),decoded_rate=delta(left['ne']['rates_hz'][:,j:j+1],right['ne']['rates_hz'][:,j:j+1],np.arange(1,401)),preclip=delta(left['ne']['decoder_preclip'][:,j:j+1],right['ne']['decoder_preclip'][:,j:j+1],np.arange(1,401)),postclip=delta(left['ne']['decoder_postclip'][:,j:j+1],right['ne']['decoder_postclip'][:,j:j+1],np.arange(1,401)),decoded_command=delta(left['timing']['decoded'][:,j:j+1],right['timing']['decoded'][:,j:j+1],np.arange(1,401)),applied_command=delta(left['co']['ctrl'][:,j:j+1],right['co']['ctrl'][:,j:j+1],np.arange(4000)*.1)))
        metrics['comparison_scope']='Current sample17 preclip calibration, not historical match. Candidate uses its own rates; no target equality constraint after closed-loop branching.'
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(4,2,figsize=(14,14))
        for arm,a in data.items():
            tt=a['tr']['time']*1000;ne=a['ne'];ax[0,0].plot(tt,a['tr']['com'][:,2]-a['tr']['com'][0,2],label=arm);ax[0,1].plot(tt,a['angle'],label=arm);ax[1,0].plot(tt,a['foot'][:,:,2].sum(axis=1),label=arm+' feet');ax[1,0].plot(tt,a['nonfoot'][:,2],ls='--',label=arm+' nonfoot');ax[1,1].plot(ne['time_ms'],ne['spikes'].sum(axis=1),label=arm)
            for j in CHANGED:ax[2,0].plot(np.arange(4000)*.1,a['co']['ctrl'][:,j],label=arm+f' {j}')
            ax[2,1].plot(np.arange(1,401),ne['decoder_preclip'][:,55],label=arm+' preclip55');ax[2,1].plot(np.arange(1,401),ne['decoder_preclip'][:,62],label=arm+' preclip62')
            dw=metrics[arm]['command_bound_dwell'];ax[3,1].plot(np.arange(90),[d['upper_bound_1']['total_dwell_ms'] for d in dw],label=arm+' upper1')
        ax[3,0].plot(np.arange(4000)*.1,np.max(abs(left['co']['ctrl']-right['co']['ctrl']),axis=1),label='max across90')
        for aa,title in zip(ax.flat,['COM dz native','Orientation rad','Foot and nonfoot Fz native','Target-only spikes','Actual applied55/62','Own preclip55/62','Applied90 difference','Upper-bound dwell ms']):aa.set_title(title);aa.grid(alpha=.3);aa.legend(fontsize=6);aa.set_xlabel('individual clock ms')
        ax[3,1].set_xlabel('actuator index');fig.suptitle('S292 current-sample two-channel calibration: no support/walking acceptance');fig.tight_layout();fig.savefig(out/'OUTCOMES.png',dpi=140);plt.close(fig)
    except Exception as e:check('verification exception',False,error=repr(e))
    check('no simulator/network imports',not any(k=='mujoco' or k.startswith('engine.') for k in sys.modules));check('verifier unchanged',sha(__file__)==selfhash)
    result=dict(passed=all(c['passed'] for c in checks),checks=checks,failed=[c for c in checks if not c['passed']],verifier_sha256=selfhash,verifier_path=str(Path(__file__).resolve()),offline_calls=dict(model_load=0,forward=0,physics_step=0,neural_tick=0,tier_build=0),scope='Current-sample calibration and own-arm mapping/current/decoder/physics evidence, not behavioral acceptance')
    (out/'METRICS.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,default=serial)+'\n');(out/'VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=serial)+'\n');print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=result['failed'])));return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
