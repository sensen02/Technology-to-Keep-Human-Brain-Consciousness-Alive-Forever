"""S141/P1 offline independent initialized-activation evidence checker.
No simulator/runner imports or execution. S119 is diagnostic only: initialization differs.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import sys
import traceback
import xml.etree.ElementTree as ET
import numpy as np
from verify_joint_candidate_probe import (Checks, reconstruct_rows, exact, npz, js, sha,
    serial, require_integrity, reconstruct_fir, mapped, S113, S115, NPZS, JSONS,
    physical_metrics, admission_table, events, all42_limits, save_csv)

ROOT = Path(__file__).resolve().parent
REFERENCE = Path('/home/sensen/Desktop/engine修复验收/s121-constraint-row-recording/20261024T072500Z/init80ms')
FORMAL = Path('/home/sensen/Desktop/engine修复验收/s119-training-constraints/20261024T070600Z/TRAINING_CONSTRAINTS.json')
TARGET = 'ACTIVATION_INITIALIZATION_INDEPENDENT_CHECK.json'


def canonical_once(hashes):
    result = {}
    for name, digest in hashes.items():
        path = Path(name)
        if not path.is_absolute(): raise ValueError('relative commitment: '+name)
        key = str(path.resolve())
        if key in result and result[key] != digest: raise ValueError('canonical alias conflict: '+key)
        result[key] = digest
    return result


def bind(checks, hashes, label):
    """One canonical map and one SHA per canonical file, never repeated map scans."""
    canonical = canonical_once(hashes)
    for name, digest in canonical.items():
        actual = sha(name) if Path(name).is_file() else None
        checks.add(label+' '+name, actual == digest, expected=digest, observed=actual)
    return canonical


def required(checks, canonical, path, label):
    # bind() already verified the bytes: membership lookup is constant time.
    checks.add(label+' committed '+str(path.resolve()), str(path.resolve()) in canonical)


def scalar_rate(a, u, prm):
    ac = min(1., max(0., a)); uc = min(1., max(0., u))
    scale = .5 + 1.5*ac
    ta = prm[0]*scale; td = prm[1]/scale
    delta = uc-a
    if prm[2] < 1e-15:
        tau = ta if delta > 0 else td
    else:
        x = min(1., max(0., delta/prm[2]+.5))
        smooth = x*x*x*(3*x*(2*x-5)+10)
        tau = td+(ta-td)*smooth
    return delta/max(tau, 1e-15)


def scalar_step(a, u, prm, h=.000025):
    k1=scalar_rate(a,u,prm)
    k2=scalar_rate(a+.5*h*k1,u,prm)
    k3=scalar_rate(a+.5*h*k2,u,prm)
    k4=scalar_rate(a+h*k3,u,prm)
    # Native weighted accumulation order, RAW intermediate activation, no substage clamp.
    av=0.
    for k,b in ((k1,1/6),(k2,1/3),(k3,1/3),(k4,1/6)): av += b*k
    return a+h*av


def reconstruct_activation(initial, commands, prm):
    if initial.shape != (90,) or commands.shape != (6,90) or prm.shape != (90,3):
        raise ValueError('activation reconstruction dimensions')
    a=initial.copy(); rows=[a.copy()]
    for command in commands:
        for substep in range(20):
            a=np.array([scalar_step(float(a[j]),float(command[j]),prm[j]) for j in range(90)])
            rows.append(a.copy())
    return np.asarray(rows)


def xml_parameters(xml, provenance, checks):
    nodes=list(ET.parse(xml).getroot().find('actuator'))
    checks.add('90 explicit general muscle XML actuators', len(nodes)==90 and all(n.tag=='general' for n in nodes))
    prm=[]
    for key in ('dynprm','gainprm','biasprm','lengthrange'):
        width=2 if key=='lengthrange' else 10
        rows=[]
        for n in nodes:
            values=list(map(float,n.attrib[key].split()))
            rows.append(values+[0.]*(width-len(values)))
        arr=np.asarray(rows,dtype=np.float64)
        checks.add('XML compiled numeric '+key,hashlib.sha256(arr.tobytes()).hexdigest()==provenance['before']['model.actuator_'+key]['sha256'])
        if key=='dynprm': prm=arr[:,:3]
    return nodes,prm


def check_force(checks, arrays, nodes, label):
    # Independent scalar formula, not a model call; positive-force parameters only.
    from verify_muscle_reconstruction import evaluate
    force=np.empty((121,90))
    for i in range(121):
        for j in range(90):
            force[i,j]=evaluate(float(arrays['actuator_length'][i,j]),float(arrays['actuator_velocity'][i,j]),float(arrays['act'][i,j]),nodes[j])[0]
    checks.eq(label+' all121 all90 muscle force',arrays['actuator_force'],force,1e-10)
    projected=np.einsum('sud,su->sd',arrays['actuator_moment'],force)
    checks.eq(label+' reconstructed muscle J transpose force all48',arrays['qfrc_actuator'],projected,1e-10)
    return dict(force_maxabs=float(np.max(np.abs(force-arrays['actuator_force']))),projection_maxabs=float(np.max(np.abs(projected-arrays['qfrc_actuator']))))


def audit_runner(checks, pre, out):
    """Inspect, never import/exec, the producer and exact one-insertion private source."""
    runner=ROOT/'run_neural_activation_initialized_probe.py'
    tree=ast.parse(runner.read_text()); original=(ROOT/'run_extended_initialization_probe.py').read_text()
    constants={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('ANCHOR','INSERTION'):
            constants[node.targets[0].id]=ast.literal_eval(node.value)
    anchor=constants['ANCHOR']; insertion=constants['INSERTION']
    checks.add('sole initialization hook insertion',original.count(anchor)==1 and insertion=='            _initialize_release(mj,m,d,dec,release_ctrl,warmu,dest,reserve)\n')
    compiled=original.replace(anchor,anchor+insertion)
    checks.add('release insertion after zero assertion before sample0',compiled.index(insertion)>compiled.index(anchor) and compiled.index(insertion)<compiled.index('            rows = [sample()]'))
    checks.add('saved private runner exact audited source',(out.parent/'PRIVATE_RUNNER_SOURCE.py').read_text()==compiled)
    source_pre=pre['private_source_insertion']
    checks.add('private compiled source SHA',source_pre['compiled_source_sha256']==hashlib.sha256(compiled.encode()).hexdigest())
    checks.add('original source SHA',source_pre['original_sha256']==sha(ROOT/'run_extended_initialization_probe.py'))
    state_writes=[]; physical_calls=[]
    for n in ast.walk(tree):
        targets=n.targets if isinstance(n,ast.Assign) else [n.target] if isinstance(n,(ast.AugAssign,ast.AnnAssign)) else []
        for target in targets:
            text=ast.unparse(target)
            if text.startswith(('d.','m.','net.')): state_writes.append((n.lineno,ast.unparse(n)))
        if isinstance(n,ast.Call) and ast.unparse(n.func).endswith(('mj_forward','mj_step','from_xml_path','mj_resetData')): physical_calls.append((n.lineno,ast.unparse(n)))
    checks.add('producer only declared release state writes',[x[1] for x in state_writes]==['d.ctrl[:] = command','d.act[adr] = command'],observed=state_writes)
    checks.add('producer one extra forward no step/model call',len(physical_calls)==1 and physical_calls[0][1]=='mj.mj_forward(m, d)',observed=physical_calls)
    init=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='initialize')
    checks.add('only activation assignment inside one-time initializer',any('assert not event' in ast.unparse(n) for n in init.body) and all(init.lineno<line<init.end_lineno for line,_ in state_writes))
    original_tree=ast.parse(original); pose=[]
    for n in ast.walk(original_tree):
        if isinstance(n,ast.Assign):
            pose.extend(ast.unparse(n) for target in n.targets if ast.unparse(target).startswith('d.qpos'))
    checks.add('original one initial pose assignment',pose==['d.qpos[:] = q'])
    recorder_tree=ast.parse((ROOT/'run_constraint_row_recording.py').read_text())
    function=next(n for n in recorder_tree.body if isinstance(n,ast.FunctionDef) and n.name=='constraint_record')
    forbidden=[]
    for n in ast.walk(function):
        if isinstance(n,ast.Call) and any(s in ast.unparse(n.func) for s in ('mj_forward','mj_step','from_xml','net.step','mj_jac','mj_mul','mj_contactForce')): forbidden.append(ast.unparse(n))
        if isinstance(n,ast.Assign):
            forbidden.extend(ast.unparse(n) for target in n.targets if ast.unparse(target).startswith(('d.','m.','net.')))
    checks.add('S121 recorder passive',not forbidden,forbidden=forbidden)
    return dict(state_writes=state_writes,extra_calls=physical_calls,private_source_sha256=hashlib.sha256(compiled.encode()).hexdigest(),scope='Hash-bound static audit and saved explicit ledger, not runtime instrumentation proof.')


def verify(out, reference, formal_path, checks, report):
    checks.add('offline import boundary',not any(k in sys.modules for k in ('mujoco','torch','engine.neural_cond')))
    checks.add('original reference and formal identity',reference==REFERENCE.resolve() and formal_path==FORMAL.resolve())
    checks.add('one init80ms arm',out.name=='init80ms')
    pre=js(out.parent/'PREREG.json'); hashes=js(out.parent/'INPUT_HASHES.json')
    checks.add('prereg source hashes exact',pre['source_hashes']==hashes)
    cmap=bind(checks,hashes,'input'); report['input_hashes']=cmap
    for p in (ROOT/'run_neural_activation_initialized_probe.py',ROOT/'run_extended_initialization_probe.py',ROOT/'run_constraint_row_recording.py',reference/'RESULT.json',formal_path): required(checks,cmap,p,'required input')
    counts=dict(model_load=1,forward=122,step=120,neural_ticks=166)
    ledger=js(out.parent/'CALL_LEDGER.json')
    checks.add('exact refreshed-release call ledger',ledger==dict(counts,active=None,completed=['init80ms'],per_arm={'init80ms':counts}),observed=ledger)
    checks.add('complete not failed',js(out.parent/'COMPLETE.json')['status']=='COMPLETE' and not (out.parent/'FAILED.json').exists())
    result=js(out/'RESULT.json'); checks.add('RESULT call ledger',result['calls']==counts)
    checks.add('single RESULT',js(out.parent/'RESULTS.json')==[result])
    artifact_maps={}
    for label,path in [('S141',out),('S121',reference),('S113',S113),('S115',S115)]:
        artifacts=bind(checks,js(path/'RESULT.json')['hashes'],label+' artifact'); artifact_maps[label]=artifacts
        for name in (*NPZS,*JSONS): required(checks,artifacts,path/name,label)
        if label in ('S141','S121'):
            for name in ('CONSTRAINT_ROWS.npz','CONSTRAINT_ROW_METADATA.json'): required(checks,artifacts,path/name,label)
        if label!='S141': required(checks,cmap,path/'RESULT.json',label+' reference')
    report['source_audit']=audit_runner(checks,pre,out)
    artifacts=artifact_maps['S141']
    for name in ('PRE_RELEASE_ZERO_INITIALIZATION.npz','ACT_INITIALIZATION.json'):
        required(checks,artifacts,out/name,'initialization evidence')
    require_integrity(checks,'hashes and completion')
    old=reconstruct_rows(checks,reference,'S121'); new=reconstruct_rows(checks,out,'S141')
    a=old['arrays']; b=new['arrays']; init=npz(out/'INITIALIZATION.npz'); warm=npz(reference/'INITIALIZATION.npz')
    checks.add('same moment metadata',old['moment']==new['moment'])
    checks.eq('all121 native clock',b['time'],np.arange(121)*.000025,1e-12)
    checks.eq('matched baseline instants',b['time'],a['time'],0)
    for label,item in [('S121',old),('S141',new)]: checks.add(label+' recognized native rows',item['counts']['other']==0 and item['counts']['unrecognized']==0)
    for key in ('weights','actuator_pool'): checks.eq('original runtime '+key,b[key],a[key],0)
    checks.eq('original initialization W',init['weights'],warm['weights'],0)
    for key in ('spikes','features','decoded','history','release_features','release_decoded','current','body_qpos','body_qvel'):
        checks.eq('unchanged computational initialization '+key,init[key],warm[key],0)
    for name in ('NEURAL_COLD.npz','CONNECTIVITY.npz'):
        x=npz(out/name); y=npz(reference/name); checks.add('same keys '+name,set(x)==set(y))
        for key in x: checks.eq('original cold '+name+':'+key,x[key],y[key],0)
    checks.add('160 warm and six runtime ticks',init['features'].shape==(160,66,5) and b['features'].shape==(6,66,5) and b['applied'].shape==(6,90))
    manifests={str(Path(p).resolve()) for p in hashes if Path(p).name=='MANIFEST.json'}
    checks.add('one canonical pool manifest',len(manifests)==1); require_integrity(checks,'initialization and manifest')
    manifest=sorted(js(next(iter(manifests)))['rows'],key=lambda r:r['actuator_index'])
    pools=sorted({tuple(sorted(r['local_indices'])) for r in manifest if r['local_indices']}); lookup={p:i for i,p in enumerate(pools)}
    mapping=np.array([lookup.get(tuple(sorted(r['local_indices'])),-1) for r in manifest])
    checks.eq('manifest pool mapping',b['actuator_pool'],mapping,0)
    checks.add('manifest actuator identity',[r['actuator'] for r in manifest]==new['moment']['actuator_names'])
    fs,us,hist=reconstruct_fir(np.concatenate((init['spikes'],b['spikes'])),pools,mapping,b['weights'])
    _,_,wh=reconstruct_fir(init['spikes'],pools,mapping,b['weights'])
    for key,x,y in [('warm features',init['features'],fs[:160]),('warm decoded',init['decoded'],us[:160]),('release features',init['release_features'],fs[159]),('release command',init['release_decoded'],us[159]),('warm FIR',init['history'],wh),('runtime features',b['features'],fs[160:]),('runtime commands',b['applied'],us[160:]),('final FIR',npz(out/'NEURAL_FINAL.npz')['FIR_history'],hist)]: checks.eq('166 tick FIR '+key,x,y,1e-14)
    release=init['release_decoded']
    before=npz(out/'PRE_RELEASE_ZERO_INITIALIZATION.npz')
    checks.add('all original zero initialization keys',set(before)==set(warm))
    for key in warm: checks.eq('exact archived zero release '+key,before[key],warm[key],0)
    event=js(out/'ACT_INITIALIZATION.json'); adr=np.asarray(event['compiled_actadr']); num=np.asarray(event['compiled_actnum'])
    checks.add('compiled one activation per actuator identity',np.array_equal(adr,np.arange(90)) and np.array_equal(num,np.ones(90)))
    checks.add('one release activation initialization event',event['act_write_count']==1 and event['extra_forward_count']==1 and event['physical_s']==0 and event['source_s']==.08 and event['dec_interval']==159)
    for key,original_key in [('BODY_ACT_BEFORE_INITIALIZATION','body_act'),('BODY_CTRL_BEFORE_INITIALIZATION','body_ctrl'),('BODY_QPOS_BEFORE_INITIALIZATION','body_qpos'),('BODY_QVEL_BEFORE_INITIALIZATION','body_qvel')]: checks.eq('before initialization '+key,init[key],warm[original_key],0)
    checks.eq('actual initialization source command',init['activation_initialization_command'],release,0)
    for key in ('body_act','body_ctrl'): checks.eq('initialized release '+key,init[key],release,0)
    for key in ('act','ctrl'): checks.eq('refreshed sample0 '+key,b[key][0],release,0)
    for key in ('qpos','qvel'): checks.eq('original mechanical sample0 '+key,b[key][0],a[key][0],0)
    checks.add('nonzero activation intervention',np.any(release!=0) and not a['act'][0].any())
    checks.eq('20-substep controls',b['ctrl'][1:],np.repeat(b['applied'],20,axis=0),1e-14)
    checks.add('sample0 forces and acceleration refreshed',not np.array_equal(b['actuator_force'][0],a['actuator_force'][0]) and not np.array_equal(b['qacc'][0],a['qacc'][0]))
    checks.add('no external forces or warnings',not b['qfrc_applied'].any() and not b['xfrc_applied'].any() and not b['warnings'].any())
    checks.eq('runtime held tick0 actual current',b['current'],np.repeat(b['current'][0][None],6,axis=0),0)
    for name,ticks in [('NEURAL_COLD_META.json',0),('NEURAL_RELEASE_META.json',160),('NEURAL_FINAL_META.json',166)]:
        scalars=js(out/name)['neural_scalars']; checks.add('clock '+name,scalars['steps']==ticks and abs(scalars['t_ms']-.5*ticks)<1e-10)
    schedule=js(out/'SCHEDULE.json'); checks.add('six tick schedule',len(schedule)==6)
    for tick,row in enumerate(schedule):
        checks.add('consume '+str(tick),row['tick']==tick and row['consume_id']==160+tick)
        for key,value in dict(source_start_s=(160+tick)*.0005,source_end_s=(161+tick)*.0005,physical_start_s=tick*.0005,physical_end_s=(tick+1)*.0005,source_offset_s=.08).items(): checks.eq('schedule '+str(tick)+' '+key,row[key],value,1e-12)
    prov=js(out/'MODEL_PROVENANCE.json')
    for key in ('before','after'): checks.add('unchanged model '+key,prov[key]==js(reference/'MODEL_PROVENANCE.json')[key])
    xmls={str(Path(p).resolve()) for p in hashes if Path(p).suffix=='.xml'}
    checks.add('one canonical XML',len(xmls)==1); require_integrity(checks,'release and clocks')
    xml=Path(next(iter(xmls))); nodes,prm=xml_parameters(xml,prov,checks)
    acts=reconstruct_activation(release,b['applied'],prm)
    checks.eq('independent scalar RK4 all121 all90 activations',b['act'],acts,1e-12)
    checks.eq('independent zero-init baseline RK4 all121 all90',a['act'],reconstruct_activation(a['act'][0],a['applied'],prm),1e-12)
    report['force_reconstruction']={label:check_force(checks,x,nodes,label) for label,x in [('S121',a),('S141',b)]}
    formal=js(formal_path); checks.add('original exact formal prereg',pre['training_constraints']==formal)
    report['formal_constraints']=formal; report['per_arm']={}; measured={}
    for label,path in [('S113',S113),('S115',S115),('S121',reference),('S141',out)]:
        p,r,native=physical_metrics(path,xml,checks,label); measured[label]=p
        report['per_arm'][label]=dict(path=str(path),physical=p,native=native)
    for label,key in [('S113','safety_exact_bounds'),('S115','benefit_exact_bounds')]:
        for k,v in formal[key].items():
            if k=='extrema':
                for metric,sides in v.items():
                    for side,value in sides.items(): checks.add(label+' exact formal '+metric+' '+side,measured[label][k][metric][side]==value)
            else: checks.add(label+' exact formal '+k,measured[label][k]==v)
    require_integrity(checks,'scalar activation muscle force COM native and formal references')
    # Activation is the changed actual motor state, not a changed W/command candidate.
    rows,decision=admission_table(formal,measured['S141'],not np.array_equal(a['act'],b['act']))
    checks.add('17 diagnostic formal comparisons',len(rows)==17)
    report.update(diagnostic_formal_table=rows,diagnostic_S119_conjunction=all(r['passed'] for r in rows),diagnostic_helper_decision=decision,events={label:events(item) for label,item in [('S121',old),('S141',new)]},closure={label:item['states'] for label,item in [('S121',old),('S141',new)]},activation_maxabs=float(np.max(np.abs(acts-b['act']))),initialization_scope='One activation-state initialization plus matching released ctrl. Mechanical qpos/qvel unchanged; sample0 force/acceleration recomputed, not historical replay/restart. No body integration during neural initialization. S119 diagnostic only; original candidate acceptance inapplicable.')
    return old,new


def outputs(out,old,new,report):
    a=old['arrays']; b=new['arrays']; ms=b['time']*1000
    header=['criterion','actual','reference','operator','margin','passed','reference_source']
    paths=[save_csv(out,'S119_DIAGNOSTIC_TABLE.csv',header,[[r[k] for k in header] for r in report['diagnostic_formal_table']])]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,2,figsize=(15,12))
    for label,x in [('S121',a),('S141 initialized',b)]:
        _,limit,_=all42_limits(x,new['moment'])
        for ax,y in zip(axs.flat[:4],[x['load_ratio'],limit,x['qpos'][:,2]/x['qpos'][0,2],x['com_velocity_world'][:,2]]): ax.plot(ms,y,label=label)
    for j in range(10): axs[2,0].plot(ms,b['act'][:,j],label='act '+str(j)); axs[2,0].plot(ms,a['act'][:,j],ls=':',alpha=.4)
    for j in range(5): axs[2,1].step(ms,b['ctrl'][:,j],where='post',label='ctrl '+str(j))
    axs[0,0].axhline(.8,color='k',ls=':')
    axs[0,1].axhline(report['formal_constraints']['safety_exact_bounds']['extrema']['limit_excess']['max'],color='k',ls=':')
    for ax,title in zip(axs.flat,['load ratio','all42 max limit excess','root height ratio','whole-body COM vertical velocity','first 10 activations; dotted S121','released ctrl first 5; sample0 is warm final command']):
        ax.set_title(title); ax.set_xlabel('physical ms'); ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.suptitle('S141 activation-state initialization diagnostic; no physical prewarm / autonomy claim')
    fig.tight_layout(); p=out/'INIT_RELEASE_SUPPORT.png'; fig.savefig(p,dpi=140); plt.close(fig); paths.append(str(p))
    return paths


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True,type=Path); p.add_argument('--report',type=Path); p.add_argument('--reference',type=Path,default=REFERENCE); p.add_argument('--formal',type=Path,default=FORMAL); args=p.parse_args(argv)
    out=args.output.resolve(); target=(args.report or out/TARGET).resolve()
    if target.exists(): raise ValueError('never overwrite prior report; use --report for failed variants')
    checks=Checks(); report=dict(passed=False,evidence_passed=False,candidate_admitted=False,admission_evaluated=False,baseline_promotion=False,promotion=False,checkpoint_replacement=False,autonomy=False,standing_or_walking_claim=False,behavior_acceptance=False,physical_units='NOT_ESTABLISHED',verifier=str(Path(__file__).resolve()),verifier_sha256=sha(__file__),new_calls=dict(model_load=0,forward=0,step=0,neural_ticks=0))
    try:
        old,new=verify(out,args.reference.resolve(),args.formal.resolve(),checks,report)
        report['outputs']=outputs(out,old,new,report)
        checks.add('offline final boundary',not any(k in sys.modules for k in ('mujoco','torch','engine.neural_cond')))
        report['passed']=report['evidence_passed']=checks.passed
    except Exception as exc: report.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    report['helper_source_hashes']={str(Path(m.__file__).resolve()):sha(m.__file__) for name,m in tuple(sys.modules.items()) if name.startswith('verify_') and getattr(m,'__file__',None)}
    report['checks']=checks.rows
    with target.open('x') as f: f.write(json.dumps(report,ensure_ascii=False,indent=2,default=serial,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=report['passed'],diagnostic_S119_conjunction=report.get('diagnostic_S119_conjunction'),candidate_admitted=False,report=str(target),error=report.get('error')),ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
