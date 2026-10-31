"""S124/P1/S2 bounded offline XML/hash audit and conditional muscle reconstruction.
Never imports simulator/project code; no model/forward/step or XML modification.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/sensen/Desktop/engine修复验收')
OLD = BASE/'s121-constraint-row-recording/20261024T072500Z/init80ms'
NEW = BASE/'s122-secant-constraint-pair/20261024T073500Z/w36b4_secant'
S123 = BASE/'s123-muscle-compensation/20261024T074500Z'
FORMAL = BASE/'s119-training-constraints/20261024T070600Z/TRAINING_CONSTRAINTS.json'
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
OUT = BASE/'s124-muscle-reconstructability/20261024T075500Z'
SOURCE = Path('/tmp/dsh-spill-6NncVL/session-a9c6da6bad4c/d7595eff21e9-web_fetch.txt')
SOURCE_URL = 'https://raw.githubusercontent.com/google-deepmind/mujoco/3.9.0/src/engine/engine_util_misc.c'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def ah(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def js(p): return json.loads(Path(p).read_text())
def dump(p,d): Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def save(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def xmltree(path,files,stack=()):
    path=path.resolve(); assert path not in stack, 'Cyclic XML include'
    files.add(path); root=ET.parse(path).getroot()
    def expand(e):
        for i,c in reversed(list(enumerate(list(e)))):
            if c.tag=='include':
                child=xmltree(path.parent/c.attrib['file'],files,stack+(path,))
                e.remove(c)
                for n in reversed(list(child)): e.insert(i,n)
            else: expand(c)
    expand(root)
    return root


def resolve(root):
    defaults={}
    def visit(d,parent):
        state={k:dict(v) for k,v in parent.items()}
        for e in d:
            if e.tag!='default': state.setdefault(e.tag,{}).update(e.attrib)
        defaults[d.get('class','main')]=state
        for e in d:
            if e.tag=='default': visit(e,state)
    for d in root.findall('default'): visit(d,{})
    result=[]
    for i,e in enumerate(root.find('actuator')):
        cls=e.get('class','main'); inherited=defaults.get(cls,{}).get(e.tag,{})
        attrs=dict(inherited); attrs.update(e.attrib)
        result.append(dict(id=i,name=e.get('name'),tag=e.tag,class_name=cls,
                           explicit=dict(e.attrib),inherited=dict(inherited),resolved=attrs,
                           sources={k:('actuator explicit' if k in e.attrib else 'default class '+cls) for k in attrs}))
    return result


def gain_bias(length,velocity,lr,g,b):
    # Literal arithmetic branches from MuJoCo 3.9.0 source lines848..950.
    # All denominators are strictly positive in these saved parameters/states;
    # hence mjMAX(mjMINVAL,denominator) equals denominator (asserted, no default).
    assert g[2]>0 and b[2]>0
    L0=(lr[1]-lr[0])/(g[1]-g[0]); assert L0>1e-15 and L0*g[6]>1e-15
    L=g[0]+(length-lr[0])/L0; V=velocity/(L0*g[6])
    lo,hi=g[4],g[5]; a=.5*(lo+1); mid=.5*(1+hi)
    assert min(a-lo,1-a,mid-1,hi-mid,g[8]-1)>1e-15
    if not lo<=L<=hi: fl=0.
    elif L<=a: fl=.5*((L-lo)/(a-lo))**2
    elif L<=1: fl=1-.5*((1-L)/(1-a))**2
    elif L<=mid: fl=1-.5*((L-1)/(mid-1))**2
    else: fl=.5*((hi-L)/(hi-mid))**2
    y=g[8]-1
    if V<=-1: fv=0.
    elif V<=0: fv=(V+1)*(V+1)
    elif V<=y: fv=g[8]-(y-V)*(y-V)/y
    else: fv=g[8]
    G=-g[2]*fl*fv
    B0=(lr[1]-lr[0])/(b[1]-b[0]); assert B0>1e-15
    BL=b[0]+(length-lr[0])/B0; bm=.5*(1+b[5]); assert bm-1>1e-15
    if BL<=1: B=0.
    elif BL<=bm: B=-b[2]*b[7]*.5*((BL-1)/(bm-1))**2
    else: B=-b[2]*b[7]*(.5+(BL-bm)/(bm-1))
    return G,B,L,V,fl,fv


def main(out):
    assert not out.exists(), 'Fresh output only'
    assert 'mujoco' not in sys.modules
    xmlfiles=set(); tree=xmltree(XML,xmlfiles); resolved=resolve(tree)
    roots=(OLD.parent,NEW.parent,S123,FORMAL.parent)
    inputs={p.resolve() for r in roots for p in r.rglob('*') if p.is_file()}
    inputs.update(xmlfiles); inputs.update((Path(__file__).resolve(),SOURCE))
    # Bind and verify every explicit upstream input/code commitment, not just results.
    commitments={}
    for r in (OLD.parent,NEW.parent):
        commitments.update(js(r/'INPUT_HASHES.json'))
    commitments.update(js(S123/'PREREG.json')['input_sha256'])
    for r in (OLD,NEW): commitments.update(js(r/'RESULT.json')['hashes'])
    for name,h in commitments.items():
        p=Path(name); assert p.is_absolute() and p.is_file(), name
        assert sha(p)==h, 'Upstream commitment mismatch '+name
        inputs.add(p.resolve())
    before={str(p):sha(p) for p in sorted(inputs)}
    inventory=[]; metadata=[]
    # Every frozen archive and all JSON metadata in the bounded upstream roots.
    for r in roots:
        for p in sorted(r.rglob('*.npz')):
            with np.load(p,allow_pickle=False) as z:
                for k in z.files:
                    v=z[k]
                    inventory.append(dict(path=str(p),key=k,shape=list(v.shape),dtype=str(v.dtype),sha256=ah(v)))
        for p in sorted(r.rglob('*.json')):
            d=js(p)
            def walk(v,key=''):
                if isinstance(v,dict):
                    if set(('shape','dtype','sha256'))<=set(v):
                        metadata.append(dict(path=str(p),key=key,kind='HASH_ONLY_ARRAY',record=v)); return
                    for k,x in v.items(): walk(x,key+'.'+k if key else k)
                elif isinstance(v,list):
                    metadata.append(dict(path=str(p),key=key,kind='SAVED_LIST',length=len(v)))
                else: metadata.append(dict(path=str(p),key=key,kind='SAVED_SCALAR',value=v))
            walk(d)
    names=js(OLD/'MOMENT_METADATA.json')['actuator_names']
    assert names==js(NEW/'MOMENT_METADATA.json')['actuator_names']==[e['name'] for e in resolved]
    assert len(names)==90
    compiled={}; proofs=[]
    for key,n in [('lengthrange',2),('dynprm',10),('gainprm',10),('biasprm',10)]:
        value=np.zeros((90,n),dtype=np.float64)
        for i,e in enumerate(resolved):
            tokens=list(map(float,e['resolved'][key].split())); assert len(tokens)<=n
            value[i,:len(tokens)]=tokens
        # Zero-padding is accepted ONLY by matching the actual compiled digest.
        compiled[key]=value
        for r in (OLD,NEW):
            p=js(r/'MODEL_PROVENANCE.json'); assert p['xml_sha256']==sha(XML)
            for phase in ('before','after'):
                record=p[phase]['model.actuator_'+key]
                ok=record==dict(shape=list(value.shape),dtype=str(value.dtype),sha256=ah(value))
                proofs.append(dict(path=str(r/'MODEL_PROVENANCE.json'),phase=phase,field=key,passed=ok))
    # Explicitly prove runtime activation/force mode, rather than guessing defaults.
    mode_values={'actearly':np.zeros(90,dtype=bool),'forcelimited':np.zeros(90,dtype=bool),
                 'actadr':np.arange(90,dtype=np.int32)}
    for key,value in mode_values.items():
        for r in (OLD,NEW):
            p=js(r/'MODEL_PROVENANCE.json')
            for phase in ('before','after'):
                rec=p[phase]['model.actuator_'+key]
                proofs.append(dict(path=str(r/'MODEL_PROVENANCE.json'),phase=phase,field=key,
                                   passed=rec==dict(shape=list(value.shape),dtype=str(value.dtype),sha256=ah(value))))
    ready=all(p['passed'] for p in proofs)
    gaps=[]
    for r in (OLD,NEW):
        for i in range(6):
            for key in ('lengthrange','dynprm','gainprm','biasprm','length0','acc0'):
                recovered=key in compiled and ready
                gaps.append(dict(run=r.name,actuator_id=i,actuator_name=names[i],field='actuator_'+key,
                                 status='XML_VALUES_MATCH_COMPILED_SHA256' if recovered else 'HASH_ONLY_NO_VALUES',
                                 required_for_current_positive_force=('no' if key in ('length0','acc0') else 'yes'),
                                 reason=('positive gain/bias force; acc0 branch unused; length0 not read by muscle gain/bias' if key in ('length0','acc0') else 'all90 array bytes bound before/after in both runs')))
    out.mkdir(parents=True)
    dump(out/'ACTUATOR_XML_RESOLUTION.json',dict(xml_path=str(XML),xml_sha256=sha(XML),recursive_includes=[str(p) for p in sorted(xmlfiles)],all_actuators=resolved,compiled_hash_proofs=proofs,focus_ids=list(range(6))))
    dump(out/'STATIC_FIELD_INVENTORY.json',dict(scope=[str(r) for r in roots],npz_arrays=inventory,json_metadata=metadata,hash_only_is_not_values=True))
    save(out/'FIELD_GAPS.csv',gaps)
    summary=dict(scope='S124/P1/S2 bounded offline, act0..5 actual IDs; no simulation',
                 readiness='RECONSTRUCTABLE' if ready else 'NOT_RECONSTRUCTABLE',
                 force_scale='positive explicit gain and bias force individually hash-verified; no acc0 inference',
                 xml_sha256=sha(XML),compiled_proofs_passed=ready,
                 missing_nonrequired_fields=['compiled actuator_length0 values','compiled actuator_acc0 values'],
                 formula_source_url=SOURCE_URL,formula_source_local=str(SOURCE),formula_source_sha256=sha(SOURCE),
                 arrays_inventoried=len(inventory),metadata_records=len(metadata),upstream_commitments_verified=len(commitments),
                 model_load=0,forward=0,step=0,neural=0,xml_writes=0,fit_arrays=False,guessed_defaults=False,
                 training_acceptance=False,checkpoint_replacement=False,standing_or_walking_claim=False,physical_units='NOT_ESTABLISHED')
    if ready:
        rows=[]; states={}; residuals={}
        for r in (OLD,NEW):
            with np.load(r/'TRAJECTORY.npz',allow_pickle=False) as z:
                d={k:z[k] for k in ('time','act','actuator_length','actuator_velocity','actuator_force','actuator_moment')}
            assert d['act'].shape==(121,90)
            state=[]
            for t in range(121):
                sample=[]
                for i in range(6):
                    G,B,L,V,fl,fv=gain_bias(d['actuator_length'][t,i],d['actuator_velocity'][t,i],compiled['lengthrange'][i],compiled['gainprm'][i],compiled['biasprm'][i])
                    active=G*d['act'][t,i]; pred=active+B; actual=d['actuator_force'][t,i]
                    row=dict(run=r.name,sample=t,time_ms=float(d['time'][t]*1000),actuator_id=i,actuator_name=names[i],activation=float(d['act'][t,i]),normalized_length=L,normalized_velocity=V,FL=fl,FV=fv,gain=G,active_force=active,passive_muscle_force=B,reconstructed_force=pred,saved_force=float(actual),residual=float(pred-actual))
                    rows.append(row); sample.append((G,B,d['act'][t,i]))
                state.append(sample)
            states[r.name]=(np.array(state),d)
            res=max(abs(x['residual']) for x in rows if x['run']==r.name)
            residuals[r.name]=res
            assert res<1e-10, 'Force reconstruction failed'
        save(out/'MUSCLE_FORCE_RECONSTRUCTION.csv',rows)
        deriv=[]; delta=[]
        s0,d0=states[OLD.name]; s1,d1=states[NEW.name]
        for t in range(121):
            for i in range(6):
                G0,B0,a0=s0[t,i]; G1,B1,a1=s1[t,i]
                da=a1-a0; dg=G1-G0; db=B1-B0
                observed=d1['actuator_force'][t,i]-d0['actuator_force'][t,i]
                terms=(da*G0,a0*dg,da*dg,db)
                delta.append(dict(sample=t,time_ms=float(d0['time'][t]*1000),actuator_id=i,actuator_name=names[i],activation_term=terms[0],state_gain_term=terms[1],cross_term=terms[2],passive_bias_term=terms[3],saved_force_delta=float(observed),residual=float(sum(terms)-observed)))
                for run,(s,d) in states.items():
                    for dof in (6,7,8):
                        deriv.append(dict(run=run,sample=t,time_ms=float(d['time'][t]*1000),actuator_id=i,actuator_name=names[i],dof=dof,dtorque_dactivation_fixed_state=float(s[t,i,0]*d['actuator_moment'][t,i,dof]),interpretation='local fixed length/velocity/moment partial only; not neural/closed-loop sensitivity'))
        save(out/'FORCE_DELTA_DECOMPOSITION.csv',delta)
        save(out/'LOCAL_FIXED_STATE_PARTIALS.csv',deriv)
        summary.update(force_max_abs_residual=residuals,reconstructed_records=len(rows),force_delta_max_abs_residual=max(abs(r['residual']) for r in delta),derivative_scope='fixed-state activation partial only, after force reproduction; no neural/closed-loop sensitivities')
    after={str(p):sha(p) for p in sorted(inputs)}
    assert before==after
    assert 'mujoco' not in sys.modules
    summary.update(input_files_bound=len(before),all_inputs_unchanged=True)
    dump(out/'INPUT_HASHES.json',dict(before=before,after=after,all_unchanged=True))
    dump(out/'SUMMARY.json',summary)
    text='# S124 肌肉可重建性离线审计（P1/S2）\n\n'
    text+='结论：**'+summary['readiness']+'**。不是仅因 compiled 数组未单独落盘就认定不可重建。\n\n'
    text+='全部90执行器 XML lengthrange/dynprm/gainprm/biasprm 按实际顺序解析、继承默认类，构造的 float64 数组与 S121/S122 的 before/after 编译哈希完全一致。补零只在整数组哈希匹配后接受，不是猜测参数。XML 精确 SHA256：`'+sha(XML)+'`。\n\n'
    text+='act0..5 名称保持真实ID；正 gain force 与正 bias force 各自保存（bias并非gain力标度）。compiled acc0/length0 仅有哈希没有值；正 force 不走 acc0 自动缩放分支，length0 不进入所审计 gain/bias 公式，因此不阻断本次力重建。actearly=false、forcelimited=false、actadr=0..89 同样按实际编译哈希核验。\n\n'
    if ready:
        text+='按 MuJoCo 3.9.0 实际源码逐分支复算两组121样本×6肌肉 active=G*saved_act 与 passive muscle bias，总力最大绝对误差：'+json.dumps(residuals)+'。不是把 qfrc_passive 当被动肌肉力。\n\n'
        text+='力差严格分解 Δa*G0+a0*ΔG+Δa*ΔG+ΔB；局部 G*moment 偏导只在力复现通过后输出，固定长度/速度/力臂，不是神经、控制命令或闭环灵敏度。未拟合数组、未提出新权重。\n\n'
    text+='直接上游结果、代码承诺、递归XML include及正式S119约束均跑前跑后绑定。模型/forward/step/神经/XML写入均为0。绝对物理单位 NOT_ESTABLISHED；不改变S113基线/checkpoint，不宣称训练、物理预热、站立或行走成功。未改 tracker。\n'
    (out/'REPORT.zh-CN.md').write_text(text)
    dump(out/'OUTPUT_HASHES.json',{str(p):sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,default=OUT)
    main(p.parse_args().out)
