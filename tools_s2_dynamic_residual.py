"""Independent scalar endpoint arithmetic; all actual residuals REPORT_ONLY."""
import numpy as np


def entries(values, nv, nefc, sparse=False, rownnz=None, rowadr=None, colind=None, nJ=None):
    v = np.asarray(values).reshape(-1)
    if not np.isfinite(v).all(): raise ValueError('nonfinite J')
    if not sparse:
        if len(v) != nv*nefc or (nJ is not None and nJ != len(v)): raise ValueError('dense size')
        return [[(j, float(v[i*nv+j])) for j in range(nv)] for i in range(nefc)]
    if len(rownnz)!=nefc or len(rowadr)!=nefc: raise ValueError('sparse rows')
    rows=[]; count=0
    for n,a in zip(rownnz,rowadr):
        n,a=int(n),int(a)
        if n<0 or a<0 or a+n>len(v) or a+n>len(colind): raise ValueError('sparse range')
        cols=[int(c) for c in colind[a:a+n]]
        if any(c<0 or c>=nv for c in cols) or cols!=sorted(set(cols)): raise ValueError('sparse columns')
        rows.append([(c,float(v[a+k])) for k,c in enumerate(cols)]);count+=n
    if nJ is not None and count!=nJ: raise ValueError('sparse nJ')
    return rows


def stats(values, denominator=None):
    x=np.asarray(values,dtype=float)
    result={'count':int(x.size),'absmax':float(np.max(np.abs(x))) if x.size else None,'rms':float(np.sqrt(np.mean(x*x))) if x.size else None}
    if denominator is not None:
        den=np.asarray(denominator);mask=den!=0
        result.update(relative_undefined=int(np.sum(~mask)),relative_absmax=float(np.max(np.abs(x[mask])/den[mask])) if np.any(mask) else None)
    return result


def scalar(rows, qacc, smooth, aref, D, force, state, qfrc, b):
    r=[]; rb=[]; mapped=[0.]*len(qacc)
    for i,row in enumerate(rows):
        acc=0.; sm=0.
        for j,v in row:
            acc+=v*float(qacc[j]);sm+=v*float(smooth[j]);mapped[j]+=v*float(force[i])
        r.append(acc-float(aref[i]));rb.append(sm-float(aref[i])-float(b[i]))
    r=np.asarray(r); dr=np.asarray(D)*r; e=np.asarray(force)+dr; quadratic=np.asarray(state)==1
    return {'r':r,'e':e,'quadratic':quadratic,'denominator':np.maximum(np.abs(force),np.abs(dr)), 'mapping':np.asarray(qfrc)-mapped,'b_error':np.asarray(rb)}


def analyze(out):
    import json
    from pathlib import Path
    out=Path(out);manifest=json.loads((out/'manifest.json').read_text()); reports=[]; series=[]
    for rec in manifest['records']:
        with np.load(out/rec['npz'],allow_pickle=False) as a:
            results=[];types={};states={};perdof=[]
            for i in range(400):
                lo,hi=a['efc_offsets'][i:i+2];jl,jh=a['J_offsets'][i:i+2]
                rows=entries(a['J_values'][jl:jh],rec['nv'],hi-lo,bool(a['sparse'][i]),a['J_rownnz'][lo:hi],a['J_rowadr'][lo:hi],a['J_colind'][jl:jh],int(a['nJ'][i]))
                z=scalar(rows,a['qacc'][i],a['qacc_smooth'][i],a['efc_aref'][lo:hi],a['efc_D'][lo:hi],a['efc_force'][lo:hi],a['efc_state'][lo:hi],a['qfrc_constraint'][i],a['efc_b'][lo:hi]);results.append(z)
                for t in np.unique(a['efc_type'][lo:hi]):
                    mask=(a['efc_type'][lo:hi]==t)&z['quadratic'];bucket=types.setdefault(str(int(t)),[[],[]]);bucket[0].extend(z['e'][mask]);bucket[1].extend(z['denominator'][mask])
                for s in a['efc_state'][lo:hi]: states[str(int(s))]=states.get(str(int(s)),0)+1
            cat=lambda k:np.concatenate([z[k] for z in results])
            mask=cat('quadratic');e=cat('e');den=cat('denominator');mapping=np.asarray([z['mapping'] for z in results])
            reports.append({'npz':rec['npz'],'quadratic':stats(e[mask],den[mask]),'nonquadratic_count':int(np.sum(~mask)),'state_histogram':states,'per_type_quadratic':{t:stats(*v) for t,v in types.items()},'r_reference_deviation':stats(cat('r')),'b_consistency':stats(cat('b_error')),'mapping_per_dof':[dict(dof=j,unit_type=rec['dof_types'][j],**stats(mapping[:,j])) for j in range(rec['nv'])]})
            series.append(([stats(z['e'][z['quadratic']])['absmax'] for z in results],np.max(np.abs(mapping),axis=1)))
    report={'policy':'REPORT_ONLY','cases':reports,'solver_stationarity':'not measured','fullKKT':'not measured','massbalance':'not measured','physical_cause':'unknown'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,7))
    for rec,(e,m) in zip(manifest['records'],series):
        axes[0].plot(np.arange(1,401)*.025,e,label=f"{rec['dt_s']*1e6:g} us")
        axes[1].plot(np.arange(1,401)*.025,m,label=f"{rec['dt_s']*1e6:g} us")
    for ax,label in zip(axes,('QUADRATIC e absmax (model units)','Mapping absmax (mixed DOF units; display only)')):
        ax.set_ylabel(label);ax.legend();ax.grid(alpha=.3)
    axes[1].set_xlabel('Time (ms)');fig.suptitle('REPORT ONLY: not stationarity / full KKT / mass balance');fig.tight_layout();fig.savefig(out/'residual.png');plt.close(fig)
    (out/'同端点残差报告.zh-CN.md').write_text('# S2/P2 同端点残差报告\n\n手动研究实施，不是成品。两固定nativefloor pulse target0；同projected pose，10ms，RK4 dt25/12.5us，共同25us400端点。primal NEWTON/noslip0。逐标量循环独立计算r=Jqacc-aref，QUADRATIC e=f+D*r；r不应为零，e与映射均REPORT_ONLY，不套1e-10通过门槛。非QUADRATIC单列覆盖，不套标量本构式。相对分母零undefined。\n\n未构造AR/M，未测solver stationarity、完整KKT或mass balance，physical_cause unknown；未证明身体收敛、神经因果、站立行走或生物有效性。DOF映射按各自模型单位报告，混合单位曲线仅显示数值不能作为物理总范数。contact参数仅提供解释字段不是因果结论。\n\n'+json.dumps(reports,ensure_ascii=False,indent=2)+'\n\n证据：\n'+ '\n'.join(str(out/p) for p in ['manifest.json','report.json','residual.png']+[r['npz'] for r in manifest['records']])+'\n')
    return report
