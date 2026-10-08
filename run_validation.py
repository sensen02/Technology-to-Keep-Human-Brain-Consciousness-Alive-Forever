"""Run reference comparison and controls; no parameter fitting."""
from pathlib import Path
import json,re,time,hashlib,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation,PillowWriter
from matplotlib.colors import LogNorm
from model import run
from measurement import measure
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)


def load_experiment():
    text=(ROOT/'upstream/controlCaRadData.m').read_text()
    pairs=re.findall(r'\{\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*\}',text)
    data=np.array(pairs,dtype=float)
    np.savetxt(OUT/'experimental_control.csv',data,delimiter=',',header='time_s,radius_um',comments='')
    return data


def save_case(name,**kwargs):
    path=OUT/(name+'.npz')
    signature=hashlib.sha256((ROOT/'model.py').read_bytes()+(ROOT/'upstream/microtearDistribution_Andrew_dataPoints.xlsx').read_bytes()+json.dumps(kwargs,sort_keys=True).encode()).hexdigest()
    reuse=False
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata_json']))
            if meta.get('cache_signature')==signature:
                result={k:z[k] for k in z.files if k!='metadata_json'}
                result['metadata']=meta;reuse=True
    if not reuse:
        print('Running',name,kwargs,flush=True)
        result=run(**kwargs)
        result['metadata']['cache_signature']=signature
        result['metadata']['upstream_commit']=subprocess.check_output(['git','-C',str(ROOT/'upstream'),'rev-parse','HEAD'],text=True).strip()
        arrays={k:v for k,v in result.items() if k!='metadata'}
        np.savez_compressed(path,**arrays,metadata_json=json.dumps(result['metadata']))
    m=measure(result['c'],result['baseline_c'],result['polygons'],result['gcamp'],result['ablated'])
    result['measurement']=m
    np.savetxt(OUT/(name+'_radius.csv'),np.c_[result['t'],m['radius'],m['censored']],delimiter=',',header='time_s,radius_um,censored',comments='')
    return result


def main():
    experiment=load_experiment()
    cases={
        'control':{}, 'gj_10percent':{'gj_scale':.1},
        'plc_30percent':{'plc_scale':.3},'no_wound':{'wound':False},
        'tight_solver':{'rtol':1e-7},'larger_domain':{'n_side':28},
        'seed_2':{'seed':2},'seed_3':{'seed':3},
        'recommended_576':{'n_side':24},'gj_576':{'n_side':24,'gj_scale':.1},
        'plc_576':{'n_side':24,'plc_scale':.3},
        'tight_576':{'n_side':24,'rtol':1e-7},
        'seed2_576':{'n_side':24,'seed':2},'seed3_576':{'n_side':24,'seed':3}}
    results={name:save_case(name,**kw) for name,kw in cases.items()}
    control=results['control']; metrics={'scope':'Retrospective single-wound comparison, NOT independent validation','cases':{}}
    ref=experiment[(experiment[:,0]>=2.14)&(experiment[:,0]<=23.54+1e-8)]
    for name,x in results.items():
        m=x['measurement']; valid=np.isfinite(m['radius'])
        # Do not bridge censored intervals: interpolation only when neighboring values finite.
        predicted=np.interp(ref[:,0],x['t'],m['radius'])
        good=np.isfinite(predicted); err=predicted[good]-ref[good,1]
        entry={'n_cells':len(x['positions']),'comparison_points':int(good.sum()),'requested_points':len(ref),
          'mae_um':float(np.mean(abs(err))) if len(err) else None,'rmse_um':float(np.sqrt(np.mean(err**2))) if len(err) else None,
          'bias_um':float(np.mean(err)) if len(err) else None,'fraction_within_cell_diameter':float(np.mean(abs(err)<=2*np.sqrt(43/np.pi))) if len(err) else None,
          'max_complete_radius_um':m['max_complete_radius'],'censored_frames':int(m['censored'].sum()),
          'censor_reason_counts':{str(k):int(v) for k,v in zip(*np.unique(m['censor_reason'],return_counts=True))},
          'max_c_um':float(x['c'][:,~x['ablated']].max()),'min_c_um':float(x['c'].min()),
          'metadata':x['metadata']}
        for key in ['c','er','ip3','h']:
            entry[key+'_finite']=bool(np.isfinite(x[key]).all())
            entry[key+'_min']=float(x[key].min());entry[key+'_max']=float(x[key].max())
        metrics['cases'][name]=entry
    a=control['c'];b=results['tight_solver']['c']
    metrics['solver_max_abs_c_um']=float(np.max(abs(a-b)))
    metrics['solver_rms_c_um']=float(np.sqrt(np.mean((a-b)**2)))
    metrics['no_wound_max_c_drift_um']=float(np.max(abs(results['no_wound']['c']-results['no_wound']['baseline_c'])))
    # Same central coordinates must be present in enlarged domain.
    from scipy.spatial import cKDTree
    dist,idx=cKDTree(results['larger_domain']['positions']).query(control['positions'])
    metrics['domain_max_coordinate_mismatch_um']=float(dist.max())
    inner=np.linalg.norm(control['positions'],axis=1)<60
    metrics['domain_inner60_c_rms_difference_um']=float(np.sqrt(np.mean((control['c'][:,inner]-results['larger_domain']['c'][:,idx[inner]])**2)))
    mr=results['larger_domain']['measurement']['radius']-control['measurement']['radius']
    metrics['domain_radius_max_difference_um']=float(np.nanmax(abs(mr)))
    raster=measure(control['c'],control['baseline_c'],control['polygons'],control['gcamp'],control['ablated'],pixels_per_um=3.2)
    metrics['raster_refinement_max_radius_difference_um']=float(np.nanmax(abs(raster['radius']-control['measurement']['radius'])))
    x=results['recommended_576']; y=results['larger_domain']
    metrics['domain_576_vs784_radius_max_difference_um']=float(np.nanmax(abs(x['measurement']['radius']-y['measurement']['radius'])))
    dd,ii=cKDTree(y['positions']).query(x['positions'])
    mask=np.linalg.norm(x['positions'],axis=1)<60
    metrics['domain_576_vs784_inner60_c_rms_difference_um']=float(np.sqrt(np.mean((x['c'][:,mask]-y['c'][:,ii[mask]])**2)))
    metrics['solver_576_max_abs_c_um']=float(np.max(abs(x['c']-results['tight_576']['c'])))
    metrics['solver_576_rms_c_um']=float(np.sqrt(np.mean((x['c']-results['tight_576']['c'])**2)))
    metrics['t0_experiment_radius_um']=float(experiment[experiment[:,0]==0,1][0])
    metrics['t0_note']='Model injury begins at t=0; pre-existing experimental first-frame radius is not matched or time-shift fitted.'
    for name in ['recommended_576','gj_576','plc_576','larger_domain']:
        z=results[name];metrics['cases'][name]['radius_at_19_26_s_um']=float(np.interp(19.26,z['t'],z['measurement']['radius']))
    from check_measurement import check
    from model import self_test
    metrics['measurement_unit_test']=check()
    metrics['model_self_test']=self_test()
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    # Display the 576-cell refinement; preserve 400-cell arrays/metrics above.
    control=results['recommended_576']
    # Quantitative plots with experiment distinguished from simulations.
    fig,axs=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    ax=axs[0,0];ax.plot(experiment[(experiment[:,0]>=0)&(experiment[:,0]<=25),0],experiment[(experiment[:,0]>=0)&(experiment[:,0]<=25),1],'o',ms=4,color='black',label='Experimental single wound')
    for name in ['control','recommended_576','gj_576','plc_576']:
        x=results[name];ax.plot(x['t'],x['measurement']['radius'],label=name)
    ax.set(xlim=(0,25),xlabel='Time after injury (s)',ylabel='GCaMP half-height radius (um)',title='Early signal: no new biological parameter fitting');ax.legend(fontsize=8)
    ax=axs[0,1]
    for name in ['recommended_576','tight_576','larger_domain','seed2_576','seed3_576']:
        x=results[name]; pred=np.interp(ref[:,0],x['t'],x['measurement']['radius']);ax.plot(ref[:,0],pred-ref[:,1],'.-',label=name)
    ax.axhline(0,color='k',lw=.7);ax.set(xlabel='Time (s)',ylabel='Simulation - experiment (um)',title='Residuals, domain / seed / solver checks');ax.legend(fontsize=8)
    ax=axs[1,0]
    radii=np.linalg.norm(control['positions'],axis=1)
    for target in [30,45,55,60]:
        cell=np.argmin(abs(radii-target));ax.plot(control['t'],control['c'][:,cell],label=f'cell at {radii[cell]:.1f} um')
    ax.set(xlabel='Time (s)',ylabel='Cytosolic free calcium (uM)',title='Intracellular state: NOT measured ground truth');ax.legend(fontsize=8)
    ax=axs[1,1]
    for t in [2,10,20]:
        idx=np.argmin(abs(control['t']-t));ax.plot(control['measurement']['r'],control['measurement']['profiles'][idx],label=f'{control["t"][idx]:.1f} s')
    ax.set(xlabel='Distance from injury (um)',ylabel='Synthetic fluorescence (a.u.)',title='Annular profiles / finite field of view');ax.legend(fontsize=8)
    fig.suptitle('400 / 576 / 784-cell Drosophila epithelium: reduced published calcium model\nFixed cell geometry; prescribed membrane damage; no mechanical tearing simulation',fontsize=12)
    fig.savefig(OUT/'validation.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,4,figsize=(15,4),layout='constrained')
    for ax,t in zip(axs,[0,2,10,20]):
        idx=np.argmin(abs(control['t']-t));values=control['c'][idx].copy();values[control['ablated']]=np.nan
        pc=PolyCollection(control['polygons'],array=values,cmap='viridis',edgecolors='gray',linewidth=.25,norm=LogNorm(vmin=.08,vmax=600));ax.add_collection(pc);ax.autoscale_view();ax.set_aspect('equal');ax.set_title(f'{control["t"][idx]:.1f} s');ax.set(xlabel='um',ylabel='um');ax.set_facecolor('#444444')
    fig.colorbar(pc,ax=axs,label='Cytosolic calcium (uM), logarithmic scale');fig.suptitle('576 fixed cells; gray core = prescribed ablation, not viable cells');fig.savefig(OUT/'cell_snapshots.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,6));pc=PolyCollection(control['polygons'],cmap='viridis',edgecolors='gray',linewidth=.25,norm=LogNorm(vmin=.08,vmax=600));ax.add_collection(pc);ax.autoscale_view();ax.set_aspect('equal');ax.set(xlabel='um',ylabel='um');ax.set_facecolor('#444444');fig.colorbar(pc,ax=ax,label='Cytosolic calcium (uM)');title=ax.set_title('')
    def update(i):
        v=control['c'][i].copy();v[control['ablated']]=np.nan;pc.set_array(v);title.set_text(f'576 fixed cells | t={control["t"][i]:.1f} s\nPrescribed injury; no deformation');return pc,title
    animation=FuncAnimation(fig,update,frames=range(0,len(control['t']),2),interval=100)
    animation.save(OUT/'calcium_wave.gif',writer=PillowWriter(fps=10));plt.close(fig)
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__': main()
