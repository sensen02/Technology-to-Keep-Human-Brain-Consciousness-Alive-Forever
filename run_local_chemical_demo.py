"""Finite ligand vs infinite bath with an idealized barrier and receptor mask.
No ligand species is assigned: these are illustrative generic kinetics, NOT DA
calibration or physiological hormone predictions.
"""
from pathlib import Path
import json,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from engine.local_tissue import LocalTissue,FinitePool,PrescribedBath
OUT=Path(__file__).resolve().parent/'outputs'/'brain_isolation'

def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    shape=(16,8,3); spacing=(5,5,5); capacity=np.zeros(shape);capacity[8:]=2
    link=np.zeros(shape);link[0]=100
    barriers={}
    for y in range(shape[1]):
        for z in range(shape[2]):
            a=np.ravel_multi_index((7,y,z),shape);b=np.ravel_multi_index((8,y,z),shape)
            barriers[(int(a),int(b))]=0
    specs={'finite_open':{},'finite_barrier':{'barrier_edges':barriers},
           'bath_open':{},'receptor_absent':{}}
    models={}
    for name,extra in specs.items():
        models[name]=LocalTissue(shape,spacing_um=spacing,diffusion_um2_s=50,
            receptor_capacity_nM=0 if name=='receptor_absent' else capacity,
            kon_nM_inv_s=.1,koff_s=.05,response_tau_s=5,
            finite_pools=() if name=='bath_open' else (FinitePool(3000,10,link),),
            baths=(PrescribedBath(10,link),) if name=='bath_open' else (),**extra)
    traces={k:[] for k in models};dt=.25;steps=240
    for _ in range(steps):
        for k,m in models.items():
            err=m.step(dt)
            traces[k].append([m.time_s,m.free_nM[8:].mean(),m.occupancy[8:].mean(),
                              m.pool_nM[0] if m.pool_nM.size else 10,err])
    fig,ax=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for axis,name in zip(ax[0],('finite_open','finite_barrier','bath_open')):
        m=models[name]
        im=axis.imshow(m.free_nM[:,:,1].T,origin='lower',extent=(0,80,0,40),vmin=0,vmax=10,cmap='viridis')
        axis.axvline(40,color='white',ls='--')
        axis.set(title=name+' : free ligand at 60 s',xlabel='um',ylabel='um')
        fig.colorbar(im,ax=axis,label='nM (illustrative ligand)')
    for k,rows in traces.items():
        a=np.array(rows)
        for axis,col in zip(ax[1],(1,2,3)):axis.plot(a[:,0],a[:,col],label=k)
    for axis,title,ylabel in zip(ax[1],('Receptor-side concentration','Receptor occupancy','Source reservoir concentration'),('nM','fraction','nM')):
        axis.set(title=title,xlabel='s',ylabel=ylabel);axis.legend(fontsize=8)
    fig.suptitle('GENERIC CHEMICAL MODULE — finite pool, barrier, receptors; NOT calibrated hormone effects')
    fig.savefig(OUT/'local_chemical_demo.png',dpi=150);plt.close(fig)
    metrics={'scope':'generic ligand kinetics; no neurotransmitter identity or downstream neural effect assigned',
        'all_parameters':'illustrative; geometry, kinetics, receptor density, diffusion and source not measured',
        'shape':shape,'spacing_um':spacing,'dt_s':dt,'duration_s':steps*dt,'wall_seconds':time.perf_counter()-start,
        'units':{'concentration':'nM','amount':'nM*um^3','amount_to_mol':1e-24},
        'scenarios':{k:{'initial_amount':m.initial_amount,'final_amount':m.total_amount(),
             'mass_balance_error':m.mass_balance_error(),'ledger':m.ledger,
             'receptor_side_mean_free_nM':float(m.free_nM[8:].mean()),
             'receptor_side_mean_occupancy':float(m.occupancy[8:].mean())} for k,m in models.items()}}
    for m in models.values():assert abs(m.mass_balance_error())<1e-6
    assert models['finite_barrier'].free_nM[8:].max()==0
    assert models['receptor_absent'].occupancy.max()==0
    (OUT/'local_chemical_metrics.json').write_text(json.dumps(metrics,indent=2))
    np.savez_compressed(OUT/'local_chemical_traces.npz',**{k:np.array(v) for k,v in traces.items()})
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
