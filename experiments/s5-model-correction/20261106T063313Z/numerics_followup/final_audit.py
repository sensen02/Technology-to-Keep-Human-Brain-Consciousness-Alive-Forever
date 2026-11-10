"""Post-run all-cell rates, tau crosscheck and artifact audit; no simulation."""
import pickle
import numpy as np
from PIL import Image
import run as r

def main():
    summary=r.js(r.O/'SUMMARY.json');counts={};stats=[];saved={}
    for row in summary['results']:
        z=r.arrays(r.Path(row['folder'])/'NEURAL.npz');counts[row['label']]=z['counts12000']
        for w in r.WINDOWS:
            count=z['counts12000'][:w].sum(0);rate=count/(w/1000)
            saved[row['label']+'_'+str(w)+'ms_count']=count
            saved[row['label']+'_'+str(w)+'ms_Hz']=rate
            stats.append(dict(label=row['label'],window_ms=w,total_spikes=int(count.sum()),allcell_mean_hz=float(rate.mean()),percell_rate_quantiles_hz=np.quantile(rate,[0,.05,.5,.95,1])))
    comparisons=[]
    for method in r.METHODS:
        dts=[.125,.0625,.03125,.015625]
        for a,b in zip(dts[:-1],dts[1:]):
            label=lambda dt:method+'_'+str(dt).replace('.','p')+'ms'
            for w in r.WINDOWS:
                ca=counts[label(a)][:w];cb=counts[label(b)][:w]
                comparisons.append(dict(method=method,dt_ms=[a,b],window_ms=w,count_1ms=r.metrics(ca,cb),rate_1ms_hz=r.metrics(ca*1000.,cb*1000.),cumulative_cell_count=r.metrics(ca.sum(0),cb.sum(0)),percell_mean_hz=r.metrics(ca.sum(0)*1000/w,cb.sum(0)*1000/w)))
    np.savez_compressed(r.O/'WINDOW_CELL_COUNTS_RATES.npz',**saved)
    r.dump(r.O/'COUNT_RATE_DIAGNOSTICS.json',dict(stats=stats,comparisons=comparisons,meaning='All12000 cells; 1ms binned Hz/counts and per-window mean Hz. Not a new decoder.'))
    tau_saved=r.arrays(r.O/'TAU_ALL_CELLS.npz');tau_error=0.
    for dt in [.5,.25,.125,.0625]:
        cc=r.cache(dt)
        for ph,c in cc.items():
            n=c['net'];total=n.p.leak_uS+n.ge+n.gi
            expected=np.vstack([n.p.capacitance_nF/total,dt*total/n.p.capacitance_nF,n.p.capacitance_nF/(total+(n.We+n.Wi).dot(n._ring[n._ring_i]))])
            err=float(np.max(np.abs(expected-tau_saved[str(dt)+'_'+str(ph)])));tau_error=max(tau_error,err)
    assert tau_error<1e-12
    with Image.open(r.O/'LOCAL_REFINEMENT.png') as im:
        im.load();image=dict(size=im.size,mode=im.mode,decoded=True,visual_inspection=False,reason='Actual read_image attempt rejected: current model lacks image input.')
    h=r.checkhash();f=r.js(r.O/'SOURCE_FREEZE.json');assert all(r.sha(p)==v for p,v in f.items())
    r.dump(r.O/'FINAL_AUDIT.json',dict(passed=True,old_input_hash_count=len(h),source_freeze_exact=True,tau_max_error=tau_error,image=image,all_neural_jobs_finished=True,new_physical_runs=0))
    print('FINAL_AUDIT',len(h),tau_error,image,flush=True)
if __name__=='__main__':main()
