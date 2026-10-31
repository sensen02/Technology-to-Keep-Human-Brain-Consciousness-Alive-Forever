"""S299 Stage A figure: input dispersion, output rate compression, CI."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
S289 = Path('/home/sensen/Desktop/engine修复验收/s289-decoder-range-support/20261026T173400Z/decoder_original')
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output
    z=np.load(S289/'NEURAL.npz');sc=np.asarray(z['sensory_current'],float);V=np.asarray(z['decoder_motor_gains'],float)
    rates=np.asarray(z['rates_hz'],float);nz=np.flatnonzero(V!=0)
    drv=np.flatnonzero(np.abs(sc).max(0)>0);sig=sc[:,drv];keep=[]
    for j in range(sig.shape[1]):
        if not any(np.array_equal(sig[:,j],sig[:,k]) for k in keep): keep.append(j)
    U=sc[:,drv[keep]];um=U.mean(0)
    ci=json.loads((out/'COMPRESSION_INDEX.json').read_text())
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(14,10));t=np.arange(1,401)
    a0=ax[0,0]
    order=np.argsort(um)
    for r,j in enumerate(order): a0.plot(t,U[:,j]+r*0.6,lw=.8)
    a0.set_title('29 unique driven input series (offset for visibility): leg-level magnitudes differ, means %.2f..%.2f'%(um.min(),um.max()))
    a0.set_xlabel('ms')
    a1=ax[0,1]
    rm=rates[:,nz].mean(0)
    a1.hist(rm,bins=40);a1.set_title('per-channel filtered-rate means over 64 activated channels: %.1f..%.1f Hz (10-90%%: %.1f..%.1f)'%(rm.min(),rm.max(),np.percentile(rm,10),np.percentile(rm,90)))
    a1.set_xlabel('Hz');a1.set_ylabel('channels')
    a2=ax[1,0]
    a2.bar(['D_in','D_out'],[ci['D_in'],ci['D_out']],color=['tab:blue','tab:red'])
    a2.set_yscale('log');a2.set_title('compression index: D_in %.3f -> D_out %.3f, CI %.4f'%(ci['D_in'],ci['D_out'],ci['CI']))
    a3=ax[1,1];a3.axis('off')
    rs=json.loads((out/'READOUT_STRUCTURE.json').read_text())
    txt=('readout: %d pool neurons in %d distinct sets\ncross-leg shared sets: %d\noverlap of pool neurons with the 29 driven series: %d\n\ninput: %d driven cells, %d unique series\noutput: %.1f%% spread of per-channel rate means\n\nstage B/C: NOT EXECUTED (see STAGE_STATUS.json)'
         %(rs['pool_neurons'],rs['distinct_pool_sets'],rs['cross_leg_shared_count'],rs['overlap_pool_neurons_with_29_driven_series'],
           json.loads((out/'INPUT_AUDIT.json').read_text())['driven_cells'],U.shape[1],
           json.loads((out/'OUTPUT_AUDIT.json').read_text())['across_channel_spread_percent']))
    a3.text(0,0.95,txt,va='top',fontsize=10,family='monospace')
    fig.suptitle('S299 stage A: where leg information is lost (offline, zero dynamic calls)',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.95));p=out/'WORKPOINT_STAGE_A.png';fig.savefig(p,dpi=145)
    (out/'FIGURE.json').write_text(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps(dict(figure=str(p))))
if __name__=='__main__': raise SystemExit(main())
