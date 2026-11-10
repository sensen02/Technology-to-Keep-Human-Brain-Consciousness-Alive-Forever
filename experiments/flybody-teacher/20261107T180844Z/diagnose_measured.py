"""Analyze recorded measured teacher failures without changing or stepping models."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('folders',nargs='+',type=Path);args=ap.parse_args();allrows=[]
    fig,axes=plt.subplots(len(args.folders),3,figsize=(15,4*len(args.folders)),squeeze=False)
    for row,p in enumerate(args.folders):
        meta=json.loads((p/'MODEL_IDENTITY.json').read_text());result=json.loads((p/'RESULT.json').read_text());w=meta['weight']
        with np.load(p/'NATIVE.npz') as f:rows=f['rows'];contacts=f['contacts'];en=f['environmental']
        with np.load(p/'TRACKING.npz') as f:tr=f['rows']
        with np.load(p/'BODY.npz') as f:q=f['qpos'];t=f['time']
        with np.load(p/'REFERENCE.npz') as f:ref=f['qpos']
        bygeom={}
        for c in contacts:
            if c[3]>=0 or c[10]<=1e-9:continue
            names=[meta['geom_names'][int(c[1])],meta['geom_names'][int(c[2])]]
            name=next((s for s in names if s.startswith('walker/')),'UNKNOWN')
            item=bygeom.setdefault(name,dict(first_s=float(c[0]),last_s=float(c[0]),max_positive_z=0.,sample_count=0))
            item['last_s']=float(c[0]);item['max_positive_z']=max(item['max_positive_z'],float(c[10]));item['sample_count']+=1
        report=dict(folder=str(p),controls=result['controls'],duration_s=result['sim_s'],expected_controls=result['expected_controls'],nonclaw_ground_contact=bygeom,
            positive_nonclaw_peak_over_weight=float(rows[:,7].max()/w),initial_50ms_nonclaw_peak=float(rows[rows[:,0]<=.05,7].max()),later_nonclaw_peak=float(rows[rows[:,0]>.05,7].max()) if np.any(rows[:,0]>.05) else None,
            root_error_max_native=float(tr[:,1].max()),joint_tracking_rms_rad=float(np.sqrt(np.mean(tr[:,2]**2))),
            root_orientation_error_max_rad=float((2*np.arccos(np.clip(tr[:,3],0,1))).max()),
            recorded_body_sha256=hashlib.sha256((p/'BODY.npz').read_bytes()).hexdigest(),root_reference_copy=False,
            interpretation='Nonclaw tarsal contact is distinguished from torso/head contact. No pass is inferred from reference-end termination or video.')
        allrows.append(report)
        axes[row,0].plot(ref[:len(q),0],ref[:len(q),1],label='Measured-data fitted root');axes[row,0].plot(q[:,0],q[:,1],label='Physical teacher');axes[row,0].axis('equal');axes[row,0].set_title(p.name);axes[row,0].set_xlabel('x (official cm)');axes[row,0].set_ylabel('y (official cm)');axes[row,0].legend()
        axes[row,1].plot(rows[:,0],rows[:,1:7].sum(1)/w,label='Positive claw reaction');axes[row,1].plot(en[:,0],en[:,6]/w,label='Adhesive root vertical');axes[row,1].plot(en[:,0],(en[:,3]+en[:,6])/w,label='Net constraint+actuator');axes[row,1].plot(rows[:,0],rows[:,7]/w,label='Nonclaw reaction');axes[row,1].set_xlabel('Physical time (s)');axes[row,1].set_ylabel('Force / original body weight');axes[row,1].legend()
        axes[row,2].plot(tr[:,0],tr[:,1],label='Root error (cm)');axes[row,2].plot(tr[:,0],tr[:,2],label='Joint RMS error (rad)');axes[row,2].set_xlabel('Physical time (s)');axes[row,2].set_title('Different units explicitly labelled');axes[row,2].legend()
    out=args.folders[0].parent
    with (out/'MEASURED_DIAGNOSTICS.json').open('x') as f:json.dump(allrows,f,indent=2)
    fig.suptitle('Measured-reference teacher: actual tracking and support; no brain/current42 acceptance');fig.tight_layout();fig.savefig(out/'MEASURED_DIAGNOSTICS.png',dpi=150)
    print(json.dumps(allrows,indent=2))

if __name__=='__main__':main()
