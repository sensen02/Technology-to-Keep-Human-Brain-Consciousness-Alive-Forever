"""Compare optical-only pose and rendered eye samples with held reference streams."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

def stats(a):
    return {k:float(x) for k,x in zip(('median','p90','max'),np.percentile(a,[50,90,100]))}
def yaw(v):return np.degrees(np.arctan2(v[1],v[0]))
def angle_between(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return np.degrees(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1)))
def main(d):
    rig=json.loads((d/'rig.json').read_text())
    poses=json.loads((d/'estimated.json').read_text())
    refs=json.loads((d/'reference.json').read_text())
    rows=[]; errors=[]
    for k,p in poses.items():
        if p is None:continue
        t=refs[k]
        rp,rt=np.array(p['rot']),np.array(t['rot'])
        hp,ht=np.array(p['pos_mm']),np.array(t['pos_mm'])
        a=rp@rig['head_forward']
        y=dict(frame=int(k),sample='walk' if int(k)<=315 else 'head_control',
               time_ms=int(k)*2 if int(k)<=315 else '',head_x_mm=hp[0],head_y_mm=hp[1],head_z_mm=hp[2],
               yaw_deg=yaw(a),pitch_deg=np.degrees(np.arcsin(np.clip(a[2],-1,1))),
               cameras=p['visible_cameras'],tags=p['visible_tags'],reprojection_px=p['median_error_px'])
        for i,name in enumerate('xyz'):y[f'head_forward_{name}']=a[i]
        for i,name in enumerate('xyz'):y[f'head_up_{name}']=p['up'][i]
        current=dict(frame=int(k),head_um=1000*np.linalg.norm(hp-ht),
                     rotation_deg=np.degrees(Rotation.from_matrix(rp.T@rt).magnitude()),
                     forward_deg=angle_between(a,rt@rig['head_forward']))
        for e in rig['eyes']:
            side=e['name']
            pos=np.array(p['eyes'][side]['pos_mm']);axis=np.array(p['eyes'][side]['optical_axis'])
            current[side+'_eye_um']=1000*np.linalg.norm(pos-(ht+rt@e['pos_mm']))
            current[side+'_axis_deg']=angle_between(axis,-rt@np.array(e['rot'])[:,2])
            for i,name in enumerate('xyz'):
                y[f'eye_{side}_{name}_mm']=pos[i]
                y[f'eye_{side}_axis_{name}']=axis[i]
        rows.append(y);errors.append(current)
    with (d/'poses.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    walking=[e for e in errors if e['frame']<=315]
    real=np.load(d/'compound.npz')
    gold=np.load(d/'compound_reference.npz')
    common,i,j=np.intersect1d(real['frames'],gold['frames'],return_indices=True)
    y,t=real['readout'][i],gold['readout'][j]
    error=np.abs((y-t).sum(-1))
    a0=next(row['yaw_deg'] for row in rows if row['frame']==316)
    controls={str(r['frame']):float((r['yaw_deg']-a0+180)%360-180) for r in rows if r['frame']>=316}
    output=dict(resolved=len(errors),total=len(poses),walk_resolved=len(walking),
                pose={k:stats([r[k] for r in walking]) for k in errors[0] if k!='frame'},
                controls_yaw_from_fixed_body_deg=controls,render_reference_frames=common.tolist(),
                eye_brightness_error=stats(error),per_frame=errors,
                mean_eye_difference=[float(v) for v in np.mean(abs(real['readout'][:,0]-real['readout'][:,1]),axis=(1,2))])
    (d/'AUDIT.json').write_text(json.dumps(output,indent=2)+'\n')
    print({k:v for k,v in output.items() if k not in ('per_frame','mean_eye_difference')})
if __name__=='__main__':main(Path(sys.argv[1]))
