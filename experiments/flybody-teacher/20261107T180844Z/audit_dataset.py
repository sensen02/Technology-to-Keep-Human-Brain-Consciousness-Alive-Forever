"""Inventory reference schema and preserve official file-level partitions before rollout."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import h5py
ROOT=Path(__file__).resolve().parent
OUT=Path('/home/sensen/Desktop/engine修复验收/flybody-teacher/20261107T180844Z')

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()
def serial(v):
    if isinstance(v,bytes):return v.decode('utf8')
    if hasattr(v,'tolist'):return serial(v.tolist())
    if isinstance(v,list):return [serial(x) for x in v]
    return v

def main():
    ap=argparse.ArgumentParser();ap.add_argument('files',nargs='+',type=Path);args=ap.parse_args();reports=[]
    for path in sorted(args.files):
        with h5py.File(path,'r') as f:
            print(path, list(f),flush=True)
            schema=[]
            def visit(name,obj):
                if name.startswith('trajectories/') and name.split('/')[1]!=first:return
                row=dict(name=name,type=type(obj).__name__,attrs={k:serial(v) for k,v in obj.attrs.items()})
                if isinstance(obj,h5py.Dataset):
                    row.update(shape=list(obj.shape),dtype=str(obj.dtype))
                    if obj.size<500:row['values']=serial(obj[()])
                schema.append(row)
            first=next(iter(f['trajectories']))
            # Do not enumerate all per-trajectory data objects unnecessarily.
            for name,obj in f.items():
                if name=='trajectories':
                    visit(name,obj);visit(name+'/'+first,obj[first]);obj[first].visititems(lambda sub,o:visit(name+'/'+first+'/'+sub,o))
                else:
                    visit(name,obj)
                    if isinstance(obj,h5py.Group):obj.visititems(lambda sub,o:visit(name+'/'+sub,o))
            lens=f['trajectory_lengths'][()];dt=float(f['timestep_seconds'][()]);keys=list(f['trajectories'])
            assert len(keys)==len(lens) and len(set(keys))==len(keys)
            q=f['trajectories'][first]['root_qpos'][()];j=f['trajectories'][first]['qpos'][()]
            report=dict(path=str(path),sha256=digest(path),bytes=path.stat().st_size,top_attrs={k:serial(v) for k,v in f.attrs.items()},schema=schema,
                trajectories=len(keys),timestep_seconds=dt,frame_count=int(lens.sum()),length_min=int(lens.min()),length_max=int(lens.max()),length_median=float(np.median(lens)),
                index0_key=first,index0_frames=len(q),index0_total_s=(len(q)-1)*dt,index0_usable_control_steps=max(0,min(5000,len(q)-65)),
                first_root_start=q[0].tolist(),first_root_end=q[-1].tolist(),first_joint_shape=list(j.shape),first_finite=bool(np.isfinite(q).all() and np.isfinite(j).all()),
                official_partition='test' if 'test' in path.name.lower() else ('train' if 'train' in path.name.lower() else 'UNSPECIFIED'),
                original_individual_separation='NOT_ESTABLISHED_FROM_FILE_NAME',released_teacher_unseen_data='NOT_ESTABLISHED')
            reports.append(report)
            print(json.dumps({k:v for k,v in report.items() if k!='schema'}),flush=True)
    with (OUT/'DATASET_SCHEMA.json').open('x') as out:json.dump(reports,out,indent=2)

if __name__=='__main__':main()
