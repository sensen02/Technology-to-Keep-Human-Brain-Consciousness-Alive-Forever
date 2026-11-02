"""Acquire author NMF2 continuous annotations; numeric conversion only, no controller.
Usage: venv_body/bin/python -B tools/acquire_continuous_fly_kinematics.py --output PATH
Downloaded source is CSV only. No upstream code executes. Existing raw bytes are reused.
"""
import argparse, csv, hashlib, io, json, subprocess
from pathlib import Path
import numpy as np

URL = 'https://dataverse.harvard.edu/api/access/datafile/10407643'
META = 'https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/3MCEYR'
MD5 = 'ddf066d0a20ab9aac275bf3304baade0'
LEGS = ['LF','LM','LH','RF','RM','RH']
MARKERS = ['ThC','CTr','FTi','TiTa','Claw']

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def fetch(url,p):
    if not p.exists(): subprocess.run(['curl','-L','--fail','--max-time','60','-A','Mozilla/5.0',url,'-o',str(p)],check=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();p=a.output;p.mkdir(parents=True,exist_ok=True)
    raw=p/'annotated_3d_walking_kinematics.csv';fetch(URL,raw)
    assert hashlib.md5(raw.read_bytes()).hexdigest()==MD5
    fetch(META,p/'NMF2_DATAVERSE_METADATA.json')
    reader=csv.DictReader(io.StringIO(raw.read_text()));rows=list(reader);fields=reader.fieldnames
    print('CSV fields',fields)
    frame=np.array([int(r['frame_idx']) for r in rows],dtype=np.int64)
    trial=np.array([int(r['video_id']) for r in rows],dtype=np.int64)
    # Infer only the spelling convention, never reinterpret numeric axes/units.
    def col(m,c):
        choices=[m+'_'+c,m+c,m+'.'+c]
        found=[x for x in choices if x in fields]
        if len(found)!=1: raise ValueError((m,c,found,fields))
        return found[0]
    xyz=np.array([[[[float(r[col(leg+'-'+marker,c)]) for c in 'xyz'] for marker in MARKERS] for leg in LEGS] for r in rows],dtype=np.float64)
    n=len(rows);assert xyz.shape==(n,6,5,3) and np.all(np.diff(frame)==1) and len(set(trial))==1
    finite=np.isfinite(xyz);valid=finite.all(-1);time=(frame-frame[0])/120.
    bones=np.linalg.norm(np.diff(xyz,axis=2),axis=-1)
    # Geometric unsigned interior angles, not signed anatomical Euler DOFs.
    angles=np.full((n,6,3),np.nan)
    for k in range(3):
        u=xyz[:,:,k]-xyz[:,:,k+1];v=xyz[:,:,k+2]-xyz[:,:,k+1]
        den=np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1)
        ok=np.isfinite(u).all(-1)&np.isfinite(v).all(-1)&(den>0)
        angles[:,:,k][ok]=np.arccos(np.clip(np.sum(u*v,axis=-1)[ok]/den[ok],-1,1))
    # Single trial time holdout. No frame shuffled, no temporal interpolation.
    split=np.full(n,2,dtype=np.int8);split[:int(n*.6)]=0;split[int(n*.6):int(n*.8)]=1
    np.savez_compressed(p/'NMF2_CONTINUOUS.npz',xyz_source=xyz,coordinate_valid=finite,marker_valid=valid,source_frame=frame,trial_id=trial,time_relative_s=time,leg_names=np.array(LEGS),marker_names=np.array(MARKERS),xyz_axis_names=np.array(list('xyz')),bone_lengths_source=bones,geometric_interior_angle_rad=angles,angle_valid=np.isfinite(angles),split=split)
    with (p/'NMF2_CONTINUOUS_LONG.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['source_frame','relative_time_s','video_id','leg','marker','x_source','y_source','z_source','valid','split'])
        for i in range(n):
            for l,leg in enumerate(LEGS):
                for j,m in enumerate(MARKERS):w.writerow([frame[i],time[i],trial[i],leg,m,*xyz[i,l,j],int(valid[i,l,j]),int(split[i])])
    repeat={str(k):bool(np.array_equal(xyz[k:],xyz[:-k],equal_nan=True)) for k in range(1,n//2+1)}
    q=dict(shape=list(xyz.shape),frames=n,source_frame_first=int(frame[0]),source_frame_last=int(frame[-1]),nominal_hz=120,nominal_dt_s=1/120,span_s=float(time[-1]),monotonic=True,finite_coordinates=int(finite.sum()),missing_coordinates=int((~finite).sum()),all_six_legs_share_frame_and_video=True,trial_count=1,animal_identity='same video_id=0; individual identifier not published in CSV',adjacent_marker_displacement_source_max=float(np.linalg.norm(np.diff(xyz,axis=0),axis=-1).max()),bone_lengths_min=bones.min(0).tolist(),bone_lengths_max=bones.max(0).tolist(),bone_lengths_coefficient_variation=(bones.std(0)/bones.mean(0)).tolist(),bone_adjacent_jump_source_max=float(abs(np.diff(bones,axis=0)).max()),exact_periodic_repetition=repeat,left_right_exact_equal=[bool(np.array_equal(xyz[:,i],xyz[:,i+3])) for i in range(3)],angle_range_rad=[float(np.nanmin(angles)),float(np.nanmax(angles))],split_frame_counts=[int((split==k).sum()) for k in range(3)])
    dump(p/'NMF2_QUALITY.json',q)
    dump(p/'NMF2_FIELDS_AND_TRANSFORMS.json',dict(source_url=URL,doi='10.7910/DVN/3MCEYR',license='CC0-1.0 per official dataset metadata',source_md5=MD5,source_sha256=sha(raw),file_size=raw.stat().st_size,record='annotated_3d_walking_kinematics.csv / video_id0 / frames29..65',frame_sampling='video360Hz; annotation every third frame ->120Hz per paper; CSV does not contain measured timestamps',interpolation='no interpolation in this conversion; upstream 3D annotation/body alignment remains upstream processing',source_units='NOT_ESTABLISHED; no conversion to mm',source_axes='NOT_ESTABLISHED; xyz retained verbatim',root='Th fixed at(0,0,200); no measured world root translation exported',synchrony='same CSV row/frame_idx/video_id for six legs; preserve order and phase',markers=dict(ThC='thorax-coxa',CTr='coxa-trochanter',FTi='femur-tibia',TiTa='tibia-tarsus',Claw='distal claw'),angles='three-point unsigned interior angle, atan/DOF/zero/sign anatomical mapping not claimed; radians',confidence=None,reprojection_error=None,contact=None,missing='coordinate_valid/marker_valid/angle_valid; original missing values remain NaN',transforms=['numeric CSV float64 parse', 'six-leg marker field projection by exact names', '(frame_idx-first)/120 nominal relative seconds', 'Euclidean bone length', 'arccos clipped normalized dot product for interior angles'],split='one short trial: chronological60/20/20, strong dependence/too short for biological generalization; prefer add independent animal/trial',split_codes={'0':'train','1':'validation','2':'test'},no_model_training=True,no_simulation=True))
    print(json.dumps(q,ensure_ascii=False))
if __name__=='__main__':main()
