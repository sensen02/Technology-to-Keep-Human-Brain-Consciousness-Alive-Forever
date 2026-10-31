"""Local curated experimental reconstruction reader, never a controller."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S=Path('/home/sensen/Desktop/engine修复验收/real-freewalking-3d/20261027T150000Z')
O=Path('/home/sensen/Desktop/engine修复验收/s4-real-sixleg-reader/20261027T212200Z')
LEGS=['T1L','T2L','T3L','T1R','T2R','T3R']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item(),allow_nan=False)+'\n')
def load():
 manifest=json.loads((S/'FINAL_MANIFEST_SHA256.json').read_text());files=['bout_0000__kp_data.npy','info__kp_names.json','REFERENCE_METADATA.json','AUTHOR_README.md','POSTPROCESS_SOURCE.py','POSTPROCESS_CONFIG.yaml','REPORT.zh-CN.md'];hashes={str(S/f):sha(S/f) for f in files}
 for f in files:assert hashes[str(S/f)]==manifest[f]
 names=json.loads((S/'info__kp_names.json').read_text());raw=np.load(S/'bout_0000__kp_data.npy').reshape(-1,50,3).astype(float);meta=json.loads((S/'REFERENCE_METADATA.json').read_text());frames=np.arange(meta['start_frame'],meta['end_frame']+1);assert len(frames)==len(raw)==685 and len(names)==50 and len(set(names))==50
 origin=raw[:,names.index('Scutellum')];x=raw[:,names.index('Antenna_Base')]-origin;x=x/np.linalg.norm(x,axis=1)[:,None];y=raw[:,names.index('WingL_base')]-raw[:,names.index('WingR_base')];y-=np.sum(x*y,axis=1)[:,None]*x;y/=np.linalg.norm(y,axis=1)[:,None];z=np.cross(x,y);basis=np.stack([x,y,z],axis=1);feet=raw[:,[names.index(l+'_TaTip') for l in LEGS]];relative=np.einsum('fij,flj->fli',basis,feet-origin[:,None,:]);recovered=np.einsum('fij,fli->flj',basis,relative)+origin[:,None,:];np.testing.assert_allclose(recovered,feet,atol=1e-12);return names,raw,frames,origin,basis,feet,relative,hashes,meta

def main():
 O.mkdir(parents=True,exist_ok=False);names,raw,frames,origin,basis,feet,relative,hashes,meta=load();time=frames/800.;dump(O/'SOURCE_CONTRACT.json',dict(source_type=meta['source_type'],paper='https://doi.org/10.64898/2026.05.03.722293',dataset='https://drive.google.com/drive/folders/1flBiyFmJYWPA6EIN4Xoh_2Lc5VT2_AoV',file='Full_running_dataset.h5',public_file_id='1jTAdpwAaAJprKIBgn17PVJseQlFqYrSn',original_whole_file_sha256=None,whole_file_reason='not downloaded whole; reuse exact extracted original NPY and historical HTTP range block hashes',source_repository='https://github.com/elliottabe/3d_tracking_dataset',schema_commit='d346fcc50bc67e41ef57f98a44c0f53bebb2ab8f',animal_record=meta['animal_record'],bout=meta['bout'],source_frame_range=[int(frames[0]),int(frames[-1])],source_hashes=hashes,license=dict(paper='CC BY-NC4.0 per previous archived source review',external_dataset='UNRESOLVED no independent dataset license established; do not infer paper license covers dataset'),measurement='real animal multiview markerless reconstruction, curated upstream processed coordinates; not direct sensor truth',upstream_interpolation='postprocessing present; cannot certify no upstream fill/filter; this reader performs no temporal interpolation',units='SOURCE_UNITS_UNKNOWN',source_axes='NOT_ESTABLISHED',adapter_axes='Scutellum origin; x->Antenna_Base; y orthogonalized WingR->WingL; z=x cross y',timing='nominal800Hz, source frame/800; no independent acquisition timestamps',confidence=None,ground_plane=None,contact_labels=None,physical_world_speed=None,acceptance='PARTIAL: real6leg3D reader verified; physical unit/source axis/license not established'))
 np.savez_compressed(O/'REFERENCE_READOUT.npz',source_frame=frames,nominal_record_time_s=time,relative_time_s=time-time[0],raw_source_keypoints=raw,raw_missing_mask=~np.isfinite(raw),body_origin_source=origin,body_basis_rows=basis,feet_source=feet,feet_body_relative_source_units=relative);dump(O/'KEYPOINT_NAMES.json',names);dump(O/'INTEGRITY.json',dict(exit_code=0,frames=len(frames),keypoints=len(names),shape=raw.shape,missing_values=int((~np.isfinite(raw)).sum()),minimum_nominal_dt_s=float(np.diff(time).min()),maximum_nominal_dt_s=float(np.diff(time).max()),independent_timestamps_available=False,body_orientation_finite=bool(np.isfinite(basis).all()),orthogonal_basis_max_error=float(abs(basis@np.swapaxes(basis,1,2)-np.eye(3)).max()),source_reconstruction_inverse_passed=True,missing_confidence=True,missing_physical_units=True,missing_ground=True,missing_contact=True,missing_source_axis_definition=True))
 fig=plt.figure(figsize=(17,12));ax=fig.add_subplot(221,projection='3d')
 for i,leg in enumerate(LEGS):ax.plot(*relative[:,i,:].T,label=leg)
 ax.set_title('Body-relative toe trajectories / SOURCE UNITS UNKNOWN');ax.legend();snap=fig.add_subplot(222,projection='3d');frame=0;snap.scatter(*raw[frame].T,s=10,color='grey')
 for leg in LEGS:
  indices=[j for j,name in enumerate(names) if name.startswith(leg+'_')];snap.scatter(*raw[frame,indices].T,label=leg)
 snap.set_title('All50 upstream reconstructed points / source frame6555');snap.legend()
 for panel,component in [(223,0),(224,2)]:
  a=fig.add_subplot(panel)
  for i,leg in enumerate(LEGS):a.plot(time-time[0],relative[:,i,component],label=leg)
  a.set_title('Body-relative '+('forward' if component==0 else 'adapter z')+' / source units');a.set_xlabel('nominal seconds from frame/800');a.legend(ncol=3);a.grid(alpha=.2)
 fig.suptitle('Experimental multiview-derived freely running fly / not simulation / no contact inference');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(O/'REAL_SIXLEG_REFERENCE.png',dpi=130);plt.close(fig);Path('/home/sensen/Desktop/S4_REAL_SIXLEG_REFERENCE.png').write_bytes((O/'REAL_SIXLEG_REFERENCE.png').read_bytes());assert all(sha(Path(p))==v for p,v in hashes.items());dump(O/'EXECUTION.json',dict(command=sys.executable+' '+str(Path(__file__)),exit_code=0,network_download=0,model_load=0,forward=0,body_step=0,neural_step=0,failures=[]));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('verified685 frames50 points6legs; physical units/axes/license still unresolved')
if __name__=='__main__':main()
