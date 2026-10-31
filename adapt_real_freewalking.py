"""Real-video reconstructed keypoint reference, never a controller or command replay."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('/home/sensen/Desktop/engine修复验收/real-freewalking-3d/20261027T150000Z')
def main():
 names=json.loads((D/'info__kp_names.json').read_text());raw=np.load(D/'bout_0000__kp_data.npy').reshape(-1,50,3).astype(float)
 n=len(raw);start=int(np.load(D/'info__start_frames.npy')[0]);end=int(np.load(D/'info__end_frames.npy')[0]);length=int(np.load(D/'info__clip_lengths.npy')[0])
 print('FRAME_METADATA',n,start,end,length)
 # Only source-index based time is exported; no acquisition timestamp supplied.
 # Dataset frame count must match original 800Hz frame range (not blindly use repository 1kHz default).
 assert n==length and end-start in (n,n-1),(n,start,end,length)
 frames=start+np.arange(n);time=frames/800.
 center=raw[:,names.index('Scutellum')];front=raw[:,names.index('Antenna_Base')]
 x=front-center;x/=np.linalg.norm(x,axis=1)[:,None]
 lateral=raw[:,names.index('WingL_base')]-raw[:,names.index('WingR_base')]
 y=lateral-(lateral*x).sum(1)[:,None]*x;y/=np.linalg.norm(y,axis=1)[:,None];z=np.cross(x,y)
 basis=np.stack([x,y,z],axis=2)
 legs=['T1L','T2L','T3L','T1R','T2R','T3R'];tips=raw[:,[names.index(k+'_TaTip') for k in legs]]
 relative=np.einsum('tli,tij->tlj',tips-center[:,None,:],basis)
 displacement=center-center[0]
 # Unknown coordinate scale: preserve source units, never infer mm from plausible body length.
 angles={}
 for leg in legs:
  ids=[names.index(leg+s) for s in ('_Tro','_FeTi','_TiTa')]
  u=raw[:,ids[0]]-raw[:,ids[1]];v=raw[:,ids[2]]-raw[:,ids[1]]
  cosine=(u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1));a=np.degrees(np.arccos(np.clip(cosine,-1,1)))
  angles[leg]=dict(min_deg=float(np.nanmin(a)),max_deg=float(np.nanmax(a)),definition='3-point Tro-FeTi-TiTa interior angle, not model DOF')
 # Step sequence: positive-to-negative forward excursion maxima, descriptive events only.
 events=[]
 for j,leg in enumerate(legs):
  f=relative[:,j,0];velocity=np.diff(f)
  for i in np.flatnonzero((velocity[:-1]>0)&(velocity[1:]<=0))+1:
   events.append(dict(leg=leg,source_frame=int(frames[i]),time_from_recording_start_s=float(time[i]),event='local forward excursion maximum; NOT contact/swing label'))
 events.sort(key=lambda v:v['time_from_recording_start_s'])
 np.savez_compressed(D/'REFERENCE.npz',source_frame=frames,time_from_recording_start_s=time,raw_keypoints=raw,body_reference=center,body_basis_world_columns=basis,foot_world=tips,foot_body=relative,body_displacement=displacement,missing_mask=~np.isfinite(raw))
 metadata=dict(source_type='real animal video markerless 3D reconstruction, curated postprocessed kp_data; not direct sensor coordinates or simulation',bout='bout_0000',selection='lexicographically first curated running bout, complete 685 frames; no performance selection',animal_record='session1/2025_10_07_17_15_30',fps=800,time='source frame/800; original acquisition timestamps unavailable',start_frame=start,end_frame=end,length=n,position_units='SOURCE_UNITS_UNKNOWN; no mm/m conversion made',coordinate_frame='source lab frame unspecified axes; body origin Scutellum, x toward Antenna_Base, y orthogonalized WingR-to-WingL, z=x cross y',confidence=None,confidence_reason='not present in curated bout fields',contact=None,stance_swing=None,reason='no contact sensor; excursion extrema are descriptive not contact labels',leg_ids=legs,joint_ranges=angles,step_sequence=events,displacement_source_units=displacement[-1].tolist(),physical_displacement_mm=None,unit_conversion=None,missing_values=int((~np.isfinite(raw)).sum()))
 (D/'REFERENCE_METADATA.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
 # Independent loop/dot and inverse-coordinate verification, no shared einsum expression.
 r=np.load(D/'REFERENCE.npz');maxerr=0.
 for i in range(n):
  for j in range(6):
   v=tips[i,j]-center[i];expected=np.array([np.dot(v,axis) for axis in (x[i],y[i],z[i])]);maxerr=max(maxerr,float(np.max(np.abs(expected-r['foot_body'][i,j]))))
 np.testing.assert_allclose(np.einsum('tlj,tij->tli',relative,basis)+center[:,None,:],tips,atol=1e-12)
 np.testing.assert_array_equal(r['raw_keypoints'],np.load(D/'bout_0000__kp_data.npy').reshape(n,50,3))
 np.testing.assert_allclose(np.diff(time),np.full(n-1,1/800),atol=1e-12)
 assert maxerr<1e-12
 (D/'VERIFICATION.json').write_text(json.dumps(dict(passed=True,exit_code=0,frame_range_checked=True,raw_arrays_exact=True,coordinate_dot_max_error=maxerr,inverse_transform_verified=True,unit_conversion='NOT_PERFORMED_UNKNOWN_SOURCE_SCALE',independent_timestamps='UNAVAILABLE'),indent=2))
 fig,axs=plt.subplots(3,3,figsize=(15,12))
 for j,leg in enumerate(legs):
  ax=axs.flat[j];ax.plot(relative[:,j,0],relative[:,j,2]);ax.set_title(leg+' foot relative to body');ax.set_xlabel('forward (source units)');ax.set_ylabel('dorsal (source units)');ax.axis('equal')
 axs.flat[6].plot(center[:,0],center[:,1]);axs.flat[6].set_title('Scutellum lab XY (reconstructed)');axs.flat[6].axis('equal')
 axs.flat[7].plot(time-time[0],displacement);axs.flat[7].set_title('Body displacement / source units');axs.flat[7].legend(['X','Y','Z'])
 for j,leg in enumerate(legs):axs.flat[8].plot(time-time[0],relative[:,j,0],label=leg)
 axs.flat[8].legend(fontsize=7);axs.flat[8].set_title('Six feet forward excursion; not contact')
 fig.suptitle('Real freely running fly / 3D reconstructed pose / unknown coordinate units / no contact truth');fig.tight_layout();fig.savefig(D/'REAL_WALKING_REFERENCE.png',dpi=140);plt.close(fig)
 Path('/home/sensen/Desktop/REAL_FLY_FREEWALKING_REFERENCE.png').write_bytes((D/'REAL_WALKING_REFERENCE.png').read_bytes())
 print(json.dumps(metadata,ensure_ascii=False))
if __name__=='__main__':main()
