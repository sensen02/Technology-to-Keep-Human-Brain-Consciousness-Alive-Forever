"""Read-only reconstruction of the same-individual BANC handedness deposit.
Originals unmodified; does not execute downloaded code or train an SNN.
"""
from pathlib import Path
import sys,json,hashlib,urllib.request,csv
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s4-banc-individual-behavior/20261101T211209Z')
sys.path.insert(0,str(O/'REPRO_ENV'))
import h5py
P=O/'ORIGINAL/BANC_fly_behavior';RAW=P/'raw_data';PREFIX='10-25-2020-19-16-33__'
COMMIT='4e76312dff341b934b3d75e23cd57b08e6eb02f5'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def main():
    assert (O/'DOWNLOAD_VERIFICATION.json').exists() and not (O/'AUDIT.json').exists()
    inputs=[Path(r['file']) for r in json.loads((O/'DOWNLOAD_VERIFICATION.json').read_text())['files']]
    hashes={str(p):sha(p) for p in inputs}
    dump(O/'ANALYSIS_PREREGISTRATION.json',dict(scope='S1/S3/S4 raw behavior audit, all96 cohort crosschecks, identify individual73 only from supplied notebook; no SNN or physics training',csv_tolerance=dict(centroid_abs=.000051,time_abs=.000501,rBias_abs=5.1e-10,nTurns_exact=True),timestamp='Retain both legacy float32 cumsum and float64 sum of original intervals; do not silently mix.',counts='Report raw arm records, all consecutive transitions, and upstream convention removing first arm record before differencing separately; do not force582 denominator.',split='Prospective contiguous60/20/20 of canonical float64 wall time; only training segment estimates constant-right baseline. Missing coordinates/dropped flags retained, no interpolation.',download_source_sha256=hashes,dependencies=dict(h5py=h5py.__version__,numpy=np.__version__),no_model_or_neural_changes=True))
    src=O/'UPSTREAM_SOURCE';src.mkdir()
    urls={
      'analyze_ymaze.m':f'https://raw.githubusercontent.com/de-Bivort-Lab/margo/{COMMIT}/examples/Y-Maze/analyze_ymaze.m',
      'RawDataMap.m':f'https://raw.githubusercontent.com/de-Bivort-Lab/margo/{COMMIT}/data%20classes/RawDataMap.m',
      'Data-Outputs.md':'https://raw.githubusercontent.com/wiki/de-Bivort-Lab/margo/Data-Outputs.md'}
    for name,url in urls.items():
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'data-audit'}),timeout=40) as f:data=f.read()
        (src/name).write_bytes(data)
    dump(O/'UPSTREAM_PROVENANCE.json',{name:dict(url=url,sha256=sha(src/name),commit=COMMIT if name!='Data-Outputs.md' else None) for name,url in urls.items()})
    notebook=json.loads((P/'example_code/BANC_Behavior.ipynb').read_text());text='\n\n'.join(''.join(c['source']) for c in notebook['cells'])
    (O/'NOTEBOOK_SOURCE.txt').write_text(text)
    assert 'generated from fly 73' in text and 'index 72' in text
    with h5py.File(P/'meta_data'/f'{PREFIX}Y-Maze_CSxw118b_F_1-96_Day1.mat') as h:
        group=h['#refs#/oi'];n=int(group['num_frames'][0,0]);m=int(group['num_traces'][0,0]);orientation=group['roi/orientation'][()].ravel();num_dropped=group['num_dropped'][()].ravel()
        pars=h['#refs#/Bm'];units=''.join(chr(int(c)) for c in pars['units'][()].ravel());mm_per_pix=float(pars['mm_per_pix'][0,0]);target_rate=float(pars['target_rate'][0,0]);roi=group['roi/bounds'][()][:,72]
        assert n==99426 and m==96 and units=='pixels'
    dt=np.fromfile(RAW/f'{PREFIX}time.bin',dtype='<f4');assert len(dt)==n and np.all(dt>0) and np.isfinite(dt).all()
    t32=np.cumsum(dt,dtype=np.float32);t64=np.cumsum(dt,dtype=np.float64)
    turns=np.fromfile(RAW/f'{PREFIX}Turns.bin',dtype='u1').reshape(n,m)
    cent=np.memmap(RAW/f'{PREFIX}centroid.bin',dtype='<f4',mode='r',shape=(n,2,m))[:,:,72].copy()
    drop=np.unpackbits(np.fromfile(RAW/f'{PREFIX}dropped_frames.bin','u1'),bitorder='little').reshape(n,m)
    assert np.array_equal(drop.sum(0),num_dropped)
    reference=np.genfromtxt(P/'example_csv/BANCr_rBias_nTurns.csv',delimiter=',',skip_header=1)
    cohort=[];record73=None
    for j in range(m):
        idx=np.flatnonzero(turns[:,j]>0);arms=turns[idx,j].astype(int);delta=np.diff(arms)
        right=np.isin(delta,[1,-2]) if orientation[j] else np.isin(delta,[-1,2]);legacy=right[1:]
        row=dict(fly_one_based=j+1,orientation=int(orientation[j]),raw_arm_records=len(idx),all_transition_count=len(right),all_right=int(right.sum()),upstream_bias_count=len(legacy),upstream_bias_right=int(legacy.sum()),upstream_bias=float(legacy.mean()),csv_n_turns=int(reference[j,1]),csv_rBias=float(reference[j,0]))
        assert len(right)==int(reference[j,1])
        assert abs(legacy.mean()-reference[j,0])<=5.1e-10
        cohort.append(row)
        if j==72:record73=row;frames=idx[1:];right73=right
    assert record73 is not None
    official_cent=np.genfromtxt(P/'example_csv/BANCr_centroid.csv',delimiter=',',skip_header=1)
    assert len(official_cent)==n
    assert np.array_equal(np.isfinite(cent),np.isfinite(official_cent[:,1:]))
    xyerr=float(np.nanmax(abs(cent-official_cent[:,1:])))
    terr=float(np.max(abs(t32-official_cent[:,0])))
    assert xyerr<=.000051 and terr<=.000501
    own_drop=drop[:,72].astype(bool);finite=np.isfinite(cent).all(1);valid=finite&~own_drop
    signals={}
    for field in ['speed','orientation','area','majorAxisLength','minorAxisLength']:
        signals[field]=np.memmap(RAW/f'{PREFIX}{field}.bin',dtype='<f4',mode='r',shape=(n,m))[:,72].copy()
    total=float(t64[-1]);boundary=[.6*total,.8*total]
    split=np.searchsorted(boundary,t64,side='right').astype('u1');event_split=split[frames];included=np.arange(len(frames))>0
    sections=[]
    for k,label in enumerate(['train','validation','test']):
        mask=(event_split==k)&included
        sections.append(dict(split=label,frames=int((split==k).sum()),valid_coordinates=int(((split==k)&valid).sum()),bias_events=int(mask.sum()),right_events=int(right73[mask].sum()),right_fraction=float(right73[mask].mean())))
    ptrain=sections[0]['right_fraction'];baseline=[]
    for k,label in enumerate(['validation','test'],start=1):
        y=right73[(event_split==k)&included].astype(float)
        def nll(p):return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
        baseline.append(dict(split=label,n=len(y),train_constant_p=ptrain,NLL_train_constant=nll(ptrain),NLL_balanced=nll(.5)))
    with (O/'TURN_EVENTS.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['raw_frame_zero_based','time_float64_s','time_legacy_float32_s','from_arm','to_arm','right','included_in_upstream_rBias','split'])
        seq=turns[:,72];nz=np.flatnonzero(seq>0)
        for k,frame in enumerate(frames):writer.writerow([int(frame),float(t64[frame]),float(t32[frame]),int(seq[nz[k]]),int(seq[frame]),int(right73[k]),int(k>0),int(event_split[k])])
    np.savez_compressed(O/'FLY73_ALIGNED.npz',frame=np.arange(n),dt_s=dt,time_float64_s=t64,time_legacy_float32_s=t32,centroid_pixels=cent,dropped_tracking=own_drop,finite_centroid=finite,valid_tracking=valid,split=split,turn_event_frames=frames,turn_right=right73,included_in_upstream_bias=included,**signals)
    dump(O/'COHORT_RECOMPUTE.json',cohort)
    audit=dict(individual=dict(source='BANC_Behavior.ipynb markdown and code: MATLAB fly73/Python index72',fly_one_based=73,trial_date='2020-10-25',row=record73),metadata=dict(frames=n,flies=m,units=units,mm_per_pix_metadata=mm_per_pix,mm_per_pix_caveat='1.0 under pixels mode is not an established physical mm calibration',target_rate_hz=target_rate,roi_bounds_pixels=roi,roi_orientation=int(orientation[72]),hdf5_locator='refs/oi(meta), refs/Bm(parameters), actual snapshot paths; no MATLAB object execution'),timing=dict(duration_float64_s=total,duration_float32_s=float(t32[-1]),max_csv_time_error_float32=terr,float32_minus_float64_s=float(t32[-1]-t64[-1]),median_interval_s=float(np.median(dt)),effective_rate_hz=float(n/total),max_interval_s=float(dt.max()),intervals_above_point2_s=int((dt>.2).sum())),quality=dict(valid_centroid_frames=int(finite.sum()),invalid_centroid_frames=int((~finite).sum()),dropped_frames=int(own_drop.sum()),finite_and_not_dropped_frames=int(valid.sum()),max_raw_centroid_vs_csv_error_pixels=xyerr),count_conventions=dict(raw_arm_records=583,all_transition_count=582,all_right=410,all_right_fraction=410/582,upstream_bias_count=581,upstream_bias_right=409,upstream_bias_fraction=409/581,explanation='Upstream analyze_ymaze removes the first nonzero arm record, then differences the remaining arms. CSV n_turns counts all original transitions, while CSV rBias matches the first-transition-excluded denominator for all96 flies. Do not force these denominators equal.',upstream_code_version_caveat='Pinned public MARGO commit is retrieved reference code, not proven to be the exact 2020 acquisition commit.'),cohort_checks=dict(csv_count_exact_all96=True,max_bias_error=float(max(abs(r['upstream_bias']-r['csv_rBias']) for r in cohort))),signals_available=['planar centroid pixels','elapsed intervals','ellipse orientation degrees, axis not directed heading','speed pixels/s','area/ellipse lengths','maze-arm entry events','tracking quality','ROI geometry'],signals_absent=['raw video in this17-object deposit','3D joint angles','six-foot contacts and forces','EMG or muscle activation','simultaneous neural activity','reconstructed per-neuron sensory input'],prospective_split=sections,baseline=baseline,training_executed=False,source_hashes_unchanged=hashes=={str(p):sha(p) for p in inputs},calls=dict(model_load=0,physics_step=0,neural_step=0,SNN_training=0))
    assert audit['source_hashes_unchanged'];dump(O/'AUDIT.json',audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()))
if __name__=='__main__':main()
