"""Independent stdlib binary indexing checks; no MATLAB execution or SNN training."""
from pathlib import Path
import json,hashlib,csv,struct,math
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s4-banc-individual-behavior/20261101T211209Z')
P=O/'ORIGINAL/BANC_fly_behavior';R=P/'raw_data';PRE='10-25-2020-19-16-33__'

def main():
    download=json.loads((O/'DOWNLOAD_VERIFICATION.json').read_text());total=0
    for row in download['files']:
        md=hashlib.md5();sha=hashlib.sha256();size=0
        with Path(row['file']).open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):md.update(block);sha.update(block);size+=len(block)
        assert size==row['bytes'] and md.hexdigest()==row['md5'] and sha.hexdigest()==row['sha256'];total+=size
    a=json.loads((O/'AUDIT.json').read_text());z=np.load(O/'FLY73_ALIGNED.npz')
    assert 'generated from fly 73' in (O/'NOTEBOOK_SOURCE.txt').read_text()
    raw=(R/f'{PRE}Turns.bin').read_bytes();frames=[j for j,c in enumerate(raw[72::96]) if c];seq=[raw[j*96+72] for j in frames]
    pairs=list(zip(seq[:-1],seq[1:]));rights=[(b-c) in (-1,2) for c,b in pairs]
    assert len(seq)==583 and len(pairs)==582 and sum(rights)==410
    assert sum(rights[1:])==409 and abs(sum(rights[1:])/len(rights[1:])-a['individual']['row']['csv_rBias'])<5.1e-10
    assert frames[1:]==z['turn_event_frames'].tolist() and rights==z['turn_right'].tolist()
    # Independent little-endian decoding and sequential float32 rounding of time intervals.
    dt=[v[0] for v in struct.iter_unpack('<f',(R/f'{PRE}time.bin').read_bytes())];N=len(dt);legacy=[];running=0.
    for v in dt:running=struct.unpack('<f',struct.pack('<f',running+v))[0];legacy.append(running)
    assert legacy==z['time_legacy_float32_s'].tolist()
    assert abs(math.fsum(dt)-float(z['time_float64_s'][-1]))<1e-9
    binary=(R/f'{PRE}centroid.bin').read_bytes()
    sample_frames=[0,1,100,1000,50000,90000,N-1]
    for frame in sample_frames:
        for axis in (0,1):
            v=struct.unpack_from('<f',binary,4*(frame*192+axis*96+72))[0];stored=float(z['centroid_pixels'][frame,axis])
            assert (math.isnan(v) and math.isnan(stored)) or v==stored
    rawdrop=(R/f'{PRE}dropped_frames.bin').read_bytes();drop=[bool(rawdrop[j*12+9]&1) for j in range(N)]
    assert drop==z['dropped_tracking'].tolist() and sum(drop)==532
    events=list(csv.DictReader((O/'TURN_EVENTS.csv').open()))
    assert len(events)==582
    for j,event in enumerate(events):
        assert int(event['raw_frame_zero_based'])==frames[j+1]
        assert int(event['right'])==int(rights[j]) and int(event['included_in_upstream_rBias'])==int(j>0)
        assert float(event['time_float64_s'])==float(z['time_float64_s'][frames[j+1]])
    cohort=list(csv.DictReader((P/'example_csv/BANCr_rBias_nTurns.csv').open(encoding='utf-8-sig')))
    assert len(cohort)==96 and int(cohort[72]['n_turns'])==582
    for s,label in enumerate(['train','validation','test']):
        es=[e for e in events if int(e['split'])==s and e['included_in_upstream_rBias']=='1'];row=a['prospective_split'][s]
        assert len(es)==row['bias_events'] and sum(int(e['right']) for e in es)==row['right_events']
    report=dict(passed=True,files_md5_and_sha256=len(download['files']),bytes=total,events=582,upstream_bias_n=581,right_in_bias=409,centroid_independent_frames=sample_frames,dropped_tracking=532,time_float32_sequential_exact=True,decode_method='stdlib byte slicing/struct and per-frame bit access, separate from main numpy reshape/unpackbits/cumsum',calls=dict(model=0,neural=0,training=0),limits='Selected centroid-frame scalar checks, full time/turn/drop records; no replay of experimental acquisition or reconstruction of missing sensory/leg/neural signals.')
    (O/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
