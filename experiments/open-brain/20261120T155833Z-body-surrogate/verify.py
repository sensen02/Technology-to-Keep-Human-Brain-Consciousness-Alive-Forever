"""Build an inference fixture, check decoded data alignment and report command coverage."""
from paths import *
from train import load
from model import FeedbackSampler
from skeleton import ObservedBody
import torch
import hashlib
import subprocess
import sys
import tempfile

def main():
    ref=read(ARCHIVE/'seed11701/INPUTS.json')
    an=read(BASE/'20261117T235930Z-ascending-reference-fit/BINDING.json')['records']
    roots=ref['dn_roots']+[b['banc_root'] for b in an]
    catalog=read(BASE/'20261117T051249Z-bidirectional/CATALOG.json')['ports']
    binding={'frame_ms':5,'an_bin_ms':1,'dn_roots':roots[:17],'an_roots_left_right':roots[17:],
        'groups':[catalog[r]['group'] for r in roots],
        'legs':LEGS,'markers':['thorax_center_tag','thorax_forward_tag','thorax_side_tag']+
        [leg+'_'+joint for leg in LEGS for joint in ('knee','ankle','tip')]}
    (HERE/'IO.json').write_text(json.dumps(binding,indent=2))
    counts=np.zeros(17,dtype=int)
    for f in sorted((HERE/'data').glob('*')):
        with np.load(f/'nerve.npz') as z,np.load(f/'external.npz') as ext,np.load(f/'matrix.npz') as mat:
            assert z['roots'].tolist()==roots and z['dn'].shape==(20000,17) and z['an'].shape==(20000,2)
            assert mat['adc'].shape==(100000,64) and ext['markers_mm'].shape==(400,21,3)
            if int(f.name.split('_',1)[0]) in (10901,11311):
                counts+=z['check_dn_arrived'].sum(0)
    _,p,f,dn,_=next(row for row in load(11701) if row[0].endswith('intact'))
    inputs=(p[100:120],p[101:121],f[101:121],dn[101:121])
    np.savez_compressed(HERE/'example_input.npz',**{k:x.numpy() for k,x in zip(
        ('previous_mm','current_mm','touch_uN','dn_counts'),inputs)})
    with tempfile.TemporaryDirectory() as d:
        out=Path(d)/'next.npz'
        command=[sys.executable,str(HERE/'predict.py'),str(HERE/'example_input.npz'),str(out)]
        subprocess.run(command,check=True)
        net=ObservedBody(HERE/'weights.pt').eval()
        with torch.no_grad():
            poses,logits=net(*inputs)
        with np.load(out) as z:
            assert np.allclose(z['next_mm'],poses.numpy(),rtol=0,atol=1e-5)
            assert np.allclose(z['an_prob'],logits.sigmoid().numpy(),atol=1e-6)
            spikes=z['an_body_source'].reshape(-1,2)
            assert all(np.all(np.diff(np.flatnonzero(spikes[:,side]))>=3) for side in range(2))
            test_count=int(spikes.sum())
    result=dict(validation_seed=11311,test_seed=11701,development_dn_arrivals=counts.tolist(),
                active_development_dn=int((counts>0).sum()),silent_development_groups=[
                    catalog[r]['group'] for r,x in zip(roots,counts) if x==0],successful_cli_frames=20,
                fixture_an_events=test_count,sha256_weights=hashlib.sha256((HERE/'weights.pt').read_bytes()).hexdigest())
    (HERE/'verification.json').write_text(json.dumps(result,indent=2))
    print(result)

if __name__=='__main__':main()
