"""Independent, frozen-model check using the newer intact event-anchored P1 run."""
import argparse
from paths import HERE,BASE,read,np,Path
SOURCE=Path('/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261118T052426Z-event-anchored-repair/FORMAL/seed11903')
dest=HERE/'data/11903_clean_on'
def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('stage',choices=['external','electrode'])
    stage=cli.parse_args().stage
    dest.mkdir(parents=True,exist_ok=True)
    mode='clean_on'
    meta=read(SOURCE/'INPUTS.json')
    lane=meta['modes'].index(mode)
    if stage=='external':
        from observe import acquire,decode,mujoco
        model=mujoco.MjModel.from_binary_path(str(SOURCE/(mode+'.mjb')))
        pixels,maps=acquire(model,SOURCE/mode/'TRACE.npz',np.random.default_rng(11903+150*lane))
        pos,force=decode(pixels,maps)
        np.savez_compressed(dest/'external.npz',pixel_xy=pixels,taxels=np.vstack(maps),
                            offsets=np.r_[0,np.cumsum(list(map(len,maps)))],markers_mm=pos,force_uN=force)
    else:
        from electrode import library,render,extract,WIDTH
        roots=meta['dn_roots']+[r['banc_root'] for r in read(BASE/'20261117T235930Z-ascending-reference-fit/BINDING.json')['records']]
        assert roots==read(HERE/'IO.json')['dn_roots']+read(HERE/'IO.json')['an_roots_left_right']
        wave,chol=library(roots)
        with np.load(SOURCE/'NEURAL.npz') as z:
            down,up=z['dn_arrived'][:,lane],z['an_emitted'][:,lane]
        raw=render(np.c_[down,up],wave,chol,np.random.default_rng(11903*17+lane))
        calls=extract(raw,wave,chol)
        dn=np.zeros_like(calls[:,:17])
        dn[WIDTH-1:]=calls[:-WIDTH+1,:17]
        np.savez_compressed(dest/'matrix.npz',adc=raw,uv_per_code=.02,dt_us=20)
        np.savez_compressed(dest/'nerve.npz',roots=roots,dn=dn,an=calls[:,17:],
                            check_dn_arrived=down,check_an_emitted=up)
if __name__=='__main__':main()
