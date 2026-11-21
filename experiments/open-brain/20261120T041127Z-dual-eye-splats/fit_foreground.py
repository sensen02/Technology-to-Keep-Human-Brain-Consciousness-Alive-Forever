"""Prepare and fit per-frame Gaussians from calibrated silhouettes and photo colors."""
import json,os,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter
from seed_rgb import write_gaussians

base=Path(sys.argv[1]).resolve()
frames=list(json.loads(Path(sys.argv[2]).read_text()))
exe=Path(sys.argv[3]).resolve()
models=base/'models';models.mkdir(exist_ok=True)
logs=base/'logs';logs.mkdir(exist_ok=True)
def prepare(folder):
    meta=json.loads((folder/'pose.json').read_text())
    grid=np.mgrid[-.245:.245:.0045,-.193:.193:.0045,-.12:.105:.0045].reshape(3,-1).T
    xyz=grid+np.array(meta['target'])
    count=np.zeros(len(xyz),np.int32);colors=np.zeros_like(xyz)
    color_count=np.zeros(len(xyz),np.int32)
    train=[r for i,r in enumerate(meta['frames']) if i not in [2,8]]
    for cam in train:
        c=np.array(cam['transform_matrix'])
        cv=(xyz-c[:3,3])@c[:3,:3]@np.diag([1.,-1.,-1.])
        px=np.floor(cv[:,0]/cv[:,2]*cam['fl_x']+cam['cx']).astype(int)
        py=np.floor(cv[:,1]/cv[:,2]*cam['fl_y']+cam['cy']).astype(int)
        inside=(cv[:,2]>0)&(px>=0)&(py>=0)&(px<cam['w'])&(py<cam['h'])
        px=np.clip(px,0,cam['w']-1);py=np.clip(py,0,cam['h']-1)
        with Image.open(folder/cam['file_path']) as im:rgba=np.asarray(im.convert('RGBA'))
        mask=maximum_filter(rgba[:,:,3],size=3)
        count+=inside&(mask[py,px]>=100)
        have=inside&(rgba[py,px,3]>=128)
        colors[have]+=rgba[py[have],px[have],:3]
        color_count+=have
    hit=(count>=len(train)-1)&(color_count>=2)
    if hit.sum()<80:raise RuntimeError(f'Insufficient silhouettes for {folder}: {hit.sum()}')
    xx=xyz[hit]
    col=colors[hit]/color_count[hit,None]/255.
    write_gaussians(folder/'init.ply',xx,col,np.full(xx.shape,.0032))
    (folder/'transforms.json').write_text(json.dumps(dict(frames=train,ply_file_path='init.ply')))
    return len(xx)
for k in frames:
    folder=base/k
    if not (folder/'pose.json').exists():
        print('Waiting for capture of',k,flush=True)
        import time
        for _ in range(100):
            time.sleep(.5)
            if (folder/'pose.json').exists():break
    n=prepare(folder)
    subenv=dict(os.environ,DRI_PRIME='1',RUST_LOG='brush_cli=info')
    with (logs/(k+'.log')).open('w') as f:
        subprocess.run([str(exe),str(folder),'--total-steps','1500','--max-splats','22000',
                      '--export-path',str(models),'--export-name',f'{k}.ply','--sh-degree','0',
                      '--refine-every','150','--growth-stop-iter','1050',
                      '--mean-noise-weight','2.0','--match-alpha-weight','1.0'],env=subenv,stdout=f,stderr=f,check=True)
    if not (models/f'{k}.ply').exists():raise RuntimeError(k+' model export missing')
    print('Frame',k,'initial voxels',n,flush=True)
