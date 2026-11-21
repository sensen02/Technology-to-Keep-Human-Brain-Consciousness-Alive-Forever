"""Score held-out photo views and assemble color and retina comparison figures."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flygym.vision.retina import Retina
import render

def measure(a,b):
    d=a-b
    ax=gaussian_filter(a,(1.5,1.5,0));bx=gaussian_filter(b,(1.5,1.5,0))
    s2=gaussian_filter(a*a,(1.5,1.5,0))-ax*ax
    t2=gaussian_filter(b*b,(1.5,1.5,0))-bx*bx
    st=gaussian_filter(a*b,(1.5,1.5,0))-ax*bx
    ss=((2*ax*bx+.0001)*(2*st+.0009))/((ax**2+bx**2+.0001)*(s2+t2+.0009))
    return dict(psnr=float(-10*np.log10(max(np.mean(d*d),1e-15))),
                mean_abs=float(np.mean(abs(d))),ssim=float(np.mean(ss)))
def rgb(path):return np.asarray(Image.open(path).convert('RGB'),np.float32)/255
def metrics(pairs):
    table={n:measure(rgb(a),rgb(b)) for n,a,b in pairs}
    return dict(count=len(table),mean_psnr=float(np.mean([v['psnr'] for v in table.values()])),
                mean_ssim=float(np.mean([v['ssim'] for v in table.values()])),per_view=table)
def retina_results(out,poses):
    retina=Retina();readout=[]
    for k in poses:
        out_frame=[]
        for side in 'LR':
            im=np.asarray(Image.open(out/'combined'/f'eye_{k}_{side}.png').convert('RGB'))
            x=retina.raw_image_to_hex_pxls(retina.correct_fisheye(im))
            out_frame.append(x)
        readout.append(out_frame)
    ids=np.array([int(k) for k in poses])
    np.savez_compressed(out/'compound.npz',frames=ids,walking_time_ms=np.where(ids<=315,ids*2,np.nan),
        eye=['L','R'],channel=['yellow','pale'],readout=np.asarray(readout),ommatidia_id_map=retina.ommatidia_id_map,
        pale_type_mask=retina.pale_type_mask)
    return np.asarray(readout),retina.ommatidia_id_map.astype(int)

def main(folder,old,out,model,fg):
    meta=json.loads((folder/'scan/calibration.json').read_text())
    poses=json.loads((old/'estimated.json').read_text())
    s=[r for r in meta['frames'] if r['split']=='holdout']
    k=[r for r in meta['frames'] if r['split']=='eyecheck']
    test=lambda records: [(Path(r['file_path']).stem,folder/'scan'/r['file_path'],out/'environment'/Path(r['file_path']).name) for r in records]
    audit=dict(environment_holdout=metrics(test(s)),eye_environment=metrics(test(k)),composite_vs_reference=metrics([
        (f'{frame}_{side}',old/'eyes'/f'{frame}_{side}.png',out/'combined'/f'eye_{frame}_{side}.png')
        for frame in poses for side in 'LR']),body_holdout={},static_gaussians=len(render.GaussianField(model).center))
    for frame in poses:
        m=json.loads((folder/'foreground'/frame/'pose.json').read_text())
        g=render.GaussianField(fg/(frame+'.ply'))
        for i in [2,8]:
            c=m['frames'][i];a=g.project(c,near=.003)
            z,*projected=a
            _,trans=render.raster(c['w'],c['h'],np.argsort(z,kind='stable'),*projected,c['fl_x'],c['fl_y'],c['cx'],c['cy'])
            ref=np.asarray(Image.open(folder/'foreground'/frame/c['file_path']).convert('RGBA'))[...,3]>=128
            pred=(1-trans)>=.5
            intersection=(ref&pred).sum()
            union=(ref|pred).sum()
            audit['body_holdout'][frame+f'_{i}']=float(intersection/union)
    lengths=list(audit['body_holdout'].values())
    audit['foreground_median_mask_iou']=float(np.median(lengths))
    values,mask=retina_results(out,poses)
    prior=np.load(old/'compound.npz')['readout']
    errors=np.abs(values.sum(axis=-1)-prior.sum(axis=-1))
    audit['compound_abs_error']=dict(zip(['median','p90','max'],[float(v) for v in np.percentile(errors,[50,90,100])]))
    (out/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    cases=[211,317,318]
    fig,axs=plt.subplots(3,5,figsize=(12,7),layout='constrained')
    titles=['Tracking camera','3DGS left eye','3DGS right eye','Left ommatidia','Right ommatidia']
    for i,n in enumerate(cases):
        vals=values[list(poses).index(str(n))].sum(-1)
        for j,ax in enumerate(axs[i]):
            ax.set_axis_off()
            if j==0:im=Image.open(old/'external'/f'{n}_H02_NW.png')
            elif j<3:im=Image.open(out/'combined'/f'eye_{n}_{"LR"[j-1]}.png')
            else:im=np.where(mask>0,vals[j-3][np.maximum(mask-1,0)],np.nan)
            ax.imshow(im,cmap='gray',vmin=0,vmax=.67)
            if i==0:ax.set_title(titles[j],fontsize=10)
            if j==0:ax.text(0,-.045,{211:'Walking • 422 ms',317:'+25° turn',318:'−25° turn'}[n],transform=ax.transAxes)
    fig.savefig(out/'DUAL_SPLATS.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,4,figsize=(11,6),layout='constrained')
    for j,r in enumerate(s[:4]):
        n=Path(r['file_path']).name
        for i,ax in enumerate(axs[:,j]):
            ax.set_axis_off()
            ax.imshow(Image.open((folder/'scan'/r['file_path']) if not i else (out/'environment'/n)))
            if i==0:ax.set_title('Held-out RGB')
            else:ax.set_title(f"3DGS • {audit['environment_holdout']['per_view'][Path(n).stem]['psnr']:.1f} dB")
    fig.savefig(out/'HOLDOUT.png',dpi=135)
    print([(k,v['mean_psnr'],v['mean_ssim']) for k,v in audit.items() if isinstance(v,dict) and 'mean_ssim' in v])
    print('mask IoU',audit['foreground_median_mask_iou'],'retina',audit['compound_abs_error'])
if __name__=='__main__':main(*(Path(x) for x in sys.argv[1:]))
