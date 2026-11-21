from pathlib import Path
import json,numpy as np,sys
from PIL import Image
from render import GaussianField,raster
cfg=json.load(open('scan/calibration.json'))
for name in sys.argv[1:]:
 p=Path(name);g=GaussianField(p)
 rms=[]
 for cam in [c for c in cfg['frames'] if c['split']=='holdout']:
    z,*arr=g.project(cam,correct_near=True)
    img,_=raster(cam['w'],cam['h'],np.argsort(z),*arr,cam['fl_x'],cam['fl_y'],cam['cx'],cam['cy'])
    ref=np.array(Image.open('scan/'+cam['file_path']),dtype=float)
    rms.append(-10*np.log10(np.mean(((img-ref)/255.)**2)))
 print(name,np.mean(rms),np.round(rms,2),flush=True)
