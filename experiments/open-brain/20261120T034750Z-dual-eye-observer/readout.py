"""Read each rendered eye using the installed project retina geometry."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from flygym.vision.retina import Retina

def main(folder,sequence,out):
    poses=json.loads((folder/sequence).read_text())
    source=folder/('eyes' if sequence=='estimated.json' else 'eye_reference')
    retina=Retina()
    frames=[];corrected=[];responses=[]
    for key,pose in poses.items():
        frame=int(key)
        if pose is None or any(not (source/f'{frame}_{side}.png').is_file() for side in 'LR'):
            continue
        seen=[];hexes=[]
        for side in 'LR':
            with Image.open(source/f'{frame}_{side}.png') as im:
                rgb=np.asarray(im.convert('RGB'))
            assert rgb.shape==(retina.nrows,retina.ncols,3)
            image=retina.correct_fisheye(rgb)
            hexes.append(retina.raw_image_to_hex_pxls(image))
            seen.append(image)
        frames.append(frame)
        corrected.append(seen);responses.append(hexes)
    frames=np.asarray(frames)
    masks=retina.ommatidia_id_map
    np.savez_compressed(out,frames=frames,walking_time_ms=np.where(frames<=315,frames*2,np.nan),
                        eye=np.asarray(['L','R']),channel=np.asarray(['yellow','pale']),
                        readout=np.array(responses),ommatidia_id_map=masks,
                        pale_type_mask=retina.pale_type_mask)
    corrected=np.array(corrected)
    grid=folder/('retina_frames' if sequence=='estimated.json' else 'retina_reference')
    grid.mkdir(exist_ok=True)
    for i,frame in enumerate(frames):
        for e,side in enumerate('LR'):
            Image.fromarray(corrected[i,e]).save(grid/f'{frame}_{side}.png')
    print('readout',np.array(responses).shape,
          hashlib.sha256(np.ascontiguousarray(masks).tobytes()).hexdigest())

if __name__=='__main__':
    main(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]))
