"""Stage image-only Brush datasets, with holdout folders separate from training."""
import argparse,json,os
from pathlib import Path

def main(scan,out,partial):
    frames=json.loads((scan/'calibration.json').read_text())['frames']
    out.mkdir(exist_ok=False)
    (out/'images').mkdir()
    train=[]
    for row in frames:
        key=row['file_path']
        exists=(scan/key).is_file()
        if row['split']!='train':continue
        if partial and not exists:continue
        os.link(scan/key,out/key)
        train.append({k:row[k] for k in ('file_path','transform_matrix','fl_x','fl_y','cx','cy','h','w')})
    assert len(train)==116 or partial
    (out/'transforms.json').write_text(json.dumps(dict(camera_model='OPENCV',frames=train),indent=2))
    print('RGB training observations',len(train))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scan',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--partial',action='store_true');a=p.parse_args()
    main(a.scan,a.out,a.partial)
