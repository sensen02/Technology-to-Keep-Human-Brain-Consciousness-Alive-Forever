"""Extract different measured phases from an already verified simulation video."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);args=ap.parse_args()
    folder=args.folder
    manifest=json.loads((folder/'VIDEO_MANIFEST.json').read_text());video=folder/'WALK_4x_slow.mp4'
    assert sha(video)==manifest['video_sha256']
    sheet=Image.new('RGB',(1024,1440),(20,20,25));selected=[];reader=imageio.get_reader(video)
    for i,t in enumerate([1.05,1.18,1.31,1.44,1.57,1.70,1.83,1.96]):
        item=min(manifest['mapping'],key=lambda x:abs(x['time']-t))
        im=Image.fromarray(reader.get_data(item['frame'])).resize((512,360))
        sheet.paste(im,((i%2)*512,(i//2)*360));selected.append(dict(requested=t,**item))
    reader.close()
    image=folder/'PHASE_PREVIEW.png'
    if image.exists():image.read_bytes()
    sheet.save(image)
    p=folder/'PHASE_PREVIEW.json'
    if p.exists():p.read_text()
    p.write_text(json.dumps(dict(video=str(video),video_sha256=sha(video),frames=selected,
        source='unaltered decoded frames from saved-physics video; no motion reconstruction or temporal extrapolation'),indent=2)+'\n')
    print(json.dumps(dict(image=str(image),source_video_unchanged=True,phases=len(selected))),flush=True)


if __name__=='__main__':main()
