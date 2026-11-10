"""Select different gait phases from the already-rendered video; no motion generation."""
import json
import hashlib
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image
FOLDER=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T081009Z/neural/teacher_gain8_wave_4s')

if __name__=='__main__':
    manifest=json.loads((FOLDER/'VIDEO_MANIFEST.json').read_text())
    video=FOLDER/'WALK_4x_slow.mp4'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(video)==manifest['video_sha256']
    reader=imageio.get_reader(video)
    sheet=Image.new('RGB',(1024,1440),(20,20,25));selected=[]
    for i,t in enumerate([1.05,1.15,1.32,1.49,1.66,1.83,2.00,2.16]):
        item=min(manifest['mapping'],key=lambda row:abs(row['time']-t))
        frame=Image.fromarray(reader.get_data(item['frame'])).resize((512,360))
        sheet.paste(frame,((i%2)*512,(i//2)*360))
        selected.append(dict(requested_time=t,**item))
    reader.close();sheet.save(FOLDER/'PHASE_PREVIEW.png')
    (FOLDER/'PHASE_PREVIEW.json').write_text(json.dumps(dict(
        video=str(video),sha256=sha(video),frames=selected,
        provenance='unaltered frames decoded from verified source video at distinct gait phases'),indent=2)+'\n')
    print('Eight distinct-phase frames selected from unchanged video')
