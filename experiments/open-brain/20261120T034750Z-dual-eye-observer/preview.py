"""Make a slowed sample review with separate walking and fixed-body turn sections."""
import json,subprocess,sys
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

d=Path(sys.argv[1])
poses=json.loads((d/'estimated.json').read_text())
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
                    '-s','1008x420','-framerate','20','-i','-','-c:v','libx264','-pix_fmt','yuv420p',
                    '-threads','2','-movflags','+faststart',str(d/'DUAL_EYES.mp4')],stdin=subprocess.PIPE)
for name,pose in poses.items():
    if pose is None:continue
    frame=int(name)
    canvas=Image.new('RGB',(1008,420),'#f4f1e8')
    draw=ImageDraw.Draw(canvas)
    for i,(kind,title) in enumerate([('external','External camera'),('L','Left eye'),('R','Right eye')]):
        path=d/('external' if kind=='external' else 'eyes')/(f'{frame}_H02_NW.png' if kind=='external' else f'{frame}_{kind}.png')
        with Image.open(path) as img:
            img=img.convert('RGB')
            img.thumbnail((314,340),Image.Resampling.LANCZOS)
            canvas.paste(img,(336*i+(336-img.width)//2,45+(340-img.height)//2))
        draw.text((336*i+15,12),title,font=font,fill='#26393c')
    tag=f'Walking sample    {2*frame:d} ms' if frame<=315 else f'Fixed-body control    {frame-316}/3'
    draw.text((16,391),tag,font=font,fill='#26393c')
    for _ in range(5 if frame<=315 else 14):p.stdin.write(canvas.tobytes())
p.stdin.close()
if p.wait()!=0:raise SystemExit('ffmpeg failed')
