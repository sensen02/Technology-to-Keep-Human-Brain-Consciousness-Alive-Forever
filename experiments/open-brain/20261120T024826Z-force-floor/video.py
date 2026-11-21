"""Synchronized review of external measurements, slowed from 500 Hz to 30 fps."""
from pathlib import Path
import csv,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
d=Path(__file__).resolve().parent
ref=np.load(d/'insert/external/H02_NW.npz')
pixels=ref['background'];palette=ref['palette']
loads=np.array([float(r['fz_uN']) for r in csv.DictReader((d/'insert/feet.csv').open())]).reshape(-1,6)
colors=['#d94449','#d77a27','#b8a024','#198760','#2973ae','#a13c8d']
legs='LF LM LH RF RM RH'.split()
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
large=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
pipe=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x448','-r','30','-i','-','-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(d/'WALK_LOADS.mp4')],stdin=subprocess.PIPE)
for i,diff in enumerate(ref['delta']):
    pixels^=diff
    frame=Image.new('RGB',(960,448),'#f6f5ed')
    frame.paste(Image.fromarray(palette[pixels]),(0,0))
    draw=ImageDraw.Draw(frame)
    draw.text((691,20),f'{2*(i+1):03d} ms',font=large,fill='#182e30')
    draw.text((691,59),'Foot load (µN)',font=small,fill='#182e30')
    for k,leg in enumerate(legs):
        y=94+45*k
        draw.text((690,y),leg,font=small,fill=colors[k])
        x0=744;x1=x0+int(loads[i,k]*23)
        draw.line((x0,y+9,x1,y+9),fill=colors[k],width=9)
        draw.text((888,y),f'{loads[i,k]:+.2f}',font=small,fill='#182e30')
    draw.text((690,388),'500 Hz capture',font=small,fill='#445951')
    draw.text((690,411),'30 fps playback',font=small,fill='#445951')
    pipe.stdin.write(frame.tobytes())
pipe.stdin.close()
if pipe.wait():raise RuntimeError('ffmpeg failed')
