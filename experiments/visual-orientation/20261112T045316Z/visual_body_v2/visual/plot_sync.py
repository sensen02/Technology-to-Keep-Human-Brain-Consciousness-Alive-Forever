"""Actualmoving-head body8visual signals andseparatecontrols; savedRGB only."""
import argparse,os
from contracts import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-after-parent-release',action='store_true');a=p.parse_args()
 if not a.execute_after_parent_release:print('CODE_ONLY no arrays/plot');return
 for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):require(os.environ.get(key)=='1',key+'=1 required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from PIL import Image,ImageDraw,ImageFont
 data={condition:load(condition) for condition in IDS};empty=data['V_empty'];out=HERE/'SYNCHRONIZED_EVIDENCE';out.mkdir(exist_ok=False);hashes={str(HERE/'plot_sync.py'):sha(HERE/'plot_sync.py')};fig,axes=plt.subplots(8,3,figsize=(18,25),sharex=True);metrics=[]
 for row,condition in enumerate(IDS):
  case=data[condition];hashes.update(case['source_hashes']);d=case['decisions'];t=case['trace'];r=case['result'];e=d['time']-5.8;bt=t['time']-5.8;require(np.array_equal(t['time'],empty['trace']['time']),'matchedV_emptytime')
  axes[row,0].plot(e,d['raw_released_capture_time_s']-5.8,label='actual released capture time',lw=1);axes[row,0].plot(e,d['image_capture_time_s']-5.8,label='actor-used capture time',ls='--',lw=1);axes[row,0].set_ylabel(condition+'\ncapture ANN time (s)');axes[row,0].set_title(('CONTROL' if case['iscontrol'] else 'NORMAL')+' | RGB only / moving head',fontsize=9)
  axes[row,1].plot(e,d['raw_turn'],label='neural raw turn');axes[row,1].plot(e,d['command'][:,1],label='applied high-level turn',ls='--');axes[row,1].set_ylim(-1.05,1.05);axes[row,1].set_ylabel('Turn')
  current=np.degrees(case['yaw']);baseline=np.degrees(empty['yaw']);axes[row,2].plot(bt,current,label='actual body delta yaw');axes[row,2].plot(bt,baseline,label='matched V_empty',ls='--');axes[row,2].plot(bt,current-baseline,label='minus V_empty',ls=':');axes[row,2].set_ylabel('Yaw from handover (deg)');axes[row,2].set_title('Original5 '+('PASS' if r['engineering_pass'] else 'FAIL')+' | '+('Control' if case['iscontrol'] else 'Function')+' '+('PASS' if case['functional_pass'] else 'FAIL'),fontsize=9)
  for ax in axes[row]:
   ax.axvspan(-5.8,0,color='gray',alpha=.08);ax.axvline(0,color='black',lw=.6);ax.grid(alpha=.2)
   event_times=() if condition=='V_empty' else (.6,4.8) if condition in ('V_left_to_right','V_right_to_left','V_flip_fixedframe') else (.6,3.6)
   for event in event_times:ax.axvline(event,color='darkorange',ls=':',lw=.5);ax.axvline(event+.02,color='gold',ls='--',lw=.5)
  for ax in axes[row]:ax.legend(fontsize=6,loc='best')
  metrics.append(dict(condition=condition,functional_pass=case['functional_pass'],iscontrol=case['iscontrol'],flags=case['functional_flags'],metrics=case['functional_metrics'],original_engineering_pass=r['engineering_pass']))
 for ax in axes[-1]:ax.set_xlabel('Actual ANN elapsed (s); source event and +20ms release distinguished')
 fig.suptitle('ACTUAL MOVING-HEAD RGB / BODY CLOSED LOOP: NORMAL6 AND CONTROL2 SEPARATE\nCamera50Hz /20ms delay, CNN held16 /GRU5ms; recorded eye images, not static-library replay',fontsize=14);fig.tight_layout(rect=(0,0,1,.96));path=out/'EIGHT_CONDITIONS_SYNC.png';fig.savefig(path,dpi=150);plt.close(fig)
 # Three times percontrol show genuine actual-released versusactor-used eye pairs.
 panel=Image.new('RGB',(512,1080),'white');draw=ImageDraw.Draw(panel);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12);records=[];y=0
 for condition in CONTROLS:
  case=data[condition];d=case['decisions'];rgb=case['rgb'];cap=case['captures']
  for elapsed in (4.82,6.,9.5):
   index=int(np.argmin(abs(d['time']-(5.8+elapsed))));raw=int(d['raw_released_image_capture_id'][index]);actor=int(d['image_capture_id'][index]);draw.text((4,y+2),f'{condition} ANN {float(d["time"][index]-5.8):.3f}',font=font,fill='black');draw.text((4,y+20),f'RAW RELEASED {raw} | ACTOR USED {actor}; LEFT/RIGHT perpair',font=font,fill='black')
   for k,img in enumerate((rgb[raw,0],rgb[raw,1],rgb[actor,0],rgb[actor,1])):panel.paste(Image.fromarray(img).resize((128,128),Image.Resampling.NEAREST),(128*k,y+46))
   records.append(dict(condition=condition,decision_index=index,decision_time_s=float(d['time'][index]),actual_released_id=raw,actor_used_id=actor,actual_capture_time_s=cap[raw]['capture_time_s'],actor_capture_time_s=cap[actor]['capture_time_s']));y+=180
 panel_path=out/'CONTROLS_ACTUAL_VS_ACTOR_EYES.png';panel.save(panel_path);unchanged(hashes);fresh(out/'PLOT.json',dict(status='ACTUAL_SAVED_OUTPUTS_PLOT_COMPLETE_REVIEW_PENDING',plot=str(path),plot_sha256=sha(path),control_eye_panel=str(panel_path),control_eye_panel_sha256=sha(panel_path),rows=metrics,eye_records=records,source_hashes=hashes,physics_steps=0,Core_constructions=0,model_forward_calls=0,static_image_clamp_not_claimed=True,parent_visual_review='PENDING'))
 print(str(out/'PLOT.json'))
if __name__=='__main__':main()
