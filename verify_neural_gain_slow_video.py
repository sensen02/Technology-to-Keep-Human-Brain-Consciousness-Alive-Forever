"""Decode saved video and verify exact frame receipts; no model or neural calls."""
from pathlib import Path
import json,hashlib,subprocess
import numpy as np,imageio.v2 as imageio
O=Path('/home/sensen/Desktop/engine修复验收/s4-neural-gain-render/20261028T012000Z');D=Path('/home/sensen/Desktop/S4_NEURAL_GAIN_FULLBODY_SLOW_COMPARISON.mp4')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_read_frames:format=duration','-of','json',str(D)],text=True));s=probe['streams'][0];assert int(s['nb_read_frames'])==300 and s['r_frame_rate']=='30/1' and s['width']==2160 and s['height']==600;assert abs(float(probe['format']['duration'])-10)<.001
 reader=imageio.get_reader(D);errors={};n=0
 for i,f in enumerate(reader):
  n+=1
  if i in [0,45,90,299]:
   a=imageio.imread(O/f'FRAME_{i:03d}.png');err=float(np.abs(a.astype(float)-f.astype(float)).mean());assert err<12;errors[str(i)]=err
   if i==90:imageio.imwrite(O/'DECODED_FRAME_090.png',f)
 reader.close();assert n==300
 receipts=json.loads((O/'FRAME_RECEIPTS.json').read_text());assert len(receipts)==900
 pr=json.loads((O/'PREREGISTRATION.json').read_text());paths=[Path(k) for k in pr['inputs_sha256'] if k.endswith('BODY.npz')];z=[np.load(p) for p in paths]
 for r in receipts:
  a=z[r['panel']];idx=r['saved_index'];assert r['qpos_sha256']==hashlib.sha256(a['qpos'][idx].tobytes()).hexdigest();assert r['physical_time_s']==float(a['time'][idx])
 assert all(sha(Path(k))==v for k,v in pr['inputs_sha256'].items())
 result=dict(exit_code=0,ffprobe=probe,decoded_frames=n,representative_frame_mean_absolute_errors=errors,all900_receipts_match_saved_states=True,all_inputs_unchanged=True,model=0,forward=0,step=0,neural=0)
 (O/'VIDEO_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(result))
if __name__=='__main__':main()
