"""Independent algebraic raw observation -> configured neural current reconstruction."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output;arm=p/'init80ms';cfg=json.loads((arm/'ENCODER_PARAMETERS.json').read_text());groups=cfg['groups'];c=cfg['config'];errors=[]
 with np.load(arm/'TRAJECTORY.npz') as z:a={k:z[k].copy() for k in z.files}
 for tick in range(len(a['current'])):
  raw=json.loads((arm/f'OBSERVATION_{tick:02d}.json').read_text());cur=np.zeros(cfg['n_neurons'])
  for leg,q in raw.items():
   vals={'claw_position':q['tibia_joint_dev_rad']/c['reference_angle_rad'],'hook_direction':q['tibia_joint_vel_rad_s']/c['reference_velocity_rad_s'],'hairplate_angle':q['multi_joint_dev_rad']/c['reference_angle_rad'],'campaniform_strain':q['contact_force_norm']/c['reference_force'],'bristle_touch':q['foot_contact'],'club_vibration':0.,'multidendritic_noxious':0.}
   for ch,v in vals.items():cur[np.asarray(groups[leg][ch],dtype=int)]+=c['current_nA_per_unit']*v
  err=float(abs(cur-a['current'][tick]).max());assert err<1e-14;errors.append(err)
 with (p/'PARENT_ENCODING_CHECK.json').open('x') as f:json.dump(dict(passed=True,independent_linear_encoding_maxabs=max(errors),ticks=len(errors),checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),f,indent=2)
 print('Independent raw-to-current algebra passed',errors)
if __name__=='__main__':main()
