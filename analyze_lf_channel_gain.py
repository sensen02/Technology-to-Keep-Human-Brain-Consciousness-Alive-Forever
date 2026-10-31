"""S151 two common windows and six-leg allocation, saved arrays only."""
from pathlib import Path
import argparse,json,csv
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify_muscle_reconstruction import evaluate
REF=Path('/home/sensen/Desktop/engine修复验收/s149-half-gain-physical/20261024T213108429114Z')
def load(p):
 with np.load(p/'init80ms/TRAJECTORY.npz') as z:return {k:z[k].copy() for k in z.files}
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output;data={'S149 all0.5':load(REF),'S151 LF4=0.25':load(p)};nodes=list(ET.parse('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml').getroot().find('actuator'));legs=json.loads((p/'init80ms/COM_CONTACT_METADATA.json').read_text())['foot_order'];summ={};records=[]
 for window,n in [('early_common',min(11,*[len(x['time']) for x in data.values()])),('full_common',min(len(x['time']) for x in data.values()))]:
  fig,axs=plt.subplots(3,3,figsize=(15,11));summ[window]={}
  for name,a in data.items():
   t=a['time'][:n];active=np.zeros((n,90));bias=active.copy()
   for i in range(n):
    for j,node in enumerate(nodes):
     f,g,b=evaluate(a['actuator_length'][i,j],a['actuator_velocity'][i,j],a['act'][i,j],node);assert abs(f-a['actuator_force'][i,j])<1e-10;active[i,j]=g*a['act'][i,j];bias[i,j]=b
   power=np.sum(active*a['actuator_velocity'][:n],1);cum=np.r_[0,np.cumsum((power[1:]+power[:-1])/2*np.diff(t))];perleg={}
   for j,leg in enumerate(legs):
    f=a['foot_force'][:n,j,2];peak=int(f.argmax());perleg[leg]=dict(peak_Fz=float(f[peak]),peak_ms=float(t[peak]*1000),initial_Fz=float(f[0]),final_Fz=float(f[-1]),minimum_Fz=float(f.min()),Fz_impulse=float(np.trapezoid(f,t)),final_contacts=int(a['contacts'][n-1,j]));axs.flat[j].plot(t*1000,f,label=name);axs.flat[j].set_title(leg+' foot Fz (native)')
   summ[window][name]=dict(duration_ms=float(t[-1]*1000),legs=perleg,COM_vz_max=float(a['com_velocity_world'][:n,2].max()),COM_vz_min=float(a['com_velocity_world'][:n,2].min()),COM_vz_final=float(a['com_velocity_world'][n-1,2]),COM_az_final=float(a['com_acceleration_world'][n-1,2]),COM_height_change=float(a['com_position_world'][n-1,2]-a['com_position_world'][0,2]),max_limit_excess=float(a['limit_excess'][:n].max()),active_work_native=float(cum[-1]),load_final=float(a['load_ratio'][n-1]))
   axs.flat[6].plot(t*1000,a['com_velocity_world'][:n,2],label=name);axs.flat[6].set_title('COM vertical velocity (native)');axs.flat[7].plot(t*1000,cum,label=name);axs.flat[7].set_title('Active muscle signed work (native)');axs.flat[8].plot(t*1000,a['load_ratio'][:n],label=name);axs.flat[8].set_title('Load ratio');
   for i in range(n):records.append([window,name,i,float(t[i]),*[float(v) for v in a['foot_force'][i,:,2]],float(a['load_ratio'][i]),float(a['com_velocity_world'][i,2]),float(a['com_acceleration_world'][i,2]),float(power[i])])
  axs.flat[8].axhline(.8,color='red')
  for ax in axs.flat:ax.set_xlabel('Actual physical ms');ax.legend()
  fig.suptitle('S151 LF channel4 intervention: '+window+'; terminal failure sample included');fig.tight_layout();fig.savefig(p/(window.upper()+'.png'),dpi=140);plt.close(fig)
 with (p/'LF_COMPARISON.csv').open('x') as f:w=csv.writer(f);w.writerow(['window','arm','sample','time_s',*[x+'_Fz' for x in legs],'load','COM_vz','COM_az','active_power']);w.writerows(records)
 with (p/'LF_COMPARISON.json').open('x') as f:json.dump(dict(passed=True,windows=summ,physical_units='NOT_ESTABLISHED',prewarm_success=False),f,indent=2)
 print(json.dumps(summ))
if __name__=='__main__':main()
