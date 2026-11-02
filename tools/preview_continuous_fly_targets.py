"""Standalone browser preview from validated recorded points, not a simulation."""
from pathlib import Path
import json,hashlib
import numpy as np
S=Path('/home/sensen/Desktop/engine修复验收/s1-continuous-fly-kinematics/20261102T004326Z')
O=Path('/home/sensen/Desktop/engine修复验收/s1-motion-target-delivery/20261102T004326Z')

def main():
 O.mkdir(parents=True,exist_ok=False)
 with np.load(S/'NMF2_CONTINUOUS.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
 assert a['xyz_source'].shape==(37,6,5,3) and np.isfinite(a['xyz_source']).all() and np.all(np.diff(a['source_frame'])==1)
 with np.load(S/'caltech_angles/train_neutral.npz',allow_pickle=False) as z:
  assert len(z['angles_rad'])==173292 and z['angles_rad'].shape[1]==20
  assert np.isfinite(z['angles_rad'][z['angle_mask']]).all()
  animals={split:set(z['animal_id'][z['split']==split]) for split in ('train','val','test')}
  assert all(not(animals[x]&animals[y]) for x,y in [('train','val'),('train','test'),('val','test')])
 with np.load(S/'caltech_angles/test_neutral.npz',allow_pickle=False) as z:assert not z['admission_mask'].any() and not z['angle_mask'].any()
 payload=dict(xyz=a['xyz_source'].tolist(),frames=a['source_frame'].tolist(),time=a['time_relative_s'].tolist(),legs=a['leg_names'].tolist(),markers=a['marker_names'].tolist())
 # Every displayed point is embedded verbatim; no interpolation, IK, root motion or physics.
 html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>果蝇真实六腿标注记录</title>
<style>body{font:16px system-ui;background:#101820;color:#e2ecf5;margin:24px;max-width:1200px}canvas{background:#172633;width:31%;height:auto;border:1px solid #50616c;border-radius:8px}p{line-height:1.6}input{width:70%}button{padding:8px 16px}code{overflow-wrap:anywhere}#legend span{margin-right:18px}</style>
<h1>真实六腿3D标注：逐帧浏览</h1><p>这是实验标注数据浏览，非物理仿真或神经控制结果。37帧、名义120 Hz、跨度0.300秒。坐标原始单位和轴方向未确认；胸部为固定参考，不能从此图判断自由身体推进或足部承重。</p>
<p><button id="play">播放一遍</button> <input id="frame" type="range" min="0" max="36" value="0"> <span id="label"></span></p>
<div><canvas id="xy" width="600" height="500"></canvas> <canvas id="xz" width="600" height="500"></canvas> <canvas id="yz" width="600" height="500"></canvas></div><p id="legend"></p>
<p>连线仅连接同一腿的相邻标注点：<span id="markers"></span>。显示所有原坐标，不补点、不平滑、不对称复制；连线不表示物理碰撞或肌肉。</p>
<p>较长补充数据为487个独立运动段、20个派生关节角，最长单段名义约1.997秒；不是长连续3D原位置。上游测试部分因六腿复制和列数异常已隔离。没有将不同段拼接成一条长轨迹。</p>
<p>来源：<a style="color:#8dd3ff" href="https://doi.org/10.7910/DVN/3MCEYR">NeuroMechFly v2官方记录</a>。原坐标NPZ：<code>__SOURCE__</code></p>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
const d=JSON.parse(document.getElementById('data').textContent), colors=['#ff8b76','#ffd166','#a3dc9e','#66d8ff','#c8a2ff','#f5a8d1'];
let timer=null;const slider=document.getElementById('frame');
document.getElementById('markers').textContent=d.markers.join(' → ');
document.getElementById('legend').innerHTML=d.legs.map((name,i)=>`<span style="color:${colors[i]}">${name}</span>`).join('');
function draw(){const f=+slider.value;document.getElementById('label').textContent=`原帧 ${d.frames[f]} / ${d.frames.at(-1)}，相对时间 ${d.time[f].toFixed(4)} s`;
for(const [id,x,y] of [['xy',0,1],['xz',0,2],['yz',1,2]]){const c=document.getElementById(id),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);
const flat=d.xyz.flat(2),xs=flat.map(p=>p[x]),ys=flat.map(p=>p[y]),loX=Math.min(...xs),hiX=Math.max(...xs),loY=Math.min(...ys),hiY=Math.max(...ys),scale=Math.min((c.width-60)/(hiX-loX),(c.height-80)/(hiY-loY));
g.fillStyle='#e2ecf5';g.font='18px system-ui';g.fillText(`${'XYZ'[x]} / ${'XYZ'[y]} 原始坐标投影`,20,26);
for(let leg=0;leg<6;leg++){g.strokeStyle=colors[leg];g.fillStyle=colors[leg];g.lineWidth=3;g.beginPath();d.xyz[f][leg].forEach((p,i)=>{const px=30+(p[x]-loX)*scale,py=c.height-30-(p[y]-loY)*scale;if(i)g.lineTo(px,py);else g.moveTo(px,py);});g.stroke();
d.xyz[f][leg].forEach(p=>{g.beginPath();g.arc(30+(p[x]-loX)*scale,c.height-30-(p[y]-loY)*scale,5,0,Math.PI*2);g.fill();});}}}
slider.oninput=()=>{if(timer){clearInterval(timer);timer=null;}draw();};
document.getElementById('play').onclick=()=>{if(timer)clearInterval(timer);slider.value=0;draw();timer=setInterval(()=>{if(+slider.value>=36){clearInterval(timer);timer=null;return;}slider.value=+slider.value+1;draw();},1000/120);};draw();
</script></html>'''
 html=html.replace('__SOURCE__',str(S/'NMF2_CONTINUOUS.npz')).replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False))
 path=O/'六腿真实标注浏览.html';path.write_text(html)
 # Read generated payload back independently and compare every coordinate/time to the source.
 embedded=json.loads(path.read_text().split('<script id="data" type="application/json">')[1].split('</script>')[0])
 assert np.array_equal(np.asarray(embedded['xyz']),a['xyz_source']) and np.array_equal(embedded['frames'],a['source_frame']) and np.array_equal(embedded['time'],a['time_relative_s'])
 record=dict(passed=True,source=str(S/'NMF2_CONTINUOUS.npz'),source_sha256=hashlib.sha256((S/'NMF2_CONTINUOUS.npz').read_bytes()).hexdigest(),displayed_coordinates=3330,html_roundtrip_coordinates_exact=True,derived_angles_finite_and_upstream_test_quarantined=True,animal_group_split_disjoint=True,rendering_inspected=False,reason='Interactive artifact generated and payload verified; no browser/image viewing tool used.',model_load=0,neural_step=0,body_step=0,training=0)
 (O/'PREVIEW_VERIFICATION.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False,indent=2));print(path)
if __name__=='__main__':main()
