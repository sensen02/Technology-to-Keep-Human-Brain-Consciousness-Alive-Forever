"""Read-only numpy/XML/source reconciliation; never imports simulator or runners."""
from pathlib import Path
import hashlib,json,subprocess,sys
import numpy as np
import xml.etree.ElementTree as ET
P=Path(__file__).parent
O=Path('/home/sensen/Desktop/engine修复验收/s4-initial-pose-version-audit/20261028T016000Z')
D=Path('/home/sensen/Desktop/S4_INITIAL_POSE_VERSION_AUDIT.md')
CURRENT=Path('/home/sensen/Desktop/engine修复验收/_历史归档/actual-neural-matched-20261018T031000Z/initial_runtime_state.npz')
S7=Path('/home/sensen/Desktop/engine修复验收/s7-posture-ik/20261023T021141Z/DERIVED_INITIAL.npz')
S3=Path('/home/sensen/Desktop/engine修复验收/s2-stance-s3b/20261023T060000Z')
XML=P/'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
NEW=Path('/home/sensen/Desktop/engine修复验收/s4-dynamic-neural-gain/20261028T011000Z')
H=Path('/home/sensen/Desktop/engine修复验收/s2-release-250ms/20261027T161000Z/normal/BODY.npz')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def same_numeric(a,b):
 if a is None or b is None:return a==b
 try:
  x=np.fromstring(a,sep=' ');y=np.fromstring(b,sep=' ')
  return x.shape==y.shape and bool(np.allclose(x,y,rtol=0,atol=1e-14))
 except ValueError:return a==b

def main():
 O.mkdir(parents=True,exist_ok=False)
 sources=[CURRENT,S7,S3/'STANCE2_CALIBRATED.npz',S3/'S3V2_MODEL_a_springref_stance.xml',S3/'REPORT.zh-CN.md',S3/'build2.py',XML,H,NEW/'train_01/BODY.npz',NEW/'train_02/BODY.npz',Path(__file__)]+[P/n for n in ['run_one_second_motor_pair.py','run_dynamic_neural_gain_calibration.py','run_neural_muscle_closure.py','engine/canonical_physics.py','render_neural_gain_slow_comparison.py','engine/embodied/initial_pose_contract.py']]
 hashes={str(x):sha(x) for x in sources};dump(O/'INPUT_CODE_HASHES.json',hashes)
 a=ET.parse(XML).getroot();b=ET.parse(S3/'S3V2_MODEL_a_springref_stance.xml').getroot()
 def ordered_joints(root):
  out=[]
  def walk(body):
   out.extend(x for x in body.findall('joint') if x.get('name'))
   for child in body.findall('body'):walk(child)
  for body in root.findall('./worldbody/body'):walk(body)
  return out
 ja=ordered_joints(a);jb=ordered_joints(b)
 assert [x.get('name') for x in ja]==[x.get('name') for x in jb]
 hinges=[x.get('name') for x in ja if x.get('type')!='free'];assert len(hinges)==42
 z=np.load(S3/'STANCE2_CALIBRATED.npz');assert set(str(x) for x in z['hinge_names'])==set(hinges)
 q=np.load(CURRENT)['qpos'];variants={'runtime_Oct18':q,'S7_fixed_IK':np.load(S7)['new_qpos'],'S3V2_fixed_calibrated':z['qpos'],'historical_baseline_video':np.load(H)['qpos'][0],'latest_train01_video':np.load(NEW/'train_01/BODY.npz')['qpos'][0],'latest_train02_video':np.load(NEW/'train_02/BODY.npz')['qpos'][0]}
 comparisons={}
 for k,v in variants.items():
  comparisons[k]=dict(root_qpos=v[:7].tolist(),same_as_runtime=bool(np.array_equal(v,q)),max_abs_qpos_delta=float(abs(v-q).max()),changed_component_count=int(np.count_nonzero(v!=q)),max_abs_hinge_delta_rad=float(abs(v[7:]-q[7:]).max()),hinge_rows=[dict(name=n,current=float(q[7+i]),comparison=float(v[7+i]),delta_rad=float(v[7+i]-q[7+i])) for i,n in enumerate(hinges)])
 comparisons['S7_fixed_IK'].pop('hinge_rows')
 comparisons['S7_fixed_IK']['max_abs_last42_qpos_component_delta_rad']=comparisons['S7_fixed_IK'].pop('max_abs_hinge_delta_rad')
 dump(O/'POSE_VERSION_COMPARISON.json',dict(joint_order_same=True,shape=[49],layout='MuJoCo body preorder: direct joints before child bodies, NOT XML raw joint iterator; S7 raw qpos differences only, no certified named rows',comparisons=comparisons))
 by_a={x.get('name'):x for x in ja};by_b={x.get('name'):x for x in jb};diff={}
 for field in ['springref','range','stiffness','damping','axis','pos']:
  rows=[dict(name=n,current=by_a[n].get(field),fixed=by_b[n].get(field)) for n in hinges if not same_numeric(by_a[n].get(field),by_b[n].get(field))];diff[field]=dict(numerically_changed_count=len(rows),rows=rows)
 wings={k:{n:next(x for x in r.iter('body') if x.get('name')==n).attrib for n in ['LWing','RWing']} for k,r in [('current',a),('fixed_S3V2',b)]}
 range_checks={}
 for version,v in [('S3V2_fixed_calibrated',variants['S3V2_fixed_calibrated'])]:
  range_checks[version]={}
  for model_name,joints in [('current_canonical',by_a),('fixed_S3V2',by_b)]:
   rows=[]
   for i,n in enumerate(hinges):
    limits=np.fromstring(joints[n].get('range','-inf inf'),sep=' ');value=float(v[7+i])
    if value<limits[0]-1e-12 or value>limits[1]+1e-12:rows.append(dict(name=n,value=value,range=limits.tolist()))
   range_checks[version][model_name]=dict(violations=len(rows),rows=rows)
 other={}
 for tag,selector,fields in [('actuator','./actuator/*',['gainprm','biasprm','dynprm','lengthrange','ctrlrange']),('geom','.//geom',['solref','solimp','rgba','friction','contype','conaffinity']),('site','.//site',['pos','quat','size']),('body','.//body',['pos','quat']),('tendon','./tendon/*',['range','springlength','stiffness','damping'])]:
  aa={x.get('name'):x for x in a.findall(selector) if x.get('name')};bb={x.get('name'):x for x in b.findall(selector) if x.get('name')}
  other[tag]={field:[dict(name=n,current=aa[n].get(field),fixed=bb[n].get(field)) for n in aa.keys()&bb.keys() if not same_numeric(aa[n].get(field),bb[n].get(field))] for field in fields}
 dump(O/'MODEL_DIFF.json',dict(current_xml=str(XML),fixed_xml=str(S3/'S3V2_MODEL_a_springref_stance.xml'),joint_diff=diff,wing_body_diff=wings,saved_pose_range_checks=range_checks,other_attribute_diffs=other,layout_note='Stored hinge_names is source list order, not qpos order. build2.py assigns q[compiled adr[name]]; both XMLs have identical free+42 hinge body-preorder (body direct joints first, then child bodies). S7 only raw qpos component comparison is certified here.'))
 run=subprocess.run(['git','diff','HEAD','--','run_neural_muscle_closure.py','engine/canonical_physics.py','engine/embodied/initial_pose_contract.py','run_one_second_motor_pair.py','run_dynamic_neural_gain_calibration.py','render_neural_gain_slow_comparison.py'],cwd=P,capture_output=True,text=True);assert run.returncode==0
 (O/'CURRENT_VS_HEAD.diff').write_text(run.stdout)
 text=f'''# 初始姿态版本回退核对（保存数据／XML，只读）

## 已确认的原因
本轮运行入口重选了2026-10-18的历史INITIAL和canonical XML，未沿用2026-10-23已修正站姿及配套模型。不是你之前的修复失效，也不是最近增益改变了t=0姿态；是运行分支选择回到了更早的姿态/模型组合。

- 最新入口 `{P/'run_dynamic_neural_gain_calibration.py'}` 第11行从 `{P/'run_one_second_motor_pair.py'}` 导入INITIAL。
- 后者第9行硬编码 `{CURRENT}`，并非S7 IK或S3V2站姿文件。
- `{P/'run_neural_muscle_closure.py'}` 的prepare_body显式输入qpos时原样使用，不自动选择最新修复，不负责从任意旧初态恢复修后站姿。
- 最新视频的历史基线、train01、train02第0帧均与Oct18 INITIAL逐值一致，最大误差0。这只证明忠实使用了旧姿态，不能证明接入了你修过的姿态。

## 旧修复与当前实取值

|版本|根高度（模型长度）|与当前根/42关节关系|
|---|---:|---|
|当前Oct18 INITIAL及三栏视频|2.0422429167667704|根单位四元数，三栏与旧初态完全相同|
|S7修后IK new_qpos|1.42194447|49/49分量不同，后42分量最大差2.5384539236 rad；未独立具名核对|
|S3V2修后校准站姿|1.7869570400592367|48/49分量不同，42关节全部不同，最大差2.1084925725 rad|

旧修复证据：`{S7}` 与 `{S3/'REPORT.zh-CN.md'}`。S7的恒定张力10ms只是非神经短窗诊断；S3V2静态平衡门失败，不能把它称为自主站立成功。这里核对的是已做姿态修复是否被沿用，不给行为通过。

## 配套XML也没有沿用
当前XML `{XML}`，SHA256 {sha(XML)}。
S3V2配套XML `{S3/'S3V2_MODEL_a_springref_stance.xml'}`，SHA256 {sha(S3/'S3V2_MODEL_a_springref_stance.xml')}。

按body直接joint优先再子body的编译顺序，逐名匹配关节顺序相同；42个springref数值改变、9个range数值改变（其余字符串差异含浮点格式，不能记为42个实际范围变更）；关节axis/pos/stiffness/damping数值不变。S3V2左右翅body具有约100°后掠/10°旋转及z偏置，而当前左右翅body无quat、z=.1782/.1783。因此不仅腿姿态，之前折翅派生也未选入本轮。没有翅关节qpos可单独恢复该模型级变更。

|当前与S3V2实质数值差异|数量|
|---|---:|
|actuator gainprm / dynprm / lengthrange|各90|
|actuator biasprm / ctrlrange|各0|
|geom solref / rgba|72 / 65|
|geom friction / solimp / contype / conaffinity|各0|
|site pos|185|
|body pos / quat（左右翅）|各2|
|joint springref / range|42 / 9|
|joint axis / pos / stiffness / damping|各0|

185处site位置变更说明旧分支还包含肌腱附着几何改动，不只是改了站姿文件或翅的显示透明度。

**不能只把修后qpos塞进当前XML就宣称修好，也不能盲目恢复旧XML全部参数**：修后姿态配套9项扩展关节范围与42项被动参考；而build2.py第91–103行还包含肌肉gainprm[2]×10、dynprm[0]=0.0005和geom接触solref[0]=0.002等物理变更。MODEL_DIFF逐名列出gain/bias/dyn/lengthrange/ctrlrange及接触参数实差。恢复姿态/翅几何与恢复这些物理假设必须分开决策，不能把旧分支完整视为单纯姿态修复。折翅此前也只是有声明的视觉近似，不能凭报告认证生理解剖正确。

最近渲染只把71个mesh设为不透明并开放geomgroup，揭露了当前分支本来保存的外形；它没有改变保存qpos。完整可见不等于姿态正确。当前git差异中的prepare_body没有新增改姿逻辑；工作树HEAD比较不能证明所有历史入口同版，尤其新入口为未追踪研究脚本。完整差异见CURRENT_VS_HEAD.diff。HEAD本身早于Oct23修复，不能当作最后批准版本；新研究入口未跟踪、模型outputs资产被Git忽略，所以Git diff无XML内容不代表没有版本回退，须以封存模型SHA与具名属性比较为准。tracker当前索引没有S7/S3V2修复条目，且相对HEAD有大量旧日志删除，这形成了历史索引缺口；本轮只记录该事实，不据此证明某次删除是唯一原因，不恢复或改写全部历史。

## 结论与下一步
责任在本轮入口/版本管理：把“canonical+历史初态未改”误当作“已经修过的初始化被保留”。先停止把这些旧姿态运行用作修后站姿结论。待明确要恢复的修后几何版本后，应明确绑定站姿文件、所需关节范围/参考和翅几何，并把旧分支肌力/激活时间常数/接触修改另列为待裁决的物理假设，不能盲目全量覆盖旧XML。先核对保存姿态/模型，不在本轮直接改模型或跑新动力学。本核对未执行任何模型加载、forward、身体步、神经步、解码、渲染或修复。

## 保留文件及重要程度
- 核心必需：`{O/'POSE_VERSION_COMPARISON.json'}`，逐关节旧修复/当前/视频首态数值。
- 核心必需：`{O/'MODEL_DIFF.json'}`，配套模型被动参考/范围/双翅差异。
- 核心必需：`{D}` 与 `{O/'REPORT.zh-CN.md'}`，本次原因和边界。
- 重要辅助：`{O/'CURRENT_VS_HEAD.diff'}`、`{O/'INPUT_CODE_HASHES.json'}`、`{O/'EXECUTION.json'}`、`{O/'MANIFEST_SHA256.json'}`，修改与身份记录。
- 重要辅助：`{Path(__file__)}`，只读复核入口。

失败与修正：013000Z只读入口先因源名字列表顺序误断言退出1；014000Z运行退出0但事后发现将XML原始joint迭代误作MuJoCo顺序，具名行/范围检查无效。015000Z改为body直接joint优先再递归子body顺序；最终016000Z进一步删除未独立具名认证的S7行和范围判断，S3V2修后站姿在自身模型范围违规0、在当前canonical范围违规9。013/014为重要辅助的失败证据，015为已替代中间核对，均非最终结论，不覆盖旧目录。桌面报告是当前可变交付镜像，旧归档中指向桌面报告的哈希不再代表该镜像现值；旧目录内部文件不变。S7仅认证49原始qpos分量差，未独立绑定S7原模型的具名索引，不据其范围检查作结论。

清理：本轮未生成待弃模拟/渲染或可安全删除的临时中间文件，未删除任何历史或用途不明文件。013/014保留用于错误复核，不作为并列有效版本；已封档与用户修复资产全部保留，不以日期判断删除。
'''
 (O/'REPORT.zh-CN.md').write_text(text);D.write_text(text)
 assert all(sha(Path(p))==v for p,v in hashes.items())
 dump(O/'EXECUTION.json',dict(command=sys.executable+' '+str(Path(__file__)),exit_code=0,model_load=0,forward=0,step=0,neural=0,render=0,decoder=0,inputs_unchanged=True,git_diff_exit_code=run.returncode,prior_readonly_failure=dict(path='/home/sensen/Desktop/engine修复验收/s4-initial-pose-version-audit/20261028T013000Z',exit_code=1,reason='Stored hinge_names list order differs from XML declaration order; corrected validation to set equality while qpos layout follows same XML tree in both models. No simulator imported.')) )
 (O/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 dump(O/'MANIFEST_SHA256.json',{str(x):sha(x) for x in O.iterdir() if x.is_file()}|{str(D):sha(D)})
 print('OUTPUT',O);print('POSE_COUNTS',[(k,v['changed_component_count']) for k,v in comparisons.items()]);print('MODEL_COUNTS',{k:v['numerically_changed_count'] for k,v in diff.items()})
if __name__=='__main__':main()
