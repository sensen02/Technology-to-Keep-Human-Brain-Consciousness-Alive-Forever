"""S2/S4 static kinematic target contract. Never imports a simulator or network.
Reads XML, archived numerical arrays and local experimental reconstruction only.
Outputs are an interface specification and audit, never a synthetic training set.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
DESKTOP = Path('/home/sensen/Desktop/engine修复验收')
SOURCE = DESKTOP/'real-freewalking-3d/20261027T150000Z'
NMF = DESKTOP/'s5-standard-gait-reference/20261028T040000Z'
RUN = DESKTOP/'s5-banc888-physical-handover/20261101T220018Z/RUN'
FIT = DESKTOP/'s4-banc888-closedloop-parameter-fit/20261101T222509Z'
OUT = DESKTOP/'s4-kinematic-training-interface/20261102T004326Z'
MODEL_SHA = '3470c72f7e68f181ec1a80a5dc5fbff182b750091e7a1ddc7ccfbfb590b80cb6'
LEGS = dict(T1L='LF', T2L='LM', T3L='LH', T1R='RF', T2R='RM', T3R='RH')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def dump(p, obj):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=convert)+'\n')


def vec(text, fallback):
    return [float(x) for x in (text or fallback).split()]


def target_schema():
    def arr(items, n=None):
        d = dict(type='array', items=items)
        if n is not None: d.update(minItems=n, maxItems=n)
        return d
    def obj(props, required):
        return dict(type='object', properties=props, required=required, additionalProperties=False)
    number = dict(type='number')
    string = dict(type='string', minLength=1)
    boolean = dict(type='boolean')
    vector = arr(number, 3)
    matrix = arr(vector, 3)
    hash_schema = dict(type='string', pattern='^[0-9a-f]{64}$')
    record = obj(dict(path=string, sha256=hash_schema, role=string), ['path','sha256','role'])
    status = dict(type='string', enum=['observed_reconstruction','derived','inferred_label','not_available'])
    root = obj(dict(position=vector, basis_world_columns=matrix, valid=boolean,
                    observed_vs_derived=status, method=string, frame_id=string,
                    position_unit=string, translation_observable=boolean,
                    orientation_observable=boolean),
               ['position','basis_world_columns','valid','observed_vs_derived','method','frame_id',
                'position_unit','translation_observable','orientation_observable'])
    root['description'] = 'Derived body reference, not necessarily model free root or COM. Invalid roots must be null, not fabricated.'
    label = obj(dict(values=arr(dict(type=['number','null'])), valid_mask=arr(boolean),
                     observed_vs_derived=dict(type='string',enum=['derived','inferred_label','observed_reconstruction']),
                     method=string, units=string), ['values','valid_mask','observed_vs_derived','method','units'])
    sample = obj(dict(time_s=number, source_frame=dict(type='integer'),
                      keypoints=arr(dict(oneOf=[vector,dict(type='null')])), valid_mask=arr(boolean),
                      observed_vs_derived=dict(type='string',enum=['observed_reconstruction']),
                      root=dict(oneOf=[root,dict(type='null')]), root_valid=boolean,
                      confidence=arr(dict(type=['number','null'])), phase=label, footcontact=label),
                 ['time_s','source_frame','keypoints','valid_mask','observed_vs_derived','root','root_valid'])
    source = obj(dict(dataset=string, release_or_commit=string, individual_id=dict(type=['string','null']),
                      individual_identity_status=dict(type='string',enum=['verified','source_record_only','unknown']),
                      session_id=string, bout_id=string, measurement=string, upstream_processing=string,
                      raw_files=arr(record), license_status=string),
                 ['dataset','release_or_commit','individual_id','individual_identity_status','session_id','bout_id',
                  'measurement','upstream_processing','raw_files','license_status'])
    frames = obj(dict(source_frame_id=string, source_axes=string, length_unit=string,
                      meters_per_source_unit=dict(type=['number','null']), body_frame_definition=string,
                      root_to_model_transform=dict(type=['object','null']), model_xml_sha256=hash_schema,
                      model_length_unit_status=string),
                 ['source_frame_id','source_axes','length_unit','meters_per_source_unit','body_frame_definition',
                  'root_to_model_transform','model_xml_sha256','model_length_unit_status'])
    clock = obj(dict(time_origin=string, nominal_fps=dict(type=['number','null']),
                     acquisition_timestamps_available=boolean, synchronization=string,
                     gap_policy=string, interpolation_provenance=string),
                ['time_origin','nominal_fps','acquisition_timestamps_available','synchronization','gap_policy','interpolation_provenance'])
    schema = obj(dict(schema_version=dict(const='1.0.0'), source=source,
                      coordinate_contract=frames, timing=clock, keypoint_names=arr(string),
                      skeleton_edges_by_name=arr(arr(string,2)), samples=arr(sample),
                      derivations=arr(record), split_contract=obj(dict(grouping=string,
                          train=string,validation=string,holdout=string,fit_on_train_only=boolean),
                          ['grouping','train','validation','holdout','fit_on_train_only'])),
                 ['schema_version','source','coordinate_contract','timing','keypoint_names',
                  'skeleton_edges_by_name','samples','derivations','split_contract'])
    schema.update({'$schema':'https://json-schema.org/draft/2020-12/schema',
                   '$id':'urn:cell-wound:kinematic-target:1.0.0',
                   'title':'Real six-leg 3D observation contract; no angle replay or force inference',
                   'description':'Required and optional properties are explicit. No example data are generated.',
                   'x-semantic-validation':[
                       'samples nonempty; strictly increasing time_s; source_frame retained, never concatenate gaps',
                       'keypoint_names unique; each sample keypoints and valid_mask length equals keypoint_names length',
                       'valid_mask true iff corresponding point is finite and accepted; invalid point must be null',
                       'root_valid iff root is non-null and root.valid; basis columns right handed and orthonormal; root method and provenance required',
                       'confidence if supplied has point count length; absent confidence means unknown, not 1',
                       'phase/footcontact values and masks match declared six-leg order recorded in method; inferred labels never force targets',
                       'all floats finite; JSON NaN/Infinity prohibited; all raw_files and derivations sha256 verified',
                       'unknown scales represented null; no SI/root/world loss until independently calibrated',
                       'root_to_model_transform if supplied includes hashable calibration artifact via derivations; never fit on holdout',
                       'anatomical names/candidate mappings require validation before contributing to losses; unresolved mappings masked out',
                       'observed_reconstruction means upstream video reconstruction, not unprocessed/direct sensor truth',
                       'do not treat this schema as dataset readiness; external source identity and license remain separate gates']})
    return schema


LOSS = '''# S2/S4 六腿3D监督接口：规格，不是训练或行为验收结果

本次只静态解析实际3470 XML与已有compiled数组，不导入MuJoCo/身体/神经网络，不执行compile、forward、step、训练或联网。目标是连续六腿3D与身体root监督；迷宫仅次要行为任务。没有生成fake/synthetic dataset，没有更改runtime/XML/defaults/tracker，也不提交。

## 身份、关节与点映射

实际模型为 `/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z/RUN/MODEL.xml`，sha256 `3470c72f7e68f181ec1a80a5dc5fbff182b750091e7a1ddc7ccfbfb590b80cb6`。43关节=自由根+42hinge，nq49/nv48，90肌肉、71mesh；不是75肌肉。75是历史五腿镜像派生数量口径，不能作为全模型计数。实际映射74条非空、16 unresolved，74条对应66不同pool，保持同池decoder绑定。

JSON的joints保存42条名称、所在body/父body、关节顺序、局部axis/pos、qpos/dof地址、弧度range、springref及与已有compiled逐项比对。XML yaw/pitch/roll名称并不承诺常见Euler轴：例如Coxa_yaw局部轴是[1,0,0]，Coxa_roll为[0,0,1]。同body多个hinge按模型顺序组合，不能只按关节名字改符号。父body不是前一hinge；joint_order_in_body另列。

NMF每腿7 DOF含Coxa yaw/pitch/roll、trochanterfemur pitch/roll、Tibia pitch、Tarsus1 pitch；当前7 DOF含Coxa三轴、Trochanter三轴、Tibia pitch，没有Tarsus1 hinge。两边都是42不能证明角度可复制。NMF右侧legacy roll/yaw另有API符号转换。NMF脚点是tarsus5原点，不是TaTip或接触面。685数据则为多视角重建50点，骨架节点顺序与数组字段顺序不同，必须按名字重排；本工具已保存边名与data索引。

映射是候选而非解剖认证：ThxCx→Coxa原点、Tro→前足Trochanter原点/中后足Femur原点、FeTi→Tibia原点、TiTa→Tarsus1原点，均需源定义/标注与模型解剖对应验证。TaT1/TaT3无精确源定义，不猜；TaTip只能考虑Tarsus5上的待标定局部点，没有已知distal offset，不把原点当爪尖。源未提供中后足ThxCx点，不补零。marker位置局部[0,0,0]只是明确的模型body原点；非接触点、非新写入site。

## 框架、尺度与时间

原始lab坐标/Scutellum bodyframe/模型Thorax free-root frame/COM不能混用。685 body原点Scutellum，x朝Antenna_Base，y由WingR→WingL正交化，z=x×y；基矩阵按世界列存储，p_body=R^T(p_world-o)。当前自由根位姿属于Thorax，其原点不等于Scutellum；需有来源的固定外参(T_B→root)，不能默认零。NMF参考是thorax局部坐标；其单位也不能直接混入685未知源单位或模型长度。

685帧用原source_frame/800，nominal dt=.00125s，非独立采集时间戳。NMF processed step dt=.003s、45点/步、对称/闭合/平滑；33个FK相位含重复端点且六腿同相位不是同步六腿记录。禁止把它当连续动物训练样本，禁止把800Hz与NMF dt混用。维持逐帧有效mask和缺口；不得跨缺口插值为观测，已有上游插值/滤波显式记derived lineage。

源长度未知时仅允许声明的形状归一化损失；不能获得真实m/s或N。每腿骨长从训练集有效点对估计（robust median+离散度），按当前模型同名marker骨段作局部方向重定向，保留左右/前中后各段独立长度。禁止一个总比例掩盖所有形态差异。缺段/映射不确定则mask，不猜。归一化后也不宣称SI校准。root translation loss只有外参/尺度可追溯时启用；body形状与root旋转可在其各自可观测范围先评估。

## 可实施损失与边界

先冻结真实来源/数据哈希、模型、映射版本、训练/验证/留出（优先按动物/会话；单bout时间留出要承认相邻相关），只在train估外参/骨长/normalizer/权重，验证选参后一次holdout，禁止按验证或失败结果重写目标。

1. root姿态对齐：由真实身体点推导R_obs与o_obs，保留derived标签、有效性、残差；用训练集外参把预测模型root变到同一观察frame。旋转用SO(3)测地角（quaternion绝对点积或log），不用直接Euler逐分量差。允许单次初始朝向/原点对齐；禁止每帧最佳配准消掉真实身体运动或腿误差。
2. Cartesian主损失：L_pts = sum(w_tk * valid_tk * Huber((p_pred_tk - p_retarget_tk)/s_leg,k)) / sum(w_tk * valid_tk)。零有效数项返回“不可评估”，不能记0成功。p_pred从同次真实物理轨迹的固定body marker读取；retarget来自真实原始点和冻结骨长变换，不直接复制source角度。可分别报告bodyframe形状、root旋转、可标定后world位移，不能混一个单位。
3. 时间损失：在相邻有效且间隔合规的真实时间上作速度/加速度或腿间相对轨迹损失；保留实际dt，先声明滤波/噪声界。不得用动态时间弯曲或逐腿自由相位把不同步六腿伪装同步成功。相位只能辅助描述并报告推断方法/置信度。
4. 生物与物理分列：视频推断phase/footcontact只是inferred_label，可以辅助验收步序；不是force、接触位置、承重或感觉电流真值。力/摩擦/滑移/非足支撑需来自实际物理量或独立传感器。未知地面不凭足z最小值伪造真值。现有支撑loss与3D运动loss并列，防止身体倒地后score下降冒充行走。
5. IK只离线参考：在冻结原生关节轴/范围下拟合可达目标，报告不可达残差、限位、腿段交叉/自碰撞、根姿态和marker误差；IK解不当真实动物角度，更不作运行每步控制。动态可执行性还需另行物理验收。运行链必须SNN→原decoder共享池→native muscle→物理身体→实际感觉；优化器只离线选择固定参数，不产生每步动作/root平移/姿态修正。

## 有限可追溯参数群与结构缺口

现训练只有全局exc_gain、inh_gain、threshold_shift三个常量；其中前两者是gain，第三是阈值shift，不能把三参数统称三个gain。其支撑loss没有六腿3D目标；三个自由度无法独立指定42关节/66池时序，也没有训练覆盖证明。loss减小不证明动作维度补齐，更不识别唯一内部神经状态。

未来先从小的具名参数群开始：按有来源的腿/运动池/感觉类别，给“既有非零边”正倍率和有界阈值offset；可先六腿×E/I入边组及六腿MN阈值组，再按数据支持拆分有文献/注释身份的拮抗池。分组须互斥或显式乘积组合，记录每条边/神经元所属组、旧新值与来源/hash，正倍率用exp(log_gain)，设预登记界与趋近1正则。保留root顺序、实际mask、E/I、延迟、同池decoder绑定、16unresolved，不允许0倍率熄灭通路或每执行器独立增益打散同池绑定。冻结肌肉几何、参数和模型身份；传感阈值只在单位可解释后另标工程/实测。建议不是本轮实现或训练结论。

实际mask=0的边，任何有限正倍率仍=0；历史3455实际输入中3086是出度0非读出、其中3073有rawtier输出的证据单列在tracker。不能靠降阈值声称补回断路，也不能把raw边直接当已准入E/I。任何结构修复需独立来源表(root版本/原始边/递质预测/受体假设/准入差异/新旧mask哈希)及匹配sham/切断对照，不能伪装为现有权重微调。身体非负肌肉力矩方向/66共享绑定缺口也不由神经loss自动解决。

## target_schema.json的必需与可选字段

必需：schema版本；dataset/release/session/bout/individual身份状态；每个原始输入完整路径+SHA256；测量与上游处理声明；坐标/尺度/身体frame/模型hash；实际时间来源/同步/缺口策略；明确keypoint_names和按名称骨架边；每帧time/source_frame/原始重建keypoints/valid_mask/observed_vs_derived/root或null/root_valid；派生文件hash及split。unknown individual或scale必须显式null/unknown，不借BANC fly73身份套给另一实验动物。

可选：逐点confidence（缺失不填1）、phase和footcontact（方法+mask+inferred标签）；root缺失时null/false，禁止构造虚假root。source keypoints标observed_reconstruction只指上游视频重建，不能说未平滑的直接测量。observed/derived是字段级来源，不是质量认证。schema的x-semantic-validation列出JSON Schema无法跨数组验证的长度、唯一性、正交性与哈希规则；装载器必须逐项执行。当前本地数据仍缺可靠绝对尺度、接触真值、独立采集时间戳与精确marker定义；它是可用形状参考，不是全部训练准入门已过。

## 文件管理

核心必需：工具、JOINT_AND_MARKER_MAPPING.json/CSV、target_schema.json、LOSS_SPEC.zh-CN.md、SOURCE_IDENTITY.json、VERIFICATION.json、INDEPENDENT_VERIFICATION.json。重要辅助：REPORT.zh-CN.md、MANIFEST_SHA256.json、CLEANUP_LEDGER.json、工具快照与独立验证源。没有生成新数据集或复制大原始数据；没有确认可弃产物，实际清理0，既有用户资料与冻结证据不改。
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args(); out = args.out
    if out.exists(): raise RuntimeError('Refuse overwrite existing evidence directory')
    inputs = [RUN/'MODEL.xml', RUN/'MAPPING.json', FIT/'COMPILED_MODEL_ARRAYS.npz',
              FIT/'BODY_IDENTITIES.json', FIT/'SOURCE_IDENTITY.json',
              SOURCE/'info__kp_names.json', SOURCE/'SKELETON.json', SOURCE/'REFERENCE.npz',
              SOURCE/'bout_0000__kp_data.npy', SOURCE/'REFERENCE_METADATA.json',
              SOURCE/'AUTHOR_README.md', SOURCE/'POSTPROCESS_SOURCE.py', SOURCE/'POSTPROCESS_CONFIG.yaml',
              NMF/'REFERENCE_METADATA.json', NMF/'REPORT.zh-CN.md', NMF/'STANDARD_GAIT_FK.npz',
              ROOT/'prepare_standard_gait_reference.py', ROOT/'adapt_real_freewalking.py',
              ROOT/'GOAL_ROADMAP.zh-CN.md',ROOT/'GOAL_TRACKER.zh-CN.md',
              ROOT/'tools/train_banc888_checkpoint_parameters.py', Path(__file__).resolve()]
    hashes = {str(p):sha(p) for p in inputs}
    assert hashes[str(RUN/'MODEL.xml')] == MODEL_SHA
    fit_identity = json.loads((FIT/'SOURCE_IDENTITY.json').read_text())['sha256']
    assert fit_identity[str(RUN/'MODEL.xml')] == MODEL_SHA
    xml = ET.parse(RUN/'MODEL.xml').getroot()
    assert xml.find('compiler').get('angle') == 'radian'
    assert not xml.findall('include')
    arrays = np.load(FIT/'COMPILED_MODEL_ARRAYS.npz', allow_pickle=False)
    identity = json.loads((FIT/'BODY_IDENTITIES.json').read_text())
    bodies = []; joints = []; body_ids = {'world':0}; dof = 0; qpos = 0
    def walk(parent, parent_name):
        nonlocal dof, qpos
        for body in parent.findall('body'):
            name = body.get('name'); bid = len(bodies)+1; body_ids[name] = bid
            bodies.append(dict(name=name, body_id=bid, parent=parent_name,
                parent_id=body_ids[parent_name], pos_parent=vec(body.get('pos'),'0 0 0'),
                quat_parent_wxyz=vec(body.get('quat'),'1 0 0 0')))
            order = 0
            for joint in body:
                if joint.tag not in ('joint','freejoint'): continue
                typ = 'free' if joint.tag == 'freejoint' else joint.get('type','hinge')
                assert typ in ('free','hinge'), typ
                jid = len(joints); axis = vec(joint.get('axis'),'0 0 1') if typ == 'hinge' else [0.,0.,1.]
                row = dict(name=joint.get('name'), joint_id=jid, type=typ, body=name, body_id=bid,
                    parent_body=parent_name, parent_body_id=body_ids[parent_name], joint_order_in_body=order,
                    axis_body_local=axis, anchor_body_local=vec(joint.get('pos'),'0 0 0'),
                    dof_address=dof, qpos_address=qpos,
                    range_rad=vec(joint.get('range'),'0 0') if typ=='hinge' else None,
                    limited=joint.get('limited','true')=='true' if typ=='hinge' else False,
                    springref_rad=float(joint.get('springref','0')) if typ=='hinge' else None)
                joints.append(row); order += 1; dof += 6 if typ=='free' else 1; qpos += 7 if typ=='free' else 1
            walk(body,name)
    walk(xml.find('worldbody'),'world')
    assert [j['name'] for j in joints] == identity['joint_names']
    assert (qpos,dof,len(joints)) == (49,48,43)
    for j in joints:
        i = j['joint_id']
        assert int(arrays['jnt_type'][i]) == (0 if j['type']=='free' else 3)
        assert int(arrays['jnt_dofadr'][i]) == j['dof_address']
        assert int(arrays['jnt_qposadr'][i]) == j['qpos_address']
        assert int(arrays['jnt_bodyid'][i]) == j['body_id']
        if j['type']=='hinge':
            np.testing.assert_allclose(arrays['jnt_axis'][i], j['axis_body_local'], atol=1e-15,rtol=0)
            np.testing.assert_allclose(arrays['jnt_pos'][i], j['anchor_body_local'], atol=1e-15,rtol=0)
            np.testing.assert_array_equal(arrays['jnt_range'][i],j['range_rad'])
            assert bool(arrays['jnt_limited'][i]) == j['limited']
    for b in bodies:
        i=b['body_id']; assert int(arrays['body_parentid'][i]) == b['parent_id']
        np.testing.assert_allclose(arrays['body_pos'][i],b['pos_parent'],atol=1e-15,rtol=0)
    actuators = list(xml.find('actuator'))
    assert len(actuators)==90 and [a.get('name') for a in actuators]==identity['actuator_names']
    assert np.all(arrays['actuator_gaintype']==2) and np.all(arrays['actuator_biastype']==2)
    assert len(xml.findall('./asset/mesh')) == 71 and len(arrays['name_meshadr'])==71
    assert len(arrays['name_eqadr'])==0 and np.all(arrays['body_mocapid']==-1)
    mapping=json.loads((RUN/'MAPPING.json').read_text())['rows']
    assert [r['actuator'] for r in mapping] == identity['actuator_names']
    mapped=[r for r in mapping if r['local_indices']]
    pools={tuple(r['local_indices']) for r in mapped}
    assert (len(mapped),len(mapping)-len(mapped),len(pools))==(74,16,66)
    names=json.loads((SOURCE/'info__kp_names.json').read_text())
    skeleton=json.loads((SOURCE/'SKELETON.json').read_text())
    assert len(names)==len(set(names))==50 and set(names)==set(skeleton['node_names'])
    edges=[[skeleton['node_names'][a],skeleton['node_names'][b]] for a,b in skeleton['edges']]
    edges_in_data=[[names.index(a),names.index(b)] for a,b in edges]
    raw=np.load(SOURCE/'bout_0000__kp_data.npy',allow_pickle=False).reshape(-1,50,3)
    ref=np.load(SOURCE/'REFERENCE.npz',allow_pickle=False)
    np.testing.assert_array_equal(raw,ref['raw_keypoints']); assert raw.shape==(685,50,3)
    R=ref['body_basis_world_columns']; eye=np.eye(3)
    orth_error=float(np.max(abs(np.swapaxes(R,1,2)@R-eye)))
    assert orth_error<1e-12 and np.all(np.linalg.det(R)>0)
    frames=ref['source_frame']; times=ref['time_from_recording_start_s']
    np.testing.assert_allclose(times,frames/800,atol=1e-12,rtol=0)
    markers=[]
    for source_leg, leg in LEGS.items():
        candidates={'ThxCx':leg+'Coxa','Tro':leg+('Trochanter' if leg.endswith('F') else 'Femur'),
                    'FeTi':leg+'Tibia','TiTa':leg+'Tarsus1'}
        for name in names:
            if not name.startswith(source_leg+'_'): continue
            suffix=name.split('_',1)[1]; body=candidates.get(suffix)
            if body:
                assert body in body_ids
                candidate=dict(body=body,body_id=body_ids[body],local_position=[0,0,0],definition='body origin, not validated source marker')
                status='ANATOMICAL_NAME_CANDIDATE_NOT_VALIDATED'
            elif suffix=='TaTip':
                candidate=dict(body=leg+'Tarsus5',body_id=body_ids[leg+'Tarsus5'],local_position=None,
                               definition='unknown distal offset, NOT tarsus5 origin or measured contact')
                status='DISTAL_OFFSET_UNRESOLVED'
            else:
                candidate=None; status='SOURCE_POINT_DEFINITION_NOT_ESTABLISHED_DO_NOT_GUESS'
            markers.append(dict(source='freewalking685',source_keypoint=name,source_index=names.index(name),
                leg=leg,source_definition='key name and upstream skeleton edge only; precise landmark not established locally',
                candidate=candidate,status=status,loss_enabled=False))
    nmf=json.loads((NMF/'REFERENCE_METADATA.json').read_text())
    for leg in LEGS.values():
        markers.append(dict(source='NMF_processed_FK',source_keypoint=leg.lower()+'_tarsus5_origin',source_index=None,
            leg=leg,source_definition=nmf['foot_definition'],
            candidate=dict(body=leg+'Tarsus5',body_id=body_ids[leg+'Tarsus5'],local_position=[0,0,0],definition='homologous body origin; morphology/frame retarget still required'),
            status='DEFINED_SOURCE_BODY_ORIGIN_NOT_TATIP_OR_FORCE',loss_enabled=False))
    hips=[r for r in joints if r['type']=='hinge']; assert len(hips)==42
    for leg in LEGS.values(): assert sum(j['name'].startswith('joint_'+leg) for j in hips)==7
    result=dict(model_identity=dict(xml=str(RUN/'MODEL.xml'),sha256=MODEL_SHA,nq=qpos,nv=dof,hinges=42,
        muscles=90,meshes=71,free_root='thorax_free',free_root_body='Thorax',nmocap=0,neq=0,
        mapped=74,unresolved=16,distinct_shared_pools=66,xml_angle_unit='radian',
        lengths='model_native; no calibrated conversion to freewalking source or NMF implied'),
        joints=hips,free_joint=joints[0],bodies=bodies,markers=markers,
        source_skeleton=dict(data_keypoint_names=names,node_names=skeleton['node_names'],
            order_equal=names==skeleton['node_names'],edges_by_name=edges,edges_in_data_order=edges_in_data),
        source_dof_order=nmf['source_dof_order'],
        dof_difference='NMF Tarsus1 pitch vs current Trochanter yaw; no one-to-one angle copy',
        marker_policy='All candidates disabled for training until semantic/calibration gates; no guessed source landmark')
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'JOINT_AND_MARKER_MAPPING.json',result)
    columns=['row_type','name','leg','body','parent_body','joint_id','joint_order_in_body','dof_address',
             'qpos_address','axis_body_local','anchor_body_local','range_rad','springref_rad','source',
             'source_index','status','candidate','loss_enabled']
    with (out/'JOINT_AND_MARKER_MAPPING.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for j in hips:
            r={k:j[k] for k in columns if k in j};r.update(row_type='joint',leg=j['name'][6:8]);w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
        for m in markers:
            r={k:m[k] for k in columns if k in m};r.update(row_type='marker',name=m['source_keypoint']);w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
    dump(out/'target_schema.json',target_schema())
    (out/'LOSS_SPEC.zh-CN.md').write_text(LOSS)
    dump(out/'SOURCE_IDENTITY.json',dict(repository=str(ROOT),requested_head='c8a08d3f258b4dbeede978ddabd1d828f4fa17d7',
        source_hashes=hashes,compiled_array_archive_binding='FIT/SOURCE_IDENTITY.json binds archived compile execution to exact MODEL hash; this audit statically crosschecks joint/body arrays',
        upstream_data='video multiview reconstruction with upstream processing; not direct point/force sensors'))
    verification=dict(passed=True,xml_sha256=MODEL_SHA,joints_checked=42,bodies_checked=len(bodies),
        compiled_arrays_archive_keys=len(arrays.files),joint_name_order_exact=True,compiled_axis_range_address_parent_pass=True,
        source_frames=685,source_points=50,raw_exact=True,skeleton_order_equal=names==skeleton['node_names'],
        skeleton_name_remapping_verified=True,orthonormal_basis_max_error=orth_error,
        timestamp_source_frame_div_800=True,source_units='UNKNOWN',marker_candidates=len(markers),
        unresolved_no_guess=sum(m['candidate'] is None for m in markers),
        counters=dict(download=0,model_load=0,compile=0,forward=0,body_step=0,network_load=0,neural_step=0,training=0),
        exploratory_failure='Read-only array inspection requested absent names key: KeyError; resolved with archived BODY_IDENTITIES joint_names. No output/dynamics generated.',
        source_hashes_unchanged=all(sha(Path(p))==h for p,h in hashes.items()),command=' '.join(sys.argv),exit_code=0)
    dump(out/'VERIFICATION.json',verification)
    dump(out/'CLEANUP_LEDGER.json',dict(actual_deleted=[],reason='No created disposable files; no original data or frozen evidence modified',
         importance={'核心必需':['JOINT_AND_MARKER_MAPPING.json','JOINT_AND_MARKER_MAPPING.csv','LOSS_SPEC.zh-CN.md','target_schema.json','SOURCE_IDENTITY.json','VERIFICATION.json','INDEPENDENT_VERIFICATION.json'],
                     '重要辅助':['REPORT.zh-CN.md','CLEANUP_LEDGER.json','MANIFEST_SHA256.json','audit_kinematic_training_interface.py','VERIFY_INDEPENDENT.py'],
                     '可再生临时':[],'待确认':[]}))
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (out/'REPORT.zh-CN.md').write_text('# S2/S4静态运动学训练接口审计\n\n实际3470完整模型：42hinge，自由根，90肌肉/74mapped/16unresolved/66共享池，71mesh。XML与已有compiled轴/范围/地址/父body核验通过。完整数据685×50×3与原数组exact，source skeleton按名称重排，body basis正交。详见LOSS_SPEC.zh-CN.md与机器映射。\n\n这是未来真实数据监督契约，未训练、未证明物理行走或来源marker精确对应。没有synthetic数据、联网、模型或网络加载、compile/forward/step；未修改runtime/XML/defaults/tracker或提交。source hash与验证文件见清单；独立验证完成后另记INDEPENDENT_VERIFICATION.json。\n\n文件重要程度及实际清理在CLEANUP_LEDGER.json；实际删除0。所有标为候选的marker默认loss_enabled=false，主agent需补真实源定义/外参/尺度后逐项准入。\n')
    print(json.dumps(verification,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
