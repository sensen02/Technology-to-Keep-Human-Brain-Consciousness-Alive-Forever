# 真实头部双视野工程相机：只读调研与零步预检入口

本委托只读取源码与写文本/Python源码，未导入Torch/NumPy/MuJoCo、加载数组、构造Core、forward、渲染或训练。父代理拥有唯一重任务权限。当前实现是32×32RGB双透视工程相机，不是已验证生物复眼；没有更换现有身体或控制器。

## 已核对的安装源码/API

- `/run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/mujoco/include/mujoco/mjvisualize.h:205`：`mjvGLCamera`有pos、forward、up；frustum_center、frustum_width（渲染不用width）、frustum_bottom/top/near/far及orthographic。
- `/run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/mujoco/rendering/classic/renderer.py:299`：`Renderer.update_scene`调用mjv_updateScene；因此必须在每次update_scene之后设置`renderer.scene.camera[0]`和`[1]`，否则姿态会被覆盖。
- 同一renderer源码184行直接mjr_render当前scene，不再次计算camera；两个GLCamera对同一单眼设置相同值，避免单目绘制中的双视图平均偏移。需要父实际EGL验证绑定与图像，不把文本字段存在视为运行通过。
- `/run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/flygym/compose/fly/base_fly.py:549`：add_vision会增加sensorbody和camera；它适合FlyGym自己的新构造体，不适合对现有冻结compiled身体就地增加ncamera。因此本轮不用它，也不重新compile XML。
- `/run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/flygym/simulation.py:408`：get_raw_vision实际MuJoCo RGB后correct_fisheye，默认眼renderer隐藏geomgroup1/2以避自遮挡。本轮不照抄隐藏身体的做法。
- `/run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/flygym/assets/model/neuromechfly/vision.yaml`：157°、512×450、721ommatidia、fisheye3.8/zoom2.72，配置含hidden_segments；与当前派生Head坐标未标定，不能直接套用眼位/鱼眼参数声称生物有效。
- `/run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/flygym/vision/retina.py`：Retina载入compound_eye.npz和Numba计算，raw_image_to_hex_pxls才是721×2通道变换。32×32RGB不能无声明地传给固定512×450idmap。
- `/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/vision.py` 已有OmmatidiaFrontEnd光学/对比度工程编码，文档明确非真实运动检测器；不复用旧BodyBackend/CPG控制路线。

## 最小光学实现

`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/camera.py`：真实Head xpos/xmat变换本地眼位与forward/up，左/右光轴±25°、俯角−15°、垂直FOV100°、32×32RGB，near=.005/far20原生长度。眼位在已审核头mesh AABB前/左右边界之外额外10%extent，z偏移+.15；声明工程近似，避免相机在不透明头内部。所有71mesh不隐藏、不改alpha，self-occlusion自然保留。

光学相机仅在内存MjvScene中定义，不改MjModel相机数组、XML、惯量、碰撞、力矩、根轨迹或物理参数。目标用scene-only高对比竖柱，遮挡用opaque box；普通深度遮挡由实际RGB渲染产生，不从坐标手工画像素。装饰明确objid=-1/objtypeUNKNOWN/segid=-1，只RGB不做受污染segmentation。目标坐标仅环境侧场景构建，未来policy只接真实RGB/编码特征。

## 父实际执行

先读 `/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/PREREG_CAMERA_PROBE.json`；所有光学参数/少量场景已在任何数值运行前写定。

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/probe.py --parent-resource-released
```

输出独占新目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/CAMERA_PROBE/`：EYE_PANEL.png、每条件左右实际图、RAW_RGB.npz、PREFLIGHT.json、IDENTITY.json。已有目录拒绝覆盖；真实运行失败时先读日志/检查输出，禁止盲重跑。

probe读取已归档neutral的3个实际姿态（handover、中段、末端），在私有可视Core恢复保存qpos/qvel，只mj_forward，mj_step/step1/step2替为抛错。compiled identity运行前后核验、所有71mesh alpha1、qpos/qvel不变。左右/off/遮挡/无目标同遮挡比较，高对比目标固定world。光轴±20°是显式counterfactual optical-only诊断，不冒称真实头部转动或神经动作；另两个保存姿态使用真实head transform，均需父实际查看图像判定可见性。self-occlusion没隐藏，低分辨率可见性尚未实际验证，失败应先修光学再训练。

## 因果时序（统一父契约）

相机50Hz每20mscapture@v，图像release@v+.020；motor/行为200Hz每5ms查询，保持最后已释放图像。初始20ms validfalse+zero占位，不称实际曝光。camera.PendingVision把capture/query分开，禁止pending覆盖和非连续20ms/5ms时钟；probe使用真实已渲染RGB检查初始valid、20ms延迟和中间5msheldframe无未来泄漏。当前没有行为网络forward或部署成功声明。

## 后续需要父核实

实际眼图中的左右像素方向、竖柱尺寸/颜色与地面可分辨性、遮挡变化以及完整身体自遮挡；必须看到真实图后再考虑灰度化/视觉encoder。若改参数，保存首次失败/原参数身份，在新scope或追加跑前登记中改，不能把看图后选择当原预登记成功。
