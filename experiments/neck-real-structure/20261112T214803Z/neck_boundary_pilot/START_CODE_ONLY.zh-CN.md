# 两条真实身份纤维的边界传输pilot：父执行入口

状态固定为 **SCOPED_TWO_FIBRE_TRANSPORT_NOT_FULL_NECK_ADMISSION**。

只选择真实BANC v888的DN720575941352057904和AN720575941351919408，原始root_888、super_class／cervical_connective元数据与两个源定义平面的实际SWC相交提供身份依据。不是从SWC的label2推断上下行，也不在颈部虚构突触／胞体／RNN。

本模块只是同一轴突跨边界的身份保持活动传输。dt=.001秒、delay=.002秒、gain1、无量纲活动±1是显式工程配置，不是实测传导速度／膜电位／刺激幅度。没有膜模型、ANN、完整身体或电极耦合。两条样本纤维不覆盖3699注释候选，不能授予完整颈部或脑训练准入。

## 父顺序执行（子没有执行真实绑定或真实源传输）

1. 玩具测试，可单独复核：

`python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T214803Z/neck_boundary_pilot/test_boundary.py`

实际子已执行16项测试全PASS/exit0，仅合成临时fixtures自动清理，无真实源活动传输。

2. 生成真实SOURCE_BINDINGS并取得stdout中的sha256：

`python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T214803Z/neck_boundary_pilot/build_source_bindings.py --execute-parent-source-binding`

输出新文件：

`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T214803Z/neck_boundary_pilot/SOURCE_BINDINGS.json`

builder读取并核对metadata、独立geometrycheck、公开generation下载receipt、SWC和所有引用源的SHA；还解析NECK_POINT_AND_SAMPLE_GEOMETRY_AUDIT.json中的source_hashes，把实际v888_meta.feather／颈部point tables／SWC底层文件纳入bindings并与报告中expected SHA比较，不只固定嵌入的报告文本。每次传输操作重新核这些源(包括约57MB的元数据文件)，因此这是小步数审计内核，不是高性能实时长回合实现；实际两个版本ID／方向／plane相交条件不成立就拒绝写入。已有文件不覆盖。builder只做源绑定，尚未传输活动。

3. 将步骤2实际输出的SHA填入下列参数，执行小信号／单纤维cut／restore工程数值测试：

`python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T214803Z/neck_boundary_pilot/run_pilot.py --execute-parent-pilot --source-bindings-sha256 ACTUAL_SOURCE_BINDINGS_SHA256_FROM_STEP2`

输出INPUTS_BEFORE、TRANSPORT_JOURNAL、PILOT_RESULT均只写本pilot目录，存在则拒绝盲重跑。正常DN/AN包在第2工程tick到达，切断DN丢其在途包且保留AN传输，restore时钟不reset、只恢复后续新包，旧包不能复活。全部步骤与数据文件SHA在操作前后核验；若科学源或源码改变，必须拒绝，不自动重绑哈希。

BoundaryTransport需要调用者提供预期bindingSHA，send/advance/set_cut均重新核源。包只接受指定方向、原始ID、单个有界值、当前capture_step和绑定SHA，任何关节动作／目标坐标／隐藏字段、错序ID、未来时间、NaN均拒绝。它不证明全程序没有其他旁路；完整三部分程序仍需独立代码审计和未来因果实验。

## 限制与后续

full接口契约保持原NOT_ADMITTED_NECK_SOURCE_PENDING，不改MODEL_CONTRACT.json，不据此训练脑或运行身体。geometry的两个curated seed planes不是完整外科cut geometry；本cut是工程传输关闭，不是模拟组织损伤，restore不是生理修复。真实电生理、时延、受体／递质、刺激伪迹与电极场没有由本pilot建立。

父实际源测试通过后，仅可说两条有来源身份的纤维边界可按工程规则传输／切断／恢复。扩大群体覆盖或训练人工脑，必须另有覆盖、动力学和接口来源证据与预登记，不自动扩大本pilot结论。
