# S2/P2 惯性质量布局纯数组合同

手动研究原型，非成品。本轮不import MuJoCo、不创建模型/forward/step；不改失败runner或任何历史哈希。只验证两个独立解码算法，不能冒充安装wheel数值或真实stationarity结果。

固定3.9.0源码依据：https://github.com/google-deepmind/mujoco/blob/3.9.0/src/engine/engine_support.c （mj_fullM沿i,parent[i],...顺序读输入，diag first）；https://github.com/google-deepmind/mujoco/blob/3.9.0/src/engine/engine_core_smooth.c （M是下三角CSR，diag在行末；mapM2M桥接qM）。父上轮和独立审查已读取，本文不新增网络或模型动作。

树状qM输入：parent长度nv，根=-1，其余0<=parent[i]<i，因此无环且索引拓扑有序。按行i及其祖先链，依次消费qM，先对角，再最近祖先至根；精确长度等于所有链长之和，无padding，双向对称写入，非边保持0。拒绝错误parent顺序/环、错长度/非finite/非整数索引。

CSR M输入：rowadr/rownnz各nv，非空行、连续无padding、从0完整消费全部values/colind。列必须严格递增、0<=col<=row，对角row必须在行末（且只一次）。逐下三角元素镜像一次，不能加倍对角、不能平均对称化。拒绝重复、越界、上三角、缺对角、重叠/空隙/未消费尾部、错误dtype及非finite。这里是本适配层严格子合同，不宣称支持任意一般CSR或padding。

nv=0拒绝，nv<=256是本工具分配上界（非RSS界）。输入索引必须整数非bool，uint64越界在int64转换前拒绝。values必须实数finite，float转换后仍finite；complex/object/bool拒绝。解码只保证结构/对称，不保证SPD或物理有效性；SPD交给后续stationarity工具，不自动jitter或改值。

独立oracle：测试用指定dense SPD及parent先构造两种不同序列，期望dense直接列常量；随机有分支forest造tree支持矩阵、严格对角占优保证SPD，独立encode两种格式后各解码与dense比较。非identity列序和非对角值必须覆盖，坏布局错位应被拒绝；故意把CSR values当qM时可能长度相同且结构接受但数值错，必须展示此类结构校验无法辨认buffer身份。调用者必须保留来源字段和双解码交叉，不靠SPD或长度推断来源。

未来新采集器使用d.qM给mj_fullM，d.M给CSR解码，另用mj_mulM的基向量逐列交叉；真实输入需同端点先保存raw和phase/failure状态再分析。此合同不执行该未来采集，不修改上一轮的错误声明。桌面样例明确SYNTHETIC，测试前后历史90路径及失败4source哈希只读检查；无新仿真图。
