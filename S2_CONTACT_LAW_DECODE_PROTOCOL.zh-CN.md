# S2/P2 同端点接触力解码预注册（计算前）

有界离线手动研究，非成品。输入唯一固定目录 `/home/sensen/Desktop/engine修复验收/S2-raw-contact-20261014T204339Z-7ba64625` 的 manifest 及两 native_floor pulse NPZ；不生成新动力学，不编辑生产，不 forward/step，不搜索参数，不依赖安装/服务/硬件/reset。

## 实际已读 schema
两 NPZ 各400端点；contact_offsets(401)、contact_local6(N,6)、contact_efc_address(N)、contact_dim(N)、contact_friction(N,5)；efc_offsets(401)、efc_force/state/type/id(M)。manifest 每record cone=0，compiled没有cone但record本身是编译时记录（原归档runner可审阅）。source_hashes_before/after十文件和Python/MuJoCo3.9.0环境已读。使用saved cone=0，不需要模型加载；unknown cone拒绝计算，另行冻结cp model onlyload核实身份，无forward/step。

## 官方依据（只取源码，不远程执行）
- https://raw.githubusercontent.com/google-deepmind/mujoco/3.9.0/src/engine/engine_util_misc.c ：mju_decodePyramid，dim1 normal=p[0]；其余 normal顺序求和2*(dim-1)行，分量i+1=(p[2i]-p[2i+1])*friction[i]。
- https://raw.githubusercontent.com/google-deepmind/mujoco/3.9.0/src/engine/engine_core_util.c ：mj_contactForce先六维清零，有效contact且efc_address>=0时按cone解码，否则零；非pyramidal复制dim分量。
- https://raw.githubusercontent.com/google-deepmind/mujoco/3.9.0/include/mujoco/mjtype.h ：cone0 pyramidal/1 elliptic；constraint state0 SATISFIED(零cost)、1 QUADRATIC、2 LINEARNEG、3 LINEARPOS、4 CONE。state histogram仅端点约束行分布，不是跟踪物理点的状态变化。

## 固定计算与验收
每端点独立slice efc，address只在该端点有意义。不跨时钟匹配contact/efc行。独立本地六维decoder不用MuJoCo解码函数；dim允许1/3/4/6，所需行1或2*(dim-1)，friction至少dim-1且finite/nonnegative；inactive address -1返回六零，其他负address或越界拒绝。验证offset/ncon/nefc/有限值、efc_type(5 frictionless/6 pyramidal)和efc_id对应端点contact索引。

固定工程绝对容差1e-10，无相对放宽：max所有六分量abs(decoded-saved)<=1e-10；统计不通过contact数、最大误差端点/contact/component、inactive数/残差。对联系contact的active rows统计min、严格负数数及低于-1e-10数，>=-1e-10仅金字塔工程非负通过，不把所有efc（如equality/limit）当接触。所有state按官方enum解码，未知拒绝。

保存全400端点max decode residual、max局部线性力范数（前三分量）和max局部力矩范数（后三分量，不混单位），画残差/局部力范数图并实际read_image。所有输入archive文件与十source、工具和protocol前后SHA256；核对历史source身份。独立复算不导入工具，使用官方本地mju_decodePyramid或独立矩阵算法核对最大残差/count/hash。

## 无法计算的动态接触律
NPZ不含efc_AR、efc_b、efc_vel、efc_aref、efc_R/D、efc_J、qacc/qacc_smooth、contact solref/solreffriction/solimp/includemargin等字段；qvel/position/dist/friction/state不足以重建AR*f+b/KKT或速度/solref接触律残差。不从未记录字段造数、不重跑forward补算。完整动态律残差未计算、物理差峰原因仍unknown、收敛/生物/站立行走均未证明。missing字段清单随JSON记录；下一最小动作仅起草未来最小观测需求，不执行新动态。
