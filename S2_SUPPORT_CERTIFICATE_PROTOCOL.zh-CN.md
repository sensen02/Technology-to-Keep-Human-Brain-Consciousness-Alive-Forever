# 固定diamond矩阵不可行证书协议（计算前）

手动研究非成品，仅离线A门证据，不加载MuJoCo/神经/轨迹，不改变姿态/摩擦/增益。输入为 `/home/sensen/Desktop/engine修复验收/S2-support-gate-20261014T172624/native_diamond.npz`、同目录zero_diamond.npz，使用原保存A,b,lo,hi。先冻结输入SHA。

令A=[R,C]，活化l<=a<=u、反力lambda>=0。若存在y满足C^T y<=0及 y^T b > sum_i max(l_i (R^T y)_i,u_i(R^T y)_i)，则对任意允许x均y^TAx<=上述上界<y^Tb，故该保存系统不可行。y的各分量是方程线性组合权重，不是生物力或单肌原因。

用原scale将行无量纲数值预处理为As,bs，求dual LP：max y^Tbs-sum(z)，z_i>=l_i Ri^Ty且z_i>=u_i Ri^Ty，Cs^Ty<=0，-1<=y<=1；每case10s/maxiter10000，失败只INDETERMINATE。LP给候选，不作证书。

候选转回原行y/scale，用活跃contact零点/肌肉零点/候选box边界重建一个精确有理向量。每个输入float64视作其精确二进制有理数，Fraction Gaussian消元最多48维；只在接近候选的活跃等式中挑独立方程，若不满秩或符号不满足就不宣称成功。精确验证所有C列点积<=0与严格正gap，不用epsilon吞掉无上界contact上的正违约。保存有理numerator/denominator、精确gap、dot结果/上界及浮点展示。此证明对象是存储的float矩阵精确有理化系统，不包括几何/J/肌力采样误差或原生接触兼容性；无区间误差包络不推广连续物理不可行。

独立脚本仅标准库Fraction读取证书与原NPZ重算（NPZ用NumPy），不重新求LP/运行模型。输出权重最大行和对分离gap的贡献仅定位组合约束，不把最大行硬称根因。若无证书不换姿态/参数制造结论。可绘方程权重图并实际打开，不用混合力/力矩范数作生物指标；本轮无全量测试。结论若成立可停止复证该矩阵，下一步才核对输入力域/接触Jacobian定义是否准确代表原生编译模型，不能直接改mu/adhesion或进入神经站立。
