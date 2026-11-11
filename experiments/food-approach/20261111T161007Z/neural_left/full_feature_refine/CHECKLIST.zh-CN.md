# LEFT full feature refinement（独立新阶段）

- [x] PREREG先于代码，parent4000 SHA9c58固定、假设非根因声明。
- [x] 实现strict LEFT_FEATURE_REFINE_V1及fresh8Adam、base-only frozen验证，原adapter43242/8张量。
- [x] 实际预检76项PASS：parent全权重/teacherflow mean+双hidden exact，生产/scan全BPTT梯度/unitAdam最大误差1.1746506545229352e-13，base0grad/neutralexact。
- [x] 固定2000额外updates完成（约1143s），新curve1..2000；save1000/2000 live-save exact，新进程独立保存审核各15项PASS。bash-38 exit0收集。
- [x] prepare12项PASS/50路径preCore；final2000 only LEFT完整1232000原生步，独立实际审核140项PASS；原slip/all6/nativeNF0/yaw30门不变。bash-39/40/41全部exit0收集。
- [x] LEFT行为严格FAIL：yaw29.65128828979569°<30°；maxslip.5780968089711384>.5；六腿LF1/其余2、1ms/nativeNF0、extra0、其余四工程门PASS；neutral依门跳过，无扩budget/structure/物理重跑。
- [x] 所有job CPU affinity1及OPENBLAS/OMP/MKL1，最大1heavy，均结束收齐，资源已释放父。actual拟合MSE=.001003827564677095；finalSHA47e2ec012c016d4c912518ad1d7a326c080d0c65ecf8778a2016c937e58cf838。

当前：FIXED_FEATURE_REFINE_COMPLETE_LEFT_STRICT_FAIL。较原head28.64885°/slip.763168改善，但未达门；不据此认定旧冻结特征是孤立根因。原HEAD4000失败与全部本轮证据保留，tracker父负责。已运行源码未修改，未渲染。
