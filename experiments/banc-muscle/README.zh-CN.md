# 独立 BANC—肌肉身体研究

本项目使用已有 BANC 连接组和规范 MuJoCo 肌肉身体。范围包括周边感觉观测、腹神经索局部神经回路、运动神经元到肌肉的工程映射、足地承重及后续闭环研究。电极及独立颈部仿真排除在本项目范围之外。

当前基线和证据位于 `20261120T061904Z-independent/`。该目录中的图、物理、训练接口和验收单独登记。`INTERFACE.json` 中可用于后续学习的功能候选初始全部关闭，既有原始突触和待学工程权重各有独立来源。

局部线性翻译与原边 92%–108% 的首轮闭环对照见 [试验报告](20261120T074805Z-linear-interfaces/REPORT.zh-CN.md)。该派生件可读入逐连接参数，首轮为未训练的工程接口，12 个站立回合均未达标。

当前站姿和感觉预热入口为 [run_stance.py](run_stance.py)。在项目根目录运行 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv_body/bin/python -B experiments/banc-muscle/run_stance.py /tmp/banc-warm-release`。默认先用原生肌肉稳定 250 ms，再预热前足感觉 250 ms，然后完全撤手观察 250 ms。可用 `--mode all` 重跑四组对照。已通过算法承重和触觉预热的是 1 ms 派生地面，神经接管后的持续站立仍未通过。[站姿、接触参数和验收](20261120T081056Z-stance-warmup/REPORT.zh-CN.md)

分批 CMA-ES 站立候选支持上述原入口的 `--parameters`。名义初态首次越门 141.1 ms，两个保留初速分别 142.4 / 42.2 ms；完整 250 ms 站立仍未通过。[训练、权重和消融结果](20261121T041548Z-cma-stance/REPORT.zh-CN.md)

[十倍时长站立续训](20261121T043642Z-tenfold-stance/REPORT.zh-CN.md) 从上述 CMA-ES 候选继续训练 54 分 36 秒。名义初态首次越门推迟到 151.3 ms。三个新留出中两组改善，一组提前 4.2 ms，完整 250 ms 站立为 0/3。可用同一入口的 `--parameters` 加载本次实验的 `SELECTED.npz`。
