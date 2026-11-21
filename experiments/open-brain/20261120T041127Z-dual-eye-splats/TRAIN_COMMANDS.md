# 训练与复算

照片采集运行 Blender 4.5.16。capture_scan.py 采集初始 116 张训练照、8 张留出验证照及 12 个空箱眼位参考。capture_detail.py 补充 40 张长基线照。capture_low.py 补充 64 张低机位照。capture_foreground.py 对每个时刻拍 12 张可见身体部分的照片。所有采集脚本最后三个参数依次为 Blender 的输出目录、当前分片序号和分片总数。

Brush v0.3.0 Linux 二进制从项目官方发布获取，压缩包 SHA-256 为 4f0f9a8785d1951c62df26aae247c02c5bba32b00f40b06df4e1c9b867399e20。运行于 RX 9070 XT / Vulkan，固定默认种子 42。以下命令从实验目录执行，环境变量 py 和 bl 可从上轮测力实验 setup.sh 载入。

```bash
brush="$PWD/brush-app-x86_64-unknown-linux-gnu/brush_app"
export DRI_PRIME=1 OPENBLAS_NUM_THREADS=1
PYTHONPATH="$PWD/cv" "$py" seed_rgb.py scan detail ../20261120T013500Z-optical-touch/predictions.csv seeded
"$brush" seeded --total-steps 18500 --export-path "$PWD/seeded-checkpoints" --max-splats 260000 --export-every 6000 --sh-degree 1 --refine-every 200 --growth-stop-iter 13800 --mean-noise-weight 5
python3 add_low.py
"$brush" all_training --start-iter 18500 --total-steps 24000 --export-path "$PWD/low-checkpoints" --max-splats 260000 --export-every 6000 --sh-degree 1 --refine-every 200 --growth-stop-iter 0 --mean-noise-weight .6 --lr-mean 8e-6 --lr-mean-end 1e-6
PYTHONPATH="$PWD/cv" "$py" fit_foreground.py foreground ../20261120T034750Z-dual-eye-observer/estimated.json "$brush"
```

OpenCV 4.11.0.86 用于图像特征匹配。所有环境训练数据含 RGB 和光学标定。浅色沙面缺少特征，因此初始局部平面通过上一项由散斑照片估到的地面读数拟合。初始化与续训采用训练图像的子目录，留出的图像不进入优化。身体重建每个时刻各用 10 张训练照，另留两台机位评分。采集时用可见部位的透明背景分割生成 alpha，模型仅接收 RGBA 和外部相机标定。

发布的高斯复算包可独立运行 `bash replay.sh /一个尚不存在的输出目录`。输入为 1 个环境 PLY、20 个身体 PLY、数值标定和八机位解出的眼位。需要 Python、NumPy、SciPy、Pillow 和 Numba。渲染不打开 Blender、原始照片或几何文件。取近处与眼睛相距较近的高斯，在每个像素上用完整三维协方差求射线的高斯密度峰值来处理大视角透视。其余使用椭圆投影。环境和身体高斯合并深度排序、做透明度累积。
