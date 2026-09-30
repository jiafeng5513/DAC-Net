# DAC-Net(Deep Aberration Correction Network)

CSST 巡天望远镜 PSF 椭率测量中的**波像差校正**——蒙特卡洛误差仿真 + 多目标优化补偿求解器。

本仓库是论文《基于多目标优化的天文望远镜点扩散函数椭率测量中的波像差校正》
(*Applied Optics*, Vol. 63, No. 3, 743–756, 2024, [doi:10.1364/AO.504477](https://doi.org/10.1364/AO.504477),

中文译文见 `docs/`)中蒙特卡洛误差仿真的**正问题数据生成代码**——按给定 RMS 水平随机生成
准直器(平行光管)与干涉仪(GXSS)的条纹 Zernike 波像差,叠加逐样本测量误差,
经 FFT 传播得到 PSF,并按四极矩定义计算椭率。

## 项目背景

中国空间站巡天望远镜(CSST)将用于弱引力透镜(WGL)探测。WGL 在星系椭率上引起的
剪切形变远小于星系自身的内生椭率,因此望远镜的 PSF 椭率成为 WGL 探测中重要的
系统误差来源,必须在地面实验室中精确测量,指标为**椭率测量误差 < 0.01**。

论文分析了两种常用地面测量方法及其误差源:

1. **波像差数值计算法**:干涉仪测波前 → Zernike 系数 → FFT 计算 PSF 椭率。
   大口径长焦系统波前测量误差不可忽略:一阶像散(Z5–Z6)对椭率的影响达 0.034,
   一阶彗差(Z7–Z8)达 0.012,均超出 0.01 指标;测量必需的 2 m 平面镜还存在显著面形误差。
2. **星点目标成像法**:准直器 + 星孔 + 探测器成像。准直器口径不小于被测望远镜(2 m),
   其波像差使 PSF 椭率均值漂移超过 0.01,同样不可忽略。

论文提出的解决方法是:准直器与离轴光学系统各用"理想薄透镜 + Zernike 相位面"等效,
二者之间插入由 Zernike 系数描述的"虚拟透镜"补偿面,以实测系统 PSF(PSF-M)为目标,
用 NSGA-II 多目标优化(椭率差、EE80 半径差、PSF 形态差三个评价函数)求解补偿面系数。
算例:目标 PSF 椭率 0.0349,引入准直器波像差后变为 0.0560,补偿后恢复到 0.0345,
全部补偿区间均包含目标值且范围小于 0.01。

本仓库的仿真代码即用于这类蒙特卡洛误差分析:为"含误差的系统 PSF"批量生成带随机
波像差(Z5–Z36 条纹 Zernike 项)的样本。仿真常量与论文一致:像面网格间距 0.4 μm、
变换网格 1024×1024、波长 632.8 nm 体系,椭率计算取 PSF 中心 0.5″ 有效半径窗口;
波像差 RMS 水平为可配置参数(默认 1/14 λ)。

## 目录结构

```
DAC-Net/
├── docs/                          # 论文:英文原版 PDF + 中文译文(docx 及预览 pdf)
├── src/
│   ├── matlab/                    # 原始 MATLAB 代码及参考数据
│   │   ├── sim_mainV4_Monte_Carlo.m   # 主仿真程序(生成波像差 → PSF → 椭率)
│   │   ├── zernike_fun.m / RandNum.m / cartesian2polar.m / cac_ellipticity_fun.m
│   │   ├── generate_exitpupil.m       # ExitPupil.mat 的一次性生成脚本
│   │   ├── ExitPupil.mat              # 光瞳数据(仿真输入,仅取尺寸)
│   │   ├── initial.mat / ansmat.mat   # MATLAB 实跑存档(验证用的参考输出)
│   └── python/                    # 全向量化 Python 移植(与 MATLAB 原版逐位一致)
│       ├── elli_error_generate/       # 仿真包(正向仿真 + 论文多目标优化补偿求解器)
│       ├── sim_main_v4.py             # 仿真命令行入口
│       ├── solve_compensation.py      # 补偿求解器命令行入口(NSGA-II,论文 §2.E-4)
│       ├── tests/                     # Octave + Python 等价性验证及求解器性质测试
│       ├── output/ · solver_out/      # 仿真 / 求解产物(生成,已忽略)
│       └── README.md                  # Python 移植版详细文档
└── .venv/                         # Python 环境(已加入 .gitignore)
```

原始 MATLAB 代码头部署名为"徐天晓"(见 `matlab/sim_mainV4_Monte_Carlo.m` 调试日志)。

## 快速开始

```bash
# 安装依赖(仓库根目录,仅需一次)
python3 -m venv .venv
.venv/bin/pip install -r src/python/requirements.txt

# 运行仿真(2 个蒙特卡洛样本,约 3 秒)
.venv/bin/python src/python/sim_main_v4.py --n 2 --seed 42 --out-dir ./output
```

输出 `initial.{npz,mat}`(理论波前/PSF)与 `ansmat.{npz,mat}`(逐样本
`Deta_error`、`PSF`、椭率),`.mat` 为 MATLAB 兼容格式,字段名与原版一致。

Python 版还包含论文核心方法的**补偿求解器**(等效模型 + 虚拟透镜 + NSGA-II
三目标优化)。搜索边界按测量误差先验逐系数缩放(`--bound-k`),合成验证下补偿后
椭率误差均值 ~1e-3(逐样本最优可达 2e-4),满足论文 0.01 指标;`--jobs` 跨样本并行:

```bash
.venv/bin/python src/python/solve_compensation.py \
    demo --samples 4 --pop 180 --gen 120 --seed 1 --jobs 4 --out ./solver_out
```

完整参数、Python API、逐位复现 MATLAB 运行的方法见
[src/python/README.md](src/python/README.md)。

## 正确性验证

Python 移植经三层验证,与 MATLAB 原版达到机器精度一致(PSF 最大绝对差异 ≤ 4.4e-16,
椭率差异 ~1.7e-16,详见 python/README.md):

```bash
octave --no-gui src/python/tests/verify_components.m
.venv/bin/python src/python/tests/compare_components.py
octave --no-gui src/python/tests/verify_python_mat.m
```

按此顺序运行;后两项依赖中间产物(`test_output/` 与 `/tmp` 下的系数文件),
首次运行前请按 python/README.md 中的"--coeffs 逐位复现"一节生成。

## 备注

- `matlab/` 下的 `*.mat` 为参考数据与仿真输入,纳入版本管理;Python 侧生成的
  `output/`、`solver_out/` 及 `python/` 根目录下的输出文件均为可再生产物,已被 `.gitignore` 排除。
- 基金:国家重点研发计划(2021YFC2206303)、国家自然科学基金(12005033)、
  中国科学院青年创新促进会(2023235)。
