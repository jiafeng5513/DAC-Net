# Elli_error_generate — Python 移植版

`Elli_error_generate` MATLAB 仿真程序的 Python 移植:为 CSST(巡天空间望远镜)生成平行光管 + 干涉仪(GXSS)的 Zernike 波像差,经 FFT 传播得到 PSF 并计算椭率,支持蒙特卡洛批量样本生成。

与原版逐像素双重循环不同,本移植**全向量化**:2 样本完整仿真约 2.5–3 秒(MATLAB 原版为小时级)。数值结果与原版 MATLAB 运行**逐位一致**(PSF 最大绝对差异 ≤ 4.4e-16,见[验证](#验证))。

## 目录结构

```
python/
├── elli_error_generate/        # 仿真包
│   ├── zernike.py              # zernike_fun.m 的向量化移植(Z5..Z36,标准/条纹两种排序)
│   ├── rand_num.py             # RandNum.m:固定 RMS 的随机 Zernike 系数
│   ├── ellipticity.py          # cac_ellipticity_fun.m:PSF 四极矩椭率(含 e1/e2 分量)
│   ├── simulation.py           # 主流程:光瞳网格、波像差、PSF、蒙特卡洛循环、结果保存
│   ├── merit.py                # 求解器观测量:EE80 半径、PSF 形态差
│   ├── solver.py               # 补偿求解器:批量正向模型 + NSGA-II(论文式 8-10 评价函数)
│   └── experiment.py           # 合成验证协议(demo/archived 两种数据来源 + Pareto 统计)
├── sim_main_v4.py              # 仿真命令行入口(sim_mainV4_Monte_Carlo.m 的移植)
├── solve_compensation.py       # 求解器命令行入口(论文多目标优化补偿实验)
├── tests/
│   ├── verify_components.m     # Octave 组件级验证(跑原始 MATLAB 函数)
│   ├── verify_python_mat.m     # Octave 回环验证(Python 写的 .mat 是否为合法 struct)
│   ├── compare_components.py   # Python 侧等价性对比(26 项)
│   └── test_solver.py          # 求解器性质测试(e1/e2 解析、EE80 暴力交叉验证、批量正向一致性)
├── output/                     # 仿真产物 initial/ansmat.{npz,mat}(生成,已忽略)
├── solver_out/                 # 求解产物 compensation_*.{json,npz}(生成,已忽略)
└── requirements.txt
```

## 安装

```bash
python3 -m venv .venv
.venv/bin/pip install -r python/requirements.txt
```

依赖:numpy、scipy(mat 文件读写)、命令行绘图可选 matplotlib。

## 命令行用法

```bash
.venv/bin/python python/sim_main_v4.py --n 2 --seed 42 --out-dir ./output
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--n` | 2 | 蒙特卡洛样本数(原 MATLAB 的 `sim_total_num`) |
| `--nflag` | 1 | 0 = 标准 Zernike,1 = 条纹(fringe)Zernike |
| `--rms-parallel` | 1/14 | 平行光管波像差 RMS(单位:波长) |
| `--rms-gxss` | 1/14 | 干涉仪波像差 RMS |
| `--radius-ex` / `--radius-in` | 231 / 0 | 光瞳外/内半径(像素) |
| `--seed` | 无 | 随机种子(Python 侧可复现) |
| `--data` | `../matlab/ExitPupil.mat` | 光瞳数据文件(仅用于确定网格尺寸) |
| `--out-dir` | `.` | 输出目录 |
| `--no-mat` | 关 | 只写 `.npz`,跳过 MATLAB 兼容 `.mat` |
| `--coeffs` | 无 | 注入固定系数 `.npz`,逐位复现某次运行(见下) |

### 逐位复现一次 MATLAB 运行

从 MATLAB 运行存档的 `initial.mat` / `ansmat.mat` 中提取系数,注入 Python 端口即可得到与 MATLAB **逐位相同**的 PSF 和椭率:

```python
import scipy.io as sio, numpy as np
init = sio.loadmat('../matlab/initial.mat')['initial']
ans  = sio.loadmat('../matlab/ansmat.mat')['ansmat']
np.savez('coeffs.npz',
         Anm_parallel=np.asarray(init['parallel_zernike'][0,0]).ravel(),
         Anm_GXSS=np.asarray(init['GXSS_zernike'][0,0]).ravel(),
         Deta_error=np.stack([np.asarray(ans['Deta_error'][0,i]).ravel()
                              for i in range(ans.shape[1])]))
```

```bash
.venv/bin/python python/sim_main_v4.py --n 2 --coeffs coeffs.npz --out-dir ./output
```

## Python API

```python
from elli_error_generate import SimConfig, run_simulation, save_results

config = SimConfig(sim_total_num=100, seed=42)
initial, ansmat = run_simulation(config, '../matlab/ExitPupil.mat')
save_results(initial, ansmat, './output')   # 写出 npz + mat
# ansmat[i]['PSF'], ansmat[i]['ellipticity_parallel_GXSS'], ansmat[i]['Deta_error']
```

## 求解器:多目标优化补偿(论文方法)

在已验证的正向核之上实现了论文 §2.E–§4 的补偿求解:以实测 PSF(PSF-M)为目标,
优化"虚拟透镜"补偿面的 Zernike 系数,使仿真 PSF(PSF-S)逼近 PSF-M,评价函数为
论文式 (8)–(10):椭率差+长轴差、EE80 半径差、PSF 形态差(权重 w₁–w₄ 默认 0.5,
对应论文 §4.B),NSGA-II 参数默认种群 180、120 代、SBX 交叉概率 1、多项式变异 0.1
(论文 §4.A)。补偿后系统(被测系统等效透镜+虚拟透镜,去掉准直器)的椭率即测量结果。

```bash
# 合成实验:按论文协议生成 PSF-M 与带误差模型,验证补偿恢复精度(4 样本并行)
.venv/bin/python src/python/solve_compensation.py \
    demo --samples 4 --pop 180 --gen 120 --seed 1 --out ./solver_out --jobs 4

# 用 MATLAB 存档的 ansmat 样本做 PSF-M(补偿需吸收未知的 Deta_error)
.venv/bin/python src/python/solve_compensation.py \
    archived --sample 1 --pop 120 --gen 60 --seed 4 --out ./solver_out
```

`demo` 模式(论文协议):真值波前生成干净台架 PSF-M;模型用"测得的"系数(真值 +
高斯测量误差,σ 按论文表 1 的 Z5–Z9 折算);真值椭率 = 被测系统单独光瞳的椭率。
`archived` 模式:PSF-M = `matlab/ansmat.mat` 样本(含未知扰动 `Deta_error`),
同时报告系数恢复误差 `‖x − Deta_error‖`。

### 命令行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--samples`(demo)/ `--sample`(archived) | 4 / 1 | 样本数 / ansmat 样本序号(1 起) |
| `--pop` / `--gen` | 180 / 120 | NSGA-II 种群 / 迭代代数(论文 §4.A) |
| `--cx-prob` / `--mut-prob` | 1.0 / 0.1 | SBX 交叉 / 多项式变异概率 |
| `--bound-k` | 6 | **demo 逐系数搜索边界 = bound_k × 测量误差 σ**(见下) |
| `--bound` | 0.2 | 标量系数边界(archived 模式 / 回退) |
| `--jobs` | 1 | demo 跨样本并行进程数(样本相互独立,结果与串行一致) |
| `--fast` | 关 | complex64/float32 FFT 路径(略快、省内存;会改变结果) |
| `--chunk` | 16 | 每批 FFT 的样本数(调并行度 / 内存) |
| `--progress` / `--no-progress` | 开 | 每代进度行(`--jobs>1` 时自动关闭,改为逐样本完成提示) |
| `--seed` | 无 | 随机种子 |
| `--out` | 无 | 结果保存目录(`compensation_summary.json` + `compensation_results.npz`) |

### 搜索边界:按测量误差先验缩放(收敛关键)

补偿量的真解是 `x = −ε`(ε 为测量误差),其量级仅 ~0.01–0.04;若在过宽的 `±0.2`
盒子里搜索,32 维下 NSGA-II 无法收敛到这一极小区域。因此 demo 默认按**逐系数**
`±bound_k·σ`(σ 为 `meas_sigma`,`bound_k=6` ≈ 6σ,安全覆盖)设定边界,把搜索空间
收缩到误差实际所在的尺度。实测该项把补偿后椭率误差从 ~0.02 降到 ~0.002(11×),
Pareto 区间包含真值比例从 0.5 升到 0.75。

**验证结果**(本仓库实测,4 样本 seed=1,pop 120 × gen 100,`--jobs 4` 约 6.5 分钟):
- 性质测试 13/13 通过:e1/e2 与解析值、旋转不变性、EE80 与暴力环积分交叉验证、
  批量正向模型与已验证单样本正向逐位一致(≤4.4e-16)。
- 补偿后椭率误差均值 **1.9e-3**(逐样本最优:样本 2/3/4 达 2e-4~3e-4;样本 1
  真值最小、最难),Pareto 区间含真值 3/4——满足论文 0.01 指标。
- archived 实验:补偿后最优误差 ~5e-5,系数恢复 ‖x−Deta‖ ≈ 0.41(PSF 域匹配
  不保证系数域唯一,论文亦指出此多对一性)。

### 性能

正向模型批量化 + `scipy` 多线程 FFT(`workers`);EE80 半径只对内层像素排序
(Markov 不等式保证 `(2.5·RMS)²` 内含 ≥84% 能量,与全排序逐位一致);PSF 与评价
函数分块流式计算,避免驻留完整 `(pop,1023,1023)` 数组。`--jobs N` 跨样本并行
(样本独立,结果不变),每进程 FFT 线程自动设为 `cpu/jobs` 以避免线程超订。

## 输出格式

每次运行写出 4 个文件:

- `initial.npz` / `initial.mat` — 理论(无测量误差)波前:三套 Zernike 系数、相位屏、广义光瞳函数、理论 PSF、`ellipticity_GXSS`。`.mat` 版为 MATLAB struct,字段名与原版一致。
- `ansmat.npz` / `ansmat.mat` — 每个蒙特卡洛样本的 `Deta_error`(32 维)、`PSF`(1023×1023)、`ellipticity_parallel_GXSS`(标量)。`.mat` 版为 1×n struct array,与原版相同(`ansmat(i).PSF` 可直接索引)。

## 验证

三层验证(见 `tests/`,需 Octave,仅验证用途):

1. **组件级**:Octave 运行原始 `zernike_fun.m` / `cac_ellipticity_fun.m` / `RandNum.m`(固定输入),与 Python 对比——zernike 标量/网格、两种排序,椭率(轴对齐 + 旋转 45° 高斯),RandNum 归一化公式(含 fringe 系数 `Coff_ZFC_Z5_Z36`),全部通过。
2. **全链路**:把 MATLAB 实跑存档中的 `Anm_parallel` / `Anm_GXSS` / `Deta_error` 注入 Python,与 `../matlab/` 下原版 `initial.mat` / `ansmat.mat` 对比——26/26 项通过,PSF 最大绝对差异 3.3e-16,椭率差异 ~1.7e-16,`Deta_error` 逐位相同。
3. **回环**:Python 写出的 `.mat` 在 Octave 中按合法 struct / struct array 加载,用原始 MATLAB 椭率函数从存储 PSF 重算,与存储值一致(9e-15)。

复现验证:

```bash
octave --no-gui tests/verify_components.m
.venv/bin/python tests/compare_components.py
octave --no-gui tests/verify_python_mat.m   # 需先按上文 --coeffs 生成 test_output
```

## 与 MATLAB 原版的差异说明

- **随机数流**:Python 的 RNG 与 MATLAB `rand` 在分布上等价,但流不可对齐——这是唯一刻意的行为差异。用 `--seed` 保证 Python 侧可复现,用 `--coeffs` 逐位复现 MATLAB 运行。
- **`cartesian2polar.m`**:未单独移植。其转置约定(原版 `r=r'`、`t` 的分象限 `asin`/`acos` 计算)已证明等价于 `t = arctan2(y, x)`,并在 `simulation.build_pupil_grid` 中以向量化形式精确复现(`r[i,j]=√(x[j]²+y[i]²)/R`、`t[i,j]=atan2(y[i],x[j])`),这是结果逐位一致的关键。
- **`generate_exitpupil.m`**:未移植。这是一次性数据生成脚本,依赖未随包的 `zernike_1_14_ss.mat`;其产物 `ExitPupil.mat` 被直接读取(主程序仅使用其尺寸推导网格大小 `samplingpoint_N = 1023`)。
- **性能**:原版对 1023×1023 网格逐像素调用 `zernike_fun`(每次 32 项),Python 版一次性向量化求值。
- 原版中 `ExitPupil_parallel` 与 `ExitPupil_GXSS` 在 `radius_in=0` 时完全相同(均为全圆),本移植保留两个变量以保持结构对应。

## 原始 MATLAB 代码

见 `../matlab/`(`sim_mainV4_Monte_Carlo.m` 及支撑函数,作者:徐天晓,2026.9.23 调试日志)。
