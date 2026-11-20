# STATUS — 交接记录 / handoff

**最后更新:** 2026-11-20 13:25
**状态:** 代码完整、测试通过、数据已策管、**骨架划分训练部分完成并已中断**。
测试集至今未被读取。README 尚未撰写（因为还缺完整结果）。

实测数字一律见 [`results/experiment_log.md`](results/experiment_log.md)，
不在本文件重复，以免两处不一致。

---

## 1. 任务定义

对 `admet-triage-dengue` 的重新定义。旧名字暗示了一个站不住的因果链条
（ADMET 模型 → 抗病毒活性）。现在的任务单一、明确、可证伪：

| 项目 | 定义 |
|---|---|
| 预测终点 | **logD<sub>7.4</sub>** — 辛醇/水分配系数（pH 7.4），实验值 |
| 数据 | MoleculeNet *Lipophilicity*（ChEMBL3301363，AstraZeneca 存入），公开 |
| 实际入库 | **4200 化合物**，0 个无法解析，0 个重复结构，logD ∈ [−1.5, 4.5]，sd 1.203 |
| 划分 | Bemis–Murcko 骨架划分 3360 / 420 / 420（确定性，无 RNG）+ 随机划分对照 |
| 调参 | 仅在 train 上做 5 折 CV；模型族选择看 valid；**test 只碰一次** |
| 指标 | RMSE、MAE、R²、Spearman ρ、Kendall τ，全部带 95% bootstrap 区间 |
| 原始数据摘要 | sha256 `aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd`（每次载入都校验） |

**明确不声称的事：** 本项目与登革热、抗病毒活性、CYP3A4 抑制、或任何生物学终点
**没有**建立联系。logD 是理化性质。它与抗病毒活性的关系在本项目中
**未被检验，也未被假设**。

## 2. 单一写入者纪律

本目录由**一个**会话负责写入（`src/logd`）。

`src/admet/`、`docs/`、`results/tables/` 属于另一个并行会话，
**保持原样，未被删除、覆盖或移动**，等待比较与验收后再由用户决定去留。
其 nearest-neighbour 相似性审计的思路已被吸收（见 `src/logd/similarity.py`），
但没有改动它的任何文件。

## 3. 环境：为什么是容器而不是 venv

这台机器的 Windows 11 **Smart App Control 处于 Enforced 状态**
（`VerifiedAndReputablePolicyState = 1`），会拦截 RDKit 的未签名原生 DLL：

```
CodeIntegrity 事件 3033 / 3077:
  RDKitDataStructs-....dll did not meet the Enterprise signing level requirements
  Policy ID {0283ac0f-fff1-49ae-ada1-8a933130cad6}
```

因此 `.venv` 里的 rdkit **无法 import**。
关闭 Smart App Control 是**不可逆**的（重新开启需要重装 Windows），
所以没有去碰这个安全设置，而是改用 Linux 容器运行。副作用是环境固定、更可复现。

- 镜像：`logd-baseline:0.1`（`Dockerfile`，python:3.13-slim + `requirements.txt` 全部钉版本）
- 容器内实测：rdkit 2026.09.2 · sklearn 1.9.1 · numpy 2.5.5 · pandas 3.1.1 · scipy 1.18.1
- GPU：RTX 5060 Laptop 8151 MiB，**本项目未使用**（见第 6 节）
- 货币成本：**0 美元**

## 4. 已完成

```
src/logd/
  config.py       所有影响结果的常数集中在此
  data.py         下载 / 标准化 / 去重 / QC 账本（强制对账）
  splits.py       骨架 + 随机划分，assert_no_leakage（分子级 & 骨架级）
  featurize.py    ECFP4(2048) / RDKit 2D 描述符 / clogP；纯函数
  similarity.py   nearest-neighbour Tanimoto 审计 + 按相似度分箱误差
  models.py       8 个模型，3 个是基线
  metrics.py      5 指标 + bootstrap CI
  train.py        CV 在 train、选择看 valid、模型冻结落盘
  evaluate.py     测试集只读一次，且写审计日志
  provenance.py   环境 / git / GPU / 成本快照
tests/            49 个测试（40 已通过；similarity 的 9 个待随下次回归跑）
Dockerfile, requirements.txt, run.ps1, pytest.ini, LICENSE
```

### 关键设计决定（已写进代码注释，不要无意改掉）

- **不中和电荷。** logD7.4 本身就是 pH 依赖量，重写电离态会悄悄改变标签含义。
  只剥离反离子（取最大片段）。
- **矛盾的重复测量直接丢弃，不平均。** 平均两个互相矛盾的测量会造出一个
  从未被测过的标签。容差 0.5 log 单位。
- **排除 `Ipc` 描述符。** 它是邻接矩阵信息量的指数，中等大小分子就溢出成 +inf。
- **所有 imputer / scaler / VarianceThreshold 都在 Pipeline 内**，CV 只在训练折上拟合。
- **RF `n_jobs=1`**：GridSearchCV 已经在并行，嵌套并发曾压垮容器引擎。
- **`evaluate.py` 没有任何 `fit` 调用**，且必须显式传 `--confirm-final-evaluation`。

## 5. 还没做

1. `hgb_desc`、`hgb_desc_ecfp4` 未训练（Run 1 在此崩溃，Run 2 被主动停止）。
2. 随机划分（random split）整组未训练。
3. `results/validation_*.json` 未生成（train.py 在全部模型跑完后才写）。
4. nearest-neighbour 相似性审计**代码已就绪但未运行**。
5. 测试集评估**未运行**，这是刻意的。
6. `README.md` 未写——**必须等有完整实测结果之后再写**，禁止先写数字再跑。
7. git 已 `init`（main 分支），但**尚未提交**。

恢复步骤与资源门槛见 `results/experiment_log.md` 的"恢复计划"。

## 6. 刻意不做（范围控制）

- **D-MPNN / Chemprop / ChemBERTa 等 GPU 模型。** 4200 个分子、210 维描述符，
  描述符模型在 CPU 上几秒就跑完（`svr_desc` 12 秒）。为了"用上 RTX 5060"
  而引入 PyTorch（Blackwell sm_120 需 cu128+）不划算，且 GPU 正被 Ollama 占用。
  README 里要**诚实写明 GPU 未被使用及原因**，而不是硬凑一个 GPU 模型。
- 任何通用平台 / 框架 / Web 界面。
- 多终点、多任务学习。
- 继续加复杂模型。`svr_desc` 已经够用，优先保住成果。

## 7. 第二步（等 Grok 返回 Alchemistry 资料后）

在拿到官方规则、数据结构和评价指标之前，**不要**改动本项目去迁就比赛。
拿到之后要回答三个事实问题：

1. 比赛的预测终点是什么？结合亲和力 / 活性，还是理化性质？
   如果是前者，本项目的 logD 模型**不能**直接用——终点不同，不是调参问题。
2. 比赛给的训练数据是什么结构？有没有标签？配体是否同系列？
3. Kendall τ 是在多少个化合物上算的？τ = 0.3 在那个 n 下的随机波动范围是多少？
   `metrics.bootstrap_ci` 正好用来回答这个问题。

**一个已经测到的、与此直接相关的事实：** 零参数的 Crippen clogP 计算器
在本数据的验证集上就有 **τ = 0.317**。所以"τ ≥ 0.3"本身不构成成绩，
必须先知道免费基线在**他们的**数据上是多少。

**不得为了达到 τ = 0.3 而伪造、挑选或反复试探盲测结果。**
盲测答案在任何情况下不进入本仓库。与 Grok 的 Alchemistry 模型保持独立，
暂不合并预测。

可以迁移的是**方法论**，不是权重：严格划分、泄漏断言、零参数基线、
bootstrap 区间、测试集访问日志。
