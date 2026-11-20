# STATUS — 交接记录 / handoff

**最后更新:** 2026-11-20 19:0x（测试集评估完成后补记）
**状态:** 代码完整、49 项测试通过、数据已策管、**骨架与随机两套划分的 8 个模型
全部训练完成**、泄漏审计完成、模型选择已冻结（`results/freeze_manifest.json`）、
`README.md` 已撰写、**最终测试集评估已完成**、申请用简介已撰写
（`docs/PROJECT_BRIEF.md`）。**项目已可交付。**

**`src/logd` 的测试集已评分，恰好一次**（2026-11-20 18:56 UTC，
`results/test_access_log.jsonl` 的 `run_index: 1`，绑定 commit `d97df34`）。
此前拦着没跑的 Smart App Control 问题，是通过先做 RDKit 预检、确认可特征化后
再运行来规避的——这样中途被拦不会留下"有日志条目但没有结果"的记录。
结果见 README §7 与 `results/test_scaffold.json`：
预先登记的 `svr_desc` 测试 RMSE **0.809 [0.738, 0.881]**，
最强基线 `clogp_linear` 为 1.099 [1.030, 1.167]，差 0.290 log 单位且区间不重叠。

⚠ 本仓库**整体**仍然不是"测试集只被访问过一次"：并行的 `src/admet` 有它自己的
测试集结果（`results/tables/`、`docs/ADMET_SESSION_HANDOFF.md`）。两套实现的
策管与划分不同，是两批不同的化合物，**不可混为一谈，也不可把分数并列比较**。

实测数字一律见 [`results/experiment_log.md`](results/experiment_log.md)，
不在本文件重复，以免两处不一致。

---

## 1. 任务定义

对 `admet-triage-dengue` 的重新定义。旧名字暗示了一个站不住的因果链条
（ADMET 模型 → 抗病毒活性）。现在的任务单一、明确、可证伪：

| 项目 | 定义 |
|---|---|
| 预测终点 | **logD<sub>7.4</sub>** — 辛醇/水分配系数（pH 7.4），实验值 |
| 数据 | MoleculeNet *Lipophilicity*（AstraZeneca logD7.4，经 ChEMBL 分发），公开。**存放编号刻意不断言**，见 `LICENSE` 与比较报告 L4 |
| 实际入库 | **4200 化合物**，0 个无法解析，0 个 canonical SMILES 重复，logD ∈ [−1.5, 4.5]，sd 1.203 |
| ⚠ 重复的真实情况 | "0 个重复结构"**只在 SMILES 粒度成立**。按 InChI 连接层（忽略立体与质子化）则有 **82 组 / 180 行**重复。由于 Morgan 指纹未启用手性、描述符全为 2D，这 82 组在**特征空间里逐位相同**，有效唯一特征向量 = 4102 (4200 − 180 + 82)。骨架划分下无害（对映体共享 Murcko 骨架，必然同分区），**随机划分下是真实泄漏**。见比较报告 L5 |
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

## 3. 环境：Smart App Control 与容器的真实关系

这台机器的 Windows 11 **Smart App Control 处于 Enforced 状态**
（`VerifiedAndReputablePolicyState = 1`），会拦截 RDKit 的未签名原生 DLL：

```
CodeIntegrity 事件 3033 / 3077:
  RDKitDataStructs-....dll did not meet the Enterprise signing level requirements
  Policy ID {0283ac0f-fff1-49ae-ada1-8a933130cad6}
```

**这个拦截是间歇性的，不是恒定的。** 实测时间线（同一个 `.pyd` 文件）：

1. 最初 `.venv` 里的 rdkit 无法 import → 当时改用容器。
2. 数分钟后 SAC 的信誉查询自行放行 → 原生 `.venv` 可用，
   Run 3 / Run 4 两次完整训练 + 49 项测试（1.49s 全绿）都在原生环境跑完。
3. 之后几分钟内**又被拦下**（`cDataStructs.pyd`，事件 3033/3077/3118）。
   等待并重试 15 次 / 每次 60 秒，未放行。

所以结论不是"容器是必需的"，也不是"原生可用"，而是：
**这台机器上的原生环境不可靠但可用，容器才是可复现的那条路。**
关闭 Smart App Control 是**不可逆**的（重新开启需要重装 Windows），
因此没有去碰这个安全设置。

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
  audit.py        只读结构的泄漏审计（不读标签、不加载模型、不消耗测试集额度）
tests/logd_suite/ **49 个测试全部通过**（1.49s，原生 venv，未用 Docker）
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

1. ~~`hgb_desc`、`hgb_desc_ecfp4` 未训练~~ **已完成**（Run 3/4，原生，未用 Docker，
   无资源异常，故未触发"放弃 boosting 退回 SVR"的预案）。
2. ~~随机划分整组未训练~~ **已完成**。
3. ~~`results/validation_*.json` 未生成~~ **已生成**（两套划分各一份）。
4. ~~nearest-neighbour 相似性审计未运行~~ **已完成**，见 `results/leakage_audit_logd.json`
   与比较报告第 9 节。结论：骨架划分 test 最近邻中位数 0.457、ECFP4 相同孪生 **0 例**；
   随机划分 0.694、**14 例**。主划分干净，随机划分确有泄漏（故仅作对照）。
5. ~~测试集评估**未运行**~~ **已运行，恰好一次**（见本文件开头与 README §7）。
   随机划分按设计**不做**测试集评估，仍然只有验证集数字。
6. ~~`README.md` 未写~~ **已写**，并在评估后更新：§7 从"尚未评分"改写为实际结果，
   §1 状态表、§8 的局限（单一划分只给一个数字）与 §10 的 Kendall τ 说明一并更正。
7. ~~git 尚未提交~~ 已有多次提交，main 分支。
8. **唯一剩下的一步：** `python -m logd.evaluate --split scaffold
   --confirm-final-evaluation`。等 SAC 放行 RDKit 即可执行。
   只评估骨架划分；随机划分只作验证集对照，不花掉第二次测试额度。

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
