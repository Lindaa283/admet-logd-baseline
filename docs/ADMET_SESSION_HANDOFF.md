# src/admet 会话交接记录 / handoff

**最后更新:** 2026-11-20 13:45
**状态:** 已按用户指令**停止**。不再修改、训练或提交代码。
**写入者归属:** 本会话只写 `src/admet/`、`tests/test_*.py`、`results/tables/`、
`results/figures/`、`docs/`。`src/logd/` 属于另一会话，**未被读取以外的任何方式触碰**。

---

## 1. 已停止的内容（确认）

| 项目 | 状态 |
|---|---|
| 全量 pipeline 运行（`n_seeds=3`, bootstrap 1000） | 启动于 13:40:50，**已 kill**，未完成 |
| 残留 python 进程 (PID 8744, 44900) | 已 kill；当前 `Get-Process python` 为空 |
| 新的训练 / Docker / 占内存进程 | 未启动，也不会启动 |
| git commit | **未执行**。我的改动留在工作区未提交 |

**Docker 注意：** 我在 13:0x 为了排查 RDKit 曾启动 Docker Desktop，随后**主动关闭**
（释放约 9 GB commit）。现在 `Docker Desktop` 与 `com.docker.backend` **又在运行**——
那是 `src/logd` 会话的容器方案所需，**不是我启动的，我也没有再去关它**。
当前内存：FreePhysical 0.4 GB，commit 58.4 / 63.6 GB。内存仍然紧张。

---

## 2. 一个重要的环境事实（两个会话独立复现）

Windows 11 Smart App Control 处于 **Enforced**
（`HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy\VerifiedAndReputablePolicyState = 1`，
`Win32_DeviceGuard.UsermodeCodeIntegrityPolicyEnforcementStatus = 2`），
最初拦截了新装 wheel 的原生 DLL：`rdkit`、`scikit-learn`、甚至 `pandas` 一度无法 import。
**我没有改动这个安全设置**（关闭 SAC 不可逆）。

但与 `src/logd` 的结论不同的是：**等待若干分钟后，SAC 的信誉判定自行放行了**。
`pandas` 先从 BLOCKED 变 OK，随后 `rdkit 2026.09.2` 和 `scikit-learn 1.9.1`
在 `.venv` 里**原生 import 成功**，无需容器。

> 可核实：本仓库 `.venv` 中直接
> `python -c "import rdkit, sklearn; print(rdkit.__version__, sklearn.__version__)"`
> → `2026.09.2 1.9.1`。
> 本会话全部结果都是在这个原生 venv（非容器）里跑出来的。

这对项目的意义：容器**不是必需**的，可作为可复现性加分项而非前置条件。
另：24 逻辑核上 OpenBLAS 会因 per-thread 缓冲在高 commit 下直接报
`Memory allocation still failed after 10 retries`，
因此 `src/admet/__init__.py` 在 import numpy **之前**把线程数钉为 4。

---

## 3. smoke test 实测结果（已完成，exit 0，耗时 739 s）

设置：`n_seeds=1`、`bootstrap_resamples=50`、确定性骨架划分、原生 venv。
**bootstrap 只有 50 次重采样**，所以下面的区间精度不足以发表，但点估计是实测值。

### 数据策管（`results/tables/qc_report.json`）

4200 原始行 → **4102 化合物**；0 个无法解析；0 个缺失；0 个超范围；
logD ∈ [−1.5, 4.5]，mean 2.186，median 2.36，sd 1.203；
单位检查：100% 落在 (−3, 7)，range 6.00 → 判定为 log10 分配系数。
2408 个 Bemis–Murcko 骨架。

### 测试集指标（确定性骨架划分，test n≈820）

| 模型 | test MAE | 95% CI (50×bootstrap) | RMSE | R² | Spearman ρ |
|---|---|---|---|---|---|
| **morgan_desc_hgb** | **0.586** | [0.560, 0.630] | **0.763** | **0.613** | **0.758** |
| descriptor_ridge | 0.725 | [0.694, 0.774] | 0.896 | 0.466 | 0.670 |
| morgan_rf | 0.739 | [0.710, 0.785] | 0.933 | 0.422 | 0.663 |
| morgan_ridge | 0.878 | [0.845, 0.923] | 1.147 | 0.125 | 0.577 |
| clogp_calibrated | 0.891 | [0.852, 0.942] | 1.079 | 0.226 | 0.451 |
| constant（训练均值） | 1.001 | [0.964, 1.050] | 1.227 | −0.000 | n/a |

**怎么读这张表：**
- 最好的模型比**零参数 Crippen clogP 的线性校准版**好 0.305 log 单位 MAE（相对降低 34%），
  两者的 bootstrap 区间不重叠（[0.560, 0.630] vs [0.852, 0.942]）。
- 比常数预测器好 0.415 log 单位。常数预测器 R² ≈ 0 正是预期（定义如此）。
- `morgan_ridge`（指纹 + 线性）几乎打不过 clogP：说明**增益来自非线性 + 描述符**，
  不是来自"用了指纹"。这一行是防止过度解读的关键对照。
- **未声称任何突破。** RMSE 0.763 与公开文献在同一 benchmark 骨架划分上的
  图神经网络结果大致同量级——这是"简单方法已经够用"的证据，不是超越谁的证据。
  要下这个结论必须按同一划分重跑对照，本会话**没有**做。

### 泄漏审计（`results/tables/leakage_audit.json`）

- 骨架划分：train/val 与 test **共享骨架 0 个**；test 到 train 的最近邻
  Tanimoto 中位数 **0.442**。
- 随机划分（对照）：**共享骨架 255 个**，最近邻相似度中位数 **0.677**。
  → 随机划分确实把同系列类似物放在了两侧，这就是不用它当主指标的量化理由。

---

## 4. 与 src/logd 的一处实质分歧：重复化合物 82 vs 0

`src/logd` 的 STATUS.md 记录"**0 个重复结构**"；本会话记录
"**82 个化合物被测过多次**（涉及 180 行）"。两者都对，差别在**判定同一化合物的粒度**：

| 粒度 | 结果 |
|---|---|
| 规范 SMILES 完全相同 | 4200 个唯一 → **0 重复** |
| 完整 InChIKey | 4200 个唯一 → **0 重复** |
| InChIKey **第一段**（骨架连接层，忽略立体与质子化） | **4102 个唯一 → 82 组重复** |

即这 82 组是**立体异构体 / 质子化态变体**，不是完全相同的字符串。

**为什么我认为这件事必须处理（可核实的依据）：**
不含手性信息的 Morgan 指纹与 2D 描述符，对一对对映体是**逐位相同**的。
所以若 R-型在 train、S-型在 test，模型拿到的特征向量完全一致——
这是教科书式的泄漏，而按完整 InChIKey 去重**检测不到**。
且 logD 是非手性环境下的理化量，一对对映体的真实 logD 相同。
实测这 82 组的标签差：mean 0.153、median 0.10、**max 1.55** log 单位
→ 由此估出一个实验噪声下限 ≈ 0.076 MAE（均值差的一半）。

**本会话的处理：**按连接层分组后取 **median**，并把 `label_spread` 留在
`results/tables/clean_dataset.csv` 里备查。

**必须承认的代价（已记入待办，未解决）：**
连接层也会把**顺反异构**和部分**非对映异构**合并，而它们的 logD 可以真的不同。
82/4200 占比很小，但 max spread 1.55 提示其中确实混有真实差异。
`src/logd` 的"矛盾测量直接丢弃、容差 0.5 log"策略在这一点上**可能更稳**。
建议 ChatGPT 审核时把这两种策略并排比较，而不是直接采纳任一方。

---

## 5. 已完成 / 未完成

### 已完成
```
src/admet/
  download.py    公开数据下载 + SHA-256 钉版 + licence/provenance 记录
  chem.py        标准化（取最大片段剥反离子）、InChIKey 身份键、24 个可解释描述符
  scaffold.py    Bemis–Murcko 骨架、骨架划分（确定性 + 带种子）、随机划分对照
  features.py    Morgan count 指纹 (r=2, 2048)、描述符矩阵、最近邻 Tanimoto
  qc.py          缺失/解析失败/单位/范围/重复/重复标签离散度 → QCReport
  leakage.py     划分审计（行重叠、化合物身份、骨架共享、指纹同一性），失败即中止
  models.py      6 级模型阶梯；超参只在 validation 选，test 只评一次
  metrics.py     MAE/RMSE/R²/Spearman/Pearson/bias + percentile bootstrap CI
  plots.py       7 张图
  pipeline.py    端到端编排
tests/test_{chem,scaffold,metrics,features,qc,leakage,models}.py
results/tables/  qc_report, leakage_audit, metrics×3, predictions, clean_dataset
results/figures/ 01–07 png（均来自已完成的 smoke run）
```

### 未完成
1. **全量运行未完成**：`n_seeds=3` + bootstrap 1000 的正式运行被按指令中止。
   现有数字全部来自 `n_seeds=1` / bootstrap 50 的 smoke run。
2. **种子稳健性与随机划分汇总不完整**：`metrics_split_family_summary.csv`
   目前只含 1 个 scaffold 种子和 1 个 random 种子，不足以给出 ±sd。
3. **README.md / RESULTS.md / LIMITATIONS.md / DATA_SOURCES.md 未撰写。**
   （licence 调研已完成，见下节，但未落盘成文档。）
4. **申请用项目简介（RA/硕士）未撰写。**
5. **测试套件未跑完**（见第 6 节）。
6. 未与 `src/logd` 做同划分、同指标的并排对照。

### 已完成的 licence 调研（尚未写进文档，供 ChatGPT 直接用）
- ChEMBL 数据许可：**"Creative Commons Attribution-ShareAlike 3.0 Unported license"**，
  原文见 https://www.ebi.ac.uk/chembl/ 的 "Current Release"。
- EMBL-EBI terms（https://www.ebi.ac.uk/about/terms-of-use，Data §3）原文：
  "EMBL-EBI itself places no additional restrictions on the use or redistribution
  of the data … other than those provided by the original data owners"；
  General §2 要求署名。
- DeepChem 文档 cheatsheet 原文：
  "LIPO | Experimental results of octanol/water distribution coefficient
  (logD at pH 7.4) | Molecules | 4 200"，ref → arXiv:1703.00564。
- DeepChem **代码**为 MIT（LICENSE 文件为 "Copyright 2017 PandeLab" + MIT 正文）。
- **未能核实**：MoleculeNet 官网 datasets 页 404；**数据文件自身的 licence
  （区别于代码）两处都未声明**；ChEMBL deposition 的确切 ID 未核实
  （STATUS.md 写 CHEMBL3301363，网页调研得到的是 DOI 10.6019/chembl3301361，
  两者均未经我独立确认）。我已把 `download.py` 里原先硬写的
  `CHEMBL3301363` 改成不断言具体 ID 的措辞。

---

## 6. 两个被我修改了预期的测试——理由与可核实依据

**前提声明：我没有为了让测试通过而改动任何生产代码，也没有放宽任何阈值。**
两处都是**测试本身写错了**。两处的原始失败输出都在上游会话日志里，关键证据抄录如下。

### 6.1 `test_seeded_splits_are_reproducible_and_seed_dependent`（test_scaffold.py）

原断言：`scaffold_split(LUMPY, seed=3)` 与 `seed=4` 的 test 索引必须不同。

失败输出的关键证据（两个数组逐元素相同）：
```
assert not np.array_equal(
  array([40, 41, ..., 64]),      # seed=3 的 test
  array([40, 41, ..., 64]))      # seed=4 的 test
```

**为什么这是测试错、不是代码错（可自行验算）：**
`LUMPY` = 40×core_a + 25×core_b + 15×core_c + 20 个单例 = 100 行；
fractions (0.7, 0.1, 0.2) → train 目标 70、val 目标 10。
划分规则是整组贪心填充：能进 train 就进 train，否则能进 val 就进 val，否则进 test。
在这个容量下可达到的划分组合**极少**——例如 core_c(15) 永远进不了 val（15 > 10），
core_a(40)+core_b(25)=65 一旦都先到就占满 train。
因此**两个不同种子落到同一划分是算法的正常行为**，不是缺陷：
算法保证的是"骨架不跨划分"与"同种子可复现"，
**从未保证"不同种子必然产生不同划分"**——当分组少且极不均衡时根本不可能保证。

**改法：**拆成两个测试。
(a) 同种子可复现 —— 保留原断言；
(b) 种子敏感性 —— 换成 40 个等大分组（`[f"core_{i//5}" for i in range(200)]`）
这一真正能体现该性质的 fixture，断言 6 个种子至少产生 2 种不同划分。
**骨架不跨划分**这一核心保证仍对 `seed=None,0,1,2` 全部断言，未削弱。

### 6.2 `test_unit_check_accepts_a_log_scale_column`（test_qc.py）

原断言：对 QC fixture，`report.unit_check` 必须以 `"consistent"` 开头。

失败输出的关键证据：
```
'SUSPECT: only 91.7% of values fall in (-3.0, 7.0); the column may not be a log-scale value'
```

**为什么这是测试错（自相矛盾，可直接看 fixture）：**
该 fixture 是我**故意**植入异常值来测越界检查的，其中一行就是
`("CHEMBL_wild", "CCCCCCCCCCCCCCCC", 99.00)`。
12 个有效标签里 1 个是 99.0 → 落在 (−3, 7) 的比例 = 11/12 = **91.7% < 95% 阈值**
→ `unit_check` 返回 SUSPECT。**这正是正确行为。**
而且**同一文件里的另一个测试** `test_out_of_range_labels_are_flagged`
明确断言这个 99.0 必须被标记为越界。
也就是说：我原来的断言与 fixture 的设计意图、以及同文件另一个测试**互相矛盾**。

**改法：**
(a) 对这个植入了坏值的 fixture，断言它**必须**返回 SUSPECT（符合代码应有行为）；
(b) **另建**一个干净 fixture（6 个正常 logD 值，无越界）来验证 "consistent" 分支。
**阈值 95%、范围 (−3, 7) 一个字都没改**，判定逻辑也没改。

### 6.3 这两处修改的验证状态（**重要，尚未完成**）

修完后重跑测试套件的那条命令**被用户中断**，所以**没有一次完整的绿色运行可以引用**。
中断前写出的部分进度（`results/tables/pytest_admet.txt`，84 字节）是：
```
........................................................................ [ 77%]
```
72 个点、**0 个 F**，到 77% 为止。按我给 pytest 的文件顺序
（chem → scaffold → metrics → features → qc → leakage → models），
被修改的两个测试都落在这已完成的 77% 之内，因此它们**确实通过了**；
但 `test_models.py` 与 `test_leakage.py` 的一部分**没跑到**。

**请 ChatGPT 用这一条命令自行核验**（只跑我的测试，不碰 `tests/logd/`，
`-p no:cacheprovider` 避免写缓存）：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_chem.py tests\test_scaffold.py tests\test_metrics.py tests\test_features.py tests\test_qc.py tests\test_leakage.py tests\test_models.py -p no:cacheprovider -q
```

---

## 7. 磁盘上的一处不一致（请勿误读）

被中止的全量运行**已经覆盖了几个文件**，它们现在与其余文件不同步：

| 文件 | 时间 | 来自哪次运行 |
|---|---|---|
| `results/tables/qc_report.json` | 13:33:45 | 被中止的全量运行（内容应与 smoke 相同，数据步是确定性的） |
| `results/tables/clean_dataset.csv` | 13:34:08 | 被中止的全量运行（同上） |
| `results/tables/leakage_audit.json` | 13:34:18 | **被中止的全量运行：含 7 个划分**（比 smoke 的 3 个更全） |
| `results/tables/metrics_*.csv`、`predictions_*.csv` | 13:32:27–28 | **已完成的 smoke run：只含 3 个划分** |
| `results/figures/01–07*.png` | 13:32:29–30 | **已完成的 smoke run** |
| `results/run_metadata.json` | 13:32:31 | **已完成的 smoke run**（记录 `n_seeds: 1`） |
| `results/tables/pytest_admet.txt` | 13:35:56 | 被中断的测试运行，**不完整** |

即：`leakage_audit.json` 描述 7 个划分，而 metrics / figures 只覆盖 3 个。
**第 3 节表格里的数字全部取自自洽的 smoke run**（13:32:27–31 那批）。
要得到完全自洽的一套产物，需要重跑一次完整 pipeline——**我没有跑，按指令停止**。

---

## 8. 我对共享文件做过的事（已被对方修复，如实记录）

我写 `.gitignore` 时**覆盖**了一个已存在的 `.gitignore`（当时未察觉它已存在），
丢掉了 `data/processed/` 与 `artifacts/` 两条规则。
`src/logd` 会话随后发现并补回，并在注释里写明原因
（否则 `git add -A` 会提交一个 81 MB 的随机森林），对应提交
`346f4d9 fix: re-ignore artifacts/ and data/processed/ after .gitignore rewrite`。
**这是我的失误**，起因是在同一目录里有两个写入者而我开工时不知情。
此后我只写自己归属的路径，未再触碰共享根文件。

当前工作区未提交的改动中属于我的是：
`src/admet/download.py`（licence 措辞修正）、`tests/test_qc.py`、`tests/test_scaffold.py`
（第 6 节的两处测试修正）、以及 `results/` 下的新产物。
`src/logd/train.py` 的改动不是我做的。**我没有执行任何 commit。**

---

## 9. 如果保留 src/admet，建议下一步

1. 重跑一次完整 pipeline 得到自洽产物与可发表精度的区间：
   `python -c "from admet import pipeline; pipeline.run(n_seeds=3, bootstrap_resamples=1000)"`
   （约 30 min，4 线程，纯 CPU，$0；注意届时内存状况）
2. 把第 4 节的重复化合物分歧与 `src/logd` 的策略并排对照后再定一种。
3. 用同一划分、同一指标对照 `src/logd` 的 8 个模型，再决定保留哪一套。
4. 文档（README / RESULTS / LIMITATIONS / DATA_SOURCES）与申请用简介仍需撰写；
   第 5 节末的 licence 原文可直接引用。
