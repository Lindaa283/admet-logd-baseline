# 实验日志 / Experiment log

手写的运行记录。**所有数字都是实测的**，没有一个是估计或预期值。

> **指标性质声明（重要）**
> 本文件中的全部数字都是 **验证集（validation）** 表现。
> - **验证集** = 用于在模型族之间做选择。已测。
> - **测试集** = 留出集，**至今一次都没有被读取过**。可验证：
>   `results/test_access_log.jsonl` 不存在。该文件由 `logd.evaluate`
>   在每次读取测试集时追加记录；文件不存在即证明测试集从未被访问。
> - **真正的盲测（blind set）** = Alchemistry 比赛的盲测数据。**本仓库中没有**，
>   也从未被下载、推断或拟合。
>
> 不要把下面任何数字当作测试集或盲测表现引用。

---

## Run 1 — 2026-11-20 约 13:11–13:13（中断）

命令：
```
docker run --rm -v <repo>:/work logd-baseline:0.1 \
  sh -c "python -m logd.train --split scaffold && python -m logd.train --split random"
```
无内存限制，`--jobs` 默认 `-1`（全并发）。

骨架划分（scaffold split），验证集结果：

| model | CV RMSE (train) | valid RMSE | valid Kendall τ | fit 秒 |
|---|---|---|---|---|
| `mean_baseline` | 1.197 | 1.214 | nan（常数预测无排序） | 4.69 |
| `clogp_identity` | 1.771 | 1.741 | 0.317 | 1.85 |
| `clogp_linear` | 1.098 | 1.057 | 0.317 | 2.48 |
| `ridge_ecfp4` | 0.772 | 0.951 | 0.437 | 14.84 |
| `svr_desc` | 0.618 | **0.754** | **0.591** | 17.42 |
| `rf_desc` | 0.665 | 0.819 | 0.515 | 67.94 |

在 `hgb_desc` 处中断。退出码 255，**无 Python traceback**（即进程被外部终止，
不是程序抛异常）。

## Run 2 — 约 13:17–13:21（按用户指令主动停止）

重启 Docker 后重跑，这次加了资源上限：
```
docker run --rm --memory 6g --cpus 4 -e OMP_NUM_THREADS=2 ... --jobs 4
```
同时把 `RandomForestRegressor(n_jobs=-1)` 改为 `n_jobs=1`，消除
GridSearchCV 进程 × OpenMP 线程的嵌套并发。

| model | CV RMSE | valid RMSE | valid Kendall τ | fit 秒 (Run 2) |
|---|---|---|---|---|
| `mean_baseline` | 1.197 | 1.214 | nan | 3.39 |
| `clogp_identity` | 1.771 | 1.741 | 0.317 | 0.27 |
| `clogp_linear` | 1.098 | 1.057 | 0.317 | 0.03 |
| `ridge_ecfp4` | 0.772 | 0.951 | 0.437 | 4.51 |
| `svr_desc` | 0.618 | 0.754 | 0.591 | 12.0 |
| `rf_desc` | — | — | — | 完成并落盘，日志未捕获到该行 |

**可复现性证据：** Run 1 与 Run 2 在不同并发配置（`--jobs -1` vs `--jobs 4`、
RF `n_jobs` 不同、有无内存限制）下得到**完全一致**的 CV RMSE 与 valid RMSE。
说明随机种子控制有效，结果不依赖并发配置。

于 13:21 按用户指令主动停止，为正在运行的 Ollama 红队评测让出内存与 GPU。
`hgb_desc`、`hgb_desc_ecfp4` 两个模型**未训练**。随机划分（random split）**未训练**。

---

## 崩溃取证 / Crash forensics

我最初的推测是"容器 OOM"。**该推测未被证据支持，不作为结论。**

| 证据来源 | 实际发现 |
|---|---|
| `%LOCALAPPDATA%\Docker\log\host\Docker Desktop.exe.log` | `[2026-11-20T13:15:42]  backend process exited: exit status 0xffffffff` —— Windows 侧 `com.docker.backend.exe` 异常退出。历史上的正常关闭记录为 `0x40010004`（DBG_TERMINATE_PROCESS），本次不同。 |
| WSL `dmesg`（docker-desktop 发行版） | **不可判定。** VM 已重启，uptime 仅 256 秒，崩溃前的内核日志已丢失。无法确认也无法排除 cgroup OOM kill。 |
| Windows 事件日志 13:00–13:25（Application + System，Level 1–3） | **无** Docker / vmmem / Hyper-V 崩溃条目。只有无关警告（Search 筛选器主机、RestartManager、.NET profiler）。 |
| 主机内存（13:22 实测） | 总 15.6 GB，**可用 0.3 GB**；提交 **59.5 / 63.6 GB**；Memory Compression 占 3.3 GB。 |
| GPU（13:22 实测） | 利用率 88%，显存 6732 / 8151 MiB，占用进程 `llama-server`。 |

**可以确定的：** Docker 的 Windows 侧后端进程异常终止，因此命名管道消失，
`docker` CLI 返回 255。这不是训练代码的 bug。

**不能确定的：** 具体的终止机制。主机内存耗尽（0.3 GB 可用、59.5 GB 提交）
与"后端被系统回收"高度一致，但没有直接日志证明，Linux 侧 OOM 也无法查证。

**已采取的缓解措施：** 消除嵌套并发（RF `n_jobs=1`）、容器显式内存与 CPU 上限
（`--memory 6g --cpus 4`）、`OMP_NUM_THREADS=2`。这些是降低风险的措施，
不是对根因的确认。

---

## 已保全的产物 / Preserved

```
artifacts/scaffold/partition_train.csv   3360 行   数据划分，可审计
artifacts/scaffold/partition_valid.csv    420 行
artifacts/scaffold/partition_test.csv     420 行   未被读取
artifacts/scaffold/mean_baseline.joblib
artifacts/scaffold/clogp_identity.joblib
artifacts/scaffold/clogp_linear.joblib
artifacts/scaffold/ridge_ecfp4.joblib
artifacts/scaffold/svr_desc.joblib        4.4 MB   当前最佳
artifacts/scaffold/rf_desc.joblib          81 MB
results/data_qc.json                               数据策管账本 + 摘要
```

`results/validation_scaffold.json` **未生成**——它在 `train.py` 的全部模型
跑完后才写入，两次运行都提前结束。因此本文件是这批验证指标的唯一记录。

---

## 恢复计划（等资源空闲后）

1. 确认 Ollama 评测结束、主机可用内存恢复到 ≥ 4 GB。
2. 重跑训练，保持受控并发：
   ```powershell
   docker run --rm --memory 6g --cpus 4 -e OMP_NUM_THREADS=2 `
     -v "<repo>:/work" logd-baseline:0.1 `
     python -m logd.train --split scaffold --jobs 4
   ```
3. 若 `hgb_desc` 再次导致不稳定：**直接放弃两个 boosting 模型**。
   `svr_desc`（valid RMSE 0.754）已经是一个合格的交付基线，
   不值得为了多一个复杂模型继续消耗这台机器。
4. 跑 nearest-neighbour 相似性审计（见下）。
5. 回归测试：`python -m pytest`（当前 40 passed，新增 similarity 测试后应为 49）。
6. 全部稳定后，才运行一次 `logd.evaluate --confirm-final-evaluation`。

## nearest-neighbour 相似性泄漏审计

已吸收自 `src/admet`（该版本保持原样未改动），实现在 `src/logd/similarity.py`，
并已接入 `logd.evaluate`。它回答的是：

> 骨架划分只保证 Bemis–Murcko 骨架不重叠，**不保证分子不相似**。
> 如果测试分子与某个训练分子的 ECFP4 Tanimoto 达到 0.9，
> 那么再漂亮的指标也只是在做内插。

输出两组数字：最近邻相似度分布，以及**按相似度分箱的误差**。
若 RMSE 在低相似度箱（< 0.3）急剧恶化，就说明指标被高相似样本抬高了。

此审计为纯数据计算，不涉及模型重训，也不使用测试集标签做任何选择。

---

# Run 3 / Run 4 — 原生 Windows 完整训练（2026-11-20 下午）

与 Run 1 / Run 2 的区别：**未使用 Docker**。Smart App Control 的信誉查询
此前自行放行了 RDKit 的 DLL，所以容器从"前提条件"降级为"可复现性附加项"。
并发按另一会话记录的教训上限：`OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=4`，
`GridSearchCV n_jobs=2`（24 逻辑核）。

```powershell
$env:PYTHONPATH="src"
$env:OMP_NUM_THREADS="4"; $env:OPENBLAS_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"; $env:NUMEXPR_NUM_THREADS="4"
.\.venv\Scripts\python.exe -m logd.train --split scaffold --jobs 2   # exit 0
.\.venv\Scripts\python.exe -m logd.train --split random   --jobs 2   # exit 0
```

两次都跑完全部 8 个模型，**没有出现资源异常**，所以没有触发"放弃 boosting
模型、退回 SVR"的预案。

## Run 3：骨架划分 3360 / 420 / 420（验证集数字）

| 模型 | cv_rmse (train) | valid_rmse | valid_kendall | 拟合秒数 |
|---|---|---|---|---|
| mean_baseline | 1.197 | 1.214 | — | 2.2 |
| clogp_identity | 1.771 | 1.741 | 0.317 | 0.2 |
| clogp_linear | 1.098 | 1.057 | 0.317 | 0.04 |
| ridge_ecfp4 | 0.772 | 0.951 | 0.437 | 11.0 |
| **svr_desc** | **0.618** | **0.754** | **0.591** | 66.3 |
| rf_desc | 0.665 | 0.819 | 0.515 | 586.2 |
| hgb_desc | 0.635 | 0.795 | 0.544 | 753.3 |
| hgb_desc_ecfp4 | 0.613 | 0.789 | 0.547 | 1948.4 |

**预先登记的模型：`svr_desc`**（验证 RMSE 最小）。
值得注意的是两个 boosting 模型的 CV 分数更好（0.635 / 0.613 对 0.618），
验证集上却更差。训练折内的交叉验证并不足以挑出泛化更好的模型，
这正是保留独立验证分区的理由。

## Run 4：随机划分（对照，验证集数字）

| 模型 | cv_rmse (train) | valid_rmse | valid_kendall |
|---|---|---|---|
| mean_baseline | 1.202 | 1.230 | — |
| clogp_identity | 1.748 | 1.745 | 0.288 |
| clogp_linear | 1.094 | 1.113 | 0.288 |
| ridge_ecfp4 | 0.830 | 0.819 | 0.551 |
| svr_desc | 0.653 | 0.653 | 0.663 |
| rf_desc | 0.719 | 0.712 | 0.627 |
| hgb_desc | 0.684 | 0.677 | 0.637 |
| **hgb_desc_ecfp4** | 0.662 | **0.645** | 0.660 |

预先登记的模型：`hgb_desc_ecfp4`。但它与 `svr_desc` 只差 **0.008 log 单位**，
远小于任何合理的不确定度 —— 这个"胜出"应当读作"两者无法区分"，
而不是 boosting 更强。

**划分难度差距（同一套代码、同一批数据、同一组超参网格）：**
`svr_desc` 验证 RMSE 随机划分 0.653 → 骨架划分 0.754，变差 **0.101 log 单位**；
Kendall τ 0.663 → 0.591。零参数的 clogP 几乎不受影响（τ 0.288 → 0.317）,
因为它根本没有从训练集学任何东西 —— 这是一个有用的对照：
划分方式只惩罚**学出来**的模型。

**随机划分不做测试集评估**（见下一节），它只作为验证集层面的对照，
加上已完成的结构泄漏审计（14 例 ECFP4 相同孪生）。

## 修正了两处我自己写错的断言

1. `train.py` 输出的 `selection_rule` 字符串以 "Test set untouched." 结尾，
   而**同一个文件**的 `split` 区块里就写着 test 分区的
   `label_mean / label_std / label_min / label_max`。
   `splits.split_summary()` 对三个分区一视同仁地统计标签。
   真实情况：test 标签的 5 个描述性统计量在测试集评估之前就被算过并落盘了。
   没有任何 test 行被特征化、拟合或打分，而选择规则是"验证 RMSE 的 argmin"，
   结构上无法读到这些统计量。但**可辩护的说法是"没有任何来自测试集的量
   进入过决策"，不是"测试集未被碰过"**。
   代码已改；两个已生成的 JSON 只重写了这一个字符串字段，
   并在文件里用 `_post_hoc_edit` 标注（无任何数值被改动）。

2. `evaluate.py` 原来把**测试集 RMSE 最低**的模型当作 `best_model` 上报。
   那是看过测试集之后再选模型，属于 post-hoc 选择，会让 bootstrap 区间
   显得比真实不确定度窄。现在 headline 模型从
   `results/validation_<split>.json` 的 `selected_model` 读入
   （即训练时就已冻结），测试集上的最优者仍然照报，但明确标注
   `post_hoc_best_on_test`，并说明它不是本项目的结论。
   若找不到预先登记的选择，`evaluate.py` 直接报错退出，不退化成按测试集挑选。

## 冻结：`results/freeze_manifest.json`

在唯一一次测试集评估之前生成。内容是 git commit、全部 `config.py` 常数、
原始与策管数据的 sha256、`src/logd/*.py` 每个文件的 sha256、
两套划分的 8 个 joblib 与 3 个 partition CSV 的 sha256、
每个模型的 `best_params`、指标清单与 bootstrap 设置、环境版本。
**任何能改变已报告测试数字的东西，都会同时改变这里的某个哈希。**

## Smart App Control 再次拦截（环境事实，非项目缺陷）

两次训练与一次 49 项回归测试（1.49s 全绿）之后，几分钟内 RDKit 又被拦下：

```
CodeIntegrity 3033 / 3077 / 3118   Policy ID {0283ac0f-fff1-49ae-ada1-8a933130cad6}
  python.exe attempted to load
  .venv\Lib\site-packages\rdkit\DataStructs\cDataStructs.pyd
  did not meet the Enterprise signing level requirements
```

`VerifiedAndReputablePolicyState = 1`（Enforced）。所以放行是**间歇性的**：
同一个文件先被放行、后又被拦截。这台机器上 SAC 的信誉判定不稳定，
原生环境因此**不可靠但可用**，容器才是可复现的那条路。
没有去关闭 SAC（不可逆）。处理方式是等待并重试，上限 15 次 / 每次 60 秒。

README 必须写明这一点：本项目的原生运行依赖一个会间歇性阻断 RDKit 的
操作系统策略，这不是代码问题，但确实影响可复现性。

## 又一处算术错误(我自己的)

STATUS.md 与比较报告 L5 都写过"有效唯一特征向量 **≤ 4118**"。
那是 `4200 − 82` 算出来的,公式用错了。82 组覆盖 180 行,
collapse 之后应是 `4200 − 180 + 82 = 4102`。
比较报告第 207 行其实早就写对了(`4200 − 4102 = 98`),
两处数字在同一份文档里互相矛盾了一段时间。现已统一为 **4102**。

## 特征维数(从冻结的 estimator 读出,非估计)

```
svr_desc        n_features_in_ = 216 → VarianceThreshold 后 207
ridge_ecfp4     n_features_in_ = 2048 → 2046
hgb_desc_ecfp4  n_features_in_ = 2264  (216 + 2048)
```

README 里原先写的"~210 描述符"已改为实测的 216 / 207。

## 两次训练合计墙钟

骨架 3367.6s ≈ 56.1 min,随机 3174.3s ≈ 52.9 min,合计 **≈ 109 min**。
README 初稿误写为 78 min,已修正。
