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
