# Project brief for RA / MSc applications

Ready-to-use text for emails to PIs, CV entries and personal statements, plus
the questions a reviewer will ask and the honest answers. English first, then
Chinese.

Every number here traces to [`results/test_scaffold.json`](../results/test_scaffold.json)
and the README's §7. **Do not round them up and do not drop the baseline.** The
baseline comparison is the reason this project is worth showing at all.

---

## 1. The 90-word version (cold email to a PI)

> I built a small, fully auditable baseline for predicting logD<sub>7.4</sub>
> from 2D structure, using 4,200 public experimental measurements from ChEMBL.
> Evaluated on a Bemis–Murcko scaffold split — so no framework is shared
> between training and test — an SVR on RDKit descriptors reaches
> RMSE 0.81 (95% CI 0.74–0.88) against 1.10 for a calibrated Crippen cLogP
> baseline. The model was selected on validation and the test set was scored
> once, with the access logged. Code, data provenance and 49 tests are in the
> repository; the limitations are written down rather than omitted.

## 2. The 250-word version (SOP / project section)

> **Predicting logD<sub>7.4</sub> from molecular structure: a reproducible
> baseline.** Lipophilicity at physiological pH is one of the properties that
> most often decides whether a compound survives early development, and it is a
> standard benchmark for molecular property prediction.
>
> I used the 4,200-compound AstraZeneca logD<sub>7.4</sub> set deposited in
> ChEMBL (CC BY-SA 3.0, distributed as MoleculeNet *Lipophilicity*), pinned by
> SHA-256 so the exact input bytes are recorded. Because medicinal-chemistry
> datasets are built from congeneric series, a random split would place close
> analogues on both sides and overstate performance; I therefore evaluated on a
> Bemis–Murcko scaffold split, and measured the difference rather than assuming
> it — the same model and hyperparameter grid lost 0.10 log units of validation
> RMSE moving from a random to a scaffold split.
>
> Against a ladder of baselines (training mean; raw Crippen cLogP; cLogP
> rescaled on training data), an RBF SVR on 2D descriptors achieved test
> RMSE 0.809 (95% bootstrap CI 0.738–0.881), MAE 0.609, R² 0.572 and
> Kendall τ 0.553 on 420 held-out compounds — an improvement of 0.290 log units
> over the best baseline, with non-overlapping intervals. The headline model was
> frozen on validation before the test set was read, and the single test
> evaluation is recorded in an append-only access log.
>
> A nearest-neighbour analysis shows the gain is real but local: test MAE rises
> from 0.44 for close analogues to 0.78 for novel chemotypes, whereas the
> zero-parameter baseline is flat — the model's edge comes from proximity to
> training chemistry and decays with novelty.

## 3. CV / résumé bullets

> **logD<sub>7.4</sub> prediction baseline** · Python, RDKit, scikit-learn
> - Built an end-to-end ADMET property-prediction pipeline on 4,200 public
>   ChEMBL measurements: curation and QC ledger, Bemis–Murcko scaffold
>   splitting, leakage assertions, 8 models, bootstrap confidence intervals.
> - Test RMSE 0.809 (95% CI 0.738–0.881) vs 1.099 for a calibrated cLogP
>   baseline on scaffold-disjoint held-out compounds; model frozen on
>   validation, test set scored once under an append-only access log.
> - 49-test regression suite; SHA-256-pinned data and code provenance; runs on
>   CPU at zero cost.

---

## 4. What to claim, and what not to

**Say this** — it is supported:

- A descriptor model beats a calibrated zero-training-data cLogP baseline on
  unseen chemotypes, by more than the uncertainty.
- The evaluation protocol is auditable: pre-registered model, single logged
  test read, hash-pinned inputs.
- The error grows with distance from the training set, and that was measured.

**Never say this:**

| Don't claim | Why it is wrong |
|---|---|
| "state of the art" / "outperforms deep learning" | No published model was re-run on this exact partition. Without that, any comparison is between different splits and is meaningless. |
| anything about dengue, antivirals, CYP inhibition, toxicity | logD<sub>7.4</sub> is a physicochemical property. No biological endpoint was tested or assumed. The repo was renamed away from `admet-triage-dengue` for exactly this reason. |
| "a drug-discovery model" | It predicts one physicochemical property. Say that. |
| "the test set was never touched" (repo-wide) | False. A second implementation in the same repository, `src/admet/`, produced its own test-set results. The correct, narrower claim is that the `src/logd` test partition was scored once. |
| "0.796 RMSE" (the boosting model) | That was the best *after* seeing test labels. The defensible number is the pre-registered 0.809. |

**If a result sounds impressive, check whether the baseline is in the same
sentence.** RMSE 0.81 means little alone; "0.81 against 1.10 for cLogP, with
non-overlapping intervals" is the actual finding.

---

## 5. How to use this in applications

1. **Make the repository presentable before sending anything.** The README is
   the deliverable; make sure §7 (the test result) is the part a skimmer hits.
   Nothing personal belongs in the repo — `.gitignore` already blocks CVs,
   transcripts and drafts, and that should stay true.
2. **Match it to the group, not the other way round.** Read two or three recent
   papers from the group first. If they do physics-based free-energy work or
   structure-based design, this project is evidence of *engineering discipline
   and honest evaluation*, not of domain overlap — say so explicitly rather
   than stretching the relevance.
3. **Lead with the protocol, not the score.** Scaffold splitting, a frozen
   pre-registered model, a logged single test read, and baselines that are hard
   to beat are the parts a computational PI will actually react to. Plenty of
   applicants have a model; far fewer can show they did not fool themselves.
4. **Offer the one-screen version.** Link the README and name the single
   comparison (0.809 vs 1.099). Do not paste the whole metrics table into an
   email.
5. **Bring the negative results.** `results/experiment_log.md` records the runs
   that crashed and the two false claims that were found and corrected in the
   code. Volunteering that is more persuasive than a clean story.
6. **Know the limitations before being asked.** §8 of the README. Being the
   person who raises the weakness first is the whole impression you want.

### Natural next steps, if asked "what would you do next?"

- **Repeat the split.** One scaffold partition gives one number; 5–10
  partitions would give a distribution and show whether the 0.29 gap is stable.
- **Add a second endpoint** (e.g. microsomal clearance, also public in ChEMBL)
  to test whether the pipeline transfers or was tuned to lipophilicity.
- **Predict uncertainty, not just values.** The similarity analysis shows error
  is predictable from distance to training data — a calibrated per-compound
  error bar is more useful to a chemist than a point estimate.
- **Compare against a published model on this exact partition**, which is the
  only way to make a "competitive with X" statement legitimate.

---

## 6. 中文版

### 90 字版（给导师的冷邮件）

> 我做了一个可完整审计的 logD<sub>7.4</sub>（pH 7.4 辛醇/水分配系数）预测基线，
> 数据是 ChEMBL 的 4200 个公开实验值。在 Bemis–Murcko 骨架划分下
> （训练集与测试集不共享任何分子骨架），基于 RDKit 描述符的 SVR 达到
> RMSE 0.81（95% 置信区间 0.74–0.88），而校准后的 Crippen cLogP 基线是 1.10。
> 模型在验证集上选定，测试集只评估过一次且有访问记录。代码、数据溯源与
> 49 项测试都在仓库里，局限性是写出来的而不是省略掉的。

### 可以说什么，不能说什么

**可以说：** 描述符模型在未见过的骨架上，以超过不确定度的幅度，击败了一个
零训练数据的校准 cLogP 基线；评估流程可审计（预先登记模型、单次带日志的测试集
读取、哈希钉定输入）；误差随着与训练集的距离上升，且这一点是实测的。

**绝对不要说：**

- **"达到 SOTA" / "超过深度学习"** —— 没有任何已发表模型在这个划分上被重跑过。
  不同划分之间的比较没有意义。
- **任何涉及登革热、抗病毒活性、CYP 抑制、毒性的说法** —— logD 是理化性质，
  本项目没有检验也没有假设任何生物学终点。仓库从 `admet-triage-dengue` 改名
  正是为了这件事。
- **"这是一个药物发现模型"** —— 它预测一个理化性质，就这么说。
- **"本仓库的测试集从未被碰过"** —— 这是假的。同一仓库里的第二套实现
  `src/admet/` 已经产出过测试集结果。只能窄到「`src/logd` 的测试分区被评估过一次」。
- **"RMSE 0.796"** —— 那是看过测试集标签之后才最好的 boosting 模型。
  可辩护的数字是预先登记的 **0.809**。

**判断标准：如果一个结果听起来很厉害，检查基线是否在同一句话里。**
单说 RMSE 0.81 没有意义；「0.81 对 cLogP 的 1.10，且置信区间不重叠」才是结论。

### 怎么用于申请

1. **先把仓库整理到能直接发出去**，README 就是交付物，确保别人一眼看到 §7。
   仓库里不能有任何个人材料（`.gitignore` 已经挡住简历、成绩单、申请稿）。
2. **去匹配课题组，而不是让课题组来匹配你。** 先读那个组两三篇近期论文。
   如果他们做的是基于物理的自由能计算或结构导向设计，那这个项目证明的是
   **工程纪律和诚实的评估能力**，不是领域重叠 —— 就这么直说，别硬拗相关性。
3. **先讲流程，再讲分数。** 骨架划分、冻结的预先登记模型、带日志的单次测试集
   读取、以及很难打败的基线 —— 这些才是计算方向的 PI 真正会有反应的部分。
   有模型的申请者很多，能证明自己没有自欺的很少。
4. **提供一屏能看完的版本。** 给 README 链接，点名那一个对比（0.809 对 1.099），
   不要把整张指标表粘进邮件。
5. **主动带上负面结果。** `results/experiment_log.md` 记录了崩掉的运行，以及代码里
   被发现并改正的两处错误断言。主动说这些，比讲一个干净的故事更有说服力。
6. **在被问之前就知道局限在哪**（README §8）。做那个自己先把弱点说出来的人，
   这就是你想留下的全部印象。

### 如果被问「下一步做什么」

- **重复划分。** 一个骨架划分只给一个数字；跑 5–10 个划分才能给出分布，
  才能说明 0.29 的差距是否稳定。
- **加第二个终点**（例如肝微粒体清除率，ChEMBL 上同样公开），
  检验这套流程是能迁移的，还是只是被调成适配亲脂性的。
- **预测不确定度，而不只是预测值。** 相似性分析已经说明误差可以由
  「与训练集的距离」预测 —— 给化学家一个逐化合物的误差条，比给一个点估计有用得多。
- **在同一个划分上对比已发表模型**，这是让「与 X 相当」这种说法合法的唯一办法。
