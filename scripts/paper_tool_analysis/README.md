# Paper Tool Analysis Scripts

这个目录是一套离线优先（offline-first）的论文分析流水线，目标是把训练日志转成：
- 可复现统计指标（工具调用、skill 生命周期、失败模式）
- 可解释语义标签（可选 LLM）
- 可直接粘进论文的图、表、案例文档

---

## 0. 你需要准备什么输入

### 输入目录 A：`tool_log_dir`
目录内需要包含：
- `skills_<step>.json`
- `tool_rollouts_<step>.json`

### 输入目录 B：`rollout_data_dir`
这是“样本级 jsonl 目录”：
- train：通常指向 `rollout/train/<RUN_ID>`
- eval：通常指向 `validation_data/val/<RUN_ID>`
- 目录内需要包含：`<step>.jsonl`

### 输入目录 C：`skill_store_dir`（可选，但强烈建议）
目录内如果存在：
- `_skill_archive/_events.jsonl`

则流水线可以进一步分析：
- skill 的 `created / deleted / validation_cleanup / periodic_step_cleanup / incorrect_trajectory`
- train 中的 cleanup 与 eval 中无 cleanup 的差异

如果你不填：
- 脚本会自动尝试从 `tool_log_dir` / `rollout_data_dir` 推断 `skills/<phase>/<RUN_ID>`
- 若推断失败，会自动降级为“无 archive 模式”，不会报错退出

通常这些目录来自训练脚本中的：
- `+trainer.tool_log_dir=...`
- `trainer.rollout_data_dir=...`
- `+trainer.skill_store_dir=...`

---

## 1. 推荐用法（配置文件驱动，不再手写长参数）

### 第一步：编辑配置文件
打开：
- `./scripts/paper_tool_analysis/pipeline_config.yaml`

至少要填这两个字段：
- `inputs.tool_log_dir`
- `inputs.rollout_data_dir`

如果你想完整统计 skill 生灭与 cleanup 原因，再额外填：
- `inputs.skill_store_dir`

### 第二步：一条命令启动
```bash
python ./scripts/paper_tool_analysis/run_pipeline.py
```

说明：
- 这条命令会按配置自动执行全流程。
- 你不需要再写 `--tool-log-dir ... --rollout-data-dir ...` 这种长参数链。

---

## 2. 配置参数逐项详解（每个参数都在这里）

下面所有参数都来自 `pipeline_config.yaml`。

### A. `inputs`

#### `inputs.tool_log_dir`
- 类型：`string`
- 必填：是
- 作用：指定训练时产出的 tool 日志目录，供 unified 数据构建使用。
- 目录中必须包含：`skills_<step>.json`、`tool_rollouts_<step>.json`。
- 示例：`/path/to/verl_step_records/tool_logs/train/<RUN_ID>`
- 常见错误：
  - 填到上一级目录，导致找不到 `skills_*.json`。
  - 填到评测日志目录（eval）而不是训练目录（train）。

#### `inputs.rollout_data_dir`
- 类型：`string`
- 必填：是
- 作用：指定样本 jsonl 目录，供样本级和 tool_call 级事件抽取。
- 目录中必须包含：`<step>.jsonl`。
- 示例：`/path/to/verl_step_records/rollout/train/<RUN_ID>`
- eval 示例：`/path/to/verl_step_records/validation_data/val/<RUN_ID>`
- 常见错误：
  - 路径存在但目录为空。
  - 文件名不是 step 数字风格，导致 step 推断异常。

#### `inputs.skill_store_dir`
- 类型：`string`
- 必填：否
- 作用：指定 skill store 目录，用于读取 `_skill_archive/_events.jsonl` 并做 skill 生灭/cleanup 归因。
- 示例：`/path/to/verl_step_records/skills/train/<RUN_ID>`
- 留空行为：
  - 自动尝试从 `tool_log_dir` / `rollout_data_dir` 推断。
  - 推断不到时继续跑，但 `death_reason` 只能退化为 `snapshot_disappearance` 这类弱归因。

### B. `outputs`

#### `outputs.root_dir`
- 类型：`string`
- 必填：否（建议保留默认）
- 作用：所有分析产物的根目录。
- 影响：下游 `analysis_cache/results/paper_bundle` 都在这个根目录下生成。

#### `outputs.analysis_cache_dir`
- 类型：`string`
- 必填：否
- 默认：`analysis_cache`
- 作用：存放统一数据集（`unified.parquet/jsonl`）和构建摘要。
- 建议：一般不改；改了要确保团队成员知道新路径。

#### `outputs.results_dir`
- 类型：`string`
- 必填：否
- 默认：`results`
- 作用：存放指标结果与中间分析结果（`tool_dynamics.json` 等）。

#### `outputs.paper_bundle_dir`
- 类型：`string`
- 必填：否
- 默认：`paper_bundle`
- 作用：存放论文可直接使用的图表、latex 表格和案例文档。

### C. `runtime`

#### `runtime.python_bin`
- 类型：`string`
- 必填：否
- 默认：空（自动使用当前解释器）
- 作用：强制指定运行各子脚本的 Python 路径。
- 什么时候需要填：
  - 机器上有多个 Python 环境。
  - 你要固定到某个 conda/venv 解释器。

#### `runtime.fail_fast`
- 类型：`bool`
- 必填：否
- 默认：`true`
- 作用：控制失败策略。
- `true`：任何一步失败就立即停止（推荐，方便定位问题）。
- `false`：失败步骤报 warning 后继续跑后续步骤。

#### `runtime.resume`
- 类型：`bool`
- 必填：否
- 默认：`false`
- 作用：控制是否增量运行。
- `false`：每次从头运行所有步骤，重新生成所有输出文件。
- `true`：跳过已存在输出文件的步骤（节省时间）。适用于断点后继续运行。
- 命令行覆盖：也可以通过 `--resume` 命令行参数覆盖配置文件设置。
- 跳过逻辑：
  - `build_unified` → 检查 `unified.parquet`
  - `tool_dynamics` → 检查 `tool_dynamics.json`
  - `skill_lifecycle` → 检查 `skill_lifecycle.csv`
  - `surprise_miner` → 检查 `surprises.json`
  - `paper_bundle` → 检查 `paper_bundle_dir/` 目录是否非空
  - `skill_semantic_clustering` / `create_skill_by_datasource`：始终执行（无独立输出文件）

### D. `pipeline`

#### `pipeline.build_unified`
- 类型：`bool`
- 默认：`true`
- 作用：是否执行 `build_unified_dataset.py`。
- 关闭场景：你已经有最新 `unified.parquet`，只想重跑后续分析。

#### `pipeline.tool_dynamics`
- 类型：`bool`
- 默认：`true`
- 作用：是否执行 `tool_dynamics_metrics.py`。

#### `pipeline.skill_lifecycle`
- 类型：`bool`
- 默认：`true`
- 作用：是否执行 `skill_lifecycle_metrics.py`。

#### `pipeline.skill_semantic_clustering`
- 类型：`bool`
- 默认：`true`
- 作用：是否执行 `skill_semantic_clustering_llm.py`。

#### `pipeline.surprise_miner`
- 类型：`bool`
- 默认：`true`
- 作用：是否执行 `surprise_miner_llm.py`。

#### `pipeline.paper_bundle`
- 类型：`bool`
- 默认：`true`
- 作用：是否执行 `make_paper_bundle.py` 生成论文可交付物。

#### `pipeline.create_skill_by_datasource`
- 类型：`bool`
- 默认：`true`
- 作用：是否执行 `create_skill_by_datasource.py`，统计各数据集触发 `create_skill` 的次数分布。
- 特点：不依赖 `unified.parquet`，直接读取 `tool_rollouts_*.json`，可独立运行。

### E. `params.build_unified`

#### `params.build_unified.max_rollout_files`
- 类型：`int`
- 默认：`0`
- 作用：限制读取 rollout jsonl 文件数量。
- 语义：
  - `0`：读取全部（推荐）
  - `N>0`：只读最后 N 个文件（按 step 排序）
- 适用场景：快速 smoke test、超大日志先做局部验证。

#### `params.build_unified.include_skill_archive`
- 类型：`string`
- 默认：`"auto"`
- 可选值：`"auto"` / `"true"` / `"false"`
- 作用：控制是否读取 `_skill_archive/_events.jsonl`。
- 推荐：`auto`

#### `params.build_unified.obs_error_markers`
- 类型：`list[string]`
- 默认：`["[stderr]"]`
- 作用：定义哪些 marker 会被视为 `tool_obs_has_error=true`。
- 典型用途：统计 run_skill 里“invalid_reason 为空但 obs 已经报错”的隐性失败。

### F. `params.tool_dynamics`

#### `params.tool_dynamics.top_k_tools`
- 类型：`int`
- 默认：`15`
- 作用：控制逐步工具曲线和摘要里保留多少个工具。

#### `params.tool_dynamics.rolling_window`
- 类型：`int`
- 默认：`5`
- 作用：控制窗口类惊喜挖掘和后续图表平滑的默认窗口大小。

#### `params.tool_dynamics.top_k_invalid_reasons`
- 类型：`int`
- 默认：`12`
- 作用：控制 invalid reason 面积图和摘要表保留的原因数量。

### G. `params.skill_lifecycle`

#### `params.skill_lifecycle.uplift_window`
- 类型：`int`
- 默认：`5`
- 作用：定义 skill 创建前后窗口，用于计算 uplift 指标。
- 解释：
  - pre 窗口：`[first_seen_step - window, first_seen_step - 1]`
  - post 窗口：`[first_seen_step + 1, first_seen_step + window]`
- 取值建议：`5~15`
- 风险：
  - 太小：噪声高。
  - 太大：可能混入阶段迁移，解释性下降。

#### `params.skill_lifecycle.reuse_window`
- 类型：`int`
- 默认：`10`
- 作用：定义“创建后多久成功复用”的观测窗口，用于 ecology 分析和 highlights。

#### `params.skill_lifecycle.min_calls_for_skill_stats`
- 类型：`int`
- 默认：`3`
- 作用：控制 skill headline summary 里保留的最小调用门槛。

### H. `params.skill_semantic_clustering`

#### `params.skill_semantic_clustering.min_calls`
- 类型：`int`
- 默认：`1`
- 作用：筛掉调用次数太少的 skill，避免语义分类证据不足。
- 取值建议：
  - 数据量小：`1`
  - 数据量大：`3~10`

#### `params.skill_semantic_clustering.max_skills`
- 类型：`int`
- 默认：`100`
- 作用：限制参与语义分类的 skill 数量（按调用次数排序后截断）。
- 影响：
  - 值大：覆盖更多 skill，但耗时更高（LLM 模式尤其明显）。
  - 值小：速度快，但长尾 skill 可能被裁掉。

### I. `params.surprise_miner`

#### `params.surprise_miner.min_effect_size`
- 类型：`float`
- 默认：`0.05`
- 作用：控制 obs error / invalid reason / recovery 类 findings 的最小变化幅度。

#### `params.surprise_miner.min_calls`
- 类型：`int`
- 默认：`8`
- 作用：控制 hero tool / hero skill / specialization 等候选的最小样本量。

#### `params.surprise_miner.max_findings`
- 类型：`int`
- 默认：`20`
- 作用：控制最终 surprises 输出最多保留多少条 finding。

### J. `params.paper_bundle`

#### `params.paper_bundle.top_k_plot_items`
- 类型：`int`
- 默认：`12`
- 作用：控制 obs error curve、invalid reason area、hero/skill 图表保留的项目数。

### K. `params.create_skill_by_datasource`

#### `params.create_skill_by_datasource.top_n`
- 类型：`int`
- 默认：`10`
- 作用：控制终端打印 data_source 排名时显示的 Top-N 数量。
- 取值建议：
  - 数据集种类少（<10）：设为实际数量或更大值以看到全部。
  - 数据集种类多：`10~20` 观察头部分布即可。
- 注意：不影响 CSV/JSON 文件的输出，文件中始终写入全部数据集。

### I. `llm`

#### `llm.enabled`
- 类型：`bool`
- 默认：`false`
- 作用：是否启用 LLM 分析。
- 影响范围：
  - `skill_semantic_clustering_llm.py`
  - `surprise_miner_llm.py`
- 建议：
  - 先用 `false` 跑通全链路。
  - 再切到 `true` 提升语义可读性。

#### `llm.base_url`
- 类型：`string`
- 必填条件：`llm.enabled=true` 时必填
- 作用：OpenAI 兼容接口地址。
- 示例：`https://api.openai.com` 或内部网关地址。

#### `llm.model`
- 类型：`string`
- 必填条件：`llm.enabled=true` 时必填
- 作用：指定分类/改写所用模型。
- 示例：`gpt-4o-mini`。

#### `llm.api_key`
- 类型：`string`
- 必填：否
- 作用：显式传 key 给脚本。
- 留空行为：自动读取环境变量 `OPENAI_API_KEY`。
- 安全建议：优先走环境变量，不要把密钥提交到仓库。

#### `llm.mode_when_enabled`
- 类型：`string`
- 默认：`llm`
- 可选：`llm` / `auto`
- 作用：启用 LLM 时传给子脚本的 `--mode`。
- 建议：`llm`（行为可预测）。

#### `llm.mode_when_disabled`
- 类型：`string`
- 默认：`heuristic`
- 可选：`heuristic` / `auto`
- 作用：禁用 LLM 时传给子脚本的 `--mode`。
- 建议：`heuristic`（彻底离线、结果稳定）。

---

## 3. 产物说明

### `analysis_cache/`
- `unified.parquet`
- `unified.jsonl`
- `build_summary.json`

### `results/`
- `tool_dynamics.csv`
- `tool_dynamics.json`
- `tool_dynamics_by_step.csv`
- `tool_failure_breakdown.csv`
- `trajectory_quality_by_step.csv`
- `trajectory_quality_summary.json`
- `skill_lifecycle.csv`
- `skill_creation_rate.csv`
- `skill_ecology_by_step.csv`
- `skill_archive_events.csv`
- `skill_monitoring_by_step.csv`
- `skill_lifecycle_summary.json`
- `skill_taxonomy.json`
- `surprises.json`
- `surprises.md`
- `analysis_overview.md`
- `create_skill_total_by_datasource.csv`（各数据集 total / correct / valid 三口径汇总表）
- `create_skill_by_step_datasource.csv`（按 step × data_source 宽表，行=step，列=数据集）
- `create_skill_used_count_by_step.json`（各 step 的全量 `used_count` 原始记录）

### `paper_bundle/`
- `fig_*.png`
- `table_main.tex`
- `table_ablation.tex`
- `table_top_failure_reasons.tex`
- `table_top_hero_skills.tex`
- `qualitative_cases.md`
- `analysis_overview.md`
- `bundle_manifest.json`

### 全流程总览：默认一共会产出多少个文件

如果按 `run_pipeline.py` 的 7 个阶段全开来跑，当前代码会稳定产出 **37 个文件**。

- `analysis_cache/`：3 个
  - `unified.parquet`
  - `unified.jsonl`
  - `build_summary.json`
- `results/`：19 个
  - 6 个工具/轨迹质量相关文件
  - 6 个 skill 生命周期/生态相关文件
  - 1 个 skill 分类文件
  - 2 个 surprise 挖掘文件
  - 1 个 overview 文档
  - 3 个 create_skill 数据集分布文件
- `paper_bundle/`：15 个
  - 8 张图
  - 4 个 LaTeX 表
  - 2 个 Markdown 文档
  - 1 个 manifest

这里要区分两件事：

- “产物文件数”是 37
- “结构化表/数据文件”只是其中一部分；另外还有 JSON 摘要、Markdown 报告、PNG 图、LaTeX 表格

如果只从“表”的角度理解，这个目录主要产出三类结构化数据：

- unified 宽表：`unified.parquet` / `unified.jsonl`
- 指标表：多个 `csv`
- 摘要对象：多个 `json`

### 全流程总链路：7 个阶段分别做什么

#### 第 1 步：`build_unified_dataset.py`

输入四类原始日志：

- `skills_<step>.json`
- `tool_rollouts_<step>.json`
- `rollout_data_dir/<step>.jsonl`
- `_skill_archive/_events.jsonl`（可选）

输出 3 个文件：

- `analysis_cache/unified.parquet`
- `analysis_cache/unified.jsonl`
- `analysis_cache/build_summary.json`

它做的事不是“直接算指标”，而是先把不同来源标准化成一张宽表。最重要的是统一出这些 `row_type`：

- `sample`
- `tool_call`
- `skill_new_event`
- `skill_snapshot`
- `tool_summary`
- `tool_rollout_example`
- `skill_archive_event`

后面的脚本几乎都只读 `unified.parquet`，所以这一步决定了后续所有指标的口径。

#### 第 2 步：`tool_dynamics_metrics.py`

输入：

- `analysis_cache/unified.parquet`

输出 6 个文件：

- `results/tool_dynamics.csv`
- `results/tool_dynamics.json`
- `results/tool_dynamics_by_step.csv`
- `results/tool_failure_breakdown.csv`
- `results/trajectory_quality_by_step.csv`
- `results/trajectory_quality_summary.json`

它回答的是：

- 工具总体成功率、失败率、obs error 率是多少
- 每一步每个工具的调用量和失败率怎么变化
- 哪些 `invalid_reason` 在什么 step、什么数据集里爆发
- 轨迹质量有没有随训练改善

#### 第 3 步：`skill_lifecycle_metrics.py`

输入：

- `analysis_cache/unified.parquet`

输出 6 个文件：

- `results/skill_lifecycle.csv`
- `results/skill_creation_rate.csv`
- `results/skill_ecology_by_step.csv`
- `results/skill_archive_events.csv`
- `results/skill_monitoring_by_step.csv`
- `results/skill_lifecycle_summary.json`

它回答的是：

- 一个 skill 什么时候 pending create
- 什么时候 promoted
- 什么时候第一次进 selected skill 集合
- 什么时候第一次被调用、第一次成功复用
- 最后为什么死掉
- 每一步 skill 生态是扩张、复用还是清理

#### 第 4 步：`skill_semantic_clustering_llm.py`

输入：

- `analysis_cache/unified.parquet`

输出 1 个文件：

- `results/skill_taxonomy.json`

它把 skill 按语义打标签，回答“这些 skill 大致属于什么能力类型”。

#### 第 5 步：`surprise_miner_llm.py`

输入：

- `analysis_cache/unified.parquet`
- 第 2、3 步的结果文件

输出 2 个文件：

- `results/surprises.json`
- `results/surprises.md`

它不重新定义基础指标，而是从已有指标里挖“值得写进论文的规律”。

#### 第 6 步：`make_paper_bundle.py`

输入：

- `analysis_cache/unified.parquet`
- `results/tool_dynamics.json`
- `results/skill_lifecycle.csv`
- `results/surprises.json`
- 以及同目录下其他 `results/*.csv/json`

输出 15 个文件：

- 8 张图：`fig_tool_adoption_curve.png`、`fig_skill_lifecycle_hist.png`、`fig_tool_success_reward_dual_axis.png`、`fig_datasource_tool_heatmap.png`、`fig_obs_error_curve.png`、`fig_invalid_reason_stacked_area.png`、`fig_skill_birth_death_timeline.png`、`fig_trajectory_quality_curve.png`
- 4 个 LaTeX 表：`table_main.tex`、`table_ablation.tex`、`table_top_failure_reasons.tex`、`table_top_hero_skills.tex`
- 2 个 Markdown：`qualitative_cases.md`、`analysis_overview.md`
- 1 个 manifest：`bundle_manifest.json`

它的职责是“消费前面已经算好的结果，再打包成论文资产”，不是重新发明一套指标。

#### 第 7 步：`create_skill_by_datasource.py`

输入：

- `tool_rollouts_<step>.json`

输出 3 个文件：

- `results/create_skill_total_by_datasource.csv`
- `results/create_skill_by_step_datasource.csv`
- `results/create_skill_used_count_by_step.json`

它是这个目录里唯一一个刻意绕过 unified 的脚本，因为它要同时保留：

- `used_count` 的全量计数
- `rollouts` 的样例级 data_source 分布

这两个统计粒度不能混为一谈。

### 每个目录里的文件到底是怎么处理出来的

#### A. `analysis_cache/`

##### 1. `unified.parquet`

这是整个分析目录的核心宽表。它把原始日志展开成统一 schema，并把常用指标平铺成列。

关键处理：

- `skills_*.json` 生成 `skill_snapshot`
- `tool_rollouts_*.json` 生成 `tool_summary` 和 `tool_rollout_example`
- `*.jsonl` 生成 `sample`、`tool_call`、`skill_new_event`
- `_skill_archive/_events.jsonl` 生成 `skill_archive_event`
- 再把 step 级 skill meta 回填到同一步的其他行

关键口径：

- `tool_raw_success = invalid_reason 为空`
- `tool_effective_success = raw_success 且 obs 没有 error marker`
- `trajectory_key = request_id 非空时用 request_id，否则回退 uid`

##### 2. `unified.jsonl`

和 `unified.parquet` 是同一份逻辑数据，只是换成 JSONL 便于抽样检查和 grep。分析口径和 `unified.parquet` 完全一致。

##### 3. `build_summary.json`

这是构建摘要，不是分析指标表。主要记录：

- 读到了哪些输入目录
- 一共生成了多少行 unified
- archive 是否被纳入
- phase / tool_variant 的检测结果

#### B. `results/` 中的工具与轨迹质量文件

##### 1. `tool_dynamics.csv`

这不是“一张单一指标表”，而是两个 section 拼在一起：

- `tool_share_by_step`
  - 每个 `step × tool_kind` 的调用份额
  - `share = calls / total_calls_within_step`
- `datasource_tool_preference`
  - 每个 `data_source × tool_kind` 的调用份额
  - `share = calls / total_calls_within_datasource`

它适合做透视和画 share 曲线。

##### 2. `tool_dynamics.json`

这是工具动态的全局摘要，主要字段包括：

- `overall_raw_success_rate`
- `overall_failure_rate`
- `overall_effective_success_rate`
- `overall_obs_error_rate`
- `failure_reason_distribution`
- `corr_num_tool_calls_vs_score`
- `corr_num_tool_calls_vs_is_correct`
- `corr_num_tool_calls_vs_total_reward`
- `top_tools_by_calls`
- `top_tool_kinds_by_calls`
- `top_invalid_reasons`

关键口径：

- `overall_raw_success_rate = mean(tool_raw_success)`
- `overall_effective_success_rate = mean(tool_effective_success)`
- `overall_obs_error_rate = mean(tool_obs_has_error)`
- 相关性是在样本级上，把每条 trajectory 的工具调用数 merge 回 `sample` 后再算 Pearson 相关

##### 3. `tool_dynamics_by_step.csv`

行粒度：

- `step × tool_name × tool_kind`

关键指标：

- `calls`
- `invalid_rate`
- `raw_success_rate`
- `effective_success_rate`
- `obs_error_rate`
- `mean_processing_time_ms`
- `p50_processing_time_ms`
- `p90_processing_time_ms`
- `mean_turn_index`

关键口径：

- `invalid_rate = mean(1 - tool_raw_success)`
- `raw_success_rate = mean(tool_raw_success)`
- `effective_success_rate = mean(tool_effective_success)`
- `calls` 对 `tool_summary` 行会用 `tool_used_count` 做权重；对逐次调用行默认每行记 1 次

##### 4. `tool_failure_breakdown.csv`

行粒度：

- `step × tool_name × tool_kind × data_source × invalid_reason`

关键指标：

- `count`
- `share_within_step_tool = count / 该 step 该 tool 的总失败数`
- `share_within_step_tool_datasource = count / 该 step 该 tool 该数据集的总失败数`

它用来回答“某种失败是全局常见，还是只在某一步、某工具、某数据源里局部爆发”。

##### 5. `trajectory_quality_by_step.csv`

行粒度：

- `step`

关键指标：

- `num_samples`
- `avg_num_turns`
- `valid_traj_rate`
- `no_loss_on_traj_rate`
- `avg_tool_calls`
- `avg_tool_penalty`
- `avg_latency_penalty`
- `avg_skill_reward`
- `avg_format_reward`
- `avg_total_reward`
- `avg_tool_processing_time_sec`
- `avg_tool_queue_time_sec`
- `avg_response_size_mb`
- `traj_stop_reason_distribution_json`

关键口径：

- 先用 `step + trajectory_key` 把 `tool_call` 数 merge 回 `sample`
- 再按 step 对样本级列做均值
- `traj_stop_reason_distribution_json` 是该 step 内 stop reason 的归一化分布

##### 6. `trajectory_quality_summary.json`

这是 `trajectory_quality_by_step.csv` 的全局摘要，主要保留：

- 样本总数
- 全局 `mean_num_turns`
- 全局 `valid_traj_rate`
- 全局 `no_loss_on_traj_rate`
- 全局 `traj_stop_reason_distribution`
- 一共覆盖多少个 step

#### C. `results/` 中的 skill 生命周期与生态文件

##### 1. `skill_lifecycle.csv`

这是 skill 侧最核心的主表，行粒度是：

- `skill_name`

关键处理链路：

- 先从 `skill_snapshot` 建 base
- 再把 `skill_new_event`、`skill_archive_event`、规范化后的 `tool_call` 名字补进 base
- 对每个 skill 回头查：
  - `s_pending`
  - `s_promoted`
  - `s_calls`
  - `s_deleted`

关键指标：

- `first_pending_create_step`
- `first_promoted_step`
- `first_selected_step`
- `first_used_step`
- `first_effective_reuse_step`
- `promotion_lag_steps`
- `first_use_lag_steps`
- `first_success_lag_steps`
- `first_seen_step`
- `last_seen_step`
- `lifespan_steps`
- `is_dead`
- `death_step`
- `death_reason`
- `death_group`
- `selected_steps`
- `latest_used_times`
- `total_calls`
- `total_direct_calls`
- `total_run_skill_calls`
- `raw_success_rate`
- `effective_success_rate`
- `obs_error_rate`
- `peak_calls`
- `peak_step`
- `mean_calls`
- `burst_score`
- `is_burst`
- 三个 uplift 指标：`total_reward_uplift / is_correct_uplift / score_uplift`

关键口径：

- `promotion_lag_steps = first_promoted_step - first_pending_create_step`
- `first_use_lag_steps = first_used_step - first_anchor`
- `first_success_lag_steps = first_effective_reuse_step - first_anchor`
- `first_anchor` 优先级：`promoted -> selected -> pending_create`
- `lifespan_steps = (death_step 或 last_seen_step) - first_seen_step + 1`
- `burst_score = peak_calls / mean_calls`
- `is_burst = peak_calls >= 3 且 burst_score >= 2.0 且 peak 后均值 <= peak_calls * 0.5`
- `death_reason` 优先信任 archive；没有 archive 才回退成 `snapshot_disappearance`

##### 2. `skill_creation_rate.csv`

行粒度：

- `first_seen_step`

关键指标：

- `new_skills`

关键口径：

- 对每个 skill 优先用 `first_pending_create_step`
- 如果没有 pending create，就回退到 `first_selected_step`
- 然后按这个 anchor 计数

它回答的是“每一步有多少新 skill 开始进入生态”。

##### 3. `skill_ecology_by_step.csv`

行粒度：

- `step`

关键指标：

- `selected_skill_count`
- `active_skill_count`
- `new_skill_count`
- `promoted_skill_count`
- `deleted_skill_count`
- `deleted_by_reason_reset`
- `deleted_by_reason_validation_cleanup`
- `deleted_by_reason_incorrect_trajectory`
- `deleted_by_reason_dedup`
- `deleted_by_reason_duplicate_existing`
- `deleted_by_reason_fit_exit`
- `deleted_by_reason_snapshot_disappearance`
- `deleted_by_reason_unknown`
- 以及从 monitoring 回填的逐步监控列

关键口径：

- `new_skill_count` 来自 `first_pending_create_step` 的 step 计数
- `promoted_skill_count` 来自 `first_promoted_step` 的 step 计数
- `deleted_skill_count` 来自 archive deleted 事件
- `active_skill_count` 由 skill 的 `active_start ~ active_end` 区间判断
- `active_start` 优先级：`promoted -> selected -> pending_create`
- `active_end` 优先级：`death_step -> last_seen_step`

##### 4. `skill_archive_events.csv`

这是 archive 原始事件的扁平化导出。行粒度是每一条 archive event。

主要字段：

- `skill_name`
- `skill_archive_event`
- `skill_archive_reason`
- `death_group`
- `skill_archive_timestamp`
- `skill_archive_global_step`
- `skill_archive_snapshot_copied`
- `skill_archive_source_path`
- `skill_archive_runtime_path`
- `skill_archive_path`
- `archive_metadata_json`

它的意义是保留强归因证据，方便后面追溯某个 skill 为什么被删。

##### 5. `skill_monitoring_by_step.csv`

行粒度：

- `step`

它从 `skill_monitoring_metrics_json` 里回填出逐步监控指标，并保留：

- `selected_skill_count`
- 以及 monitoring payload 中的所有数值项

这张表是 `skills_*.json` 中监控信号的 step 级展开版。

##### 6. `skill_lifecycle_summary.json`

这是生命周期的全局摘要，主要字段：

- `num_skills`
- `num_headline_skills`
- `num_burst_skills`
- `mean_lifespan_steps`
- `dead_skill_ratio`
- `mean_effective_success_rate`
- `mean_obs_error_rate`
- `top_death_groups`

其中 `headline_skills` 的筛选口径是：

- `total_calls >= min_calls_for_skill_stats`

#### D. `results/` 中的分类、惊喜发现与 overview 文件

##### 1. `skill_taxonomy.json`

行粒度本质上是：

- 每个 skill 一条分类记录

它先建 skill inventory：

- `skill_snapshot` 提供描述和首末 step
- `tool_call / tool_rollout_example` 提供调用次数

再按 `min_calls` 过滤、按 `max_skills` 截断，然后做两种分类之一：

- `heuristic`
  - 用关键词规则打标签
- `llm`
  - 把 name、description、examples 发给模型分类

关键字段：

- `category`
- `subcategory`
- `confidence`
- `total_calls`
- `description`
- `evidence_refs`
- `evidence_quotes`
- `rationale`
- `first_seen_step`
- `last_seen_step`

##### 2. `surprises.json`

这是惊喜发现主文件。每个 finding 都包含：

- `type`
- `magnitude`
- `stats`
- `step_window`
- `tool_name / skill_name / data_source`
- `evidence_refs`
- `claim`

内部原理是：

- 候选发现先靠确定性规则从指标表中找出来
- `llm` 模式只负责把 finding 改写成更像论文的表述

也就是说：

- LLM 不负责定义“有没有发现”
- LLM 只负责“怎么描述发现”

##### 3. `surprises.md`

这是 `surprises.json` 的人类可读版 Markdown，用于快速浏览候选亮点。

##### 4. `analysis_overview.md`

这是总览报告。它会综合：

- `tool_dynamics.json`
- `trajectory_quality_summary.json`
- `skill_lifecycle_summary.json`
- `surprises.json`

并输出一份简明文字总结。这个文件会同时出现在：

- `results/analysis_overview.md`
- `paper_bundle/analysis_overview.md`

#### E. `results/` 中的 create_skill 数据集分布文件

##### 1. `create_skill_total_by_datasource.csv`

行粒度：

- `data_source`

关键指标：

- `total_count`
- `correct_count`
- `valid_count`

三种口径分别是：

- `total`
  - rollout 样例里所有触发 `create_skill` 的记录
- `correct`
  - `is_correct=True`
- `valid`
  - `is_correct=True` 且 `invalid_reason is null`

注意：这张表的数据源是 `tool_rollouts_*.json` 的 rollout 样例，不是全量调用。

##### 2. `create_skill_by_step_datasource.csv`

行粒度：

- `step`

列结构：

- 固定列：`step`、`used_count_full`
- 其余每一列是一个 `data_source`

关键口径：

- `used_count_full` 来自 `tool_info.used_count`，是该 step `create_skill` 的全量调用次数
- 各 `data_source` 列来自 rollout 样例计数，每步最多只有少量样例

所以这张表适合看“样例分布”和“全量计数”的并置，不适合把每个 data_source 列当成真实全量调用。

##### 3. `create_skill_used_count_by_step.json`

这是最精确的 create_skill 总量摘要，记录：

- 每个 step 的 `used_count`
- 全部 step 合计总数

#### F. `paper_bundle/`

这些文件基本都不是新指标，而是对前面结果的再组织。

##### 1. 8 张图

- `fig_tool_adoption_curve.png`
  - 基于 step 级 tool kind share 曲线
- `fig_skill_lifecycle_hist.png`
  - 基于 `skill_lifecycle.csv` 中的 `lifespan_steps`
- `fig_tool_success_reward_dual_axis.png`
  - 对比 `tool_effective_success` 与 `avg score`
- `fig_datasource_tool_heatmap.png`
  - 数据集和工具类型的调用热力图
- `fig_obs_error_curve.png`
  - top 工具的 `obs_error_rate`
- `fig_invalid_reason_stacked_area.png`
  - top `invalid_reason` 的面积图
- `fig_skill_birth_death_timeline.png`
  - 基于 `skill_ecology_by_step.csv`
- `fig_trajectory_quality_curve.png`
  - 基于 `trajectory_quality_by_step.csv`

##### 2. 4 个 LaTeX 表

- `table_main.tex`
  - 阶段摘要主表
- `table_ablation.tex`
  - 相对 early stage 的变化表
- `table_top_failure_reasons.tex`
  - 顶层失败原因表
- `table_top_hero_skills.tex`
  - 高成功率、长寿命 skill 表

##### 3. 2 个 Markdown + 1 个 manifest

- `qualitative_cases.md`
  - 优先依据 `surprises.json` 的 `evidence_refs` 回查 unified 原始行来构造案例
- `analysis_overview.md`
  - 和 `results/analysis_overview.md` 语义一致
- `bundle_manifest.json`
  - 把 bundle 内所有文件路径列出来，方便下游脚本消费

### 一个贯穿全流程的小例子

下面用一个极小的虚构例子把全链路串起来。假设我们只观察一个 skill：`solve_mod_equation`。

#### 原始输入

在 `10.jsonl` 中有一条样本：

- `step=10`
- `data_source=math`
- `score=0.9`
- `is_correct=1`
- `total_reward=1.2`

这条样本的 `tool_interact_info` 中有一次：

- `tool_name=create_skill`
- `skill_name=solve_mod_equation`
- `invalid_reason=""`
- `tool_obs="created"`

同一条样本的 `batch_new_skills` 中有：

- `skill_name=solve_mod_equation`
- `created_trajectory_id=traj_001`

在 `skills_11.json` 中，这个 skill 第一次出现在 selected skill 集合里：

- `skill_name=solve_mod_equation`
- `skill_description=solve modular arithmetic equations`
- `skill_used_times=1`

在 `11.jsonl` 里又有一次调用：

- `tool_name=run_skill`
- `skill_name=solve_mod_equation`
- `invalid_reason=""`
- `tool_obs="answer=7"`
- `processing_time_ms=120`

在 `_skill_archive/_events.jsonl` 中后续有两条记录：

- `global_step=11, event=created, reason=promoted`
- `global_step=14, event=deleted, reason=periodic_step_cleanup`

#### 它会先变成哪些 unified 行

经过 `build_unified_dataset.py` 后，至少会出现这些行：

- `row_type=sample`
  - 来自 `10.jsonl` 的样本
- `row_type=tool_call`
  - 一行 `create_skill`
- `row_type=skill_new_event`
  - 来自 `batch_new_skills`
- `row_type=skill_snapshot`
  - 来自 `skills_11.json`
- `row_type=tool_call`
  - 一行 `run_skill`
- `row_type=skill_archive_event`
  - 一行 `created/promoted`
- `row_type=skill_archive_event`
  - 一行 `deleted/periodic_step_cleanup`

这时 unified 已经能回答：

- 这个 skill 被创建过
- 它进入过 selected skill 集合
- 它被实际复用过
- 它最后被 reset 类原因清理掉了

#### 它在 `tool_dynamics_by_step.csv` 里会长什么样

对于 step 11 的 `run_skill` 这一行，如果没有错误：

- `calls = 1`
- `invalid_rate = 0`
- `raw_success_rate = 1`
- `effective_success_rate = 1`
- `obs_error_rate = 0`
- `mean_processing_time_ms = 120`
- `p50_processing_time_ms = 120`
- `p90_processing_time_ms = 120`

如果同一次调用 `invalid_reason=""` 但 `tool_obs` 里带有错误 marker，比如 `[stderr]`，那么：

- `raw_success_rate` 仍然可能是 `1`
- 但 `effective_success_rate` 会掉成 `0`

这就是 raw success 和 effective success 两层口径要分开的原因。

#### 它在 `skill_lifecycle.csv` 里会得到哪些值

对于 `solve_mod_equation` 这一行：

- `first_pending_create_step = 10`
- `first_promoted_step = 11`
- `first_selected_step = 11`
- `first_used_step = 11`
- `first_effective_reuse_step = 11`
- `promotion_lag_steps = 11 - 10 = 1`
- `first_use_lag_steps = 11 - 11 = 0`
  - 因为 `first_anchor` 优先选 `first_promoted_step`
- `first_success_lag_steps = 11 - 11 = 0`
- `death_step = 14`
- `death_reason = periodic_step_cleanup`
- `death_group = reset`
- `lifespan_steps = 14 - 11 + 1 = 4`

如果它从未进入 `skills_11.json`，但有：

- `skill_new_event`
- `skill_archive_event(deleted, incorrect_trajectory)`

那么它仍然会被补进 lifecycle，只是：

- `first_selected_step = NaN`
- `description = ""`

这就是 `_build_skill_base()` 为什么不能只靠 snapshot。

#### 它在 `skill_ecology_by_step.csv` 里会如何计数

对于这个例子，生态表至少会体现：

- step 10
  - `new_skill_count = 1`
- step 11
  - `promoted_skill_count = 1`
  - `selected_skill_count >= 1`
  - `active_skill_count >= 1`
- step 14
  - `deleted_skill_count = 1`
  - `deleted_by_reason_reset = 1`

因为 `periodic_step_cleanup` 会被归并到 `reset` 这个粗粒度 death group。

#### 它在 `create_skill_total_by_datasource.csv` 里会如何体现

如果 `tool_rollouts_10.json` 的 rollout 样例中对应记录是：

- `data_source=math`
- `is_correct=True`
- `invalid_reason=None`

那么 `math` 这一行会增加：

- `total_count +1`
- `correct_count +1`
- `valid_count +1`

但要注意：

- 这只是 rollout 样例统计
- 同一步真正的 create_skill 全量调用次数要看 `used_count_full` 和 `create_skill_used_count_by_step.json`

#### surprise 和 paper bundle 在这个例子里会发生什么

如果整个实验里只有这一点点数据，通常不会形成很强的 `surprise` finding；因为很多 finding 需要 early/late 对比、显著 drop 或稳定趋势。

但 `make_paper_bundle.py` 仍然会：

- 从已有 `csv/json` 读数
- 画出稀疏但合法的图
- 生成对应的 LaTeX 表和 overview

所以这个目录的工作流可以理解成：

- `analysis_cache/` 负责把原始日志标准化
- `results/` 负责把标准化数据变成可分析指标
- `paper_bundle/` 负责把指标变成论文资产

---

## 4. 最常见问题

### Q1: 我已经有 unified.parquet，能否跳过构建？
可以。把：
- `pipeline.build_unified: false`

并确保 `outputs.root_dir/analysis_cache/unified.parquet` 已存在。

### Q2: 我不想用 LLM，如何保证全离线？
设置：
- `llm.enabled: false`
- `llm.mode_when_disabled: heuristic`

### Q3: 我想只看最近一段训练趋势，怎么加速？
设置：
- `params.build_unified.max_rollout_files: 50`

### Q4: 报错说 `llm.base_url` 为空？
这是因为你设置了：
- `llm.enabled: true`

但没填：
- `llm.base_url`
- `llm.model`

### Q5.1: 运行到一半中断了，如何继续而不重新跑完所有步骤？
设置：
- `runtime.resume: true`

这样会跳过已存在输出的步骤，直接从缺失的步骤继续执行。
也可使用命令行参数：
```bash
python run_pipeline.py --resume
```

### Q5.2: 只想单独跑 `create_skill_by_datasource`，不用跑全流程？
在 `pipeline_config.yaml` 中把其他步骤全部关掉，只留：
```yaml
pipeline:
  build_unified: false
  tool_dynamics: false
  skill_lifecycle: false
  skill_semantic_clustering: false
  surprise_miner: false
  paper_bundle: false
  create_skill_by_datasource: true
```
或直接命令行调用：
```bash
python scripts/paper_tool_analysis/create_skill_by_datasource.py \
    --tool_log_dir /path/to/tool_log_dir \
    --output scripts/paper_tool_analysis/results/ \
    --top_n 15
```

---

## 5. 代码与实现原理详解

这一节不是“怎么跑”，而是“每个脚本内部到底在做什么、为什么这么做”。

---

### 5.1 总体数据流

```
原始日志
  ├── tool_logs/
  │     ├── skills_<step>.json
  │     └── tool_rollouts_<step>.json
  ├── rollout/ 或 validation_data/
  │     └── <step>.jsonl
  └── skills/
        └── _skill_archive/_events.jsonl   (可选)
               │
               ▼
      build_unified_dataset.py
               │
               ▼
      analysis_cache/unified.parquet
               │
               ├── tool_dynamics_metrics.py
               ├── skill_lifecycle_metrics.py
               ├── skill_semantic_clustering_llm.py
               ├── surprise_miner_llm.py
               ├── create_skill_by_datasource.py   (独立直读 tool_rollouts)
               └── make_paper_bundle.py
```

整个设计的核心原则是：
- 先把各种异构日志压平到一张统一宽表。
- 后续所有分析都尽量只依赖这张宽表，避免每个脚本都重新理解原始日志格式。
- 只有 `create_skill_by_datasource.py` 例外，它刻意绕过 unified，直接读 `tool_rollouts_*.json`，因为它要保留 `used_count` 与样例采样的区别。

---

### 5.2 原始日志格式是怎么被推断出来的

这套脚本默认本地没有真实日志，所以日志格式是从训练代码反推出来的，主要来源有三类：

#### A. `skills_<step>.json`
这是 skill selection 日志。它记录：
- 当前 step 被选中的 skill 列表
- 每个 skill 的 `name / description / used_times / path`
- `selection_stats`
- `skill_monitoring_metrics`

这些字段在分析里主要用于：
- `skill_snapshot` 行
- 每步选中 skill 数量
- skill 生命周期里的 `first_selected_step / last_selected_step`
- step 级的 `skill_monitoring_by_step.csv`

#### B. `tool_rollouts_<step>.json`
这是 per-tool 汇总日志。每个工具包含：
- `used_count`
- 若干 `rollouts` 样例
- 每条样例里的 `uid / data_source / is_correct / invalid_reason / tool_prompt / tool_obs`
- 对 run_skill 来说，还可能有 `tool_obs_has_error / tool_effective_success`

这些字段在分析里主要用于：
- `tool_summary` 行：反映全量 `used_count`
- `tool_rollout_example` 行：反映少量定性样例
- `create_skill_by_datasource.py` 的三口径统计

#### C. `<step>.jsonl`
这是最细粒度的样本日志。每一行是一个 sample，里面可能有：
- 样本级字段：`input / output / gts / score / is_correct / accuracy / total_reward`
- 轨迹级字段：`traj_stop_reason`
- 奖励分解：`tool_penalty / latency_penalty / format_reward / skill_reward / skill_valid_create_reward / skill_reused_reward`
- `verl_tool_metrics`
- `tool_interact_info`：每一次 tool call 的完整轨迹
- `batch_new_skills`

这类日志是最关键的，因为：
- 它能还原“每一个样本到底调用了哪些工具”
- 它能拿到逐 step、逐 call 的 obs / invalid_reason / latency / valid_action / finish / done
- 它能把工具调用和最终 `score / reward / correctness` 对齐起来

#### D. `_skill_archive/_events.jsonl`
这是可选的 skill archive 事件流。每行通常包含：
- `event`：比如 `created` / `deleted` / `summary`
- `reason`：比如 `promoted / validation_cleanup / incorrect_trajectory / periodic_step_cleanup`
- `skill_name`
- `global_step`
- `snapshot_copied`
- `source_path / runtime_path / archive_path`
- `metadata`

它的作用是把“skill 消失了”从一个模糊的快照现象，变成一个可以解释原因的事件：
- 是验证集清理掉的
- 是错误轨迹创建后立刻清理掉的
- 是周期性 reset 清理掉的
- 还是 dedup / duplicate_existing / fit_exit 导致的

---

### 5.3 `common.py`：公共工具库的作用与原理

这个文件不直接运行，但几乎所有脚本都依赖它。

它主要做 7 件事：

#### 1. 解析 step
- 用文件名里的数字推断 step。
- 原理：`skills_10.json -> 10`，`tool_rollouts_3.json -> 3`，`17.jsonl -> 17`。
- 目的：让不同日志源可以按 step 对齐。

#### 2. 自动识别 forever tools
- 它不会 import 工具实现，而是直接用 AST 去解析 `valid_mcp_func_names`。
- 原理：静态读 Python 语法树，避免触发重依赖。
- 目的：把 `Qwen3-VL / PaddleOCR / SAM3` 这类常驻工具归一成 `forever_tool`，和动态 skill 区分开。

#### 3. 自动推断实验 phase / tool variant
- `phase` 通过 `data_source` 名字里是否有 `_train / _eval / _test / _val` 判断。
- `tool_variant` 通过实际出现过的工具名去判断是 ID 还是 OOD 工具集。

#### 4. obs error 检测
- `obs_has_error()` 会检查 observation 文本里是否出现 error marker。
- 默认 marker 是 `"[stderr]"`。
- 原理：很多 run_skill 调用 `invalid_reason` 为空，但 stdout/stderr 已经显示 runtime error，这种失败不能算成功。

#### 5. skill archive 自动推断
- 如果用户不填 `skill_store_dir`，它会尝试从：
  - `tool_logs/train/<run_id>`
  - `rollout/train/<run_id>`
  - `validation_data/val/<run_id>`
  自动替换路径锚点，推到 `skills/<phase>/<run_id>`。

#### 6. JSON / JSONL 读写与容错解析
- 提供 `read_json`、`read_jsonl`、`write_json`、`write_jsonl`。
- 提供 `parse_maybe_json_object` 这种“既能吃 dict，也能吃 JSON string”的兼容函数。

#### 7. LLM 兼容层
- `call_llm_json()` 封装了 OpenAI-compatible chat completion 调用。
- 原理：所有 LLM 脚本都只需要提供 system/user prompt，然后拿回一个 JSON dict。

---

### 5.4 `build_unified_dataset.py`：为什么要先做 unified

这是整个目录里最重要的脚本。它的任务不是“算指标”，而是把原始日志变成一张标准化宽表。

#### 它输出哪些行类型
- `sample`
- `tool_call`
- `skill_new_event`
- `skill_snapshot`
- `tool_summary`
- `tool_rollout_example`
- `skill_archive_event`

#### 为什么要把它们放在一张表里
因为后续分析经常要做跨来源 join：
- 某个 step 的 `run_skill` 失败率
- 某个 skill 第一次出现前后的 reward uplift
- 某个 cleanup event 对应的 skill 之前用了几次
- 某个 obs error spike 是否伴随 valid_traj 下降

如果继续分散在多个原始日志里，这些分析每次都要手写 join 逻辑；统一表可以把这个成本前置一次。

#### 它内部的核心做法

##### 1. 先定义稳定 schema
脚本先定义一套固定列：
- 样本级列：`score / total_reward / traj_stop_reason / verl_tool_metrics_json`
- 工具级列：`tool_name / invalid_reason / tool_obs / tool_effective_success / processing_time_ms`
- skill 级列：`skill_name / skill_status / skill_used_times`
- archive 级列：`skill_archive_event / skill_archive_reason / archive_metadata_json`

这样无论某种日志源有没有某一列，最终 parquet 都保持相同 schema。

##### 2. 把复杂对象同时保留“raw json”与“平铺字段”
例如 `verl_tool_metrics`：
- 一方面整体写入 `verl_tool_metrics_json`
- 另一方面把常用字段平铺成单独列，如：
  - `num_turns`
  - `valid_traj`
  - `no_loss_on_traj`
  - `tool_call_success`
  - `tool_processing_time_sec`

原理：
- raw json 保证不丢信息
- 平铺字段保证后续统计脚本不用每次 `json.loads()`

##### 3. 对 tool success 做两层口径
- `tool_raw_success = invalid_reason 为空`
- `tool_effective_success = raw_success 且 obs 没有 error marker`

原理：
- `invalid_reason` 更像“接口层有没有显式报错”
- `tool_effective_success` 更像“这次工具调用在语义上到底有没有成功”

##### 4. 用 `attach_step_meta()` 把 step 级 skill meta 回填到所有行
`skills_*.json` 里有：
- `selected_skill_count`
- `selected_skill_names`
- `selection_stats`
- `skill_monitoring_metrics`

这些本来只存在于 skill snapshot 文件里。
脚本会把它们按 step 回填到同一步的 sample / tool_call 行里。

目的：
- 后续做 `某一步 tool error spike 时，当时挂了多少 skill` 这种分析时，不需要再回头读 skills 文件。

##### 5. archive 是可选源
如果有 `skill_store_dir` 或自动推断到了 `_skill_archive/_events.jsonl`：
- 就加 `skill_archive_event` 行
- 否则照样正常跑

这保证了：
- train 能拿到强归因的 death reason
- eval 没 archive 也不会炸

---

### 5.5 `tool_dynamics_metrics.py`：工具侧统计具体怎么算

这个脚本不只是“工具用了多少次”，而是把工具统计分成三层：

#### 1. 全局层
输出在 `tool_dynamics.json`：
- raw success rate
- effective success rate
- obs error rate
- top tools / top invalid reasons
- 工具调用数与 `score / is_correct / total_reward` 的相关性

原理：
- 样本级好坏看最终 `sample`
- 工具级使用情况看 `tool_call`
- 先把 `(step, uid)` 上的工具调用次数聚合出来，再 merge 回 sample 做相关性

#### 2. 逐步层
输出在 `tool_dynamics_by_step.csv`：
- 每步每工具的 `calls`
- `invalid_rate`
- `raw_success_rate`
- `effective_success_rate`
- `obs_error_rate`
- `mean/p50/p90 processing_time_ms`
- `mean_turn_index`

原理：
- 先 group by `(step, tool_name, tool_kind)`
- 再在组内算 rate 和 quantile
- 这样可以直接画每一步的错误率曲线，而不是只看全局平均

#### 3. 失败分解层
输出在 `tool_failure_breakdown.csv`：
- `step × tool × data_source × invalid_reason` 的 `count`
- `share_within_step_tool`
- `share_within_step_tool_datasource`

原理：
- 不只是“哪个 reason 多”，而是“哪个 reason 在哪一步、哪个工具、哪个数据集里激增”

#### 4. 轨迹质量层
输出在：
- `trajectory_quality_by_step.csv`
- `trajectory_quality_summary.json`

主要字段：
- `avg_num_turns`
- `valid_traj_rate`
- `no_loss_on_traj_rate`
- `traj_stop_reason_distribution_json`
- `avg_tool_calls`
- `avg_tool_penalty / avg_latency_penalty / avg_skill_reward`

原理：
- `tool_dynamics` 看的是工具层
- `trajectory_quality` 看的是“整条交互轨迹质量”
- 两者要分开，否则“工具调用失败”和“整条轨迹被 mask 掉”会混在一起

---

### 5.6 `skill_lifecycle_metrics.py`：为什么从 lifecycle 升级成 ecology

旧版本只回答：
- skill 什么时候第一次出现
- 活了多久
- 是否 burst
- 是否 uplift

新版本要回答更完整的问题：
- 它是什么时候 pending create 的？
- 什么时候 promoted 成正式 skill？
- 什么时候第一次被选中？
- 什么时候第一次被成功复用？
- 最后是怎么死掉的？

#### 这个脚本融合了三类证据

##### 1. `skill_snapshot`
回答：
- first selected / last selected
- description
- latest used_times

##### 2. `tool_call`
回答：
- direct skill call 多少次
- run_skill call 多少次
- effective success rate
- obs error rate
- first used / first effective reuse

##### 3. `skill_archive_event`
回答：
- promoted
- validation cleanup
- incorrect trajectory
- periodic reset
- dedup / duplicate existing / fit exit

#### 核心实现原理

##### 1. skill 不再只靠 snapshot 建模
有些 skill：
- 在 eval 中可能没有 `skills_*.json`
- 只在 archive 里出现
- 只在 create_skill / run_skill 调用里出现

所以脚本会先取名字并集：
- snapshot 里的 skill
- skill_new_event 里的 skill
- archive 里的 skill
- tool_call 里的 skill

然后统一做 base table。

##### 2. 生灭原因不是猜，而是优先信任 archive
如果 archive 里有 deleted 事件：
- `death_reason` 直接保留原始 reason
- `death_group` 再映射成论文友好的粗粒度类别

映射示例：
- `validation_cleanup -> validation_cleanup`
- `periodic_step_cleanup -> reset`
- `semantic_dedup -> dedup`
- `incorrect_trajectory -> incorrect_trajectory`

##### 3. 没有 archive 时才退回 snapshot disappearance
如果一个 skill：
- 最后一次出现在 step 10
- 全局最大 step 是 20
- 又没有 archive deleted 事件

就把它记成：
- `death_reason = snapshot_disappearance`

这是一种弱归因，专门为 eval 场景保底。

##### 4. 逐步 ecology 不是简单计数，而是状态演化
`skill_ecology_by_step.csv` 会统计：
- `selected_skill_count`
- `active_skill_count`
- `new_skill_count`
- `promoted_skill_count`
- `deleted_skill_count`
- `deleted_by_reason_*`
- active/new skill 的 run_skill success / unsuccess calls

原理：
- 生命周期是单 skill 视角
- ecology 是 step 视角
- 论文里经常既需要“某个 skill 怎么样”，也需要“那一步 skill 生态发生了什么”

#### 从原始日志到 lifecycle 表：一条完整数据流

这一段很关键，因为 `skill_lifecycle_metrics.py` 不是“只读一个文件做 groupby”，而是把多种来源的 skill 证据拼起来，才能避免遗漏那些“创建了但没进 snapshot”“刚创建就被清理”“只在调用里出现”的 skill。

##### 1. unified 的四类 skill 相关行分别从哪里来

在 `build_unified_dataset.py` 里，和 skill 生命周期最相关的行有四类：

- `skill_new_event`
  - 来源：rollout/validation 原始 `*.jsonl` 里的 `batch_new_skills`
  - 含义：这一条 trajectory 在这个 step 里产出了一个新的 skill 候选
  - 常见字段：`skill_name`、`skill_path`、`created_trajectory_id`、`created_turn`
  - 作用：给 lifecycle 提供“最早 pending create”证据

- `tool_call`
  - 来源：原始 `*.jsonl` 里的 `tool_interact_info`
  - 含义：模型在一条 trajectory 里实际发起了一次工具调用
  - 常见字段：`tool_name`、`tool_kind`、`skill_name`、`invalid_reason`、`tool_obs`、`tool_effective_success`
  - 作用：给 lifecycle 提供“skill 被实际调用/复用过”的证据

- `skill_snapshot`
  - 来源：每一步的 `skills_*.json`
  - 含义：这一步正式进入 selected skill 集合的 skills 快照
  - 常见字段：`skill_name`、`skill_description`、`skill_used_times`
  - 作用：给 lifecycle 提供“第一次进入正式技能池”“最后一次还在池中”“description/used_times”这些信息

- `skill_archive_event`
  - 来源：`_skill_archive/_events.jsonl`
  - 含义：skill store 对某个 skill 做了创建归档、删除归档、原因标注等事件记录
  - 常见字段：`skill_name`、`skill_archive_event`、`skill_archive_reason`、`skill_archive_global_step`
  - 作用：给 lifecycle 提供“什么时候 promoted”“什么时候 deleted”“为什么被删除”这些强归因信息

##### 2. 为什么不能只靠 `skill_snapshot`

如果只拿 `skill_snapshot` 建模，会漏掉几类真实存在过的 skill：

- 新建后很快就被删掉，根本没来得及进入 snapshot
- 只在 archive 中出现，例如被标成 `incorrect_trajectory`、`validation_cleanup`、`dedup`
- 只在调用中出现，例如一次 `run_skill` 成功指向了某个 skill，但该 skill 没出现在当步 snapshot
- eval 场景没有完整 `skills_*.json`，但还有调用日志或 archive 日志

所以 lifecycle 的第一步不是直接算指标，而是先定义“哪些名字应当算作存在过的 skill”。

##### 3. `_build_skill_base()` 是怎么先把 skill 名字全集找出来的

`_build_skill_base()` 先从 `skill_snapshot` 建一个初始 base：

- `first_selected_step`
- `last_selected_step`
- `selected_steps`
- `latest_used_times`
- `description`

这一步只覆盖“进过 snapshot 的正式 skill”。

然后它再做名字并集补全。逻辑上等价于：

- 先收集 snapshot 里的所有 `skill_name`
- 再把 `skill_new_event` 里的 `skill_name` 加进来
- 再把 `skill_archive_event` 里的 `skill_name` 加进来
- 再把经过 `_skill_call_rows()` 规范化之后的 `skill_name_final` 加进来

最后，凡是“出现在这些来源里，但没出现在 snapshot base 里”的名字，都会被补成一行空壳 base 记录：

- `first_selected_step = NaN`
- `last_selected_step = NaN`
- `selected_steps = 0`
- `latest_used_times = NaN`
- `description = ""`

这一步的意义不是说“这些 skill 已经正式入池”，而是说“后续 lifecycle 统计时，不要把这些 skill 彻底漏掉”。

##### 4. `_skill_call_rows()` 到底在做什么

`tool_call` 原始记录里，“skill 是谁”并不总在同一个字段里，所以 `_skill_call_rows()` 做了一层统一解释。

它先只保留两类和 skill 复用有关的调用：

- `run_skill`
  - 这种情况下，`tool_name` 固定就是 `run_skill`
  - 真正的目标 skill 在 `skill_name` 字段里

- direct skill call
  - 这种情况下，调用根本不是通过 `run_skill`
  - `tool_name` 本身就是 skill 名
  - 代码里把“非空、不是 `create_skill`、不是 `run_skill`、也不是 `forever_tool`”的 `tool_name` 视作 direct skill call

所以这段代码：

```python
calls = calls[is_run | is_direct].copy()
calls["call_type"] = "direct"
calls.loc[is_run.loc[calls.index], "call_type"] = "run_skill"
calls["skill_name_final"] = calls["skill_name"].where(
    calls["skill_name"].str.strip() != "",
    calls["tool_name"],
)
calls = calls[calls["skill_name_final"].str.strip() != ""]
```

语义是：

- 先把无关的工具调用过滤掉，只保留 `run_skill` 和 direct skill call
- 默认把这些调用记作 `direct`
- 如果原始 `tool_name == "run_skill"`，就把 `call_type` 改成 `run_skill`
- 统一生成一个最终 skill 名字段 `skill_name_final`
  - 如果 `skill_name` 非空，就优先用 `skill_name`
  - 否则回退到 `tool_name`
- 再把最终 skill 名还是空的记录删掉

这样做完以后，后面的 lifecycle 逻辑就不用区分：

- “这是 `run_skill`，skill 名在 `skill_name`”
- “这是 direct 调用，skill 名藏在 `tool_name`”

它统一只看 `skill_name_final` 即可。

##### 5. 进入主流程后，每个 skill 会再去哪些表里找补充证据

主流程先拿到：

- `base = _build_skill_base(df)`
- `calls = _skill_call_rows(df)`
- `pending_create = skill_new_event + 合法 create_skill tool_call`
- `promoted = archive 中 event=created 且 reason=promoted`
- `deleted = archive 中 event=deleted`

然后对 `base` 里的每一个 `skill_name`，分别切出：

- `s_calls`
  - 这个 skill 的所有实际调用
- `s_pending`
  - 这个 skill 的创建候选记录
- `s_promoted`
  - 这个 skill 的 promoted 记录
- `s_deleted`
  - 这个 skill 的 deleted 记录

接着再汇总出该 skill 的关键生命周期字段。

##### 6. lifecycle 表里的关键字段分别由哪类证据决定

- `first_pending_create_step`
  - 来自 `s_pending["step"]`
  - 代表“最早什么时候进入待创建/待提升状态”
  - 证据源主要是 `skill_new_event`，另外也会吸收合法的 `create_skill` tool call

- `first_promoted_step`
  - 来自 archive 中 `event=created, reason=promoted`
  - 代表“什么时候被正式 promoted”

- `first_selected_step`
  - 来自 `skill_snapshot`
  - 代表“什么时候第一次进入 selected skill 集合”

- `first_used_step`
  - 来自 `s_calls["step"]`
  - 代表“什么时候第一次被实际调用”

- `first_effective_reuse_step`
  - 来自 `tool_effective_success == 1` 的调用
  - 代表“什么时候第一次成功复用”

- `death_step / death_reason / death_group`
  - 优先来自 archive 的 `deleted` 事件
  - 如果 archive 里没有 deleted，但它在中途从 snapshot / 调用证据中消失，就退回为 `snapshot_disappearance`

- `last_seen_step`
  - 综合 `last_selected_step` 和最后一次调用 step
  - 代表“最后一次还能观察到这个 skill 的时间”

##### 7. 一个 skill 可能出现的几条典型路径

最完整的一条路径是：

1. 原始 `jsonl` 的 `batch_new_skills` 里出现这个 skill
2. unified 中生成 `skill_new_event`
3. archive 里出现 `created + promoted`
4. 某一步 `skills_*.json` 把它带进 `skill_snapshot`
5. 后续 trajectory 里通过 `run_skill` 或 direct call 调用它
6. 最后 archive 里出现 `deleted + 某个原因`

但真实数据里常见的路径还有：

- 只创建，没 promoted
  - 会有 `skill_new_event`
  - 可能没有 snapshot
  - 最终可能直接 archive delete

- 只在 archive 里出现
  - 例如外部清理、重置、去重留下了事件
  - snapshot 不一定保留下来

- 只在调用里出现
  - 例如某次 `run_skill` 指向了 skill 名
  - 但当步没有 snapshot，或者 snapshot 数据不完整

- 创建后立刻被判为 `incorrect_trajectory`
  - 有 `skill_new_event`
  - 可能有 archive `deleted`
  - 没有机会进入 `skill_snapshot`
  - 这种就是 `_build_skill_base()` 必须补名字的典型原因

##### 8. 为什么这套设计能避免论文分析里的系统性漏数

如果一个 skill 只有正式入池之后才被统计，那么下列现象会被系统性低估：

- 创建尝试总量
- promoted 前的淘汰率
- `incorrect_trajectory` / `validation_cleanup` 这类早夭 skill 的占比
- archive 清理策略对生态规模的真实影响

而现在的做法是先把“存在过的名字”尽量补全，再分别标注：

- 它有没有进入 snapshot
- 有没有被 promoted
- 有没有被调用
- 有没有成功复用
- 最后怎么死掉

这样 lifecycle 表就不只是在统计“成熟 skill”，而是在统计整个 skill 生态中的出生、晋升、复用和淘汰过程。

---

### 5.7 `skill_semantic_clustering_llm.py`：语义分类怎么做

这个脚本的目标不是“算数值指标”，而是给 skill 一个语义标签。

#### 输入信号
- skill 的 name
- description
- first/last seen
- total calls
- 少量真实调用样例

#### 两种模式

##### `heuristic`
- 用关键词规则打标签
- 例如命中 `ocr / pdf / mineru` 就归到 `ocr_document`

##### `llm`
- 把 name + description + examples 发给模型
- 让模型输出：
  - `category`
  - `subcategory`
  - `confidence`
  - `evidence_sentences`
  - `rationale`

#### 为什么保留 heuristic
因为这条 pipeline 的设计目标是：
- 没网也能跑
- 结果基本可复现

所以 LLM 只是锦上添花，不是硬依赖。

---

### 5.8 `surprise_miner_llm.py`：亮点是怎么被挖出来的

这个脚本有两层：

#### 第一层：确定性候选发现
也就是“先找事实”，包括：
- `tool_adoption_with_non_degrading_performance`
- `failure_mode_improves_over_time`
- `skill_datasource_specialization`
- `abrupt_and_sustained_top_skill_switch`
- `obs_error_largest_drop_window`
- `invalid_reason_largest_drop_window`
- `first_successful_new_skill_reuse`
- `hero_tool`
- `hero_skill`
- `failure_spike_then_recovery`
- `skill_churn_spike`
- `created_but_never_reused_skill`
- `longest_lifespan_positive_uplift_skill`

#### 第二层：改写
- `heuristic` 模式：直接模板化生成 claim
- `llm` 模式：调用模型把候选改写成论文语言

#### 为什么要分两层
因为“发现”必须是确定的、可复现的；
而“表述”可以是 LLM 辅助的。

也就是说：
- LLM 不决定有没有发现
- LLM 只决定怎么把发现写得更像论文

---

### 5.9 `create_skill_by_datasource.py`：为什么它不走 unified

这个脚本刻意直接读 `tool_rollouts_*.json`，原因是它有两个不同粒度的数据：

#### 1. `used_count`
这是每个 step 每个 tool 的全量调用次数。

#### 2. `rollouts`
这是每步最多保留若干条样例。

如果把两者混在 unified 里，很容易误把“样例数”当成“全量调用数”。

所以这个脚本单独保留三种口径：
- `total`
- `correct`
- `valid`

其中 `valid` 的原理是：
- `is_correct=True`
- 且 `invalid_reason is null`

这和训练代码里真正能进入 skill 库的判定逻辑一致。

---

### 5.10 `make_paper_bundle.py`：论文资产为什么放最后生成

这个脚本不是做新统计，而是把前面的结果组织成：
- 图
- 表
- 总览 markdown
- qualitative cases

#### 它会生成什么

##### 图
- `fig_tool_adoption_curve.png`
- `fig_skill_lifecycle_hist.png`
- `fig_tool_success_reward_dual_axis.png`
- `fig_datasource_tool_heatmap.png`
- `fig_obs_error_curve.png`
- `fig_invalid_reason_stacked_area.png`
- `fig_skill_birth_death_timeline.png`
- `fig_trajectory_quality_curve.png`

##### 表
- `table_main.tex`
- `table_ablation.tex`
- `table_top_failure_reasons.tex`
- `table_top_hero_skills.tex`

##### 文本
- `qualitative_cases.md`
- `analysis_overview.md`

#### 原理

##### 1. 图尽量从结果文件读，而不是重复计算
例如：
- failure area 读 `tool_failure_breakdown.csv`
- skill birth/death timeline 读 `skill_ecology_by_step.csv`
- trajectory quality curve 读 `trajectory_quality_by_step.csv`

这样做的好处是：
- 图和数值表共享同一套统计口径
- 避免 bundle 里再次“偷偷重算”导致不一致

##### 2. qualitative cases 优先用 surprise 证据
它会优先取：
- hero tool / hero skill
- failure recovery
- first successful reuse
- cleanup case

也就是先读 `surprises.json` 的 `evidence_refs`，再回到 unified 里反查原始行。

这样 qualitative case 就不是随便抽样，而是和定量结论一一对应。

---

### 5.11 `run_pipeline.py`：总调度器具体做什么

这个脚本的职责很朴素，但非常重要：

#### 1. 读取 `pipeline_config.yaml`
把：
- 输入路径
- 输出路径
- 是否启用某一步
- 每一步参数
- LLM 开关
统一读出来。

#### 2. 做路径检查与目录准备
它会提前创建：
- `analysis_cache`
- `results`
- `paper_bundle`

防止子脚本因为目录不存在而失败。

#### 3. 参数透传
比如现在会把这些参数转给子脚本：
- `skill_store_dir`
- `include_skill_archive`
- `obs_error_markers`
- `top_k_tools`
- `reuse_window`
- `min_effect_size`
- `top_k_plot_items`

#### 4. 统一 resume 逻辑
如果 `runtime.resume=true`：
- 就根据关键输出文件是否存在来跳过步骤
- 避免中断后全量重跑

---

### 5.12 `pipeline_config.yaml`：为什么强调“只改配置”

因为这套脚本的目标用户不是来读源码的，而是来分析一次远程实验日志的。

所以设计原则是：
- 用户只需要改路径和几个阈值
- 然后一条 `python run_pipeline.py` 跑完整套分析

这也是为什么很多参数被前置到配置里，而不是散落在命令行里。

---

### 5.13 测试文件在验证什么

当前目录下的测试主要在做三件事：
- 伪造最小 train 日志，验证 archive / cleanup / obs error / skill ecology 是否正确
- 伪造最小 eval 日志，验证没有 `skills_*.json`、没有 archive 时是否能自动降级
- 跑一次完整 YAML pipeline smoke，验证“只填 config 就能跑通”

这类测试的意义不是验证论文结论，而是验证：
- 日志格式推断是否正确
- schema 是否稳定
- train / eval 的边界条件有没有崩

---

### 5.14 最后总结：每个脚本各自负责什么

你可以把这套代码理解成四层：

#### 第 1 层：日志理解层
- `common.py`
- `build_unified_dataset.py`

负责把“远程训练时写出来的各种日志文件”变成结构化数据。

#### 第 2 层：统计指标层
- `tool_dynamics_metrics.py`
- `skill_lifecycle_metrics.py`
- `create_skill_by_datasource.py`

负责把结构化数据变成可复现的数值指标。

#### 第 3 层：语义与亮点层
- `skill_semantic_clustering_llm.py`
- `surprise_miner_llm.py`

负责把“数字”变成“能写进论文的叙事”。

#### 第 4 层：交付层
- `make_paper_bundle.py`
- `run_pipeline.py`
- `pipeline_config.yaml`

负责把前面的东西组织起来，让用户真正方便地跑、看、交付。
