# Multimodal Orchestra 训练指南

本目录包含 Multimodal Orchestra 的训练与评测脚本，支持单节点和多节点运行。

## 目录结构

```
multimodal_orchestra/
├── single_node/
│   ├── train_8gpu.sh              # 单节点8卡训练脚本 (自动启动工具服务器)
│   └── eval_8gpu.sh               # 单节点8卡评测脚本 (val-only)
│   ├── train_8gpu_no_tool.sh      # 单节点8卡训练脚本 (不启动工具服务器)
│   ├── eval_8gpu_no_tool.sh       # 单节点8卡评测脚本 (val-only, 不启动工具服务器)
│   ├── train_8gpu_with_retry.sh   # 单节点8卡训练重试wrapper
│   └── eval_8gpu_with_retry.sh    # 单节点8卡评测重试wrapper
├── multi_node/
│   ├── train_2node_16gpu.sh       # 多节点训练脚本 (自动启动工具服务器)
│   ├── train_2node_16gpu_no_tool.sh  # 多节点训练脚本 (不启动工具服务器)
│   ├── eval_2node_16gpu.sh        # 多节点评测脚本 (val-only, 自动启动工具服务器)
│   ├── eval_2node_16gpu_no_tool.sh # 多节点评测脚本 (val-only, 不启动工具服务器)
│   ├── train_2node_16gpu_with_retry.sh  # 多节点训练重试wrapper
│   ├── eval_2node_16gpu_with_retry.sh   # 多节点评测重试wrapper
│   └── start_tool_server.sh       # [可选] 独立工具服务器启动脚本
└── README.md                      # 本文档

仓库根目录下另有工具机保活脚本:
- `scripts/tool_io_copy_delete_loop.sh`  (高频IO拷贝/删除循环)
- `scripts/tool_cpu_guard_50.sh`         (CPU占用守护，目标 >50%)
```

### 脚本说明

| 脚本 | 启动工具服务器 | 使用场景 |
|------|--------------|---------|
| `single_node/train_8gpu.sh` | 自动启动 | 单节点训练，一键运行 |
| `single_node/eval_8gpu.sh` | 自动启动 | 单节点评测（仅验证） |
| `single_node/train_8gpu_no_tool.sh` | 不启动 | 单节点训练，工具服务器外置 |
| `single_node/eval_8gpu_no_tool.sh` | 不启动 | 单节点评测（仅验证），工具服务器外置 |
| `single_node/train_8gpu_with_retry.sh` | 跟随模式 | 单节点训练失败自动重试（成功一次即停止） |
| `single_node/eval_8gpu_with_retry.sh` | 跟随模式 | 单节点评测失败自动重试（成功一次即停止） |
| `multi_node/train_2node_16gpu.sh` | 自动启动 | 多节点训练，一键运行 |
| `multi_node/train_2node_16gpu_no_tool.sh` | 不启动 | 训练，工具服务器外置 |
| `multi_node/eval_2node_16gpu.sh` | 自动启动 | 多节点评测（仅验证） |
| `multi_node/eval_2node_16gpu_no_tool.sh` | 不启动 | 评测，工具服务器外置 |
| `multi_node/train_2node_16gpu_with_retry.sh` | 跟随模式 | 多节点训练失败自动重试（成功一次即停止） |
| `multi_node/eval_2node_16gpu_with_retry.sh` | 跟随模式 | 多节点评测失败自动重试（成功一次即停止） |
| `multi_node/start_tool_server.sh` | - | [可选] 独立启动工具服务器 |

## Train / Eval 解耦说明

- 训练脚本默认关闭训练期验证：`trainer.test_freq=-1`
- 评测脚本使用纯验证模式：`trainer.val_only=True`
- 训练脚本支持可选 baseline 起点评测：`VAL_BEFORE_TRAIN=True` 时只在训练前执行一次验证
- `multi_node/train_2node_16gpu.sh` 需要先在脚本里填写 `TRAIN_DATA_PATH` 和 `VAL_DATA_PATH`
- `single_node/train_8gpu.sh` 也需要先在脚本里填写 `TRAIN_DATA_PATH` 和 `VAL_DATA_PATH`
- `single_node/eval_8gpu.sh` 和 `multi_node/eval_2node_16gpu.sh` 需要先在脚本里填写 `EVAL_TRAIN_DATA_PATH`、`EVAL_DATA_PATH`
- `EVAL_TOOL_VARIANT` 只决定 runtime 工具池：`all/id/ood`

### Skill Store 规则

- 训练脚本：默认使用 `${ROOT_DIR}/verl_step_records/skills/train/${RUN_ID}`
- 评测脚本：
  - 若设置 `SKILL_STORE_DIR`，则使用你指定的目录
  - 若未设置 `SKILL_STORE_DIR`，则冷启动空目录 `${ROOT_DIR}/verl_step_records/skills/val/${RUN_ID}`

### RUN_ID 规则（防覆盖）

- 默认：`RUN_ID=${RUN_TS}_${run_name}`（时间戳在前）
- 可选：手动导出 `RUN_ID` 复用固定实验标识
- 训练与评测输出目录按 phase 区分（`train/` 与 `val/`），避免互相覆盖

## 数据集配置

### 训练数据 (20个数据集)

**ID核心 (10个，用于训练和测试)**:
| 序号 | 数据集 | 类别 |
|------|--------|------|
| 1 | ChartQA_2000 | Chart |
| 2 | PlotQA_2000 | Chart |
| 3 | AI2D_2000 | Diagram |
| 4 | MapQA_2000 | Geospatial |
| 5 | geos_processed_2000 | Geospatial |
| 6 | unigeon_2000 | Geospatial |
| 7 | GEOQA_2000 | Math |
| 8 | geometry3k_2000 | Math |
| 9 | ScienceQA_2000 | Science |
| 10 | CLEVR_2000 | Spatial |

**ID辅助 (10个，仅用于训练)**:
| 序号 | 数据集 | 类别 |
|------|--------|------|
| 1 | LocalizedNarratives_2000 | Caption |
| 2 | DVQA_2000 | Chart |
| 3 | WebSight_2000 | Code |
| 4 | DiagramImageToText_2000 | Diagram |
| 5 | AOKVQA_2000 | General |
| 6 | VQAv2_2000 | General |
| 7 | InterGPS_2000 | Math |
| 8 | TextVQA_2000 | OCR |
| 9 | TATQA_2000 | Table |
| 10 | InfographicVQA_2000 | Doc |

### 测试数据 (20个数据集)

**ID测试 (10个)**: 与ID核心相同的数据集

**OOD测试 (10个)**:
| 序号 | 数据集 | 类别 | OOD原因 |
|------|--------|------|---------|
| 1 | ICON-QA_2000 | Diagram | 抽象图标 |
| 2 | TABMWP_2000 | Chart | 表格+数学组合 |
| 3 | CLEVR_MATH/addition | Math | CLEVR+算术 |
| 4 | CLEVR_MATH/subtraction | Math | CLEVR+算术 |
| 5 | CLEVR_MATH/subtraction_multihop | Math | 多跳推理 |
| 6 | CLEVR_MATH/adversarial | Math | 对抗样本 |
| 7 | MathVision_2000 | Math | 综合数学视觉 |
| 8 | MathVista_2000 | Math | 综合数学基准 |
| 9 | DocVQA_2000 | Doc | 文档理解 |
| 10 | HatefulMemes_2000 | General | 仇恨检测 |

### 数据目录

服务器数据根目录:
```
/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data
```

## 单节点训练 (8 GPUs)

### 快速启动

```bash
cd /path/to/verl_duo
先打开 `single_node/train_8gpu.sh`，填写：
- `TRAIN_DATA_PATH`
- `VAL_DATA_PATH`

bash examples/train/multimodal_orchestra/single_node/train_8gpu.sh
```

可选 baseline 起点评测（训练前仅一次验证）：
```bash
VAL_BEFORE_TRAIN=True bash examples/train/multimodal_orchestra/single_node/train_8gpu.sh
```

## 单节点评测 (8 GPUs)

运行前先打开脚本，填写：
- `EVAL_TRAIN_DATA_PATH`
- `EVAL_DATA_PATH`
- `EVAL_TOOL_VARIANT`

然后执行：
```bash
cd /path/to/verl_duo
bash examples/train/multimodal_orchestra/single_node/eval_8gpu.sh
```

指定已有技能目录进行评测：
```bash
SKILL_STORE_DIR=/path/to/previous/skills \
bash examples/train/multimodal_orchestra/single_node/eval_8gpu.sh
```

### 工作流程

1. 脚本自动启动工具服务器（当前默认 `python -m verl_tool.servers.serve` 启动；`uvicorn --factory` 代码块保留可切换）
2. 启动训练/评测任务
3. 任务完成后自动清理

### 关键配置

- **模型**: Qwen/Qwen3-VL-8B-Instruct
- **batch_size**: 128
- **n_gpus_per_node**: 8
- **reward_manager**: multimodal_orchestra
- **训练 tool_type**: `multimodal_processor_tool_adapt_skill_id`
- **评测 tool_type**:
  - `all -> multimodal_processor_tool_adapt_skill`
  - `id -> multimodal_processor_tool_adapt_skill_id`
  - `ood -> multimodal_processor_tool_adapt_skill_ood`

## 多节点训练 (2 Nodes x 8 GPUs = 16 GPUs)

### 方式一：一键启动（推荐）

使用 `train_2node_16gpu.sh`，脚本会自动在 head 节点启动工具服务器。

这份多节点训练脚本现在要求先在脚本里填写训练和验证路径：
- `TRAIN_DATA_PATH`
- `VAL_DATA_PATH`

然后直接运行：
```bash
bash examples/train/multimodal_orchestra/multi_node/train_2node_16gpu.sh
```

如果要做训练前 baseline：
```bash
VAL_BEFORE_TRAIN=True \
bash examples/train/multimodal_orchestra/multi_node/train_2node_16gpu.sh
```

**在 Head 节点 (假设 IP 为 192.168.1.100):**
```bash
cd /path/to/verl_duo
NODE_RANK=0 bash examples/train/multimodal_orchestra/multi_node/train_2node_16gpu.sh
```

**在 Worker 节点:**
```bash
cd /path/to/verl_duo
NODE_RANK=1 MASTER_ADDR=192.168.1.100 bash examples/train/multimodal_orchestra/multi_node/train_2node_16gpu.sh
```

### 方式二：分离工具服务器

如果工具服务器已在外部启动（例如在单独的节点/容器中），使用 `train_2node_16gpu_no_tool.sh`。

**步骤1: 启动工具服务器（任选一种方式）**

```bash
# 方式A: 使用独立脚本启动
cd /path/to/verl_duo
TOOL_SERVER_PORT=5000 bash examples/train/multimodal_orchestra/multi_node/start_tool_server.sh

# 方式B: 手动启动 (uvicorn)
export VT_TOOL_TYPE="multimodal_processor_tool_adapt_skill"
uvicorn verl_tool.servers.tool_server:create_app --factory --host 0.0.0.0 --port 5000 --workers 32

# 方式C: 手动启动 (传统方式)
python -m verl_tool.servers.serve --host 0.0.0.0 --port 5000 --tool_type "multimodal_processor_tool_adapt_skill" --workers_per_tool 32
```

**步骤2: 启动训练（指定工具服务器 URL）**

```bash
# Head 节点 (假设工具服务器在 192.168.1.50:5000)
NODE_RANK=0 TOOL_SERVER_URL=http://192.168.1.50:5000/get_observation \
    bash examples/train/multimodal_orchestra/multi_node/train_2node_16gpu_no_tool.sh

# Worker 节点
NODE_RANK=1 MASTER_ADDR=192.168.1.100 TOOL_SERVER_URL=http://192.168.1.50:5000/get_observation \
    bash examples/train/multimodal_orchestra/multi_node/train_2node_16gpu_no_tool.sh
```

### 方式对比

| 方式 | 脚本 | 适用场景 |
|------|------|---------|
| 一键启动 | `train_2node_16gpu.sh` | 简单场景，工具服务器与 head 节点同机 |
| 分离启动 | `train_2node_16gpu_no_tool.sh` | 工具服务器在单独节点，或多任务共享 |

## 多节点评测 (2 Nodes x 8 GPUs = 16 GPUs)

### 方式一：一键评测（自动启动工具服务器）

Head 节点：
```bash
NODE_RANK=0 bash examples/train/multimodal_orchestra/multi_node/eval_2node_16gpu.sh
```

Worker 节点：
```bash
NODE_RANK=1 MASTER_ADDR=192.168.1.100 bash examples/train/multimodal_orchestra/multi_node/eval_2node_16gpu.sh
```

### 方式二：外置工具服务器评测

Head 节点：
```bash
NODE_RANK=0 TOOL_SERVER_URL=http://192.168.1.50:5000/get_observation \
  bash examples/train/multimodal_orchestra/multi_node/eval_2node_16gpu_no_tool.sh
```

Worker 节点：
```bash
NODE_RANK=1 MASTER_ADDR=192.168.1.100 TOOL_SERVER_URL=http://192.168.1.50:5000/get_observation \
  bash examples/train/multimodal_orchestra/multi_node/eval_2node_16gpu_no_tool.sh
```

### 环境变量说明

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `NODE_RANK` | 节点编号 (0=head, 1+=worker) | 0 |
| `MASTER_ADDR` | Head 节点 IP | 本机 IP (head) / 必须设置 (worker) |
| `MASTER_PORT` | Ray head 端口 | 6379 |
| `DASHBOARD_PORT` | Ray dashboard 端口 | 8265 |
| `TOOL_SERVER_HOST` | 一键脚本中工具服务器主机 (`*_2node_16gpu.sh`) | `MASTER_ADDR` |
| `TOOL_SERVER_PORT` | 一键脚本内部随机端口（`30000-31000`，不可通过环境变量固定） | 随机 |
| `TOOL_SERVER_URL` | 工具服务器完整URL (train_2node_16gpu_no_tool.sh) | 必须设置 |
| `TOOL_SERVER_WORKERS` | 仅 `start_tool_server.sh` 使用的 worker 数量 | 32 |
| `TOOL_TYPE` | 工具类型 | multimodal_processor_tool_adapt_skill |

### 多节点注意事项

1. **工具服务器只需启动一次**: 所有节点共享同一个工具服务器实例
2. **确保网络连通**: 所有节点需要能访问工具服务器 URL
3. **Ray 已内置**: 不需要额外安装，verl 依赖中已包含
4. **先启动 head**: 确保 head 节点先启动，worker 再加入

## 工具服务器说明

### 启动方式

工具服务器有两种启动方式，不依赖 Ray：

#### 方式1: uvicorn --factory 模式 (推荐，支持多 worker；`start_tool_server.sh` 默认)

配置通过环境变量传入：

```bash
# 设置工具类型和其他配置
export VT_TOOL_TYPE="multimodal_processor_tool_adapt_skill"
export VT_HOST="0.0.0.0"
export VT_PORT="5000"
export VT_WORKERS_PER_TOOL=32
export VT_MAX_CONCURRENT_REQUESTS=1024
export VT_LOG_LEVEL="info"

# 启动服务器
uvicorn verl_tool.servers.tool_server:create_app \
    --factory \
    --host 0.0.0.0 \
    --port 5000 \
    --workers 32 \
    --loop uvloop \
    --http httptools
```

**重要**: `VT_TOOL_TYPE` 环境变量必须设置，否则会使用默认的 "base" 工具。

#### 方式2: 传统 python -m 启动 (备用)

如果 uvicorn/uvloop 有问题，可以使用传统方式：

```bash
python -m verl_tool.servers.serve \
    --host 0.0.0.0 \
    --port 5000 \
    --tool_type "multimodal_processor_tool_adapt_skill" \
    --workers_per_tool 32 \
    --max_concurrent_requests 1024
```

在训练脚本中切换启动方式：
- 单节点脚本: 注释/取消注释对应的启动代码块
- `start_tool_server.sh`: 设置 `USE_TRADITIONAL=1` 环境变量

### 健康检查

```bash
curl http://<host>:5000/health
```

### 调用接口

训练时使用的 URL:
```
http://<host>:5000/get_observation
```

## 常见问题

### 1. 工具服务器启动失败

检查端口是否被占用:
```bash
netstat -tuln | grep 5000
```

杀死占用进程:
```bash
fuser -k 5000/tcp
```

### 2. Ray 集群连接问题

检查 Ray 状态:
```bash
ray status
```

重启 Ray:
```bash
ray stop --force
ray start --head --port=6379  # head 节点
ray start --address=<head_ip>:6379  # worker 节点
```

### 3. NCCL 通信问题

设置调试信息:
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

### 4. OOM (内存不足)

调整以下参数:
- 降低 `gpu_memory_utilization` (如 0.7)
- 降低 `batch_size`
- 开启 `do_offload=True`

## 监控

### WandB

训练默认使用 WandB 记录日志:
- project_name: multimodal_orchestra
- experiment_name: 自动生成

### Ray Dashboard

多节点训练时可通过 Ray Dashboard 监控:
```
http://<head_ip>:8265
```

## 训练参数参考

| 参数 | 单节点 | 多节点 | 说明 |
|------|--------|--------|------|
| batch_size | 128 | 256 | 多节点可增大 |
| n_gpus_per_node | 8 | 8 | GPU数量 |
| n_nodes | 1 | 2 | 节点数量 |
| max_prompt_length | 8192 | 8192 | 最大提示长度 |
| max_response_length | 24576 | 24576 | 最大响应长度 |
| tensor_model_parallel_size | 2 | 4 | 张量并行度 |
| total_training_steps | 100 | 100 | 总训练步数 |
| save_freq | 10 | 20 | 保存频率 |
| test_freq | -1 | -1 | 训练阶段关闭周期验证 |

## 脚本改造明细（逐文件）

以下内容对应本轮实际改造，按脚本逐个说明“新增了什么”。

### A. 多节点原始脚本（保留默认自启动工具服务）

#### 1) `multi_node/train_2node_16gpu.sh`

新增内容:
- 新增计时变量:
  - `SCRIPT_START_EPOCH`
  - `JOB_START_EPOCH`
  - `JOB_END_EPOCH`
- 新增计时函数:
  - `format_duration_hms()`
  - `print_timing_summary()`
  - `on_exit()`
- 新增 `trap on_exit EXIT`:
  - 无论成功/失败，都会打印:
    - `[TIME] job_elapsed_sec=... job_elapsed_hms=...`
    - `[TIME] script_elapsed_sec=... script_elapsed_hms=...`
    - `[EXIT] exit_code=...`
- 训练主命令前后新增:
  - `JOB_START_EPOCH=$(date +%s)`
  - `job_exit_code` 捕获 `ray job submit` 返回值
  - `JOB_END_EPOCH=$(date +%s)`
- 清理阶段新增:
  - 成功打印 `Training completed successfully...`
  - 失败打印 `Training failed with exit code ...`
- 退出码透传:
  - head 节点 `exit $job_exit_code`
  - worker 节点 `wait $ray_pid` 后 `worker_exit_code=$?`，最终 `exit $worker_exit_code`

保持不变（按你的要求）:
- 该脚本始终由 head 节点自启动 tool server（不会读取外部 `TOOL_SERVER_URL` 决定是否跳过）。

#### 2) `multi_node/eval_2node_16gpu.sh`

新增内容与 `train_2node_16gpu.sh` 同结构:
- 同一套 3 个计时变量
- 同一套 3 个计时函数 + `trap on_exit EXIT`
- 评测主命令前后记录 `JOB_START_EPOCH/JOB_END_EPOCH`
- 捕获 `job_exit_code`
- 成功/失败分支日志
- head/worker 退出码透传

保持不变（按你的要求）:
- 该脚本也始终由 head 节点自启动 tool server，不依赖外部 `TOOL_SERVER_URL`。

### B. 多节点 no_tool 脚本

#### 3) `multi_node/train_2node_16gpu_no_tool.sh`

新增内容:
- 新增计时变量、计时函数、`trap on_exit EXIT`（同上）
- 新增训练主命令返回码捕获 `job_exit_code`
- 清理阶段新增成功/失败日志
- head/worker 显式退出码透传

原有语义保持:
- 仍要求外部传入 `TOOL_SERVER_URL`
- 仍不在脚本内启动 tool server

#### 4) `multi_node/eval_2node_16gpu_no_tool.sh`

新增内容:
- 新增计时变量、计时函数、`trap on_exit EXIT`
- 新增评测主命令返回码捕获 `job_exit_code`
- 清理阶段新增成功/失败日志
- head/worker 显式退出码透传

原有语义保持:
- 仍要求 `TOOL_SERVER_URL`
- 仍不自启动 tool server

### C. 多节点重试 wrapper（新增）

#### 5) `multi_node/train_2node_16gpu_with_retry.sh`

新增文件，功能:
- 支持 `MODE=auto|bundled|external`
  - `auto`: 有 `TOOL_SERVER_URL` 走 `train_2node_16gpu_no_tool.sh`，否则走 `train_2node_16gpu.sh`
- 支持参数:
  - `RETRY_SLEEP_SEC`（默认 30）
  - `HEALTH_CHECK_INTERVAL_SEC`（默认 30）
  - `HEALTH_FAIL_THRESHOLD`（默认 3）
  - `HEALTH_GRACE_SEC`（默认 180）
  - `RETRY_LOG_DIR`（默认 `logs/multimodal_orchestra_multinode/retry`）
- 失败判定:
  - 子脚本退出非 0
  - 或（仅 head，`NODE_RANK=0`）连续健康检查失败达到阈值
- 健康检查实现:
  - 调 `ray status`，解析 GPU 总量；不可用或为 0 记一次失败
- 双机协同:
  - head 做全局健康判定
  - worker 仅跟随重启（不做全局 Ray 判定）
- 成功即停:
  - 任何一次 attempt `exit_code=0`，wrapper 立即 `exit 0`，不再重试

#### 6) `multi_node/eval_2node_16gpu_with_retry.sh`

新增文件，结构与训练 wrapper 一致:
- 同样的 `MODE` 三态与参数
- 同样的 head 主控 / worker 跟随逻辑
- 同样的 Ray 健康检查
- 同样“成功一次即停止”

### D. 单节点原始脚本（默认自启动工具服务）

#### 7) `single_node/train_8gpu.sh`

新增内容:
- 新增计时变量、计时函数、`trap on_exit EXIT`
- 训练主命令前后新增 `JOB_START_EPOCH/JOB_END_EPOCH`
- 新增 `job_exit_code` 捕获
- 清理阶段按成功/失败分别打印
- 脚本结尾 `exit $job_exit_code`

原有语义保持:
- 仍在脚本内启动 tool server（默认模式）

#### 8) `single_node/eval_8gpu.sh`

新增内容:
- 新增计时变量、计时函数、`trap on_exit EXIT`
- 评测主命令前后新增 `JOB_START_EPOCH/JOB_END_EPOCH`
- 新增 `job_exit_code` 捕获
- 清理阶段按成功/失败分别打印
- 脚本结尾 `exit $job_exit_code`

原有语义保持:
- 仍在脚本内启动 tool server（默认模式）

### E. 单节点 no_tool 脚本（新增）

#### 9) `single_node/train_8gpu_no_tool.sh`

新增文件，来源于 `train_8gpu.sh` 的 no_tool 变体，改动如下:
- 文件头与用法改为“外置工具服务”
- 新增强校验:
  - 若未设置 `TOOL_SERVER_URL` 直接报错退出
- 移除脚本内工具服务启动逻辑:
  - 删除 host/port 随机端口配置
  - 删除 `python -m verl_tool.servers.serve ...` 启动段
  - 删除等待工具服务启动与 `server_pid` 存活检查
- 改为:
  - `tool_server_url="$TOOL_SERVER_URL"`
  - 打印 `Using external Tool Server URL: ...`
- 保留计时与退出码透传能力（与 train_8gpu.sh 一致）

#### 10) `single_node/eval_8gpu_no_tool.sh`

新增文件，来源于 `eval_8gpu.sh` 的 no_tool 变体，改动如下:
- 文件头与用法改为“外置工具服务”
- 未设置 `TOOL_SERVER_URL` 时直接报错退出
- 移除脚本内工具服务启动、等待、PID 检查逻辑
- 改为使用外部 `tool_server_url="$TOOL_SERVER_URL"`
- 保留计时与退出码透传能力（与 eval_8gpu.sh 一致）

### F. 单节点重试 wrapper（新增）

#### 11) `single_node/train_8gpu_with_retry.sh`

新增文件，功能:
- 支持 `MODE=auto|bundled|external`
  - `bundled` -> `train_8gpu.sh`
  - `external` -> `train_8gpu_no_tool.sh`（要求 `TOOL_SERVER_URL`）
- 支持参数:
  - `RETRY_SLEEP_SEC`（默认 30）
  - `HEALTH_CHECK_INTERVAL_SEC`（默认 30）
  - `HEALTH_FAIL_THRESHOLD`（默认 3）
  - `HEALTH_GRACE_SEC`（默认 180）
  - `ENABLE_GPU_HEALTH_CHECK`（默认 1）
  - `RETRY_LOG_DIR`（默认 `logs/multimodal_orchestra_single_node/retry`）
- GPU 健康检查:
  - 使用 `nvidia-smi --query-gpu=memory.used`
  - 汇总显存使用量
  - 连续多次为 0 视为异常，强杀子进程并重试
- 成功即停:
  - 任一 attempt 返回 0，wrapper 立刻退出，不再重试

#### 12) `single_node/eval_8gpu_with_retry.sh`

新增文件，结构与训练 wrapper 一致:
- 同样支持 `MODE=auto|bundled|external`
- 同样支持 GPU 健康检查与阈值参数
- 同样“成功一次即停止”

### G. 工具机保活脚本（新增，位于仓库根目录 `scripts/`）

#### 13) `scripts/tool_io_copy_delete_loop.sh`

新增文件，功能:
- 源目录默认:
  - `/inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/model/Qwen3-VL-32B-Instruct`
- 目标根目录默认:
  - 当前 `pwd`（可由 `IO_WORK_ROOT` 覆盖）
- 目标子目录命名:
  - 读取 `hostname -I`（兼容 `hostname -i/hostname` 兜底）
  - 空格转下划线并做安全字符过滤
- 循环逻辑:
  - `cp -a $SOURCE_DIR $TARGET_DIR`
  - `rm -rf $TARGET_DIR`
  - 无限往复
- 安全护栏:
  - 仅允许删除 `${TARGET_BASE}` 范围内目标路径
- 信号处理:
  - `SIGINT/SIGTERM/EXIT` 时清理目标目录

#### 14) `scripts/tool_cpu_guard_50.sh`

新增文件，功能:
- 目标:
  - 动态维持 CPU 占用在 `CPU_TARGET_MIN`~`CPU_TARGET_MAX`（默认 50~70）
- 默认参数:
  - `CPU_TARGET_MIN=50`
  - `CPU_TARGET_MAX=70`
  - `CPU_SAMPLE_SEC=2`
  - `CPU_MAX_WORKERS=CPU核心数`
- 实现方式:
  - 通过 `/proc/stat` 采样总 CPU 使用率
  - 动态增减 busy-loop 子进程数量
- 进程管理:
  - `spawn_worker` 启 busy 进程
  - `kill_one_worker` / `kill_all_workers` 做回收
  - 收到退出信号时清理全部子进程

## Retry Wrapper 速查

### 多节点训练 wrapper

```bash
# Head
NODE_RANK=0 MODE=bundled \
  bash examples/train/multimodal_orchestra/multi_node/train_2node_16gpu_with_retry.sh

# Worker
NODE_RANK=1 MASTER_ADDR=<head_ip> MODE=bundled \
  bash examples/train/multimodal_orchestra/multi_node/train_2node_16gpu_with_retry.sh
```

### 多节点评测 wrapper

```bash
# Head
NODE_RANK=0 MODE=bundled \
  bash examples/train/multimodal_orchestra/multi_node/eval_2node_16gpu_with_retry.sh

# Worker
NODE_RANK=1 MASTER_ADDR=<head_ip> MODE=bundled \
  bash examples/train/multimodal_orchestra/multi_node/eval_2node_16gpu_with_retry.sh
```

### 单节点训练/评测 wrapper

```bash
# 训练 bundled
MODE=bundled bash examples/train/multimodal_orchestra/single_node/train_8gpu_with_retry.sh

# 训练 external
MODE=external TOOL_SERVER_URL=http://<tool_host>:<tool_port>/get_observation \
  bash examples/train/multimodal_orchestra/single_node/train_8gpu_with_retry.sh

# 评测 bundled
MODE=bundled bash examples/train/multimodal_orchestra/single_node/eval_8gpu_with_retry.sh

# 评测 external
MODE=external TOOL_SERVER_URL=http://<tool_host>:<tool_port>/get_observation \
  bash examples/train/multimodal_orchestra/single_node/eval_8gpu_with_retry.sh
```
