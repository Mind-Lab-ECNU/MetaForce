# MetaForge

[English README](README.md)

**MetaForge** 是一个自进化多模态智能体框架，用于学习如何按需检索、适配并锻造工具。它面向复杂多模态推理任务：在这些任务中，固定工具库往往难以覆盖全部需求，而不必要的工具调用又会带来额外成本、延迟和噪声。

MetaForge 将工具增强推理建模为一个闭环过程：

- **Decide**：判断当前任务是否需要调用工具。
- **Retrieve**：从动态工具池中检索合适工具。
- **Adapt**：根据任务上下文适配工具参数。
- **Forge**：当现有工具不足时，生成新的可复用技能。
- **Recycle**：将验证后的技能回收到工具池中，供后续任务复用。

本实现基于 VerlTool 和 verl，支持多轮工具交互、GRPO 训练、动态技能管理以及多模态工具执行。

---

## 特性概览

- **自适应工具编排**：在直接回答、调用现有工具、生成新技能之间进行选择。
- **动态工具池**：统一管理可执行工具、模型工具和锻造工具。
- **在线技能锻造**：在交互过程中生成可复用技能，并在验证后注册到工具池。
- **多轮强化学习训练**：支持基于 GRPO 的多模态工具使用轨迹优化。
- **复合奖励设计**：联合监督答案质量、工具选择、参数适配、技能生成、技能复用和输出格式。
- **多模态任务支持**：覆盖视觉问答、OCR、图表/表格推理、文档理解、数学推理和网页生成等任务。

---

## 项目结构

```text
.
├── verl/                         # verl 子模块和训练后端
├── verl_tool/                    # MetaForge 工具智能体实现
│   ├── agent_loop/               # 多轮智能体交互循环
│   ├── servers/tools/            # 工具服务器和工具实现
│   └── workers/reward_manager/   # 工具使用训练的奖励管理器
├── examples/train/               # 训练和评估脚本
│   └── multimodal_orchestra/     # MetaForge 训练/评估配置
├── data_processes_final/         # 数据预处理脚本
├── data_multi_category/          # 多类别数据处理工具
├── eval_service/                 # 评估服务相关工具
├── scripts/                      # 分析和辅助脚本
├── .env.example                  # 环境变量模板
└── requirements.txt              # Python 依赖
```

---

## 安装

### 1. 创建环境

```bash
conda create -n metaforge python=3.10
conda activate metaforge
```

也可以使用 venv：

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

以可编辑模式安装训练后端和项目包：

```bash
cd verl
pip install -e .
cd ..

pip install -e .
```

### 3. 配置环境变量

复制环境变量模板：

```bash
cp .env.example .env
```

编辑 `.env`，只填写当前实验所需的模型服务和工具服务凭证：

```bash
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here
SERP_API_KEY=your_serp_key_here
GOOGLE_MAPS_API_KEY=your_google_maps_key_here
```

请不要将任何凭证硬编码到脚本中。本地密钥文件应排除在版本控制之外。

---

## 数据准备

MetaForge 默认使用 parquet 格式的训练和评估数据。典型样本包含文本 prompt、可选图像字段、答案以及奖励管理器需要的元信息。

示例目录：

```text
data/
├── train.parquet
└── val.parquet
```

训练前设置数据路径：

```bash
export TRAIN_DATA_PATH=/path/to/train.parquet
export VAL_DATA_PATH=/path/to/val.parquet
```

数据预处理相关工具位于：

- `data_processes_final/`
- `data_multi_category/`

---

## 训练

训练脚本位于：

```text
examples/train/multimodal_orchestra/single_node/
```

常用入口：

| 脚本 | 说明 |
| --- | --- |
| `train_8gpu.sh` | 单机 8 卡标准训练配置 |
| `train_4gpu.sh` | 单机 4 卡标准训练配置 |
| `train_8gpu_latency_reward.sh` | 包含延迟相关奖励组件的训练配置 |
| `train_8gpu_think.sh` | 使用 thinking-style 轨迹的训练配置 |
| `train_8gpu_internvl.sh` | 面向 InternVL 类主干模型的训练配置 |
| `train_8gpu_ablation_qwen2_5.sh` | 面向 Qwen2.5-VL 类主干模型的消融配置 |

### 示例：单机 8 卡训练

```bash
export TRAIN_DATA_PATH=/path/to/train.parquet
export VAL_DATA_PATH=/path/to/val.parquet

bash examples/train/multimodal_orchestra/single_node/train_8gpu.sh
```

启动前请根据本地环境检查并修改脚本中的：

- `model_name`：基础模型路径或 Hugging Face 模型名称
- 数据路径：`TRAIN_DATA_PATH`、`VAL_DATA_PATH`
- GPU 相关配置：GPU 数量、张量并行大小、显存利用率
- 日志后端和输出目录

训练产物默认写入 `verl_step_records/`：

```text
verl_step_records/
├── tool_logs/train/<RUN_ID>/
├── skills/train/<RUN_ID>/
├── checkpoint/train/<RUN_ID>/
└── rollout/train/<RUN_ID>/
```

---

## 评估

评估脚本位于：

```text
examples/train/multimodal_orchestra/
```

示例：

```bash
export CHECKPOINT_PATHS=/path/to/checkpoint
export VAL_DATA_PATH=/path/to/test.parquet

bash examples/train/multimodal_orchestra/eval_batch_by_ckpt.sh
```

项目中还提供了面向 CLEVR 类推理、AI2D、MathVista 等任务的评估脚本，可按实验需求选择。

---

## 工具与技能

MetaForge 支持三类工具：

1. **可执行工具**：用于代码执行、符号计算和图像处理的确定性程序。
2. **模型工具**：用于 OCR、视觉定位、文档解析、分割和图像编辑等任务的外部模型或服务。
3. **锻造工具**：智能体在交互过程中生成的可复用技能。

锻造技能通常保存在当前实验对应的技能目录下：

```text
verl_step_records/skills/train/<RUN_ID>/
```

每个技能可以包含规范说明、schema 和可执行实现文件。

---

## 配置建议

- 优先从提供的单机脚本开始，只修改路径和资源配置。
- 如果显存不足，优先降低 batch size、序列长度或显存利用率。
- 工具和模型服务凭证统一放在 `.env` 或环境变量中。
- 每个实验使用独立输出目录，避免检查点、日志和技能库混在一起。
- 调试工具行为时，优先查看 tool logs，用于定位工具选择、参数适配或执行失败问题。

---

## 说明

- 本仓库包含研究代码和实验脚本，路径、数据位置和模型名称需要根据本地环境调整。
- MetaForge 的正式引用信息将在公开版本可用后补充。
- 本项目基于 verl 和 VerlTool 生态实现。

---

## License

请参考基础项目及所用第三方依赖的许可协议。
