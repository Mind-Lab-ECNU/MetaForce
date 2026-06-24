# MetaForge

[中文文档](README_CN.md)

**MetaForge** is a self-evolving multimodal agent framework that learns to retrieve, adapt, and forge tools on demand. It is designed for multimodal reasoning tasks where a fixed tool inventory is often insufficient and unnecessary tool calls can introduce extra cost, latency, and noise.

MetaForge decomposes tool-augmented reasoning into a closed-loop process:

- **Decide** whether tool use is necessary.
- **Retrieve** suitable tools from a dynamic tool pool.
- **Adapt** tool calls to the current task context.
- **Forge** new reusable skills when existing tools are not sufficient.
- **Recycle** validated skills back into the tool pool for future use.

The implementation is built on top of VerlTool and verl, with support for multi-turn tool interaction, GRPO training, dynamic skill management, and multimodal tool execution.

---

## Highlights

- **Adaptive tool orchestration**: choose between answering directly, using existing tools, or creating new skills.
- **Dynamic tool pool**: supports executable tools, model tools, and forged tools under a unified interface.
- **Online skill forging**: synthesizes reusable skills during interaction and registers validated skills for later reuse.
- **Multi-turn RL training**: uses GRPO-style optimization for multimodal tool-use trajectories.
- **Composite reward design**: jointly supervises answer quality, tool selection, parameter grounding, skill creation, tool reuse, and output format.
- **Multimodal benchmark support**: covers visual question answering, OCR, chart/table reasoning, document understanding, math reasoning, and webpage generation tasks.

---

## Repository Structure

```text
.
├── verl/                         # verl submodule and training backend
├── verl_tool/                    # MetaForge tool-agent implementation
│   ├── agent_loop/               # Multi-turn agent interaction loops
│   ├── servers/tools/            # Tool server and tool implementations
│   └── workers/reward_manager/   # Reward managers for tool-use training
├── examples/train/               # Training and evaluation scripts
│   └── multimodal_orchestra/     # MetaForge training/evaluation recipes
├── data_processes_final/         # Data preprocessing scripts
├── data_multi_category/          # Multi-category data utilities
├── eval_service/                 # Evaluation service utilities
├── scripts/                      # Analysis and helper scripts
├── .env.example                  # Environment variable template
└── requirements.txt              # Python dependencies
```

---

## Installation

### 1. Create an environment

```bash
conda create -n metaforge python=3.10
conda activate metaforge
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Install the training backend and project package in editable mode:

```bash
cd verl
pip install -e .
cd ..

pip install -e .
```

### 3. Configure environment variables

Copy the template:

```bash
cp .env.example .env
```

Edit `.env` and fill in only the credentials required by your selected tools and model services:

```bash
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here
SERP_API_KEY=your_serp_key_here
GOOGLE_MAPS_API_KEY=your_google_maps_key_here
```

Do not hard-code credentials in scripts. Keep local secret files out of version control.

---

## Data Preparation

MetaForge expects training and evaluation data in parquet format. Typical records include text prompts, optional image references or encoded image fields, answers, and metadata required by the reward manager.

Example layout:

```text
data/
├── train.parquet
└── val.parquet
```

Set paths before training:

```bash
export TRAIN_DATA_PATH=/path/to/train.parquet
export VAL_DATA_PATH=/path/to/val.parquet
```

Data preprocessing utilities are available in:

- `data_processes_final/`
- `data_multi_category/`

---

## Training

Training scripts are provided under:

```text
examples/train/multimodal_orchestra/single_node/
```

Common entry points:

| Script | Description |
| --- | --- |
| `train_8gpu.sh` | Standard single-node 8-GPU training recipe |
| `train_4gpu.sh` | Standard single-node 4-GPU training recipe |
| `train_8gpu_latency_reward.sh` | Training recipe with latency-aware reward components |
| `train_8gpu_think.sh` | Training recipe with thinking-style trajectories |
| `train_8gpu_internvl.sh` | Training recipe for InternVL-style backbones |
| `train_8gpu_ablation_qwen2_5.sh` | Ablation recipe for Qwen2.5-VL-style backbones |

### Example: single-node 8-GPU training

```bash
export TRAIN_DATA_PATH=/path/to/train.parquet
export VAL_DATA_PATH=/path/to/val.parquet

bash examples/train/multimodal_orchestra/single_node/train_8gpu.sh
```

Before launching, check the script and adjust the following fields for your environment:

- `model_name`: base model path or Hugging Face model identifier
- data paths: `TRAIN_DATA_PATH`, `VAL_DATA_PATH`
- GPU-related settings: number of GPUs, tensor parallel size, memory utilization
- logging backend and output directories

Training artifacts are written under `verl_step_records/` by default:

```text
verl_step_records/
├── tool_logs/train/<RUN_ID>/
├── skills/train/<RUN_ID>/
├── checkpoint/train/<RUN_ID>/
└── rollout/train/<RUN_ID>/
```

---

## Evaluation

Evaluation scripts are available under:

```text
examples/train/multimodal_orchestra/
```

Example:

```bash
export CHECKPOINT_PATHS=/path/to/checkpoint
export VAL_DATA_PATH=/path/to/test.parquet

bash examples/train/multimodal_orchestra/eval_batch_by_ckpt.sh
```

Additional benchmark-specific scripts are provided for tasks such as CLEVR-style reasoning, AI2D, and MathVista.

---

## Tools and Skills

MetaForge supports three categories of tools:

1. **Executable tools**: deterministic programs for code execution, symbolic computation, and image processing.
2. **Model tools**: external models or services for OCR, visual grounding, document parsing, segmentation, and image editing.
3. **Forged tools**: reusable skills synthesized by the agent during interaction.

Forged skills are stored under the run-specific skill directory, usually:

```text
verl_step_records/skills/train/<RUN_ID>/
```

Each skill can contain a specification, schema, and executable implementation files.

---

## Configuration Tips

- Start with the provided single-node scripts and modify only paths and resource settings first.
- Reduce batch size or sequence length if GPU memory is insufficient.
- Keep tool credentials in `.env` or environment variables.
- Use a separate output directory for each experiment to avoid mixing checkpoints and skill stores.
- Inspect tool logs when debugging tool selection, parameter grounding, or execution failures.

---

## Notes

- This repository contains research code and experiment scripts. Paths, data locations, and model identifiers should be adapted to your local environment.
- Citation information for MetaForge will be updated when an official public version is available.
- This project builds on the verl and VerlTool ecosystems.

---

## License

Please refer to the license terms of the base project and all third-party dependencies used by your experiments.
