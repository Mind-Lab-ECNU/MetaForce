# 多模态数据集 split 整理（更新：2026-01-24）

## Huggingface结论总览（按你的问题顺序）

1）TableSenseAI/TabMWP 不符合我们现有数据形态，已排除。

2）achang/plot_qa 不是我们的问题场景，已排除。

3）jinaai/plotqa
- 这是 PlotQA 的测试子集（公开说明为从 PlotQA 测试集抽样）。
- 与我们脚本使用的 HuggingFaceM4/the_cauldron:plotqa 不相同：Cauldron 是训练集合集，仅提供训练 split；而 jinaai/plotqa 为测试 split 子集。两者至少在 split 上不一致，且字段结构也不同。
- **代码验证（2026-01-24）**：
  - HuggingFaceM4/the_cauldron:plotqa：仅 train split，157,070 images，字段结构为 `{"images": [...], "texts": [{"user": ..., "assistant": ...}]}`
  - jinaai/plotqa：仅 test split，1,000 rows，字段结构不同
  - The Cauldron 文档明确说明只收录 train split 并进行了去污染处理
- 结论：**确认为不同数据集/split**，进入候选测试集。

4）lmms-lab/ICON-QA
- 该数据集在 HF 上提供 val/test，无 train。
- 我们的训练数据来自 HuggingFaceM4/the_cauldron:iconqa（仅训练 split）。
- **代码验证（2026-01-24）**：
  - HuggingFaceM4/the_cauldron:iconqa：仅 train split，27,315 images
  - lmms-lab/ICON-QA：仅 val/test split，42,977 rows（val 可见）
  - The Cauldron 文档明确说明只收录 train split
- 结论：**确认为不同数据集/split**，作为测试集候选可用。

5）dali-does/clevr-math
- HF 上有两个子集配置：general 与 multihop。
- 一般理解：general 为完整配置；multihop 为多跳子集配置（仅多跳题型）。目前 HF 页面未给出更细的差异说明，只能确认二者为不同配置。
- 脚本检查：data_multi_category/scripts/math/prepare_clevr_math_2000.py
  - for split_name, split_dataset in dataset.items(): 这里循环的是 DatasetDict 里的各个 split（通常是 train/validation/test）。
  - 脚本只调用 load_dataset(数据集路径) 的默认配置，等价于 general；因此不能直接处理 multihop（除非改参数/脚本或另开配置入口）。
- 结论：general 与 multihop 可以作为两个后续处理目标，但需要区分配置分别处理。

6）hz2475/geoQA 只有 train split（HF 仅此一条），暂时没用。

7）nimapourjafar/mm_mapqa 只有 train split，暂时没用。

8）AI2D 相关
- lmms-lab/ai2d 与 lmms-lab/ai2d-no-mask 都是 HF 上的 test split 数据集；页面未给出卡片说明。
- 两者名称暗示“是否带 mask”的图像差异，但缺少官方说明与统计；从页面只能确认：都有 test split、规模与字段结构看起来一致。
- 我们的训练脚本 /Users/albertmin/train/verl_duo/data_multi_category/scripts/diagram/prepare_ai2d_2000.py 使用的是 HuggingFaceM4/the_cauldron:ai2d（训练合集，只有 train split），与 lmms-lab/ai2d 系列至少在 split 上不同，且字段结构也不同。
- lhndzn/AI2D 在 HF 上只看到单一 split（train），页面同样无说明；是否与 lmms-lab/ai2d 的 test 重复，无法仅凭页面判断，需要样本级比对。
- **代码验证（2026-01-24）**：
  - HuggingFaceM4/the_cauldron:ai2d：仅 train split，3,099 images，字段结构为 `{"images": [...], "texts": [{"user": ..., "assistant": ...}]}`
  - lmms-lab/ai2d：仅 test split，3,088 rows，lmms-eval 格式
  - lmms-lab/ai2d-no-mask：仅 test split，3,088 rows，图像不带 mask
  - lhndzn/AI2D：仅 train split，3,088 rows，CSV 格式
  - **注意**：lhndzn/AI2D 的 train（3,088）与 lmms-lab 的 test（3,088）数据量相同，可能为同源数据的不同 split 标注
- 结论：**确认 HuggingFaceM4/the_cauldron:ai2d 与 lmms-lab/ai2d 系列为不同 split**；lmms-lab/ai2d 与 lmms-lab/ai2d-no-mask 均进入候选测试集；lhndzn/AI2D 不能直接视为可拼接的 train+test，需要样本级去重/对齐确认。

## 进入候选的HuggingFace测试集（可优先考虑）

- PlotQA：jinaai/plotqa（测试子集）
- ICON-QA：lmms-lab/ICON-QA（val/test）
- CLEVR-Math：dali-does/clevr-math（general 与 multihop 两个配置）
- AI2D：lmms-lab/ai2d 与 lmms-lab/ai2d-no-mask（test）

## 补充调研结果（2026-01-24）

以下数据集已找到可用的测试集来源：

### MapQA ✅ 已找到
- **官方 GitHub**: https://github.com/OSU-slatelab/MapQA
- **下载链接**: [Google Drive](https://drive.google.com/drive/folders/12n6gjTpFuBlc-ibKhIE-F8GMgBWkSHfv)
- **数据格式**: JSON + 图像文件
- **Ground Truth**: ✅ train/dev/test 均有答案
- **规模**: ~800K 问答对，~60K 地图图像
- **说明**: 包含三个子集 MapQA-U/R/S，每个子集包含 `train-QA.json`、`dev-QA.json`、`test-QA.json`

### GeoQA ✅ 已找到
- **官方 GitHub**: https://github.com/chen-judge/GeoQA
- **下载链接**: [Google Drive](https://drive.google.com/drive/folders/1fiLTJUq7EPiZHs6AxundNfNEDLw4gtP5?usp=sharing)
- **数据格式**: `.pk` (pickle 格式)
- **Ground Truth**: ✅ 有
- **扩展版 UniGeo**: https://github.com/chen-judge/UniGeo（包含 proving_test.pk, proving_train.pk, proving_val.pk）

### FigureQA ✅ 已找到
- **官方项目页**: https://www.microsoft.com/en-us/research/project/figureqa-dataset/
- **下载链接**: [Microsoft Research](https://www.microsoft.com/en-hk/download/details.aspx?id=100635)
- **GitHub (代码生成)**: https://github.com/Maluuba/FigureQA
- **数据格式**: JSON + 图像
- **Ground Truth**: ✅ Validation 有答案 (Yes/No 二元答案)
- **说明**: 包含5种图表类型，15种问题类型

### A-OKVQA ⚠️ Test 无答案，Validation 可用
- **官方 GitHub**: https://github.com/allenai/aokvqa
- **官方网站**: http://a-okvqa.allenai.org/
- **下载命令**: `curl -fsSL https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz | tar xvz`
- **Leaderboard**: https://leaderboard.allenai.org/a-okvqa/submissions/public
- **Ground Truth**: ❌ Test 答案不公开需提交 Leaderboard；✅ **Validation 有完整答案可用于本地评测**
- **规模**: ~25K 问题
- **说明**: 需配合 COCO 2017 图像使用

### DocVQA ✅ 已找到
- **官方网站**: https://www.docvqa.org/
- **RRC 竞赛页面**: https://rrc.cvc.uab.es/?ch=17&com=downloads
- **HuggingFace (推荐)**: [lmms-lab/DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA)
- **Ground Truth**: ✅ Validation 有答案；Test 需提交评测
- **lmms-lab 版本说明**: validation (5.35k 行) 和 test 分割，总计 16,626 行，可直接用于评测

### MathVista ⚠️ testmini 有答案，test 需提交
- **官方项目页**: https://mathvista.github.io/
- **官方 GitHub**: https://github.com/lupantech/MathVista
- **HuggingFace**: [AI4Math/MathVista](https://huggingface.co/datasets/AI4Math/MathVista)
- **数据格式**: JSON + 图像
- **规模**: 6,141 样本 (testmini: 1,000 + test: 5,141)
- **Ground Truth**: ✅ **testmini (1,000 样本) 有完整答案**，用于模型开发/验证；❌ test (5,141 样本) 答案不公开，需邮件提交评测

### MathVision (MATH-V) ✅ 已找到
- **官方项目页**: https://mathllm.github.io/mathvision/
- **官方 GitHub**: https://github.com/mathllm/MATH-V
- **HuggingFace**: [MathLLMs/MathVision](https://huggingface.co/datasets/MathLLMs/MathVision)
- **数据格式**: JSONL + 图像
- **规模**: 3,040 个高质量数学问题
- **Ground Truth**: ✅ 有完整答案
- **说明**: 来自真实数学竞赛题目，覆盖 16 个数学学科，5 个难度等级

---

## 总结对比表

| 数据集 | Test 有答案 | Val 有答案 | 推荐下载来源 |
|--------|------------|-----------|--------------|
| MapQA | ✅ | ✅ | Google Drive |
| GeoQA | ✅ | ✅ | Google Drive |
| FigureQA | ⚠️ 需确认 | ✅ | Microsoft Research |
| A-OKVQA | ❌ | ✅ | AWS S3 / GitHub |
| DocVQA | ❌ 需提交 | ✅ | lmms-lab HuggingFace |
| MathVista | ❌ 需提交 | ✅ (testmini) | AI4Math HuggingFace |
| MathVision | ✅ | ✅ | MathLLMs HuggingFace |

---

## 推荐：统一评测框架

如需统一管理这些评测集，推荐使用 **VLMEvalKit**：
- GitHub: https://github.com/open-compass/VLMEvalKit
- 支持 70+ 评测基准，包括上述多个数据集
- 一键评测多个 VLM 模型
- 自动下载和处理数据集

## Qwen3-VL 的 benchmark 测试集（来自官方模型卡性能图）

Qwen/Qwen3-VL-8B-Instruct 的模型卡里给出了详细评测集合列表（以图表形式呈现，未给下载链接）。可见的测试集包括：

- STEM 与推理：MMMUval、MMMUpro_full、MathVista、MathVision、MathVerse、ZERO Bench (Math) 、ZERO Bench (Spatial)
- 通用 VQA：MMBenchDEV_EN_V1.1、RealWorldQA、MMStar、SimpleVQA
- 主观体验与指令跟随：HallusionBench、MM-MT-Bench、MIA-Bench
- 文字识别与图表/文档：MMLongBench-Doc、DocVQA_TEST、InfoVQA_TEST、AI2D_TEST、OCRBench、OCRBench_v2 (en/zh)、CC-OCRBench-overall、ChartXiv (DQ/RQ)
- 2D/3D Grounding：ODinW13、ARKitScenes、Hypersim、SUNRGBD
- 多图推理：BLINK、MUIRBench
- 具身与空间：ERQA、VSI-Bench、EmbSpatialBench、RefSpatialBench、RoboSpatialHome
- 视频：MVBench、VideoMME (w/o sub)、MVU-MCQ、LVBench、Charades-STA、Video-MME-MU
- 代理能力：ScreenSpot、ScreenSpot Pro、OSWorldG、AndroidWorld、OSWorld
- 细粒度视觉：V*、HRBench4K、HRBench8K

说明：这是模型卡里的评测清单，并非数据下载入口。后续若要对齐这些评测，需要分别寻找公开的测试集发布渠道。