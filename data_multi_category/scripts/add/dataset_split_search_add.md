# add 目录数据集的 test/val split 检索（HF/ModelScope 优先）

时间：2026-01-23

说明：本次只整理 HuggingFace / ModelScope 上“别人已处理过的”版本；若仅有 train 或仅有 test/val，也会标注出来供你判断是否可用。未写“字段”的条目通常是数据集页面未能正常展示字段（或需授权）。

---

## 一、已找到 **明确包含 test/val** 的 HF 数据集（优先候选）

### OCR-VQA
- HF：howard-hou/OCR-VQA
- Split：train / validation / test
- 主要字段（viewer）：image, image_id, questions, answers, ocr_tokens, ocr_info, title, authorName, genre, image_url, set_name

### TextVQA
- HF：lmms-lab/textvqa
- Split：train / validation / test
- 主要字段（viewer）：image_id, question_id, question, image, answers, ocr_tokens 等

### TextCaps
- HF：lmms-lab/TextCaps
- Split：train / val / test
- 主要字段（viewer）：question_id, question, image, image_id, caption_str, reference_strs 等

### FinQA
- HF：n3Er/FinQA-Infix
- Split：train / validation / test
- 主要字段（README）：id, pre_text, post_text, table 等

### TAT-QA
- HF：TableQAKit/TAT-QA
- Split：train / validation / test（viewer 可见，但解析失败）

### HiTab
- HF：kasnerz/hitab
- Split：train / test / validation（在 README 中明确说明）

### MultiHiertt
- HF：yilunzhao/MultiHiertt
- Split：train / validation / test（viewer 可见，但解析失败）

### VQAv2
- HF：HuggingFaceM4/VQAv2
- Split：train / val / test（数据集页面列出了官方分割统计；viewer 被禁用）

### VSR（Visual Spatial Reasoning）
- HF：albertvillanova/visual-spatial-reasoning
- Split：提供 random 与 zero-shot 两种划分；均含 train / dev / test

### VQA-RAD
- HF：dz-osamu/VQA-RAD
- Split：train / test

### Hateful Memes
- HF：neuralcatcher/hateful_memes
- Split：train / dev_seen / test_seen / dev_unseen / test_unseen

### TallyQA
- HF：VDebugger/tallyqa
- Split：train / validation / test（viewer 提示数据生成报错，但 split 名称可见）

### DaTikZ
- HF：nllg/datikz
- Split：train / test

### WebSight（替代候选）
- HF：haidark1/WebSightDescribed
- Split：train / valid / test
- 说明：该版本是 WebSight 的“带自然语言描述”的处理版本，字段包含 image/html/description。

---

## 二、只有 test 或 val 的 HF 数据集（可作评测但不完整）

### OK-VQA
- HF：lmms-lab/OK-VQA
- Split：仅 val2014（单一 val split）

### ST-VQA
- HF：lmms-lab/ST-VQA
- Split：仅 test

---

## 三、只找到 train（或集合中仅 train）

> 这些数据多数在 FineVision 系列合集里有 **train-only** 版本；若需要 test/val，仍需继续外部查找。

- COCOQA（FineVision_images/text 中只有 train）
- Chart2Text（FineVision_images/text 中只有 train）
- DVQA（HF 单独数据集 DavidNguyen/DVQA 为 train-only；FineVision 也为 train-only）
- Diagram-Image-to-Text（Kamizuru00/diagram_image_to_text：train-only）
- Screen2Words（Leonardo6/screen2words：train-only）
- WebSight（FineVision 里为 train-only，但可用 WebSightDescribed 作为替代）
- GeoMVerse / InterGPS / Localized Narratives / Raven / Rendered_Text / Robut_* / VisText / TQA / IAM / Visual7W / VisualMRC 等：目前仅在 FineVision_images/text 中看到 train-only

---

## 四、需要协议或页面未给出 split（暂无法确认）

- VisualMRC（NTT-hil-insight/VisualMRC）：页面需同意协议，未展示 split

---

## 五、ModelScope 结果

本轮主要在 HF 找到可用 split；ModelScope 仍需补查（尚未发现可直接用的 test/val 版本）。

