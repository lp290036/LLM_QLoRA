# LLM_QLoRA

本项目是一个面向 **联邦学习 / 分布式机器学习问答场景** 的 Qwen3.5-2B QLoRA 微调工程。工程包含训练数据、小规模测试集、QLoRA 训练脚本、基础模型对话脚本、LoRA 对话脚本，以及微调前后自动评估脚本。

> 注意：GitHub 仓库只保存必要代码、配置和小数据文件。基础模型权重、LoRA 训练输出、缓存、日志和聊天记录不会上传，需要在本地按路径准备。

## 项目目标

- 使用本地 `Qwen3.5-2B` 基座模型进行 4-bit QLoRA 微调。
- 将模型适配到联邦学习、分布式机器学习、通信压缩、优化算法等垂直领域问答。
- 通过 LoRA 低秩适配器控制训练成本，降低小数据集微调对通用能力的破坏。
- 提供基础模型与 LoRA 模型的交互式对话入口，方便人工对比效果。
- 提供离线评估脚本，对比微调前后在测试集上的 PPL、关键词命中率、重复率和生成长度。

## 目录结构

```text
.
├── FL_question.jsonl                  # 主训练集，JSONL 格式
├── README.md                          # 工程说明文档
└── qwen3.5-2b/
    ├── unsloth_train.py               # QLoRA 微调主脚本
    ├── run_chat.py                    # 基础模型交互式对话
    ├── run_chat_lora.py               # 加载 LoRA 后的交互式对话
    ├── eval_compare.py                # 微调前后自动评估对比
    ├── win_safetensors_patch.py       # Windows safetensors 无 mmap 加载补丁
    ├── fl_test.jsonl                  # 小规模评估测试集
    ├── chat_template.jinja            # Qwen chat template
    ├── config.json                    # 基座模型配置
    ├── configuration.json             # 附加配置文件
    ├── preprocessor_config.json       # 处理器配置
    ├── video_preprocessor_config.json # 视频预处理配置
    ├── LICENSE                        # Qwen 模型许可证
    └── README.md                      # 上游 Qwen 模型卡
```

未纳入 Git 的本地文件主要包括：

- `qwen3.5-2b/*.safetensors`：基础模型权重。
- `qwen3.5-2b/unsloth_output/`：训练输出和 LoRA adapter。
- `qwen3.5-2b/unsloth_compiled_cache/`、`unsloth_compiled_cache/`：Unsloth / Torch 编译缓存。
- `qwen3.5-2b/chat_logs/`：交互式对话日志。
- `*.log`、`__pycache__/`：运行日志和 Python 缓存。

## 环境要求

建议在 Windows + NVIDIA GPU 环境运行，脚本中已有针对 Windows 的 safetensors 加载补丁和缓存路径规避。

基础依赖包括：

```bash
python
pytorch
cuda
unsloth
transformers
datasets
trl
peft
bitsandbytes
```

推荐使用独立 Conda 环境，例如：

```powershell
conda activate unsloth
```

脚本目前使用本机绝对路径：

```text
D:\LLM\models\qwen3.5-2b
D:\LLM\models\FL_question.jsonl
D:\LLM\models\qwen3.5-2b\unsloth_output
```

如果项目放在其他目录，需要同步修改 `unsloth_train.py`、`run_chat.py`、`run_chat_lora.py`、`eval_compare.py` 中的路径常量。

## 数据格式

训练集 `FL_question.jsonl` 和测试集 `qwen3.5-2b/fl_test.jsonl` 均采用 JSONL，每行一个样本。

训练样本主要字段：

```json
{
  "instruction": "问题或指令",
  "input": "可选上下文",
  "output": "期望回答"
}
```

测试样本可额外包含：

```json
{
  "category": "领域或通用分类",
  "keywords": ["关键词1", "关键词2"]
}
```

训练脚本会将 `instruction + input` 作为用户输入，将 `output` 作为 assistant 答案，并通过 tokenizer 的 chat template 生成 SFT 文本。

## 代码文件说明

### `qwen3.5-2b/unsloth_train.py`

QLoRA 微调主入口。核心职责：

- 设置 Windows 运行环境变量，关闭部分异步加载和编译行为，降低 Windows 下的崩溃概率。
- 将 Torch / Triton 缓存放到 `D:\llm_cache`，规避中文用户名路径导致的编译或编码问题。
- 在导入模型加载逻辑前应用 `win_safetensors_patch.apply()`，避免 safetensors mmap 访问冲突。
- 使用 `FastLanguageModel.from_pretrained(..., load_in_4bit=True)` 以 4-bit 量化方式加载本地 Qwen3.5-2B 基座模型。
- 通过 `FastLanguageModel.get_peft_model()` 挂载 LoRA adapter。
- LoRA 目标模块包括注意力层和 MLP 线性层：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`。
- 从 `FL_question.jsonl` 读取训练样本，转换为 chat template 格式。
- 将数据按 `VAL_RATIO = 0.10` 划分训练集和验证集。
- 使用 `SFTTrainer` 执行监督微调，优化器为 `adamw_8bit`，学习率调度为 `cosine`。
- 通过 `train_on_responses_only()` 只对 assistant 回复部分计算 loss，避免模型学习 system/user 提示文本。
- 保存 LoRA adapter 和 tokenizer 到 `qwen3.5-2b/unsloth_output/final`。
- 保存训练过程 loss / eval_loss 到 `qwen3.5-2b/unsloth_output/loss_history.json`。
- 支持 `--eval` 参数，在训练后对 `fl_test.jsonl` 做一次生成测试。

主要超参数：

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `MAX_SEQ_LENGTH` | `1280` | 训练最大上下文长度 |
| `LORA_R` | `16` | LoRA 低秩维度 |
| `LORA_ALPHA` | `16` | LoRA 缩放参数 |
| `LORA_DROPOUT` | `0.05` | 抑制小数据集过拟合 |
| `NUM_EPOCHS` | `2` | 训练轮数 |
| `LEARNING_RATE` | `1e-4` | 学习率 |
| `BATCH_SIZE` | `1` | 单卡 batch size |
| `GRAD_ACCUM` | `8` | 梯度累积步数，等效 batch 为 8 |
| `NEFTUNE_ALPHA` | `5` | NEFTune 噪声强度 |
| `VAL_RATIO` | `0.10` | 验证集比例 |

运行：

```powershell
cd D:\LLM\models\qwen3.5-2b
conda activate unsloth
python unsloth_train.py
```

训练并在结束后做简单推理检查：

```powershell
python unsloth_train.py --eval
```

### `qwen3.5-2b/run_chat.py`

基础模型交互式对话入口。核心职责：

- 加载未微调的本地 Qwen3.5-2B 基座模型。
- 使用 4-bit 量化降低显存占用。
- 使用与训练脚本一致的 system prompt，保证微调前后对比相对公平。
- 维护内存中的多轮对话历史。
- 支持 `clear` 清空上下文，支持 `exit` / `quit` / `q` 退出。
- 将对话记录追加写入 `qwen3.5-2b/chat_logs/chat_history_base.txt`。

运行：

```powershell
cd D:\LLM\models\qwen3.5-2b
conda activate unsloth
python run_chat.py
```

### `qwen3.5-2b/run_chat_lora.py`

LoRA 微调后模型的交互式对话入口。核心职责：

- 先加载同一个本地 Qwen3.5-2B 基座模型。
- 再从 `qwen3.5-2b/unsloth_output/final` 加载 LoRA adapter。
- 使用与训练脚本完全一致的 system prompt。
- 交互逻辑、采样参数和日志逻辑与 `run_chat.py` 基本一致。
- 将对话记录追加写入 `qwen3.5-2b/chat_logs/chat_history_lora.txt`。

运行前需要先完成训练并生成 LoRA 权重：

```powershell
python unsloth_train.py
python run_chat_lora.py
```

### `qwen3.5-2b/eval_compare.py`

微调前后自动评估脚本。核心职责：

- 加载基础模型，并挂载 `unsloth_output/final` 中的 LoRA adapter。
- 在同一个模型对象中通过启用 / 禁用 adapter layers 对比 base 与 LoRA 输出。
- 对 `fl_test.jsonl` 中每个样本分别生成 base 回答和 LoRA 回答。
- 使用 greedy decoding，即 `do_sample=False`，减少采样随机性。
- 计算参考答案 PPL，越低表示模型对参考答案越“熟悉”。
- 计算生成长度、关键词命中率、重复 n-gram 比例和生成耗时。
- 汇总 base / LoRA 的平均指标，并保存完整结果到 `qwen3.5-2b/unsloth_output/eval_results.json`。

输出结果结构包括：

- `meta`：测试文件、LoRA 路径、解码方式、时间戳。
- `summary`：base 与 LoRA 的平均 PPL、领域 PPL、通用 PPL、关键词命中率等。
- `base`：基础模型逐题结果。
- `lora`：LoRA 模型逐题结果。

运行：

```powershell
cd D:\LLM\models\qwen3.5-2b
conda activate unsloth
python eval_compare.py
```

### `qwen3.5-2b/win_safetensors_patch.py`

Windows 下的 safetensors 加载补丁。核心职责：

- 替换 `transformers.modeling_utils.safe_open`。
- 避免使用 safetensors 默认 mmap 读取方式。
- 通过普通文件 I/O 的 `seek + read` 按需读取 tensor。
- 降低 Windows 环境下因 mmap、内存紧张或访问冲突导致的 `0xC0000005` 崩溃概率。
- 该补丁需要在 `from_pretrained()` 之前调用。

这个文件不是训练逻辑本身，但对当前 Windows 本地环境的稳定运行很关键。

## 推荐工作流

### 1. 准备本地模型

确保本地存在 Qwen3.5-2B 模型配置和权重：

```text
D:\LLM\models\qwen3.5-2b
```

GitHub 仓库不包含大模型权重，因此新机器 clone 后需要自行下载或复制模型权重文件。

### 2. 运行基础模型对话

```powershell
cd D:\LLM\models\qwen3.5-2b
conda activate unsloth
python run_chat.py
```

用于确认基座模型可正常加载和生成。

### 3. 执行 QLoRA 微调

```powershell
python unsloth_train.py
```

训练完成后会生成：

```text
D:\LLM\models\qwen3.5-2b\unsloth_output\final
```

该目录包含 LoRA adapter 和 tokenizer 文件，但不会提交到 Git。

### 4. 运行 LoRA 对话

```powershell
python run_chat_lora.py
```

用于人工检查垂直领域回答质量。

### 5. 自动评估微调效果

```powershell
python eval_compare.py
```

评估结果保存到：

```text
D:\LLM\models\qwen3.5-2b\unsloth_output\eval_results.json
```

## 当前工程限制

- 路径是硬编码的 Windows 绝对路径，不是跨机器即用配置。
- 基础模型权重和 LoRA 输出未上传，clone 后不能直接运行推理或训练。
- 训练脚本依赖 CUDA GPU；无 CUDA 环境会直接退出。
- 部分源文件中的中文注释存在编码显示异常，但 Python 语法检查通过，不影响当前解释的代码结构。
- 当前数据集是小规模垂直领域数据，评估结果更适合作为定向效果参考，不应视为通用大模型能力评测。

## Git 上传策略

`.gitignore` 已排除大文件和运行产物，避免将模型权重或缓存提交到 GitHub：

```gitignore
*.safetensors
*.bin
qwen3.5-2b/unsloth_output/
unsloth_compiled_cache/
*/unsloth_compiled_cache/
__pycache__/
*.log
chat_logs/
```

提交代码时建议继续使用显式文件路径 staging，避免误上传大模型文件。
