# -*- coding: utf-8 -*-
"""基于 Unsloth 的 Qwen3.5-2B QLoRA 微调脚本（联邦学习/分布式机器学习领域）。

设计目标：在 6GB 显卡上对垂直领域数据做 LoRA 微调，同时通过参数配置
最大限度防止「灾难性遗忘」（即微调后丢失通用对话与基础能力）。

用法：
    conda activate unsloth
    python unsloth_train.py            # 训练 + 保存 LoRA
    python unsloth_train.py --eval     # 训练并用测试集做一次推理对比
"""

# ── 必须在导入 transformers 之前先导入 unsloth ──────────────────────────
import os
# 关闭 transformers 并行异步权重加载，降低 Windows 下 mmap 瞬时提交峰值
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

import unsloth  # noqa: F401  (必须最先导入以应用补丁)
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

import sys
import json
import time
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Windows 无 mmap 加载补丁（规避 safetensors mmap 访问冲突 0xC0000005 + 省内存）──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import win_safetensors_patch
win_safetensors_patch.apply()

# ══════════════════════════════════════════════════════════════════════
#  路径配置
# ══════════════════════════════════════════════════════════════════════
MODEL_DIR  = r"D:\LLM\models\qwen3.5-2b"
TRAIN_FILE = r"D:\LLM\models\FL_question.jsonl"
TEST_FILE  = r"D:\LLM\models\qwen3.5-2b\fl_test.jsonl"
OUTPUT_DIR = r"D:\LLM\models\qwen3.5-2b\unsloth_output"
FINAL_DIR  = os.path.join(OUTPUT_DIR, "final")

# ══════════════════════════════════════════════════════════════════════
#  超参数（含防遗忘配置说明）
# ══════════════════════════════════════════════════════════════════════
MAX_SEQ_LENGTH = 1280      # 答案约 300 字（≈450 token）+ 提示，1280 足够且省显存

# —— LoRA 配置 ——
# 防遗忘要点1：仅训练低秩旁路，冻结全部原始权重 → 天然保护基座能力
# 防遗忘要点2：r 与 alpha 取相同值（缩放系数=1），更新幅度温和，不强行覆盖原分布
LORA_R        = 16
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.05       # 轻微 dropout，抑制对小数据集的过拟合

# —— 训练配置 ——
# 防遗忘要点3：少轮次 + 低学习率 + cosine 衰减 + warmup，避免在 125 条小样本上过拟合
NUM_EPOCHS          = 2
LEARNING_RATE       = 1e-4
WARMUP_RATIO        = 0.10
WEIGHT_DECAY        = 0.01     # 正则化，约束权重漂移
BATCH_SIZE          = 1
GRAD_ACCUM          = 8        # 等效 batch = 8
# 防遗忘要点4：NEFTune 给嵌入加噪声，提升泛化、降低过拟合
NEFTUNE_ALPHA       = 5
SEED                = 3407

SYSTEM_PROMPT = (
    "你是联邦学习与分布式机器学习领域的专业研究员，"
    "请针对问题给出准确、深入、结构清晰的回答。"
)


# ══════════════════════════════════════════════════════════════════════
#  数据加载
# ══════════════════════════════════════════════════════════════════════
def load_jsonl(path: str) -> list[dict]:
    """加载标准 JSONL（FL_question.jsonl 已修复为合法 JSON）。"""
    records = []
    bad = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            bad.append(lineno)
    if bad:
        print(f"  ⚠ 跳过无法解析的行: {bad}", flush=True)
    return records


def build_dataset(tokenizer, samples: list[dict]) -> Dataset:
    """把 instruction/input/output 转成带 chat 模板的纯文本训练样本。"""
    texts = []
    for s in samples:
        instruction = s.get("instruction", "").strip()
        context     = s.get("input", "").strip()
        answer      = s.get("output", s.get("response", "")).strip()
        if not instruction or not answer:
            continue
        user_content = f"{instruction}\n{context}".strip() if context else instruction
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": answer},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        texts.append(text)
    print(f"  有效训练样本: {len(texts)} 条", flush=True)
    return Dataset.from_dict({"text": texts})


# ══════════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="训练后用测试集做推理对比")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("无 CUDA，请检查 GPU 驱动与 PyTorch 安装。")

    # ── 1. 加载 4-bit 量化模型 ──────────────────────────────────────
    print("加载模型（4-bit 量化）...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = MODEL_DIR,
        max_seq_length  = MAX_SEQ_LENGTH,
        dtype           = None,      # 自动选 bf16/fp16
        load_in_4bit    = True,
        full_finetuning = False,
    )
    used  = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"模型就绪 | 显存 {used:.2f}/{total:.1f} GB", flush=True)

    # ── 2. 挂载 LoRA 适配器 ─────────────────────────────────────────
    # 防遗忘要点5：只挂注意力 + MLP 的线性层，不训练 embed_tokens / lm_head，
    #             保留词表与输出头不变，避免破坏通用语言能力
    print("挂载 LoRA 适配器 ...", flush=True)
    model = FastLanguageModel.get_peft_model(
        model,
        r                          = LORA_R,
        lora_alpha                 = LORA_ALPHA,
        lora_dropout               = LORA_DROPOUT,
        target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",
        random_state               = SEED,
        use_rslora                 = False,
        loftq_config               = None,
    )

    # ── 3. 构建数据集 ───────────────────────────────────────────────
    print(f"\n加载训练集：{TRAIN_FILE}", flush=True)
    samples = load_jsonl(TRAIN_FILE)
    print(f"  共 {len(samples)} 条原始样本", flush=True)
    train_dataset = build_dataset(tokenizer, samples)

    # ── 4. 配置 Trainer ─────────────────────────────────────────────
    trainer = SFTTrainer(
        model           = model,
        tokenizer       = tokenizer,
        train_dataset   = train_dataset,
        args = SFTConfig(
            dataset_text_field          = "text",
            max_seq_length              = MAX_SEQ_LENGTH,
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUM,
            warmup_ratio                = WARMUP_RATIO,
            num_train_epochs            = NUM_EPOCHS,
            learning_rate               = LEARNING_RATE,
            weight_decay                = WEIGHT_DECAY,
            lr_scheduler_type           = "cosine",
            optim                       = "adamw_8bit",
            neftune_noise_alpha         = NEFTUNE_ALPHA,
            logging_steps               = 1,
            seed                        = SEED,
            output_dir                  = OUTPUT_DIR,
            save_strategy               = "no",
            report_to                   = "none",
            dataset_num_proc            = 1,   # Windows 必须单进程
        ),
    )

    # ── 5. 只对「助手回答」部分计算 loss（不学习提示）──────────────────
    # 防遗忘要点6：屏蔽 system/user 部分的 loss，只学「如何作答」，
    #             不去记忆问题文本，保护指令理解能力
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part    = "<|im_start|>assistant\n",
    )

    # ── 6. 训练 ─────────────────────────────────────────────────────
    print("\n=== 开始训练 ===", flush=True)
    print(f"  样本 {len(train_dataset)} | 轮次 {NUM_EPOCHS} | 等效batch {BATCH_SIZE*GRAD_ACCUM} "
          f"| lr {LEARNING_RATE} | r {LORA_R}/alpha {LORA_ALPHA}", flush=True)
    t0 = time.time()
    stats = trainer.train()
    elapsed = time.time() - t0
    print(f"\n训练完成，耗时 {elapsed/60:.1f} 分钟", flush=True)
    print(f"  最终 loss: {stats.training_loss:.4f}", flush=True)

    # ── 7. 保存 LoRA 权重 ───────────────────────────────────────────
    os.makedirs(FINAL_DIR, exist_ok=True)
    model.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    print(f"LoRA 权重已保存到：{FINAL_DIR}", flush=True)

    # ── 8. 可选：训练后推理对比 ─────────────────────────────────────
    if args.eval:
        run_eval(model, tokenizer)


def run_eval(model, tokenizer):
    """用测试集跑一遍推理，打印结果。"""
    print("\n=== 微调后推理 ===", flush=True)
    FastLanguageModel.for_inference(model)
    test_samples = load_jsonl(TEST_FILE)
    for i, s in enumerate(test_samples, 1):
        question = s.get("instruction", "")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                input_ids=inputs, max_new_tokens=768,
                do_sample=True, temperature=0.7, top_p=0.9,
                repetition_penalty=1.1,
            )
        answer = tokenizer.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)
        print(f"\n[{i}] 问：{question}\n答：{answer}", flush=True)


if __name__ == "__main__":
    main()
