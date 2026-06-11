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
# Windows 下 eval 阶段 torch.compile/inductor 易崩溃，直接关闭
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

# 把 torch.compile / triton 的缓存目录指到纯英文路径，
# 规避中文用户名（C:\Users\刘鹏\...\torchinductor_刘鹏）导致的编译/解码报错
_CACHE_ROOT = r"D:\llm_cache"
os.makedirs(os.path.join(_CACHE_ROOT, "inductor"), exist_ok=True)
os.makedirs(os.path.join(_CACHE_ROOT, "triton"), exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_CACHE_ROOT, "inductor"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_CACHE_ROOT, "triton"))

import unsloth  # noqa: F401  (必须最先导入以应用补丁)
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

import sys
import json
import time
import hashlib
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback, set_seed

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
LATEST_LORA_FILE = os.path.join(OUTPUT_DIR, "latest_lora_dir.txt")

# ══════════════════════════════════════════════════════════════════════
#  超参数（含防遗忘配置说明）
# ══════════════════════════════════════════════════════════════════════
MAX_SEQ_LENGTH = 1280      # 答案约 300 字（≈450 token）+ 提示，1280 足够且省显存

# —— LoRA 配置 ——
# 防遗忘要点1：仅训练低秩旁路，冻结全部原始权重 → 天然保护基座能力
# 防遗忘要点2：r 与 alpha 取相同值（缩放系数=1），更新幅度温和，不强行覆盖原分布
LORA_R        = 16
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.10       # 小数据集提高 dropout，降低死记硬背

# —— 训练配置 ——
# 防遗忘要点3：少轮次 + 低学习率 + cosine 衰减 + warmup，避免在 125 条小样本上过拟合
NUM_EPOCHS          = 4        # 配合 early stopping，允许多看几轮但不强制跑满
LEARNING_RATE       = 5e-5
WARMUP_RATIO        = 0.10
WEIGHT_DECAY        = 0.01     # 正则化，约束权重漂移
BATCH_SIZE          = 1
GRAD_ACCUM          = 8        # 等效 batch = 8
# 防遗忘要点4：NEFTune 给嵌入加噪声，提升泛化、降低过拟合
NEFTUNE_ALPHA       = 5
SEED                = 3407
VAL_RATIO           = 0.10     # 留 10% 作验证集，训练中跟踪 eval_loss
EARLY_STOP_PATIENCE = 2        # eval_loss 连续 2 次不改善即停止
SAVE_TOTAL_LIMIT    = 3        # 限制 checkpoint 数量，避免输出目录膨胀
PREVIEW_SAMPLES     = 2        # 训练前打印少量渲染样本，检查模板是否正确

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


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def print_dataset_diagnostics(tokenizer, dataset: Dataset) -> None:
    if len(dataset) == 0:
        sys.exit("训练集为空：没有可用的 instruction/output 样本。")
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    lengths = [len(text_tok(x["text"], add_special_tokens=False)["input_ids"]) for x in dataset]
    lengths_sorted = sorted(lengths)
    p95 = lengths_sorted[min(len(lengths_sorted) - 1, int(len(lengths_sorted) * 0.95))]
    too_long = sum(1 for n in lengths if n > MAX_SEQ_LENGTH)
    print(
        f"  token长度 min/avg/p95/max = {min(lengths)}/{sum(lengths)/len(lengths):.1f}/{p95}/{max(lengths)}",
        flush=True,
    )
    if too_long:
        print(f"  ⚠ {too_long} 条样本超过 MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}，训练时会被截断。", flush=True)
    for i in range(min(PREVIEW_SAMPLES, len(dataset))):
        preview = dataset[i]["text"].replace("\n", "\\n").strip()
        print(f"  样本预览[{i + 1}]: {preview[:500]}", flush=True)


def assert_response_labels(trainer) -> None:
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch.get("labels")
    if labels is None:
        sys.exit("训练 batch 中没有 labels，无法确认 response-only loss mask。")
    total = labels.numel()
    trainable = int((labels != -100).sum().item())
    if trainable == 0:
        sys.exit("response-only mask 后没有任何 assistant token 参与训练，请检查 chat template 标记。")
    print(f"  label mask 检查通过：{trainable}/{total} tokens 参与 loss", flush=True)


def save_run_config(run_dir: str, train_count: int, eval_count: int, eval_steps: int) -> None:
    os.makedirs(run_dir, exist_ok=True)
    cfg = {
        "model_dir": MODEL_DIR,
        "train_file": TRAIN_FILE,
        "test_file": TEST_FILE,
        "train_sha256": sha256_file(TRAIN_FILE),
        "train_count": train_count,
        "eval_count": eval_count,
        "max_seq_length": MAX_SEQ_LENGTH,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "effective_batch": BATCH_SIZE * GRAD_ACCUM,
        "neftune_alpha": NEFTUNE_ALPHA,
        "seed": SEED,
        "val_ratio": VAL_RATIO,
        "eval_steps": eval_steps,
        "early_stop_patience": EARLY_STOP_PATIENCE,
    }
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="训练后用测试集做推理对比")
    args = parser.parse_args()

    set_seed(SEED)
    run_name = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, run_name)
    final_dir = os.path.join(run_dir, "final")

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

    # ── 3. 构建数据集（划分训练/验证集）──────────────────────────────
    print(f"\n加载训练集：{TRAIN_FILE}", flush=True)
    samples = load_jsonl(TRAIN_FILE)
    print(f"  共 {len(samples)} 条原始样本", flush=True)
    full_dataset = build_dataset(tokenizer, samples)
    print_dataset_diagnostics(tokenizer, full_dataset)
    # 留出 10% 作验证集，训练时跟踪 eval_loss，量化监控过拟合/遗忘
    split = full_dataset.train_test_split(test_size=VAL_RATIO, seed=SEED)
    train_dataset, eval_dataset = split["train"], split["test"]
    print(f"  训练集 {len(train_dataset)} 条 | 验证集 {len(eval_dataset)} 条", flush=True)

    # 验证集评估频率：每 EVAL_EVERY 个优化步评估一次
    steps_per_epoch = max(1, len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM))
    eval_steps = max(1, steps_per_epoch // 2)
    save_run_config(run_dir, len(train_dataset), len(eval_dataset), eval_steps)

    # ── 4. 配置 Trainer ─────────────────────────────────────────────
    trainer = SFTTrainer(
        model           = model,
        tokenizer       = tokenizer,
        train_dataset   = train_dataset,
        eval_dataset    = eval_dataset,
        args = SFTConfig(
            dataset_text_field          = "text",
            max_seq_length              = MAX_SEQ_LENGTH,
            per_device_train_batch_size = BATCH_SIZE,
            per_device_eval_batch_size  = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUM,
            warmup_ratio                = WARMUP_RATIO,
            num_train_epochs            = NUM_EPOCHS,
            learning_rate               = LEARNING_RATE,
            weight_decay                = WEIGHT_DECAY,
            lr_scheduler_type           = "cosine",
            optim                       = "adamw_8bit",
            neftune_noise_alpha         = NEFTUNE_ALPHA,
            logging_steps               = 1,
            eval_strategy               = "steps",
            eval_steps                  = eval_steps,
            seed                        = SEED,
            output_dir                  = run_dir,
            save_strategy               = "steps",
            save_steps                  = eval_steps,
            save_total_limit            = SAVE_TOTAL_LIMIT,
            load_best_model_at_end      = True,
            metric_for_best_model       = "eval_loss",
            greater_is_better           = False,
            report_to                   = "none",
            dataset_num_proc            = 1,   # Windows 必须单进程
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
    )

    # ── 5. 只对「助手回答」部分计算 loss（不学习提示）──────────────────
    # 防遗忘要点6：屏蔽 system/user 部分的 loss，只学「如何作答」，
    #             不去记忆问题文本，保护指令理解能力
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part    = "<|im_start|>assistant\n",
    )
    assert_response_labels(trainer)

    # ── 6. 训练 ─────────────────────────────────────────────────────
    print("\n=== 开始训练 ===", flush=True)
    print(f"  样本 {len(train_dataset)} | 轮次 {NUM_EPOCHS} | 等效batch {BATCH_SIZE*GRAD_ACCUM} "
          f"| lr {LEARNING_RATE} | r {LORA_R}/alpha {LORA_ALPHA}", flush=True)
    t0 = time.time()
    stats = trainer.train()
    elapsed = time.time() - t0
    print(f"\n训练完成，耗时 {elapsed/60:.1f} 分钟", flush=True)
    print(f"  最终 loss: {stats.training_loss:.4f}", flush=True)

    # ── 7. 保存最佳 LoRA 权重 ───────────────────────────────────────
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    with open(LATEST_LORA_FILE, "w", encoding="utf-8") as f:
        f.write(final_dir)
    print(f"最佳 LoRA 权重已保存到：{final_dir}", flush=True)
    print(f"最新 LoRA 指针已写入：{LATEST_LORA_FILE}", flush=True)

    # ── 7.1 保存 train/eval loss 曲线（供可视化看板使用）──────────────
    history = []
    for rec in trainer.state.log_history:
        if "loss" in rec or "eval_loss" in rec:
            history.append({
                "step":      rec.get("step"),
                "epoch":     rec.get("epoch"),
                "loss":      rec.get("loss"),
                "eval_loss": rec.get("eval_loss"),
            })
    loss_path = os.path.join(run_dir, "loss_history.json")
    with open(loss_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"loss 曲线已保存到：{loss_path}", flush=True)

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
