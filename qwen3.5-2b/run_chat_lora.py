# -*- coding: utf-8 -*-
"""Qwen3.5-2B 微调后模型（4-bit 基础模型 + LoRA 适配器）交互对话。

加载 unsloth_output/final 中训练好的 LoRA 适配器，可随意聊天。
对话历史保存到：D:/LLM/models/qwen3.5-2b/chat_logs/chat_history_lora.txt

用法：
    conda activate unsloth
    python run_chat_lora.py
"""
import os
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

import unsloth  # noqa: F401  必须最先导入
from unsloth import FastLanguageModel

import sys
import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Windows 无 mmap 加载补丁（规避 safetensors mmap 访问冲突）──────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import win_safetensors_patch
win_safetensors_patch.apply()

import torch
from peft import PeftModel

MODEL_DIR = r"D:\LLM\models\qwen3.5-2b"
LORA_DIR  = r"D:\LLM\models\qwen3.5-2b\unsloth_output\final"
LOG_DIR   = r"D:\LLM\models\qwen3.5-2b\chat_logs"
LOG_FILE  = os.path.join(LOG_DIR, "chat_history_lora.txt")
MODEL_TAG = "微调后(LoRA)"

# 与训练时完全一致的系统提示，否则微调效果发挥不出来
SYSTEM_PROMPT = (
    "你是联邦学习与分布式机器学习领域的专业研究员，"
    "请针对问题给出准确、深入、结构清晰的回答。"
)


def load_model():
    if not torch.cuda.is_available():
        sys.exit("无 CUDA，请检查 GPU 与 PyTorch。")
    if not os.path.isdir(LORA_DIR):
        sys.exit(f"未找到 LoRA 权重目录：{LORA_DIR}\n请先运行 unsloth_train.py 完成微调。")

    print("加载基础模型（4-bit）...", flush=True)
    model, tok = FastLanguageModel.from_pretrained(
        model_name      = MODEL_DIR,
        max_seq_length  = 4096,
        dtype           = None,
        load_in_4bit    = True,
        full_finetuning = False,
    )
    print(f"挂载 LoRA 适配器：{LORA_DIR}", flush=True)
    model = PeftModel.from_pretrained(model, LORA_DIR)
    FastLanguageModel.for_inference(model)
    model.eval()

    used  = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"就绪 [{MODEL_TAG}] | 显存 {used:.2f}/{total:.1f} GB", flush=True)
    return tok, model


def init_log():
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n\n{'=' * 70}\n")
        f.write(f"会话开始：{ts}  模型：{MODEL_TAG}  权重：{LORA_DIR}\n")
        f.write(f"{'=' * 70}\n")
    print(f"对话日志：{LOG_FILE}", flush=True)


def append_log(role: str, content: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n[{role}]\n{content}\n")
        f.write("-" * 70 + "\n")
        f.flush()


def new_history() -> list:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def chat(tok, model, history: list, prompt: str, max_new_tokens: int = 1024) -> str:
    history.append({"role": "user", "content": prompt})
    # 与训练一致：先渲染成文本，再用底层文本分词器编码（该模型是多模态处理器）
    text = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    text_tok = getattr(tok, "tokenizer", tok)
    inputs = text_tok(text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens     = max_new_tokens,
            do_sample          = True,
            temperature        = 0.7,
            top_p              = 0.9,
            repetition_penalty = 1.1,
            pad_token_id       = text_tok.eos_token_id,
        )
    reply = text_tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    history.append({"role": "assistant", "content": reply})
    torch.cuda.empty_cache()
    return reply


if __name__ == "__main__":
    tok, model = load_model()
    init_log()
    history = new_history()

    print("\n命令：clear=清空记忆  exit/quit/q=退出\n")
    while True:
        try:
            prompt = input("USER: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit", "q"):
            break
        if prompt.lower() == "clear":
            history = new_history()
            append_log("SYSTEM", "（对话历史已清空）")
            print("（对话历史已清空）\n")
            continue

        append_log("USER", prompt)
        reply = chat(tok, model, history, prompt)
        append_log(f"BOT [{MODEL_TAG}]", reply)
        print(f"BOT : {reply}\n")
