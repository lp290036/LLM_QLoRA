# -*- coding: utf-8 -*-
"""微调前后自动对比评估（离线、贪心解码）。

对 fl_test.jsonl 中每道题分别用「基座」与「LoRA」生成回答，并计算：
  - PPL（对参考答案的困惑度，越低越好）
  - 生成长度（字符数）
  - 关键词命中率（领域题）
  - 重复率（连续重复 n-gram 占比）

输出：unsloth_output/eval_results.json

用法：
    conda activate unsloth
    python eval_compare.py
"""
import os
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

import unsloth  # noqa: F401
from unsloth import FastLanguageModel

import sys
import json
import math
import re
import time
from pathlib import Path
from collections import Counter

import torch
from peft import PeftModel

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import win_safetensors_patch
win_safetensors_patch.apply()

MODEL_DIR   = r"D:\LLM\models\qwen3.5-2b"
LORA_DIR    = r"D:\LLM\models\qwen3.5-2b\unsloth_output\final"
TEST_FILE   = r"D:\LLM\models\qwen3.5-2b\fl_test.jsonl"
OUTPUT_DIR  = r"D:\LLM\models\qwen3.5-2b\unsloth_output"
RESULT_FILE = os.path.join(OUTPUT_DIR, "eval_results.json")
MAX_NEW_TOKENS = 768
NGRAM_N = 4

SYSTEM_PROMPT = (
    "你是联邦学习与分布式机器学习领域的专业研究员，"
    "请针对问题给出准确、深入、结构清晰的回答。"
)


def load_jsonl(path: str) -> list[dict]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def get_text_tokenizer(tok):
    return getattr(tok, "tokenizer", tok)


def build_messages(question: str, answer: str | None = None) -> list[dict]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


def encode_prompt(tok, question: str) -> dict:
    text = tok.apply_chat_template(
        build_messages(question), tokenize=False, add_generation_prompt=True,
    )
    text_tok = get_text_tokenizer(tok)
    return text_tok(text, return_tensors="pt")


def generate_answer(model, tok, question: str) -> str:
    inputs = encode_prompt(tok, question)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    text_tok = get_text_tokenizer(tok)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=text_tok.eos_token_id,
        )
    return text_tok.decode(
        out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()


def compute_ppl(model, tok, question: str, reference: str) -> float:
    """对参考答案计算困惑度（仅 assistant 部分）。"""
    # 完整对话（含参考答案）
    full_text = tok.apply_chat_template(
        build_messages(question, reference),
        tokenize=False, add_generation_prompt=False,
    )
    # 仅到 assistant 开头（不含参考答案）
    prompt_text = tok.apply_chat_template(
        build_messages(question),
        tokenize=False, add_generation_prompt=True,
    )
    text_tok = get_text_tokenizer(tok)
    full_ids = text_tok(full_text, return_tensors="pt")["input_ids"]
    prompt_ids = text_tok(prompt_text, return_tensors="pt")["input_ids"]
    prompt_len = prompt_ids.shape[1]

    labels = full_ids.clone()
    labels[:, :prompt_len] = -100

    full_ids = full_ids.to(model.device)
    labels = labels.to(model.device)

    with torch.inference_mode():
        out = model(input_ids=full_ids, labels=labels)
    loss = out.loss.item()
    return math.exp(min(loss, 20))


def keyword_hit_rate(text: str, keywords: list[str]) -> float | None:
    if not keywords:
        return None
    hits = sum(1 for kw in keywords if kw in text)
    return hits / len(keywords)


def repetition_rate(text: str, n: int = NGRAM_N) -> float:
    """连续重复 n-gram 占全部 n-gram 的比例。"""
    tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text)
    if len(tokens) < n * 2:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


def avg_metrics(rows: list[dict], key: str, category: str | None = None) -> float | None:
    vals = []
    for r in rows:
        if category and r.get("category") != category:
            continue
        v = r.get(key)
        if v is not None:
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


def load_model():
    if not torch.cuda.is_available():
        sys.exit("无 CUDA。")
    if not os.path.isdir(LORA_DIR):
        sys.exit(f"未找到 LoRA：{LORA_DIR}，请先运行 unsloth_train.py")

    print("加载模型...", flush=True)
    model, tok = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR, max_seq_length=4096,
        dtype=None, load_in_4bit=True, full_finetuning=False,
    )
    model = PeftModel.from_pretrained(model, LORA_DIR)
    FastLanguageModel.for_inference(model)
    model.eval()
    return model, tok


def run_variant(model, tok, samples: list[dict], use_lora: bool) -> list[dict]:
    tag = "lora" if use_lora else "base"
    if use_lora:
        model.base_model.enable_adapter_layers()
    else:
        model.base_model.disable_adapter_layers()

    rows = []
    for i, s in enumerate(samples, 1):
        q = s["instruction"]
        ref = s.get("output", "")
        kw = s.get("keywords", [])
        cat = s.get("category", "领域")

        print(f"  [{tag}] {i}/{len(samples)} {q[:40]}...", flush=True)
        t0 = time.time()
        pred = generate_answer(model, tok, q)
        gen_sec = time.time() - t0
        ppl = compute_ppl(model, tok, q, ref) if ref else None

        rows.append({
            "id": i,
            "category": cat,
            "question": q,
            "reference": ref,
            "prediction": pred,
            "ppl": round(ppl, 2) if ppl is not None else None,
            "length": len(pred),
            "keyword_hit": round(keyword_hit_rate(pred, kw), 3) if kw else None,
            "repetition": round(repetition_rate(pred), 4),
            "gen_sec": round(gen_sec, 2),
        })
        torch.cuda.empty_cache()
    return rows


def main():
    samples = load_jsonl(TEST_FILE)
    print(f"测试集 {len(samples)} 条", flush=True)

    model, tok = load_model()

    print("\n=== 基座模型（LoRA 关闭）===", flush=True)
    base_rows = run_variant(model, tok, samples, use_lora=False)

    print("\n=== LoRA 模型 ===", flush=True)
    lora_rows = run_variant(model, tok, samples, use_lora=True)

    summary = {
        "base": {
            "ppl_avg": round(avg_metrics(base_rows, "ppl") or 0, 2),
            "ppl_domain": round(avg_metrics(base_rows, "ppl", "领域") or 0, 2),
            "ppl_general": round(avg_metrics(base_rows, "ppl", "通用") or 0, 2),
            "length_avg": round(avg_metrics(base_rows, "length") or 0, 1),
            "keyword_hit_avg": round(avg_metrics(base_rows, "keyword_hit", "领域") or 0, 3),
            "repetition_avg": round(avg_metrics(base_rows, "repetition") or 0, 4),
        },
        "lora": {
            "ppl_avg": round(avg_metrics(lora_rows, "ppl") or 0, 2),
            "ppl_domain": round(avg_metrics(lora_rows, "ppl", "领域") or 0, 2),
            "ppl_general": round(avg_metrics(lora_rows, "ppl", "通用") or 0, 2),
            "length_avg": round(avg_metrics(lora_rows, "length") or 0, 1),
            "keyword_hit_avg": round(avg_metrics(lora_rows, "keyword_hit", "领域") or 0, 3),
            "repetition_avg": round(avg_metrics(lora_rows, "repetition") or 0, 4),
        },
    }

    result = {
        "meta": {
            "test_file": TEST_FILE,
            "lora_dir": LORA_DIR,
            "decode": "greedy (do_sample=False)",
            "max_new_tokens": MAX_NEW_TOKENS,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": summary,
        "base": base_rows,
        "lora": lora_rows,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存：{RESULT_FILE}", flush=True)
    print(f"  基座 PPL={summary['base']['ppl_avg']} | LoRA PPL={summary['lora']['ppl_avg']}", flush=True)
    print(f"  基座关键词命中={summary['base']['keyword_hit_avg']} | LoRA={summary['lora']['keyword_hit_avg']}", flush=True)


if __name__ == "__main__":
    main()
