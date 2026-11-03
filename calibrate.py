"""
Multi-domain calibration dataset preparation for WaterSIC PTQ.
Includes native Chat Template formatting (<|im_start|>, <|im_end|>, <think> tags)
to ensure activation distribution matches real-world conversational and reasoning inference.
"""

import os
import random
import torch
from typing import List, Dict, Any
from transformers import AutoTokenizer


# Multi-turn conversational calibration samples formatted for Chat / Instruct models
CHAT_CALIBRATION_SAMPLES = [
    [
        {"role": "system", "content": "You are a helpful, respectful, and reasoning-oriented assistant."},
        {"role": "user", "content": "Explain the mathematical difference between LDLQ Cholesky factorization and uniform scalar quantization."},
        {"role": "assistant", "content": "<think>\nThe user is asking for a comparison between two quantization methods.\n1. Uniform scalar quantization maps floating-point numbers to uniformly spaced discrete bins without considering correlations between weights.\n2. LDLQ (Lower-Diagonal-Lower Quantization) uses the Cholesky factorization of the Hessian matrix H = X^T X to sequentially eliminate quantization noise across columns.\n</think>\nUniform quantization rounds each scalar weight independently, whereas LDLQ uses the Cholesky factor of the activation covariance matrix to sequentially compensate for quantization errors in subsequent columns, achieving rate-distortion optimality."}
    ],
    [
        {"role": "system", "content": "You are an expert programmer and code analysis assistant."},
        {"role": "user", "content": "Write a high-performance PyTorch CUDA kernel launcher for 3-bit weight decompression."},
        {"role": "assistant", "content": "<think>\nNeed to demonstrate an efficient kernel launch in PyTorch C++ extension.\nUnpacking 3-bit integers requires extracting 8 numbers from 3 consecutive bytes (24 bits total).\n</think>\n```cpp\n__global__ void unpack_3bit_kernel(const uint8_t* __restrict__ packed, float* __restrict__ out, int n) {\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx * 8 < n) {\n        // Read 3 bytes and extract 8 x 3-bit values\n    }\n}\n```"}
    ],
    [
        {"role": "system", "content": "Ты продвинутый русскоязычный ИИ-помощник с глубоким логическим мышлением."},
        {"role": "user", "content": "Реши задачу: поезд проехал 240 км за 3 часа, затем увеличил скорость на 20 км/ч. Сколько времени займет путь в 400 км с новой скоростью?"},
        {"role": "assistant", "content": "<think>\n1. Исходная скорость: v1 = 240 км / 3 ч = 80 км/ч.\n2. Новая скорость: v2 = 80 + 20 = 100 км/ч.\n3. Время для 400 км: t = 400 км / 100 км/ч = 4 часа.\n</think>\nРешение по шагам:\n1. Первоначальная скорость поезда: 240 / 3 = 80 км/ч.\n2. Новая скорость после увеличения: 80 + 20 = 100 км/ч.\n3. Время на преодоление 400 км: 400 / 100 = 4 часа.\n\nОтвет: 4 часа."}
    ],
    [
        {"role": "system", "content": "You are a Mixture-of-Experts routing specialist."},
        {"role": "user", "content": "How do router logits prevent expert collapse in a 35B-A3B MoE model?"},
        {"role": "assistant", "content": "<think>\nIn MoE models with top-k gating, a load-balancing auxiliary loss encourages balanced routing probabilities across all experts, avoiding over-allocation to popular experts.\n</think>\nLoad-balancing auxiliary losses and gating jitter ensure that token routing probabilities remain distributed across all 64 experts rather than collapsing onto a dominant subset."}
    ]
]


def get_calibration_dataset(
    tokenizer: AutoTokenizer,
    nsamples: int = 128,
    seqlen: int = 2048,
    seed: int = 42,
    output_cache_path: str = None
) -> torch.Tensor:
    """
    Build tokenized calibration tensor of shape (nsamples, seqlen).
    Formats samples through tokenizer.apply_chat_template to preserve native tokens.
    """
    if output_cache_path and os.path.exists(output_cache_path):
        print(f"[Calib] Loading pre-tokenized calibration data from {output_cache_path}...")
        return torch.load(output_cache_path, map_location="cpu")

    print(f"[Calib] Compiling Chat-formatted calibration dataset: {nsamples} samples x {seqlen} tokens ({nsamples * seqlen:,} total tokens)...")
    rng = random.Random(seed)
    collected_tokens: List[int] = []
    total_needed = nsamples * seqlen

    # 1. Format conversational chat samples using model's native chat template
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        print("  [Chat Template] Applying model's native chat template (<|im_start|>, <|im_end|>)...")
        for sample in CHAT_CALIBRATION_SAMPLES:
            try:
                formatted_ids = tokenizer.apply_chat_template(sample, tokenize=True, add_generation_prompt=False)
                collected_tokens.extend(formatted_ids)
            except Exception as e:
                print(f"  [Notice] apply_chat_template error ({e}), using raw text fallback")

    # 2. Try loading diverse multi-domain text from HuggingFace
    try:
        from datasets import load_dataset
        os.environ["HF_DATASETS_DISABLE_FILE_LOCKING"] = "1"
        print("  [Data] Fetching multi-domain text (WikiText-2)...")
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        docs = list(ds["text"])
        rng.shuffle(docs)
        
        for doc in docs:
            if len(doc.strip()) > 50:
                # Wrap passages in chat format so the model router sees conversational context
                chat_doc = [
                    {"role": "user", "content": "Analyze and summarize this information:\n" + doc[:1000]},
                    {"role": "assistant", "content": "<think>\nExtracting key points from passage...\n</think>\n" + doc[1000:2000]}
                ]
                if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                    try:
                        ids = tokenizer.apply_chat_template(chat_doc, tokenize=True)
                        collected_tokens.extend(ids)
                    except Exception:
                        collected_tokens.extend(tokenizer.encode(doc, add_special_tokens=False))
                else:
                    collected_tokens.extend(tokenizer.encode(doc, add_special_tokens=False))

            if len(collected_tokens) >= total_needed:
                break
    except Exception as e:
        print(f"  [Note] Hugging Face datasets load skipped ({e}), expanding native chat pool...")

    # 3. Tile if more tokens needed
    while len(collected_tokens) < total_needed:
        for sample in CHAT_CALIBRATION_SAMPLES:
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                ids = tokenizer.apply_chat_template(sample, tokenize=True)
            else:
                text = f"<|im_start|>user\n{sample[1]['content']}<|im_end|>\n<|im_start|>assistant\n{sample[2]['content']}<|im_end|>"
                ids = tokenizer.encode(text)
            collected_tokens.extend(ids)

    # 4. Form (nsamples, seqlen) tensor
    tokens_tensor = torch.zeros((nsamples, seqlen), dtype=torch.long)
    for i in range(nsamples):
        start = i * seqlen
        end = start + seqlen
        tokens_tensor[i] = torch.tensor(collected_tokens[start:end], dtype=torch.long)

    print(f"[Calib] Done: shape={tokens_tensor.shape}, {tokens_tensor.numel():,} tokens (native chat template applied).")
    
    if output_cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_cache_path)), exist_ok=True)
        torch.save(tokens_tensor, output_cache_path)
        print(f"[Calib] Saved cache to {output_cache_path}")

    return tokens_tensor
