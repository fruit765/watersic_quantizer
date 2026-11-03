"""
WaterSIC Progressive Layer-by-Layer Quantizer.
Optimized for memory-constrained environments (e.g. Kaggle 2x T4 16GB, 30GB RAM).
Streams weights layer-by-layer, collects Hessians via forward hooks, applies WaterSIC LDLQ,
and writes directly to .wsic container.
"""

import os
import sys
import gc
import time
import json
import inspect
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

try:
    from core.profile_engine import QuantProfile
    from core.packing import pack_3bit, pack_4bit, pack_6bit, pack_8bit, pack_csr_outliers
    from core.wsic_writer import WsicContainerWriter
    from calibrate import get_calibration_dataset
    from core.zsic import compress_zsic, ZSICConfig
    HAS_WATERSIC_LIB = True
except ImportError:
    try:
        from watersic_quantizer.core.profile_engine import QuantProfile
        from watersic_quantizer.core.packing import pack_3bit, pack_4bit, pack_6bit, pack_8bit, pack_csr_outliers
        from watersic_quantizer.core.wsic_writer import WsicContainerWriter
        from watersic_quantizer.calibrate import get_calibration_dataset
        from watersic_quantizer.core.zsic import compress_zsic, ZSICConfig
        HAS_WATERSIC_LIB = True
    except ImportError:
        HAS_WATERSIC_LIB = False


def to_numpy_f16(t: torch.Tensor) -> np.ndarray:
    """Safely convert PyTorch tensor (including BF16) to float16 NumPy array."""
    return t.detach().to(torch.float16).cpu().numpy()


def pack_by_bits(codes: np.ndarray, target_bits: int) -> bytes:
    """Select packing routine by target bitrate."""
    if target_bits == 3:
        return pack_3bit(codes)
    elif target_bits == 4:
        return pack_4bit(codes)
    elif target_bits == 6:
        return pack_6bit(codes)
    elif target_bits == 8:
        return pack_8bit(codes)
    else:
        raise ValueError(f"Unsupported WaterSIC target bits for packing: {target_bits}")


def quantize_linear_layer_fallback(W: torch.Tensor, Sig_X: torch.Tensor, target_bits: int) -> Dict[str, Any]:
    """
    Fallback integer grid quantization if native ZSIC module is not compiled.
    """
    rows, cols = W.shape
    device = W.device
    
    # Simple SVD / diagonal scaling
    std_x = torch.diag(Sig_X).clamp(min=1e-6).sqrt().to(torch.float64)
    col_scales = std_x.cpu().numpy()
    row_scales = torch.ones(rows, dtype=torch.float64).numpy()

    # Normalize by scale
    W_scaled = W.to(torch.float64).cpu().numpy() / np.maximum(col_scales, 1e-6)
    
    # Min-max Lloyd-Max grid
    max_val = (1 << (target_bits - 1)) - 1
    min_val = -(1 << (target_bits - 1))
    
    scale_factor = np.quantile(np.abs(W_scaled), 0.99) / max_val
    scale_factor = max(scale_factor, 1e-7)
    row_scales *= scale_factor

    Z = np.clip(np.round(W_scaled / scale_factor), min_val, max_val).astype(np.int64)
    codes = (Z - min_val).astype(np.uint8)

    packed_bytes = pack_by_bits(codes, target_bits)
    combined_scales = np.concatenate([row_scales, col_scales]).astype(np.float64)

    return {
        "packed_qweight": packed_bytes,
        "scales_fp64": combined_scales,
        "csr_bytes": b"",
        "dequant_weight": torch.from_numpy(Z * scale_factor * col_scales).to(W.dtype).to(device)
    }


def quantize_linear_layer_watersic(
    W: torch.Tensor,
    Sig_X: torch.Tensor,
    target_bits: int,
    percdamp: float = 0.001,
    max_chunk_rows: int = 8192
) -> Dict[str, Any]:
    """
    Full WaterSIC (ZSIC) quantization with Cholesky decomposition, water-filling,
    and CSR sparse outlier extraction. Automatically chunks along rows if matrix
    is too large to fit in VRAM (e.g. embeddings or LM head with 248k rows).
    """
    if not HAS_WATERSIC_LIB:
        return quantize_linear_layer_fallback(W, Sig_X, target_bits)

    rows, cols = W.shape
    target_device = Sig_X.device

    # For massive matrices (e.g. vocab embeddings or lm_head with 248,320 rows),
    # slice into row chunks of max_chunk_rows to prevent CUDA OOM on 16GB GPUs.
    if rows > max_chunk_rows:
        num_chunks = (rows + max_chunk_rows - 1) // max_chunk_rows
        print(f"  [Memory-Safe Chunking] Tensor has {rows} rows (exceeds {max_chunk_rows}). Quantizing in {num_chunks} chunks to fit VRAM...", flush=True)

        packed_chunks = []
        t_vec_chunks = []
        col_scales_list = []
        dequant_chunks = []

        for c_idx in range(num_chunks):
            start_r = c_idx * max_chunk_rows
            end_r = min(start_r + max_chunk_rows, rows)
            w_chunk = W[start_r:end_r, :].to(target_device).contiguous()

            res_c = quantize_linear_layer_watersic(
                w_chunk, Sig_X, target_bits, percdamp=percdamp, max_chunk_rows=max_chunk_rows
            )

            packed_chunks.append(res_c["packed_qweight"])
            dequant_chunks.append(res_c["dequant_weight"].to("cpu"))

            chunk_rows = end_r - start_r
            t_vec_chunks.append(res_c["scales_fp64"][:chunk_rows])
            col_scales_list.append(res_c["scales_fp64"][chunk_rows:])

            del w_chunk, res_c
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        total_packed = b"".join(packed_chunks)
        full_t_vec = np.concatenate(t_vec_chunks)
        avg_col_scales = np.mean(col_scales_list, axis=0)
        combined_scales = np.concatenate([full_t_vec, avg_col_scales]).astype(np.float64)
        full_dequant = torch.cat(dequant_chunks, dim=0).to(target_device)

        return {
            "packed_qweight": total_packed,
            "scales_fp64": combined_scales,
            "csr_bytes": b"",
            "dequant_weight": full_dequant
        }

    # Ensure W is on target_device
    W = W.to(target_device)

    cfg = ZSICConfig(
        target_rate_bits=float(target_bits),
        percdamp=percdamp,
        binary_search=True,
        apply_rescaler=True,
        dead_dim_threshold=0.001
    )

    What, loss, rate, frame = compress_zsic(
        W,
        cfg=cfg,
        Sig_X=Sig_X,
        Sig_X_for_dead=Sig_X
    )

    Z = frame["Z"].cpu().numpy().astype(np.int64)
    rows, cols = Z.shape

    # Determine code range based on target bits
    max_code = (1 << (target_bits - 1)) - 1
    min_code = -(1 << (target_bits - 1))

    Z_clamped = np.clip(Z, min_code, max_code)
    base_codes = (Z_clamped - min_code).astype(np.uint8)
    packed_qw = pack_by_bits(base_codes, target_bits)

    # CSR outliers
    delta = Z - Z_clamped
    nz_rows, nz_cols = np.nonzero(delta)
    num_outliers = len(nz_rows)

    csr_bytes = b""
    if num_outliers > 0:
        row_counts = np.bincount(nz_rows, minlength=rows)
        row_ptr = np.zeros(rows + 1, dtype=np.uint32)
        row_ptr[1:] = np.cumsum(row_counts)
        col_idx = nz_cols.astype(np.uint16)
        deltas = delta[nz_rows, nz_cols].astype(np.int16)
        csr_bytes = pack_csr_outliers(row_ptr, col_idx, deltas, rows)

    # Scales: row_scales (t_vec) + col_scales (g_vec * alpha)
    t_vec = frame["t_vec"].cpu().numpy()
    g_vec = frame["g_vec"].cpu().numpy()
    alpha = frame["alpha"].cpu().numpy()
    col_scales = g_vec * alpha
    combined_scales = np.concatenate([t_vec, col_scales]).astype(np.float64)

    return {
        "packed_qweight": packed_qw,
        "scales_fp64": combined_scales,
        "csr_bytes": csr_bytes,
        "dequant_weight": What.to(W.dtype)
    }


def run_layer_forward(layer: nn.Module, h_in: torch.Tensor, rotary_emb: Optional[nn.Module] = None) -> torch.Tensor:
    """
    Safely invoke layer forward, automatically supplying position_embeddings / position_ids
    if required by Qwen3.5 / MoE architectures.
    """
    sig = inspect.signature(layer.forward)
    params = sig.parameters
    kwargs = {}

    if "position_embeddings" in params and rotary_emb is not None:
        seq_l = h_in.shape[1]
        p_ids = torch.arange(seq_l, device=h_in.device).unsqueeze(0).expand(h_in.shape[0], -1)
        try:
            pos_emb = rotary_emb(h_in, p_ids)
        except Exception:
            try:
                # Qwen 3.5 multi-dimensional position ids (temporal, height, width)
                p_ids_3d = p_ids.view(1, 1, -1).expand(3, h_in.shape[0], -1)
                pos_emb = rotary_emb(h_in, p_ids_3d)
            except Exception:
                pos_emb = None
        if pos_emb is not None:
            kwargs["position_embeddings"] = pos_emb
    elif "position_ids" in params:
        seq_l = h_in.shape[1]
        p_ids = torch.arange(seq_l, device=h_in.device).unsqueeze(0).expand(h_in.shape[0], -1)
        kwargs["position_ids"] = p_ids

    try:
        out = layer(h_in, **kwargs)
    except TypeError:
        out = layer(h_in)

    if isinstance(out, tuple):
        return out[0]
    return out


def run_quantization(
    model_id: str,
    profile_path: str,
    output_wsic: str,
    nsamples: int = 128,
    seqlen: int = 2048,
    devices: Any = "auto",
    device: Any = None,  # For backward-compatibility with device="cuda:0"
    dtype = None
):
    # Backward compatibility: if user passed device="cuda:0" or similar
    if device is not None and devices == "auto":
        devices = device

    # Parse and resolve target devices
    if devices == "auto" or devices is None:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            devices_list = [f"cuda:{i}" for i in range(count)]
        else:
            devices_list = ["cpu"]
    elif isinstance(devices, str):
        if "," in devices:
            devices_list = [d.strip() for d in devices.split(",") if d.strip()]
        elif devices.lower() == "auto":
            count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            devices_list = [f"cuda:{i}" for i in range(count)] if count > 0 else ["cpu"]
        else:
            devices_list = [devices.strip()]
    elif isinstance(devices, (list, tuple)):
        devices_list = list(devices)
    else:
        devices_list = ["cuda:0"] if torch.cuda.is_available() else ["cpu"]

    primary_device = devices_list[0]
    num_worker_gpus = len(devices_list)

    if dtype is None:
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    print("=" * 80)
    print("WATERSIC PROGRESSIVE LAYER-BY-LAYER QUANTIZER")
    print(f"Model ID:     {model_id}")
    print(f"Profile:      {profile_path}")
    print(f"Output WSIC:  {output_wsic}")
    if num_worker_gpus > 1 and all("cuda" in d for d in devices_list):
        gpu_details = [f"{d} ({torch.cuda.get_device_name(int(d.split(':')[-1]))})" for d in devices_list]
        print(f"Hardware:     Multi-GPU Mode active across {num_worker_gpus} GPUs:")
        for idx_d, d_str in enumerate(gpu_details):
            role = "Primary (Forward pass + LDLQ Worker)" if idx_d == 0 else "LDLQ Worker"
            print(f"              [{idx_d}] {d_str} -> {role}")
        print(f"Parallelism:  ThreadPoolExecutor(max_workers={num_worker_gpus}) for concurrent MoE experts")
    else:
        dev_desc = torch.cuda.get_device_name(0) if (torch.cuda.is_available() and "cuda" in primary_device) else "CPU"
        print(f"Hardware:     Single-Device Mode on {primary_device} ({dev_desc})")
    print("=" * 80)

    # 1. Load Profile
    profile = QuantProfile.load_from_file(profile_path)
    print(f"[Profile] Loaded profile: {profile.name} ({len(profile.rules)} rules)")

    # 2. Load Config & Tokenizer
    print("[Model] Loading configuration and tokenizer...")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check for standalone chat_template.jinja if not already loaded into tokenizer
    if getattr(tokenizer, "chat_template", None) is None:
        local_jinja = os.path.join(model_id, "chat_template.jinja")
        if os.path.exists(local_jinja):
            with open(local_jinja, "r", encoding="utf-8") as f:
                tokenizer.chat_template = f.read()
            print("[Tokenizer] Loaded local chat_template.jinja into tokenizer.")
        else:
            try:
                from huggingface_hub import hf_hub_download
                jinja_path = hf_hub_download(repo_id=model_id, filename="chat_template.jinja")
                with open(jinja_path, "r", encoding="utf-8") as f:
                    tokenizer.chat_template = f.read()
                print("[Tokenizer] Downloaded and loaded chat_template.jinja from Hugging Face Hub.")
            except Exception:
                pass

    if getattr(tokenizer, "chat_template", None):
        print(f"[Tokenizer] Active Chat Template: Jinja2 template detected ({len(tokenizer.chat_template)} chars)")
    else:
        print("[Tokenizer] Active Chat Template: Standard Qwen XML tags fallback")

    # 3. Load Calibration Data
    calib_tensor = get_calibration_dataset(tokenizer, nsamples=nsamples, seqlen=seqlen)

    # 4. Initialize WSIC Writer
    writer = WsicContainerWriter(output_wsic, config.to_dict(), default_bits=3)

    # 5. Load Model Skeleton (layer-by-layer streaming)
    print("[Model] Initializing base model...")
    model_kwargs = {
        "config": config,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True
    }
    try:
        import accelerate
        model_kwargs["device_map"] = "cpu"
    except ImportError:
        pass

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    # Identify base components
    embed_tokens = model.get_input_embeddings()
    layers = model.model.layers if hasattr(model, "model") and hasattr(model.model, "layers") else []
    lm_head = model.get_output_embeddings()
    rotary_emb = getattr(model, "rotary_emb", None)
    if rotary_emb is None and hasattr(model, "model"):
        rotary_emb = getattr(model.model, "rotary_emb", None)

    # Cache activations on CPU
    print(f"[Cache] Running token embeddings through calibration passages...")
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None and hasattr(config, "get_text_config"):
        hidden_size = config.get_text_config().hidden_size
    if hidden_size is None and hasattr(config, "text_config"):
        hidden_size = getattr(config.text_config, "hidden_size", None)
    if hidden_size is None and hasattr(embed_tokens, "weight"):
        hidden_size = embed_tokens.weight.shape[1]

    cached_h = torch.zeros((nsamples, seqlen, hidden_size), dtype=dtype, device="cpu")

    emb_device = next(embed_tokens.parameters()).device
    with torch.no_grad():
        for i in range(nsamples):
            tokens_batch = calib_tensor[i:i+1].to(emb_device)
            h = embed_tokens(tokens_batch)
            cached_h[i] = h.to("cpu")
            del tokens_batch, h
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Handle input embeddings quantization if requested by profile
    emb_rule = profile.match_tensor("token_embd.weight")
    print(f"[Embeddings] {emb_rule['action']} ({emb_rule['target_bits']} bits)")
    if emb_rule["action"] == "watersic":
        W_emb = embed_tokens.weight.data
        Sig_emb = torch.eye(W_emb.shape[1], dtype=torch.float64, device=primary_device)
        res = quantize_linear_layer_watersic(W_emb, Sig_emb, emb_rule["target_bits"])
        writer.add_quantized_watersic_tensor(
            "token_embd.weight",
            list(W_emb.shape),
            res["packed_qweight"],
            res["scales_fp64"],
            res["csr_bytes"],
            target_bits=emb_rule["target_bits"]
        )
    else:
        writer.add_unquantized_tensor("token_embd.weight", to_numpy_f16(embed_tokens.weight.data), "F16")

    # 6. Progressive Layer-by-Layer Quantization
    num_layers = len(layers)
    print(f"\n[Layers] Starting streaming quantization of {num_layers} layers...")
    if num_worker_gpus >= 2:
        print(f"[Multi-GPU] Detected {num_worker_gpus} GPUs. Parallelizing WaterSIC expert LDLQ across {', '.join(devices_list)}!")

    for layer_idx in range(num_layers):
        layer = layers[layer_idx]
        print(f"\n>>> Layer {layer_idx + 1}/{num_layers} >>>")
        t0_layer = time.time()

        # Move ONLY this layer to primary GPU for forward pass
        layer = layer.to(primary_device)
        if rotary_emb is not None:
            rotary_emb = rotary_emb.to(primary_device)

        # Find linear modules in layer
        layer_linears = {}
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                layer_linears[name] = module

        print(f"  Found {len(layer_linears)} linear projection modules.")

        # Accumulate Hessians H = X^T X on primary GPU
        hessians = {name: torch.zeros((mod.in_features, mod.in_features), dtype=torch.float64, device=primary_device)
                    for name, mod in layer_linears.items()}

        hooks = []
        for name, mod in layer_linears.items():
            def make_hook(n):
                def hook_fn(m, inp, out):
                    x = inp[0].detach()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    hessians[n].addmm_(x.t().to(torch.float64), x.to(torch.float64))
                return hook_fn
            hooks.append(mod.register_forward_hook(make_hook(name)))

        # Forward pass calibration through current layer
        total_tokens = nsamples * seqlen
        with torch.no_grad():
            for i in range(nsamples):
                h_in = cached_h[i:i+1].to(primary_device)
                _ = run_layer_forward(layer, h_in, rotary_emb)
                del h_in

        for h in hooks:
            h.remove()

        for name in hessians:
            hessians[name].div_(total_tokens)

        # Multi-GPU or Single-GPU Quantization Dispatcher
        linear_items = list(layer_linears.items())

        def quantize_single_mod(item, target_gpu):
            m_name, m_mod = item
            full_t_name = f"blk.{layer_idx}.{m_name}.weight"
            m_rule = profile.match_tensor(full_t_name)
            w_orig = m_mod.weight.data

            if m_rule["action"] == "watersic":
                t_bits = m_rule["target_bits"]
                if "cuda" in target_gpu and torch.cuda.is_available():
                    with torch.cuda.device(target_gpu):
                        w_gpu = w_orig.to(target_gpu)
                        sig_gpu = hessians[m_name].to(target_gpu)
                        q_res = quantize_linear_layer_watersic(w_gpu, sig_gpu, t_bits)
                else:
                    w_gpu = w_orig.to(target_gpu)
                    sig_gpu = hessians[m_name].to(target_gpu)
                    q_res = quantize_linear_layer_watersic(w_gpu, sig_gpu, t_bits)
                return {
                    "name": full_t_name,
                    "mod_name": m_name,
                    "action": "watersic",
                    "target_bits": t_bits,
                    "shape": list(w_orig.shape),
                    "qweight": q_res["packed_qweight"],
                    "scales": q_res["scales_fp64"],
                    "csr": q_res["csr_bytes"],
                    "dequant_weight": q_res["dequant_weight"].to(primary_device),
                    "desc": m_rule["description"]
                }
            else:
                return {
                    "name": full_t_name,
                    "mod_name": m_name,
                    "action": "keep",
                    "target_bits": 0,
                    "weight_np": to_numpy_f16(w_orig)
                }

        results = []
        if num_worker_gpus >= 2:
            import concurrent.futures
            # Split work across specified worker GPUs
            print(f"  [Multi-GPU LDLQ] Dispatching {len(linear_items)} linear modules across {num_worker_gpus} GPUs ({', '.join(devices_list)})...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_worker_gpus) as executor:
                futures = []
                for idx_m, item in enumerate(linear_items):
                    assigned_device = devices_list[idx_m % num_worker_gpus]
                    futures.append(executor.submit(quantize_single_mod, item, assigned_device))
                for f in futures:
                    results.append(f.result())
        else:
            for item in linear_items:
                results.append(quantize_single_mod(item, primary_device))

        # Apply results back and write to .wsic
        for r in results:
            if r["action"] == "watersic":
                layer_linears[r["mod_name"]].weight.data.copy_(r["dequant_weight"])
                writer.add_quantized_watersic_tensor(
                    r["name"],
                    r["shape"],
                    r["qweight"],
                    r["scales"],
                    r["csr"],
                    target_bits=r["target_bits"]
                )
            else:
                writer.add_unquantized_tensor(r["name"], r["weight_np"], "F16")

        # Save non-linear tensors (RMSNorms, biases)
        for name, param in layer.named_parameters():
            if not any(lin_name in name for lin_name in layer_linears):
                full_param_name = f"blk.{layer_idx}.{name}"
                writer.add_unquantized_tensor(full_param_name, to_numpy_f16(param.data), "F16")

        # Forward pass updated activations for next layer
        with torch.no_grad():
            for i in range(nsamples):
                h_in = cached_h[i:i+1].to(primary_device)
                h_out = run_layer_forward(layer, h_in, rotary_emb)
                cached_h[i] = h_out.to("cpu")
                del h_in, h_out

        # Offload layer back to CPU and free VRAM
        layer = layer.to("cpu")
        if rotary_emb is not None:
            rotary_emb = rotary_emb.to("cpu")
        del hessians, results
        gc.collect()
        for d in devices_list:
            if "cuda" in d and torch.cuda.is_available():
                gpu_id = int(d.split(":")[-1])
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()

        elapsed = time.time() - t0_layer
        print(f"  Completed Layer {layer_idx + 1} in {elapsed:.2f}s")

    # 7. Quantize LM Head
    if lm_head is not None:
        head_rule = profile.match_tensor("output.weight")
        print(f"\n[LM Head] {head_rule['action']} ({head_rule['target_bits']} bits)")
        W_head = lm_head.weight.data
        if head_rule["action"] == "watersic":
            Sig_head = torch.eye(W_head.shape[1], dtype=torch.float64, device=primary_device)
            res_head = quantize_linear_layer_watersic(W_head, Sig_head, head_rule["target_bits"])
            writer.add_quantized_watersic_tensor(
                "output.weight",
                list(W_head.shape),
                res_head["packed_qweight"],
                res_head["scales_fp64"],
                res_head["csr_bytes"],
                target_bits=head_rule["target_bits"]
            )
        else:
            writer.add_unquantized_tensor("output.weight", to_numpy_f16(W_head), "F16")

    # 8. Finalize .wsic Container
    print("\n[Assembly] Finalizing .wsic container...")
    writer.finalize(num_layers=num_layers)
    print("\n" + "=" * 80)
    print(" QUANTIZATION PIPELINE COMPLETE!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="WaterSIC Mixed-Bitrate Quantization Pipeline")
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID or local directory")
    parser.add_argument("--profile", type=str, required=True, help="Path to profile JSON")
    parser.add_argument("--output_wsic", type=str, required=True, help="Target .wsic file path")
    parser.add_argument("--nsamples", type=int, default=64, help="Number of calibration samples")
    parser.add_argument("--seqlen", type=int, default=2048, help="Tokens per calibration sample")
    parser.add_argument("--devices", type=str, default="auto", help="Devices to use, e.g. 'auto', 'cuda:0,cuda:1', or 'cpu'")
    args = parser.parse_args()

    run_quantization(
        model_id=args.model_id,
        profile_path=args.profile,
        output_wsic=args.output_wsic,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        devices=args.devices
    )


if __name__ == "__main__":
    main()
