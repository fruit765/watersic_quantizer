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
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from core.profile_engine import QuantProfile
from core.packing import pack_3bit, pack_4bit, pack_6bit, pack_8bit, pack_csr_outliers
from core.wsic_writer import WsicContainerWriter
from calibrate import get_calibration_dataset

# Core WaterSIC (ZSIC) quantization module (bundled in core/zsic.py)
try:
    from core.zsic import compress_zsic, ZSICConfig
    HAS_WATERSIC_LIB = True
except ImportError:
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "watersic"))
        from quant_layerwise.methods.zsic import compress_zsic, ZSICConfig
        HAS_WATERSIC_LIB = True
    except ImportError:
        HAS_WATERSIC_LIB = False


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
    percdamp: float = 0.001
) -> Dict[str, Any]:
    """
    Full WaterSIC (ZSIC) quantization with Cholesky decomposition, water-filling,
    and CSR sparse outlier extraction.
    """
    if not HAS_WATERSIC_LIB:
        return quantize_linear_layer_fallback(W, Sig_X, target_bits)

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

    # 3. Load Calibration Data
    calib_tensor = get_calibration_dataset(tokenizer, nsamples=nsamples, seqlen=seqlen)

    # 4. Initialize WSIC Writer
    writer = WsicContainerWriter(output_wsic, config.to_dict(), default_bits=3)

    # 5. Load Model Skeleton (layer-by-layer streaming)
    print("[Model] Initializing base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.eval()

    # Identify base components
    embed_tokens = model.get_input_embeddings()
    layers = model.model.layers if hasattr(model, "model") and hasattr(model.model, "layers") else []
    lm_head = model.get_output_embeddings()

    # Cache activations on CPU
    print(f"[Cache] Running token embeddings through calibration passages...")
    hidden_size = config.hidden_size
    cached_h = torch.zeros((nsamples, seqlen, hidden_size), dtype=dtype, device="cpu")

    with torch.no_grad():
        for i in range(nsamples):
            tokens_batch = calib_tensor[i:i+1].to(primary_device)
            h = embed_tokens(tokens_batch)
            cached_h[i] = h.cpu()
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
        res = quantize_linear_layer_watersic(W_emb.to(primary_device), Sig_emb, emb_rule["target_bits"])
        writer.add_quantized_watersic_tensor(
            "token_embd.weight",
            list(W_emb.shape),
            res["packed_qweight"],
            res["scales_fp64"],
            res["csr_bytes"],
            target_bits=emb_rule["target_bits"]
        )
    else:
        writer.add_unquantized_tensor("token_embd.weight", embed_tokens.weight.data.cpu().numpy(), "F16")

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
                _ = layer(h_in)
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
                    "weight_np": w_orig.cpu().numpy()
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
                writer.add_unquantized_tensor(full_param_name, param.data.cpu().numpy(), "F16")

        # Forward pass updated activations for next layer
        with torch.no_grad():
            for i in range(nsamples):
                h_in = cached_h[i:i+1].to(primary_device)
                h_out = layer(h_in)
                if isinstance(h_out, tuple):
                    h_out = h_out[0]
                cached_h[i] = h_out.cpu()
                del h_in, h_out

        # Offload layer back to CPU and free VRAM
        layer = layer.to("cpu")
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
            res_head = quantize_linear_layer_watersic(W_head.to(primary_device), Sig_head, head_rule["target_bits"])
            writer.add_quantized_watersic_tensor(
                "output.weight",
                list(W_head.shape),
                res_head["packed_qweight"],
                res_head["scales_fp64"],
                res_head["csr_bytes"],
                target_bits=head_rule["target_bits"]
            )
        else:
            writer.add_unquantized_tensor("output.weight", W_head.cpu().numpy(), "F16")

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
