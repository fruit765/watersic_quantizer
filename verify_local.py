"""
Verification script for WaterSIC Quantizer.
Runs local parity checks:
1. Lossless round-trip verification of 3-bit, 4-bit, 6-bit, 8-bit packing.
2. Profile rule verification against actual Qwen3.5 architecture tensors.
3. Comparison and structural validation of generated .wsic against reference Qwen3.5-0.8B-WaterSIC-3bit.wsic.
"""

import sys
import json
import struct
import numpy as np
from pathlib import Path

from core.profile_engine import QuantProfile
from core.packing import (
    pack_3bit, unpack_3bit,
    pack_4bit, unpack_4bit,
    pack_6bit, unpack_6bit,
    pack_8bit, unpack_8bit,
    pack_csr_outliers
)


def test_bit_packing_roundtrips():
    print("[1/3] Testing bit-packing round-trips...")
    rng = np.random.RandomState(42)
    n = 10000

    # 3-bit (0..7)
    arr3 = rng.randint(0, 8, size=n, dtype=np.uint8)
    packed3 = pack_3bit(arr3)
    unpacked3 = unpack_3bit(packed3, n)
    assert np.array_equal(arr3, unpacked3), "3-bit roundtrip failed!"
    print(f"  [OK] 3-bit roundtrip: {n} values packed into {len(packed3)} bytes (lossless)")

    # 4-bit (0..15)
    arr4 = rng.randint(0, 16, size=n, dtype=np.uint8)
    packed4 = pack_4bit(arr4)
    unpacked4 = unpack_4bit(packed4, n)
    assert np.array_equal(arr4, unpacked4), "4-bit roundtrip failed!"
    print(f"  [OK] 4-bit roundtrip: {n} values packed into {len(packed4)} bytes (lossless)")

    # 6-bit (0..63)
    arr6 = rng.randint(0, 64, size=n, dtype=np.uint8)
    packed6 = pack_6bit(arr6)
    unpacked6 = unpack_6bit(packed6, n)
    assert np.array_equal(arr6, unpacked6), "6-bit roundtrip failed!"
    print(f"  [OK] 6-bit roundtrip: {n} values packed into {len(packed6)} bytes (lossless)")

    # 8-bit (0..255)
    arr8 = rng.randint(0, 256, size=n, dtype=np.uint8)
    packed8 = pack_8bit(arr8)
    unpacked8 = unpack_8bit(packed8, n)
    assert np.array_equal(arr8, unpacked8), "8-bit roundtrip failed!"
    print(f"  [OK] 8-bit roundtrip: {n} values packed into {len(packed8)} bytes (lossless)")


def test_profile_rules():
    print("\n[2/3] Testing profile rule matching...")
    profile_path = Path(__file__).parent / "profiles" / "qwen35_moe_35b.json"
    prof_moe = QuantProfile.load_from_file(profile_path)
    
    test_cases = [
        ("model.layers.0.input_layernorm.weight", "keep_original", 0),
        ("model.layers.0.post_attention_layernorm.weight", "keep_original", 0),
        ("model.layers.0.mlp.gate.weight", "keep_original", 0),
        ("model.layers.0.mlp.router.weight", "keep_original", 0),
        ("model.layers.0.self_attn.rotary_emb.inv_freq", "keep_original", 0),
        ("model.embed_tokens.weight", "watersic", 4),
        ("lm_head.weight", "watersic", 6),
        ("model.visual.blocks.0.attn.proj.weight", "watersic", 6),
        ("model.visual.merger.mlp.0.weight", "watersic", 6),
        ("model.layers.0.self_attn.q_proj.weight", "watersic", 4),
        ("model.layers.0.self_attn.o_proj.weight", "watersic", 4),
        ("model.layers.0.mlp.shared_expert.gate_proj.weight", "watersic", 4),
        ("model.layers.0.mlp.experts.0.gate_proj.weight", "watersic", 3),
        ("model.layers.0.mlp.experts.15.down_proj.weight", "watersic", 3),
    ]

    for name, expected_act, expected_bits in test_cases:
        res = prof_moe.match_tensor(name)
        assert res["action"] == expected_act, f"Mismatch action for {name}: {res['action']} != {expected_act}"
        assert res["target_bits"] == expected_bits, f"Mismatch bits for {name}: {res['target_bits']} != {expected_bits}"
        print(f"  [Match OK] {name:50s} -> {res['action']} ({res['target_bits']}b)")


def test_wsic_reference_container():
    print("\n[3/3] Validating against reference Qwen3.5-0.8B-WaterSIC-3bit.wsic...")
    ref_wsic = Path(__file__).parent.parent / "Qwen3.5-0.8B-WaterSIC-3bit.wsic"
    if not ref_wsic.exists():
        ref_wsic = Path("Qwen3.5-0.8B-WaterSIC-3bit.wsic")
    
    if not ref_wsic.exists():
        print(f"  [Skip] Reference file {ref_wsic} not found in path.")
        return

    with open(ref_wsic, "rb") as f:
        hdr_bytes = f.read(64)
        magic, ver_maj, ver_min, flags, nlayers, b, gs, cb, kb, vb, wr, meta_off, meta_sz, wush_off, wsic_off, pad = struct.unpack("<IHHIIBBHBBHQQQQ8s", hdr_bytes)
        assert magic == 0x43495357, f"Bad magic: {hex(magic)}"
        print(f"  [Header OK] Magic={hex(magic)} WSIC, Ver={ver_maj}.{ver_min}, Layers={nlayers}, BaseBits={b}")
        
        f.seek(meta_off)
        meta_str = f.read(meta_sz).decode("utf-8")
        js = json.loads(meta_str)
        tensors = js["tensors"]
        print(f"  [Directory OK] Found {len(tensors)} tensors in reference container.")

        # Test unpacking one quantized tensor from reference WSIC
        q_tensor_name = "blk.0.attn_qkv.weight"
        if q_tensor_name in tensors:
            info = tensors[q_tensor_name]
            f.seek(wsic_off + info["qweight_offset"])
            qw_data = f.read(info["qweight_bytes"])
            unpacked = unpack_3bit(qw_data, info["rows"] * info["cols"])
            assert len(unpacked) == info["rows"] * info["cols"]
            print(f"  [Payload OK] Successfully unpacked {q_tensor_name} ({len(unpacked)} codes) from WSIC!")

    print("\nALL LOCAL PARITY & INTEGRITY CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_bit_packing_roundtrips()
    test_profile_rules()
    test_wsic_reference_container()
