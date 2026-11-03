"""
Writer for .wsic (WaterSIC Unified Model Container) supporting mixed-bitrate quantization,
metadata preservation (including vision_config and RoPE sections), and zero-copy 64-byte aligned payload.
"""

import os
import json
import struct
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from .packing import align_to, get_padding

WSIC_MAGIC = 0x43495357  # "WSIC" in little-endian
WSIC_FLAG_HAS_WUSH_KV     = (1 << 0)
WSIC_FLAG_SHARED_CODEBOOK = (1 << 1)
WSIC_FLAG_OUTLIERS_FP16   = (1 << 2)
WSIC_FLAG_HAS_VISION      = (1 << 3)


class WsicContainerWriter:
    def __init__(self, output_path: str | Path, model_config: Dict[str, Any], default_bits: int = 3):
        self.output_path = Path(output_path)
        self.model_config = model_config
        self.default_bits = default_bits
        
        self.has_vision = "vision_config" in model_config
        self.flags = WSIC_FLAG_SHARED_CODEBOOK | WSIC_FLAG_OUTLIERS_FP16
        if self.has_vision:
            self.flags |= WSIC_FLAG_HAS_VISION

        self.tensors_meta: Dict[str, Any] = {}
        self.temp_payload_path = self.output_path.with_suffix(".payload.tmp")
        self.payload_file = open(self.temp_payload_path, "wb")
        self.current_payload_offset = 0

    def add_unquantized_tensor(self, name: str, tensor_data: np.ndarray, dtype_str: str = "F16"):
        """Add unquantized FP16/BF16/F32 tensor to the payload region."""
        data_bytes = tensor_data.tobytes()
        pad = get_padding(len(data_bytes), 64)
        
        offset = self.current_payload_offset
        self.payload_file.write(data_bytes)
        self.payload_file.write(pad)
        total_len = len(data_bytes) + len(pad)
        self.current_payload_offset += total_len

        self.tensors_meta[name] = {
            "dtype": dtype_str,
            "shape": list(tensor_data.shape),
            "data_offset": offset,
            "data_bytes": len(data_bytes)
        }

    def add_quantized_watersic_tensor(
        self,
        name: str,
        shape: List[int],
        packed_qweight: bytes,
        scales_fp64: np.ndarray,
        csr_outliers_bytes: bytes = b"",
        target_bits: int = 3
    ):
        """Add WaterSIC quantized tensor with packed weights, scales, and CSR outliers."""
        # 1. Write qweight (aligned to 64 bytes)
        qw_offset = self.current_payload_offset
        self.payload_file.write(packed_qweight)
        qw_pad = get_padding(len(packed_qweight), 64)
        self.payload_file.write(qw_pad)
        self.current_payload_offset += len(packed_qweight) + len(qw_pad)

        # 2. Write scales (float64, aligned to 64 bytes)
        sc_bytes = scales_fp64.astype(np.float64).tobytes()
        sc_offset = self.current_payload_offset
        self.payload_file.write(sc_bytes)
        sc_pad = get_padding(len(sc_bytes), 64)
        self.payload_file.write(sc_pad)
        self.current_payload_offset += len(sc_bytes) + len(sc_pad)

        # 3. Write outliers if present (aligned to 64 bytes)
        ot_offset = 0
        ot_bytes_len = 0
        num_outliers = 0
        if len(csr_outliers_bytes) > 0:
            ot_offset = self.current_payload_offset
            num_outliers = struct.unpack("<I", csr_outliers_bytes[:4])[0]
            self.payload_file.write(csr_outliers_bytes)
            ot_pad = get_padding(len(csr_outliers_bytes), 64)
            self.payload_file.write(ot_pad)
            ot_bytes_len = len(csr_outliers_bytes)
            self.current_payload_offset += ot_bytes_len + len(ot_pad)

        rows = shape[0] if len(shape) >= 2 else 1
        cols = shape[1] if len(shape) >= 2 else shape[0]

        self.tensors_meta[name] = {
            "dtype": f"watersic_{target_bits}b",
            "shape": [cols, rows],  # GGUF order: dim 0 fastest
            "rows": rows,
            "cols": cols,
            "qweight_offset": qw_offset,
            "qweight_bytes": len(packed_qweight),
            "scales_offset": sc_offset,
            "scales_bytes": len(sc_bytes),
            "codebook_offset": 0,
            "outliers_offset": ot_offset,
            "outliers_bytes": ot_bytes_len,
            "num_outliers": num_outliers
        }

    def finalize(self, num_layers: int = 24, metadata_dict: Dict[str, Any] = None):
        """Finalize assembly: compile metadata JSON, assemble header, and merge files."""
        self.payload_file.close()

        full_meta = {
            "metadata": metadata_dict or {},
            "tensors": self.tensors_meta
        }
        meta_json_str = json.dumps(full_meta, indent=2, ensure_ascii=False)
        meta_json_bytes = meta_json_str.encode("utf-8")

        # Layout calculation
        header_size = 64
        meta_offset = header_size
        meta_size = len(meta_json_bytes)
        meta_pad = get_padding(meta_offset + meta_size, 64)
        
        watersic_region_offset = meta_offset + meta_size + len(meta_pad)

        # Adjust offsets in tensor metadata relative to watersic_region_offset
        # (they were already written as relative to payload start)
        header_bytes = struct.pack(
            "<IHHIIBBHBBHQQQQ8s",
            WSIC_MAGIC,
            1,  # ver_maj
            0,  # ver_min
            self.flags,
            num_layers,
            self.default_bits,
            7,  # group_size 128 (2^7)
            1 << self.default_bits,
            2,  # wush_k_bits default
            2,  # wush_v_bits default
            64, # wush_rank default
            meta_offset,
            meta_size,
            0,  # wush_kv_region_offset (size 0 if not embedded)
            watersic_region_offset,
            b"\x00" * 8
        )
        assert len(header_bytes) == 64

        # Assemble final .wsic file
        with open(self.output_path, "wb") as f_out:
            f_out.write(header_bytes)
            f_out.write(meta_json_bytes)
            f_out.write(meta_pad)
            
            # Stream payload into output file in chunks
            with open(self.temp_payload_path, "rb") as f_pay:
                while chunk := f_pay.read(64 * 1024 * 1024):  # 64 MB chunks
                    f_out.write(chunk)

        # Clean up temporary payload file
        if os.path.exists(self.temp_payload_path):
            os.remove(self.temp_payload_path)

        final_size = os.path.getsize(self.output_path)
        print(f"[WSIC Writer] Container successfully written: {self.output_path}")
        print(f"[WSIC Writer] Total size: {final_size / 1024**2:.2f} MB ({final_size} bytes)")
        return self.output_path
