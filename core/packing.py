"""
Binary packing utilities for WaterSIC quantized integers (3-bit, 4-bit, 6-bit, 8-bit)
and CSR sparse outliers with 64-byte alignment.
"""

import struct
import numpy as np


def align_to(size: int, alignment: int = 64) -> int:
    return ((size + alignment - 1) // alignment) * alignment


def get_padding(size: int, alignment: int = 64) -> bytes:
    pad_len = (alignment - (size % alignment)) % alignment
    return b"\x00" * pad_len


def pack_3bit(codes: np.ndarray) -> bytes:
    """Pack an array of uint8 values (0..7) into 3-bit packed bytes (8 codes per 3 bytes)."""
    flat = codes.reshape(-1)
    pad = (8 - (len(flat) % 8)) % 8
    if pad > 0:
        flat = np.pad(flat, (0, pad), mode="constant")
    
    blocks = flat.reshape(-1, 8).astype(np.uint32)
    b0 = (blocks[:, 0] & 7) | ((blocks[:, 1] & 7) << 3) | ((blocks[:, 2] & 3) << 6)
    b1 = ((blocks[:, 2] >> 2) & 1) | ((blocks[:, 3] & 7) << 1) | ((blocks[:, 4] & 7) << 4) | ((blocks[:, 5] & 1) << 7)
    b2 = ((blocks[:, 5] >> 1) & 3) | ((blocks[:, 6] & 7) << 2) | ((blocks[:, 7] & 7) << 5)
    
    packed = np.column_stack([b0, b1, b2]).astype(np.uint8)
    return packed.tobytes()


def unpack_3bit(packed_bytes: bytes, numel: int) -> np.ndarray:
    """Unpack 3-bit packed bytes into uint8 array."""
    data = np.frombuffer(packed_bytes, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
    b0 = data[:, 0]
    b1 = data[:, 1]
    b2 = data[:, 2]
    
    v0 = b0 & 7
    v1 = (b0 >> 3) & 7
    v2 = (b0 >> 6) | ((b1 & 1) << 2)
    v3 = (b1 >> 1) & 7
    v4 = (b1 >> 4) & 7
    v5 = (b1 >> 7) | ((b2 & 3) << 1)
    v6 = (b2 >> 2) & 7
    v7 = (b2 >> 5) & 7
    
    unpacked = np.column_stack([v0, v1, v2, v3, v4, v5, v6, v7]).reshape(-1)
    return unpacked[:numel].astype(np.uint8)


def pack_4bit(codes: np.ndarray) -> bytes:
    """Pack an array of uint8 values (0..15) into 4-bit packed bytes (2 codes per byte)."""
    flat = codes.reshape(-1)
    if len(flat) % 2 != 0:
        flat = np.pad(flat, (0, 1), mode="constant")
    pairs = flat.reshape(-1, 2).astype(np.uint8)
    packed = (pairs[:, 0] & 0x0F) | ((pairs[:, 1] & 0x0F) << 4)
    return packed.tobytes()


def unpack_4bit(packed_bytes: bytes, numel: int) -> np.ndarray:
    """Unpack 4-bit packed bytes into uint8 array."""
    data = np.frombuffer(packed_bytes, dtype=np.uint8)
    v0 = data & 0x0F
    v1 = (data >> 4) & 0x0F
    unpacked = np.column_stack([v0, v1]).reshape(-1)
    return unpacked[:numel].astype(np.uint8)


def pack_6bit(codes: np.ndarray) -> bytes:
    """Pack an array of uint8 values (0..63) into 6-bit packed bytes (4 codes per 3 bytes)."""
    flat = codes.reshape(-1)
    pad = (4 - (len(flat) % 4)) % 4
    if pad > 0:
        flat = np.pad(flat, (0, pad), mode="constant")
    blocks = flat.reshape(-1, 4).astype(np.uint32)
    b0 = (blocks[:, 0] & 0x3F) | ((blocks[:, 1] & 0x03) << 6)
    b1 = ((blocks[:, 1] >> 2) & 0x0F) | ((blocks[:, 2] & 0x0F) << 4)
    b2 = ((blocks[:, 2] >> 4) & 0x03) | ((blocks[:, 3] & 0x3F) << 2)
    packed = np.column_stack([b0, b1, b2]).astype(np.uint8)
    return packed.tobytes()


def unpack_6bit(packed_bytes: bytes, numel: int) -> np.ndarray:
    """Unpack 6-bit packed bytes into uint8 array."""
    data = np.frombuffer(packed_bytes, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
    b0, b1, b2 = data[:, 0], data[:, 1], data[:, 2]
    v0 = b0 & 0x3F
    v1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2)
    v2 = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4)
    v3 = (b2 >> 2) & 0x3F
    unpacked = np.column_stack([v0, v1, v2, v3]).reshape(-1)
    return unpacked[:numel].astype(np.uint8)


def pack_8bit(codes: np.ndarray) -> bytes:
    """Pack array of uint8 (0..255)."""
    return codes.astype(np.uint8).tobytes()


def unpack_8bit(packed_bytes: bytes, numel: int) -> np.ndarray:
    """Unpack array of uint8."""
    return np.frombuffer(packed_bytes, dtype=np.uint8)[:numel]


def pack_csr_outliers(row_ptr: np.ndarray, col_idx: np.ndarray, deltas: np.ndarray, rows: int) -> bytes:
    """
    Serialize CSR sparse outliers:
    - num_outliers (uint32)
    - rows (uint32)
    - row_ptr ((rows + 1) * uint32)
    - col_idx (num_outliers * uint16)
    - deltas (num_outliers * int16)
    """
    num_outliers = len(col_idx)
    hdr = struct.pack("<II", num_outliers, rows)
    r_bytes = row_ptr.astype(np.uint32).tobytes()
    c_bytes = col_idx.astype(np.uint16).tobytes()
    d_bytes = deltas.astype(np.int16).tobytes()
    return hdr + r_bytes + c_bytes + d_bytes
