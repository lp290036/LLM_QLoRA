# -*- coding: utf-8 -*-
"""Windows 安全的 safetensors 加载补丁。

背景：在本机 (Windows + 当前 safetensors 二进制 + 紧张的物理内存) 上，
transformers 通过 safetensors 的 mmap 读取权重时会触发访问冲突 (0xC0000005)，
且物理内存只剩约 2GB，无法把 4.24GB 模型整块读入。

方案：用纯文件 I/O (seek + read) 按需读取每个张量，完全不使用 mmap，
一次只占用单个张量的内存，规避崩溃且对内存友好。

用法：在 `from_pretrained` 之前调用 apply()。
"""
import json
import struct

import torch
import transformers.modeling_utils as _mu


# safetensors dtype 字符串 → torch dtype
_DTYPE = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
# 可选的新 dtype（旧版 torch 可能没有，存在才注册）
for _name, _attr in (("F8_E4M3", "float8_e4m3fn"), ("F8_E5M2", "float8_e5m2"),
                     ("U16", "uint16"), ("U32", "uint32"), ("U64", "uint64")):
    if hasattr(torch, _attr):
        _DTYPE[_name] = getattr(torch, _attr)


class _NoMmapReader:
    """解析 safetensors 头部，按需用 seek/read 读取张量（无 mmap）。"""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header_bytes = f.read(header_len)
        self._header = json.loads(header_bytes.decode("utf-8"))
        self._meta = self._header.pop("__metadata__", None)
        self._data_start = 8 + header_len

    def keys(self):
        return list(self._header.keys())

    def metadata(self):
        return self._meta

    def _dtype_shape(self, key):
        info = self._header[key]
        return _DTYPE[info["dtype"]], info["shape"], info["data_offsets"]

    def get_tensor(self, key):
        dtype, shape, (begin, end) = self._dtype_shape(key)
        nbytes = end - begin
        if nbytes == 0:
            return torch.empty(shape, dtype=dtype)
        with open(self.path, "rb") as f:
            f.seek(self._data_start + begin)
            buf = bytearray(f.read(nbytes))   # bytearray 可写，避免 frombuffer 告警
        t = torch.frombuffer(buf, dtype=dtype)
        if shape:
            t = t.reshape(shape)
        return t.clone()   # 脱离临时 buffer，确保内存安全

    def get_dtype(self, key):
        return self._header[key]["dtype"]

    def get_shape(self, key):
        return self._header[key]["shape"]


class _WinSafeOpen:
    """替代 safetensors.safe_open：接口兼容，但底层用 _NoMmapReader（无 mmap）。"""

    def __init__(self, path, framework="pt", device="cpu"):
        self._r = _NoMmapReader(path)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def keys(self):
        return self._r.keys()

    def metadata(self):
        return self._r.metadata()

    def get_tensor(self, key):
        return self._r.get_tensor(key)

    def get_slice(self, key):
        r = self._r

        class _Slice:
            def __getitem__(self, idx):
                t = r.get_tensor(key)
                return t if idx is Ellipsis else t[idx]

            def get_dtype(self):
                return r.get_dtype(key)

            def get_shape(self):
                return r.get_shape(key)

        return _Slice()


def apply():
    """把 transformers 权重加载用的 safe_open 替换为无 mmap 版本。"""
    _mu.safe_open = _WinSafeOpen
    return _WinSafeOpen
