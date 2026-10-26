import os
import platform
import sys
from array import array
from functools import lru_cache
from typing import Any, Optional


def _cpu_supports_avx2() -> bool:
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return False
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            return "avx2" in f.read().lower()
    except OSError:
        return False


@lru_cache(maxsize=1)
def _openssl_build_dirs() -> tuple[list, list]:
    """Include/lib dirs of the running conda env that carry openssl.

    The extension includes <openssl/sha.h> and links -lcrypto.  In a conda env
    both live under the prefix, which is not on the compiler's default search
    path, so the build failed and every L3 storage lookup raised instead of
    hitting (writes still landed on disk, reads always missed).  Point cc/ld at
    the prefix(es) that actually have the headers/library.
    """
    inc_dirs, lib_dirs = [], []
    prefixes = []
    for attr in ("prefix", "base_prefix"):
        prefix = getattr(sys, attr, None)
        if prefix:
            prefixes.append(prefix)
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        prefixes.append(conda)
    for prefix in prefixes:
        inc = os.path.join(prefix, "include")
        if os.path.isdir(os.path.join(inc, "openssl")):
            inc_dirs.append(inc)
        lib = os.path.join(prefix, "lib")
        if os.path.isdir(lib) and any(
            name.startswith("libcrypto") for name in os.listdir(lib)
        ):
            lib_dirs.append(lib)
    return inc_dirs, lib_dirs


@lru_cache(maxsize=1)
def _load_native_hash_module() -> Any:
    if sys.byteorder != "little" or not sys.platform.startswith("linux"):
        raise RuntimeError(
            "HiCache native hash is only supported on little-endian Linux"
        )

    try:
        from sglang.srt.utils.cpp_extension_loader import (
            load_extension_with_recovery,
        )

        abs_path = os.path.dirname(os.path.abspath(__file__))
        extra_cflags = ["-O3", "-std=c++17", "-DNDEBUG"]
        if _cpu_supports_avx2():
            extra_cflags.append("-mavx2")
        inc_dirs, lib_dirs = _openssl_build_dirs()
        extra_cflags += [f"-I{d}" for d in inc_dirs]
        extra_ldflags = [f"-L{d}" for d in lib_dirs] + ["-lcrypto"]
        return load_extension_with_recovery(
            name="hicache_hash_cpp",
            sources=[f"{abs_path}/hash_binding.cpp"],
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
            with_cuda=False,
        )
    except Exception as exc:
        raise RuntimeError("Failed to load HiCache native hash extension") from exc


def _native_hash_input(token_ids: Any) -> tuple[array, int, int, bool]:
    raw_token_ids = getattr(token_ids, "raw_token_ids", None)
    raw = (
        raw_token_ids()
        if raw_token_ids is not None
        else getattr(token_ids, "token_ids", token_ids)
    )

    logical_len = len(token_ids)
    is_bigram = getattr(token_ids, "is_bigram", False)

    if isinstance(raw, array) and raw.typecode in ("I", "q", "Q", "L"):
        if is_bigram and logical_len > 0 and len(raw) < logical_len + 1:
            raise ValueError("bigram token buffer is shorter than logical length")
        return raw, logical_len, 2 if is_bigram else 1, is_bigram

    if is_bigram:
        return array("I", raw[: logical_len + 1]), logical_len, 2, is_bigram

    if logical_len == 0:
        return array("I"), logical_len, 1, is_bigram

    first_token = raw[0]
    if isinstance(first_token, tuple):
        unit_width = len(first_token)
        return (
            array("I", (elem for token in raw[:logical_len] for elem in token)),
            logical_len,
            unit_width,
            is_bigram,
        )

    return array("I", raw[:logical_len]), logical_len, 1, is_bigram


def get_native_hash(
    token_ids: Any, prior_digest: Optional[bytes], page_size: Optional[int] = None
) -> str | list[str]:
    raw, logical_len, unit_width, is_bigram = _native_hash_input(token_ids)
    return _load_native_hash_module().get_hash(
        raw, logical_len, unit_width, is_bigram, prior_digest, page_size
    )
