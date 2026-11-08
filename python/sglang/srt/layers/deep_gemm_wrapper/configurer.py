import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils import (
    get_device_sm,
    is_cuda,
    is_musa,
)

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_musa = is_musa()


def _sm120_deep_gemm_apis_available() -> bool:
    try:
        import deep_gemm
    except (ImportError, OSError, RuntimeError):
        return False
    return all(
        callable(getattr(deep_gemm, name, None))
        for name in (
            "fp8_einsum",
            "m_grouped_fp8_fp4_gemm_nt_contiguous",
            "transform_sf_into_required_layout",
        )
    )


def _compute_enable_deep_gemm():
    if not (_is_cuda or _is_musa):
        return False
    if not envs.SGLANG_ENABLE_JIT_DEEPGEMM.get():
        return False

    sm_version = get_device_sm()
    if (_is_cuda and sm_version < 90) or (_is_musa and sm_version < 31):
        return False
    # SM120/SM121 support (mma.sync block-scale, no TMEM) landed in DeepGEMM#324;
    # probe every API used by the SM120 DSV4 paths since installed builds may
    # expose fp8_einsum but still predate the SM120 kernels.
    if sm_version in (120, 121) and not _sm120_deep_gemm_apis_available():
        logger.warning(
            "deep_gemm on SM120 is missing its SM120 kernels (looked for "
            "fp8_einsum, m_grouped_fp8_fp4_gemm_nt_contiguous and "
            "transform_sf_into_required_layout). DeepSeek-V4.1 will run the "
            "portable kernels instead: correct, but noticeably slower. Install "
            "the deep_gemm build shipped with this release (2.8.0 with SM120 "
            "support) to get the fast path."
        )
        return False

    try:
        import deep_gemm  # noqa: F401
    except (ImportError, OSError, RuntimeError):
        return False

    return True


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and get_platform().is_sm100
DEEPGEMM_SCALE_UE8M0 = ENABLE_JIT_DEEPGEMM and (
    get_platform().is_sm100 or get_device_sm() in (120, 121)
)
DEEPGEMM_NEED_TMA_ALIGNED_SCALES = not (DEEPGEMM_SCALE_UE8M0 or _is_musa)


_PAGED_SPARSE_OFF_REASON = ""


def _supports_paged_sparse_mqa_logits() -> bool:
    # Call-time (not import-time) evaluation: heterogeneous TP mixes SM86 and
    # SM120 ranks, and import happens before torch.cuda.set_device(rank), so an
    # import-time probe would pin cuda:0's SM onto every rank. The SM120/121
    # paged-sparse APIs landed in DeepGEMM#324 (same kernels as SM100 here).
    global _PAGED_SPARSE_OFF_REASON
    if not envs.SGLANG_ENABLE_JIT_DEEPGEMM.get():
        _PAGED_SPARSE_OFF_REASON = "env SGLANG_ENABLE_JIT_DEEPGEMM=0"
        return False
    sm = get_device_sm()
    if not (sm // 10 == 10 or sm in (120, 121)):
        _PAGED_SPARSE_OFF_REASON = (
            f"sm={sm} dev={torch.cuda.current_device() if torch.cuda.is_initialized() else 'pre-init(=0)'}"
        )
        return False
    try:
        import deep_gemm
    except (ImportError, OSError, RuntimeError) as e:
        _PAGED_SPARSE_OFF_REASON = f"deep_gemm import failed: {type(e).__name__}: {e}"
        return False
    ok = all(
        callable(getattr(deep_gemm, name, None))
        for name in (
            "get_paged_sparse_mqa_logits_metadata",
            "fp8_fp4_paged_sparse_mqa_logits",
        )
    )
    if not ok:
        _PAGED_SPARSE_OFF_REASON = (
            f"deep_gemm {getattr(deep_gemm, '__version__', '?')} lacks paged APIs"
        )
    else:
        _PAGED_SPARSE_OFF_REASON = ""
    return ok


def resolve_grouped_fp8_gemm_nt_masked():
    """The masked grouped FP8 GEMM of the installed DeepGEMM build.

    Upstream DeepGEMM (nv_dev lineage, PR #447+) exports only
    ``m_grouped_fp8_gemm_nt_masked`` (the unified fp8/fp4 kernel alias);
    sgl-deep-gemm shipped the legacy ``fp8_m_grouped_gemm_nt_masked``
    spelling. Resolved at call time so a swapped build takes effect without
    a re-import; loud failure when neither name exists.
    """
    import deep_gemm

    fn = getattr(deep_gemm, "fp8_m_grouped_gemm_nt_masked", None) or getattr(
        deep_gemm, "m_grouped_fp8_gemm_nt_masked", None
    )
    if fn is None:
        raise RuntimeError(
            f"deep_gemm {getattr(deep_gemm, '__version__', '?')} exports neither "
            "fp8_m_grouped_gemm_nt_masked nor m_grouped_fp8_gemm_nt_masked"
        )
    return fn


def __getattr__(name):
    # PEP 562: compute on EVERY access, never cache into module globals.
    # Caching (even post-init) poisons heterogeneous TP: the import storm can
    # initialize CUDA with current_device()==0 before any rank binds, so a
    # cached first read pins cuda:0's SM onto every rank. The flag is read only
    # at backend/pool init, long after run_scheduler_process bound the rank.
    if name == "DEEPGEMM_PAGED_SPARSE_MQA_LOGITS":
        return _supports_paged_sparse_mqa_logits()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
