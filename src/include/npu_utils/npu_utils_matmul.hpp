/// \file npu_utils_matmul.hpp
/// \brief NPU2 BF16 matmul backend for the open embedding engine.
/// \author FastFlowLM Team
/// \date 2026-08-30
///
/// Wraps the pre-compiled whole-array mlir-aie GEMM artifacts (xclbin + insts)
/// behind a small XRT interface. Each compiled shape is keyed by (K, N) and a
/// padded row count M (512 or 2048). The design computes
///
///     C[M, N] = A[M, K] @ B[K, N]     (BF16 in, BF16 out, fp32 accumulate)
///
/// with row-major A/B/C buffers. Callers transpose projection weights to
/// [K, N] BF16 once at load time.
///
/// The backend degrades gracefully: if the NPU or its artifacts are not
/// available, init() returns false and the engine keeps its CPU path.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace open_embedding {

class NpuMatmul {
public:
    NpuMatmul();
    ~NpuMatmul();
    NpuMatmul(const NpuMatmul&) = delete;
    NpuMatmul& operator=(const NpuMatmul&) = delete;

    /// Load compiled shape artifacts from asset_dir (xclbin + .insts files).
    /// device_id selects the NPU (default "0000:c2:00.1"; set FLM_NPU_DEVICE_ID
    /// to override). Returns false when the NPU or artifacts are unavailable.
    bool init(const std::string& asset_dir, const std::string& device_id);

    bool enabled() const;

    /// Smallest compiled padded row count for (K, N) that covers M rows;
    /// 0 if the shape is absent.
    int m_pad_for(int K, int N, int M) const;

    /// Compute C[M,N] = A[M,K] @ B[K,N] for the compiled pad M (=m_pad_for).
    /// All buffers are row-major. A/B are BF16 (uint16_t); C is FP32 (float).
    bool matmul_bf16(int M, int K, int N, const uint16_t* a, const uint16_t* b,
                     float* c);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace open_embedding