#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdlib>
#include <mutex>
#include "../util.h"
#include "../util.cuh"
#include "../graph.cuh"
#include "exl3_devctx.cuh"
#include "exl3_rdna3.cuh"
#include "exl3_rdna3_moe.cuh"

namespace
{
    std::mutex g_mtx;
    float* g_ws[MAX_DEVICES] = {};
    int* g_counters[MAX_DEVICES] = {};
    uint2* g_xh[MAX_DEVICES] = {};
    float* g_xcs[MAX_DEVICES] = {};
    // A producer wrote the transformed input of the next matmul on (A, suh_tab) into xh / xcs
    struct Prepared { const void* A; const void* suh_tab; int m, k, num_src; uint2* xh; float* xcs; };
    Prepared g_prepared[MAX_DEVICES] = {};
    // Second input workspace, written by the gate/up epilogue while the mgemm still reads g_xh
    uint2* g_xh2[MAX_DEVICES] = {};
    float* g_xcs2[MAX_DEVICES] = {};
    int* g_act_cnt[MAX_DEVICES] = {};
    const void* g_act_suh[MAX_DEVICES] = {};
    constexpr size_t XH2_BYTES = 2ull << 20;
    constexpr size_t XCS2_FLOATS = 1 << 15;
    int g_enabled = -1;
    int g_target_blocks = -1;
}

bool exl3_rdna3_enabled()
{
    #ifdef __HIP_PLATFORM_AMD__
        if (g_enabled < 0)
        {
            const char* e = std::getenv("EXL3_RDNA3_GEMM");
            g_enabled = !(e && e[0] == '0');
        }
        return g_enabled;
    #else
        return false;
    #endif
}

// Workspace and counters are allocated once per device, outside of any graph capture (prepare_ctx
// runs at model load), and never move, so captured graphs keep valid pointers
void exl3_rdna3_prepare(int device)
{
    #ifdef __HIP_PLATFORM_AMD__
        std::lock_guard<std::mutex> lock(g_mtx);
        if (g_ws[device]) return;
        c10::cuda::CUDAGuard guard(device);
        cuda_check(cudaMalloc(&g_ws[device], EXL3_RDNA3_WS_BYTES));
        cuda_check(cudaMalloc(&g_counters[device], EXL3_RDNA3_MAX_COUNTERS * sizeof(int)));
        cuda_check(cudaMemset(g_counters[device], 0, EXL3_RDNA3_MAX_COUNTERS * sizeof(int)));
        cuda_check(cudaMalloc(&g_xh[device], EXL3_RDNA3_XH_BYTES));
        cuda_check(cudaMalloc(&g_xcs[device], EXL3_RDNA3_XCS_FLOATS * sizeof(float)));
        cuda_check(cudaMalloc(&g_xh2[device], XH2_BYTES));
        cuda_check(cudaMalloc(&g_xcs2[device], XCS2_FLOATS * sizeof(float)));
        cuda_check(cudaMalloc(&g_act_cnt[device], EXL3_RDNA3_MAX_COUNTERS * sizeof(int)));
        cuda_check(cudaMemset(g_act_cnt[device], 0, EXL3_RDNA3_MAX_COUNTERS * sizeof(int)));
        cuda_check(cudaDeviceSynchronize());
    #endif
}

#ifdef __HIP_PLATFORM_AMD__
// Rows per pass: smallest instantiated MR >= m (MR 16 with multiple passes above that)
static fp_exl3_rdna3_kernel select_kernel(int size_m, int K, bool half_k, int cb, bool c_fp32, int& mr)
{
    int mr_idx;
    if      (size_m <= 1)  { mr_idx = 0; mr = 1; }
    else if (size_m <= 2)  { mr_idx = 1; mr = 2; }
    else if (size_m <= 3)  { mr_idx = 5; mr = 3; }
    else if (size_m <= 4)  { mr_idx = 2; mr = 4; }
    else if (size_m <= 5)  { mr_idx = 6; mr = 5; }
    else if (size_m <= 6)  { mr_idx = 7; mr = 6; }
    else if (size_m <= 8)  { mr_idx = 3; mr = 8; }
    else if (size_m <= 12) { mr_idx = 8; mr = 12; }
    else                   { mr_idx = 4; mr = 16; }

    switch (K)
    {
        case 1: return exl3_rdna3_get_k1(half_k, cb, mr_idx, c_fp32);
        case 2: return exl3_rdna3_get_k2(half_k, cb, mr_idx, c_fp32);
        case 3: return exl3_rdna3_get_k3(half_k, cb, mr_idx, c_fp32);
        case 4: return exl3_rdna3_get_k4(half_k, cb, mr_idx, c_fp32);
        case 5: return exl3_rdna3_get_k5(half_k, cb, mr_idx, c_fp32);
        case 6: return exl3_rdna3_get_k6(half_k, cb, mr_idx, c_fp32);
        case 7: return exl3_rdna3_get_k7(half_k, cb, mr_idx, c_fp32);
        case 8: return exl3_rdna3_get_k8(half_k, cb, mr_idx, c_fp32);
    }
    return nullptr;
}

static int target_blocks(int device)
{
    if (g_target_blocks < 0)
    {
        const char* e = std::getenv("EXL3_RDNA3_TARGET_BLOCKS");
        // Default: one full residency wave. The multiprocessor count is WGPs on gfx11 (48 on a
        // 7900 XTX) and LDS allows 6 blocks per WGP; a partial second wave roughly doubles the tail
        g_target_blocks = e ? atoi(e) : 6 * DevCtx::instance().get_num_sms(device);
    }
    return g_target_blocks;
}

// k-split so the grid (items independent block columns) covers the device; splits are whole
// 128-element Hadamard blocks
static void choose_splits(int items, int kblocks, size_t ws_per_split, int device, int& splits, int& ks_per_split)
{
    const int tb = target_blocks(device);
    splits = (tb + items - 1) / items;
    splits = MAX(1, MIN(splits, kblocks));
    while (splits > 1 && (size_t) splits * ws_per_split > EXL3_RDNA3_WS_BYTES) splits--;
    if (splits > 1 && items > EXL3_RDNA3_MAX_COUNTERS) splits = 1;
    int kb_per_split = (kblocks + splits - 1) / splits;
    splits = (kblocks + kb_per_split - 1) / kb_per_split;
    ks_per_split = kb_per_split * 8;
}

#endif

bool exl3_rdna3_prepare_input(int device, const void* A, const void* suh_tab, int m, int k, int num_src, uint2** xh, float** xcs)
{
    #ifdef __HIP_PLATFORM_AMD__
        if (!exl3_rdna3_enabled()) return false;
        static const bool fuse = !(std::getenv("EXL3_FUSE_NORM_HAD") && std::getenv("EXL3_FUSE_NORM_HAD")[0] == '0');
        if (!fuse) return false;
        if (k % 128) return false;
        if ((size_t) num_src * m * k * 2 > EXL3_RDNA3_XH_BYTES) return false;
        if ((size_t) num_src * m * (k / 128) > EXL3_RDNA3_XCS_FLOATS) return false;
        if (!g_ws[device]) exl3_rdna3_prepare(device);
        g_prepared[device] = { A, suh_tab, m, k, num_src, g_xh[device], g_xcs[device] };
        *xh = g_xh[device];
        *xcs = g_xcs[device];
        return true;
    #else
        return false;
    #endif
}

static int g_act_epi = -1;   // -1: from EXL3_ACT_EPI (default on)

void exl3_rdna3_act_epi_set(int enable)
{
    g_act_epi = enable;
}

void exl3_rdna3_arm_act_epilogue(int device, const void* down_suh)
{
    if (g_act_epi < 0) g_act_epi = !(std::getenv("EXL3_ACT_EPI") && std::getenv("EXL3_ACT_EPI")[0] == '0');
    g_act_suh[device] = g_act_epi ? down_suh : nullptr;
}

void exl3_rdna3_disarm_act_epilogue(int device)
{
    g_act_suh[device] = nullptr;
}

bool exl3_rdna3_gemm
(
    const half* A,
    const uint16_t* B,
    void* C,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool half_k,
    int cb,
    bool c_fp32,
    const half* suh,
    const half* svh,
    int device,
    cudaStream_t stream,
    Graph* graph,
    const half* A_up,
    const Exl3Rdna3GNorm* gn
)
{
    #ifdef __HIP_PLATFORM_AMD__
        if (!exl3_rdna3_enabled()) return false;
        // A producer may have written this launch's transformed input already (exl3_rdna3_prepare_input
        // with suh as the table, one source): then skip the input kernel. Either way the record is spent
        const Prepared pr = g_prepared[device];
        g_prepared[device].A = nullptr;   // this launch overwrites the input workspace
        if (!suh || !svh) return false;
        if (size_k % 128 || size_n % 128) return false;
        if (K < 1 || K > 8) return false;

        if (!g_ws[device]) exl3_rdna3_prepare(device);

        int mr;
        fp_exl3_rdna3_kernel kernel = select_kernel(size_m, K, half_k, cb, c_fp32, mr);
        if (!kernel) return false;

        const int groups = size_n / 128;
        const int kblocks = size_k / 128;


        // Rows per launch pair, bounded by the transformed-input workspace
        const int max_rows = (int) MIN((size_t) EXL3_RDNA3_XH_BYTES / ((size_t) size_k * 2), (size_t) EXL3_RDNA3_XCS_FLOATS / kblocks);
        TORCH_CHECK(max_rows >= 1, "exl3_rdna3_gemm: k too large for the input workspace");
        if (gn && size_m > max_rows) return false;   // the gate pointer is not re-based per row chunk

        const uint32_t* B32 = (const uint32_t*) B;
        int* counters = g_counters[device];
        float* ws = g_ws[device];
        uint2* xh = g_xh[device];
        float* xcs = g_xcs[device];
        const size_t c_elem = c_fp32 ? 4 : 2;

        for (int r0 = 0; r0 < size_m; r0 += max_rows)
        {
            const int m = MIN(max_rows, size_m - r0);
            const half* A_r = A + (size_t) r0 * size_k;
            void* C_r = (void*) ((char*) C + (size_t) r0 * size_n * c_elem);
            const int row_chunks = (m + mr - 1) / mr;

            int splits, ks_per_split;
            choose_splits(groups * row_chunks, kblocks, (size_t) m * size_n * sizeof(float), device, splits, ks_per_split);

            const int had_tasks = m * kblocks;
            const bool prepared = !graph && r0 == 0 && m == size_m && pr.A == (const void*) A &&
                                  pr.suh_tab == (const void*) suh && pr.m == size_m && pr.k == size_k && pr.num_src == 1;
            if (prepared)
            {
                xh = pr.xh;
                xcs = pr.xcs;
            }
            else
                exl3_rdna3_had_kernel<<<(had_tasks + 7) / 8, 256, 0, stream>>>(A_r, suh, xh, xcs, m, size_k, nullptr, A_up ? A_up + (size_t) r0 * size_k : nullptr,
                                                                               gn ? *gn : Exl3Rdna3GNorm {});
            kernel<<<dim3(groups * splits, row_chunks), EXL3_RDNA3_THREADS, 0, stream>>>
            (
                xh, B32, C_r, m, size_k, size_n, counters, xcs, ws, svh, splits, ks_per_split, Exl3Rdna3MTab {}
            );

            // Graph patching: callers bind the input (A) and output (C) pointers; the transformed
            // input, workspace and counters are static per device
            if (graph && r0 == 0)
            {
                graph->record_param((void*) exl3_rdna3_had_kernel, GP_gemm_A, 0);
                graph->record_param((void*) exl3_rdna3_had_kernel, GP_end, 0);
                graph->record_param((void*) kernel, GP_gemm_C, 2);
                graph->record_param((void*) kernel, GP_end, 0);
            }
            TORCH_CHECK(!graph || size_m <= max_rows, "exl3_rdna3_gemm: graphed matmul exceeds the input workspace");
        }
        cuda_check(cudaPeekAtLastError());
        return true;
    #else
        return false;
    #endif
}

bool exl3_rdna3_mgemm
(
    const half* A,
    const uint64_t* b_tab,
    void* C,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool half_k,
    int cb,
    bool c_fp32,
    const uint64_t* suh_tab,
    const uint64_t* svh_tab,
    const uint64_t* c_tab,
    const int* n_stride_tab,
    const int* src_tab,
    int num_entries,
    int num_src,
    int device,
    cudaStream_t stream,
    Graph* graph
)
{
    #ifdef __HIP_PLATFORM_AMD__
        if (!exl3_rdna3_enabled()) return false;
        if (size_k % 128 || size_n % 128) return false;
        if (K < 1 || K > 8) return false;
        if (num_entries < 1 || num_src < 1 || size_m < 1) return false;

        if (!g_ws[device]) exl3_rdna3_prepare(device);

        int mr;
        fp_exl3_rdna3_kernel kernel = select_kernel(size_m, K, half_k, cb, c_fp32, mr);
        if (!kernel) return false;

        // Single pass only: every source's transformed input has to fit the workspace
        const int kblocks = size_k / 128;
        if ((size_t) num_src * size_m * size_k * 2 > EXL3_RDNA3_XH_BYTES) return false;
        if ((size_t) num_src * size_m * kblocks > EXL3_RDNA3_XCS_FLOATS) return false;

        const int groups = size_n / 128;
        const int row_chunks = (size_m + mr - 1) / mr;
        const int items = groups * row_chunks * num_entries;
        if (items > EXL3_RDNA3_MAX_COUNTERS) return false;

        int splits, ks_per_split;
        choose_splits(items, kblocks, (size_t) num_entries * size_m * size_n * sizeof(float), device, splits, ks_per_split);

        uint2* xh = g_xh[device];
        float* xcs = g_xcs[device];
        const Prepared& pr = g_prepared[device];
        const bool prepared = !graph && pr.A == (const void*) A && pr.suh_tab == (const void*) suh_tab &&
                              pr.m == size_m && pr.k == size_k && pr.num_src == num_src;
        g_prepared[device].A = nullptr;
        if (prepared)
        {
            xh = pr.xh;
            xcs = pr.xcs;
        }
        const int had_tasks = size_m * kblocks;
        if (!prepared)
            exl3_rdna3_had_kernel<<<dim3((had_tasks + 7) / 8, num_src), 256, 0, stream>>>(A, nullptr, xh, xcs, size_m, size_k, suh_tab, nullptr, Exl3Rdna3GNorm {});

        Exl3Rdna3MTab mt { b_tab, svh_tab, c_tab, n_stride_tab, src_tab };

        // Gated MLP epilogue (armed by the caller): gate / up outputs contiguous in C, fp16
        const void* act_suh = g_act_suh[device];
        g_act_suh[device] = nullptr;
        const bool act = act_suh && !graph && num_entries == 2 && !c_fp32 && !c_tab && !n_stride_tab &&
                         (size_t) size_m * size_n * 2 <= XH2_BYTES && (size_t) size_m * groups <= XCS2_FLOATS &&
                         row_chunks * groups <= EXL3_RDNA3_MAX_COUNTERS;
        if (act)
        {
            mt.act_g = (const half*) C;
            mt.act_u = ((const half*) C) + (size_t) size_m * size_n;
            mt.act_suh = (const half*) act_suh;
            mt.act_xh = g_xh2[device];
            mt.act_xcs = g_xcs2[device];
            mt.act_cnt = g_act_cnt[device];
            mt.act_k = size_n;
        }
        kernel<<<dim3(groups * splits, row_chunks, num_entries), EXL3_RDNA3_THREADS, 0, stream>>>
        (
            xh, nullptr, C, size_m, size_k, size_n, g_counters[device], xcs, g_ws[device], nullptr,
            splits, ks_per_split, mt
        );
        if (act)
            g_prepared[device] = { C, act_suh, size_m, size_n, 1, g_xh2[device], g_xcs2[device] };

        // Graph patching: the input (A) only; output tables and pointers are static
        if (graph)
        {
            graph->record_param((void*) exl3_rdna3_had_kernel, GP_mgemm_A, 0);
            graph->record_param((void*) exl3_rdna3_had_kernel, GP_end, 0);
            graph->record_param((void*) kernel, GP_mgemm_C, 2);
            graph->record_param((void*) kernel, GP_end, 0);
        }
        cuda_check(cudaPeekAtLastError());
        return true;
    #else
        return false;
    #endif
}

// -------------------------------------------------------------------------------------------------
// Routed MoE experts for decode-sized batches (exl3_rdna3_moe_decode). Every (token, top-k slot) pair
// is one matmul entry with its own transformed input (the expert's suh differs), so a token batch
// needs no host-side grouping and no sync: prep (tables + gate/up input transforms) -> gate/up
// matmul -> act (silu(g) * u -> down input transform) -> down matmul -> fixed-order weighted sum.
// Pairs whose expert is not resident here (CPU split tail, other TP shard) get null trellis
// pointers; the matmul kernel skips such entries. An optional shared expert (with an optional
// sigmoid gate on the input) rides along as one extra pair per token: its own gate/up and down
// launches (it may have another bitrate), its gate dot in the prep kernel, its sum in the combine
// -------------------------------------------------------------------------------------------------

#include "exl3_rdna3_had.cuh"
#include "bits_k.cuh"

#ifdef __HIP_PLATFORM_AMD__
namespace {

// Shared expert pointer table layout (int64): g tr/suh/svh, u tr/suh/svh, d tr/suh/svh
enum { SH_GT, SH_GSUH, SH_GSVH, SH_UT, SH_USUH, SH_USVH, SH_DT, SH_DSUH, SH_DSVH, SH_N };

__global__ __launch_bounds__(256)
void moe_prep_kernel
(
    const half* __restrict__ y,
    const int64_t* __restrict__ sel,
    int topk,
    int P,                          // routed pairs; rows P .. P + T - 1 of the grid are shared-expert rows
    int num_local,
    const uint64_t* __restrict__ g_tr, const uint64_t* __restrict__ g_suh, const uint64_t* __restrict__ g_svh,
    const uint64_t* __restrict__ u_tr, const uint64_t* __restrict__ u_suh, const uint64_t* __restrict__ u_svh,
    const uint64_t* __restrict__ sh,        // shared expert table or null
    const half* __restrict__ sh_gate,       // (size_k) shared-expert input gate, or null
    float* __restrict__ gate_out,           // (T) sigmoid(y . sh_gate)
    uint64_t* __restrict__ b_tab,
    uint64_t* __restrict__ s_tab,
    uint2* __restrict__ xh,
    float* __restrict__ xcs,
    int size_k
)
{
    const int p = blockIdx.y;
    const bool shared = p >= P;
    const int t = shared ? p - P : p / topk;
    int64_t e = 0;
    bool active = true;
    const uint64_t *gt = g_tr, *gs = g_suh, *gv = g_svh, *ut = u_tr, *us = u_suh, *uv = u_svh;
    if (shared)
    {
        gt = sh + SH_GT; gs = sh + SH_GSUH; gv = sh + SH_GSVH;
        ut = sh + SH_UT; us = sh + SH_USUH; uv = sh + SH_USVH;
        if (blockIdx.x == 0)
        {
            // gate = sigmoid(y[t] . w), whole block
            __shared__ float red[8];
            float a = 0.0f;
            if (sh_gate)
                for (int i = threadIdx.x; i < size_k; i += 256)
                    a = fmaf(__half2float(y[(size_t) t * size_k + i]), __half2float(sh_gate[i]), a);
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor(a, o);
            if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = a;
            __syncthreads();
            if (threadIdx.x == 0)
            {
                float v = 0.0f;
                #pragma unroll
                for (int w = 0; w < 8; ++w) v += red[w];
                gate_out[t] = sh_gate ? 1.0f / (1.0f + __expf(-v)) : 1.0f;
            }
        }
    }
    else
    {
        e = sel[p];
        active = e >= 0 && e < num_local;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        b_tab[2 * p]     = active ? gt[e] : 0;
        b_tab[2 * p + 1] = active ? ut[e] : 0;
        s_tab[2 * p]     = active ? gv[e] : 0;
        s_tab[2 * p + 1] = active ? uv[e] : 0;
    }
    if (!active) return;
    const int kblocks = size_k / 128;
    const int task = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (task >= 2 * kblocks) return;
    const int proj = task / kblocks;
    const int c = task % kblocks;
    const int lane = threadIdx.x & 31;
    const half2* ap = (const half2*) (y + (size_t) t * size_k + c * 128 + lane * 4);
    const half* suh = (const half*) (proj ? us[e] : gs[e]);
    const int src = 2 * p + proj;
    exl3_rdna3_had::transform_block(ap[0], ap[1], suh, xh + (size_t) src * (size_k / 16) * 4,
                                    xcs + (size_t) src * kblocks, 0, c, size_k, lane);
}

__global__ __launch_bounds__(256)
void moe_act_kernel
(
    const half* __restrict__ c_gu,
    const int64_t* __restrict__ sel,
    int P,
    int num_local,
    const uint64_t* __restrict__ d_tr, const uint64_t* __restrict__ d_suh, const uint64_t* __restrict__ d_svh,
    const uint64_t* __restrict__ sh,
    uint64_t* __restrict__ b_tab,
    uint64_t* __restrict__ s_tab,
    uint2* __restrict__ xh,
    float* __restrict__ xcs,
    int size_i
)
{
    const int p = blockIdx.y;
    const bool shared = p >= P;
    int64_t e = 0;
    bool active = true;
    const uint64_t *dt = d_tr, *ds = d_suh, *dv = d_svh;
    if (shared) { dt = sh + SH_DT; ds = sh + SH_DSUH; dv = sh + SH_DSVH; }
    else
    {
        e = sel[p];
        active = e >= 0 && e < num_local;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        b_tab[p] = active ? dt[e] : 0;
        s_tab[p] = active ? dv[e] : 0;
    }
    if (!active) return;
    const int kblocks = size_i / 128;
    const int c = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (c >= kblocks) return;
    const int lane = threadIdx.x & 31;
    // Same half-precision rounding as the dense gate/up epilogue (and act_mul_kernel_h)
    auto sigmoid2 = [] (half2 x) -> half2
    {
        const half2 one = __float2half2_rn(1.0f);
        return h2rcp(__hadd2(one, h2exp(__hneg2(x))));
    };
    const size_t o = (size_t) c * 128 + lane * 4;
    const half2* gp = (const half2*) (c_gu + (size_t) (2 * p) * size_i + o);
    const half2* up = (const half2*) (c_gu + (size_t) (2 * p + 1) * size_i + o);
    half2 x01 = gp[0], x23 = gp[1];
    x01 = __hmul2(__hmul2(x01, sigmoid2(x01)), up[0]);
    x23 = __hmul2(__hmul2(x23, sigmoid2(x23)), up[1]);
    exl3_rdna3_had::transform_block(x01, x23, (const half*) ds[e], xh + (size_t) p * (size_i / 16) * 4,
                                    xcs + (size_t) p * kblocks, 0, c, size_i, lane);
}

// out[t] = sum_j w[t, j] * down(t, j) over resident pairs in slot order, + gate[t] * shared(t)
__global__ __launch_bounds__(256)
void moe_combine_kernel
(
    const float* __restrict__ c_d,
    const uint64_t* __restrict__ b_tab,
    const half* __restrict__ w,
    const float* __restrict__ gate,     // (T) or null: no shared expert
    int topk,
    int P,
    int size_n,
    float* __restrict__ out
)
{
    const int t = blockIdx.y;
    const int i = (blockIdx.x * 256 + threadIdx.x) * 4;
    if (i >= size_n) return;
    float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int j = 0; j < topk; ++j)
    {
        const int p = t * topk + j;
        if (!b_tab[p]) continue;
        const float wj = __half2float(w[p]);
        const float4 v = *(const float4*) (c_d + (size_t) p * size_n + i);
        s.x += wj * v.x; s.y += wj * v.y; s.z += wj * v.z; s.w += wj * v.w;
    }
    if (gate)
    {
        const float g = gate[t];
        const float4 v = *(const float4*) (c_d + (size_t) (P + t) * size_n + i);
        s.x += g * v.x; s.y += g * v.y; s.z += g * v.z; s.w += g * v.w;
    }
    *(float4*) (out + (size_t) t * size_n + i) = s;
}

}  // namespace
#endif

bool exl3_rdna3_moe_decode
(
    const at::Tensor& y,        // (T, H) half, routed input
    const at::Tensor& sel,      // (T, topk) int64, local expert index (outside [0, num_local): not here)
    const at::Tensor& w,        // (T, topk) half, routing weights
    const at::Tensor& g_tr, const at::Tensor& g_suh, const at::Tensor& g_svh,   // (E) int64 pointer tables
    const at::Tensor& u_tr, const at::Tensor& u_suh, const at::Tensor& u_svh,
    const at::Tensor& d_tr, const at::Tensor& d_suh, const at::Tensor& d_svh,
    double K_gu,
    double K_d,
    bool mcg,
    bool mul1,
    int64_t num_local,
    at::Tensor& tabs,           // int64 scratch, >= 6 * (T * topk + T) + T
    at::Tensor& c_gu,           // (>= 2 * (T * topk + T), I) half scratch
    at::Tensor& c_d,            // (>= T * topk + T, H) float scratch
    at::Tensor& out,            // (T, H) float, overwritten
    const c10::optional<at::Tensor>& sh_tab,    // (9) int64 shared expert pointers (SH_* order)
    const c10::optional<at::Tensor>& sh_gate,   // (H) half shared expert input gate
    double K_sh
)
{
    #ifdef __HIP_PLATFORM_AMD__
        if (!exl3_rdna3_enabled()) return false;
        const at::cuda::OptionalCUDAGuard device_guard(y.device());
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        int device;
        cudaGetDevice(&device);

        TORCH_CHECK_DTYPE(y, kHalf);
        TORCH_CHECK_DTYPE(sel, kLong);
        TORCH_CHECK_DTYPE(w, kHalf);
        TORCH_CHECK_DTYPE(tabs, kLong);
        TORCH_CHECK_DTYPE(c_gu, kHalf);
        TORCH_CHECK_DTYPE(c_d, kFloat);
        TORCH_CHECK_DTYPE(out, kFloat);
        TORCH_CHECK(y.is_contiguous() && sel.is_contiguous() && w.is_contiguous() && out.is_contiguous(),
                    "exl3_rdna3_moe_decode: contiguous inputs required");
        const bool shared = sh_tab.has_value();
        const int T = y.size(0);
        const int H = y.size(1);
        const int topk = sel.size(1);
        const int P = T * topk;
        const int Ts = shared ? T : 0;
        const int PS = P + Ts;
        const int I = c_gu.size(1);
        if (H % 128 || I % 128) return false;
        TORCH_CHECK(tabs.numel() >= 6 * PS + Ts && c_gu.size(0) >= 2 * PS && c_d.size(0) >= PS && c_d.size(1) == H,
                    "exl3_rdna3_moe_decode: scratch too small");
        if ((size_t) 2 * PS * H * 2 > EXL3_RDNA3_XH_BYTES || (size_t) 2 * PS * (H / 128) > EXL3_RDNA3_XCS_FLOATS) return false;
        if ((size_t) PS * I * 2 > XH2_BYTES || (size_t) PS * (I / 128) > XCS2_FLOATS) return false;
        if (shared)
        {
            TORCH_CHECK_DTYPE(sh_tab.value(), kLong);
            TORCH_CHECK(sh_tab.value().numel() >= 9, "exl3_rdna3_moe_decode: shared table");
            if (sh_gate.has_value()) { TORCH_CHECK_DTYPE(sh_gate.value(), kHalf); TORCH_CHECK(sh_gate.value().numel() == H, "shared gate"); }
        }

        TORCH_CHECK(!(mcg && mul1), "exl3_rdna3_moe_decode: both mcg and mul1");
        const int cb = mul1 ? 2 : (mcg ? 1 : 0);
        const BitsK bgu = bits_from_K((float) K_gu);
        const BitsK bd = bits_from_K((float) K_d);
        int mr;
        fp_exl3_rdna3_kernel k_gu = select_kernel(1, bgu.bits, bgu.half, cb, false, mr);
        fp_exl3_rdna3_kernel k_d = select_kernel(1, bd.bits, bd.half, cb, true, mr);
        fp_exl3_rdna3_kernel k_sgu = nullptr, k_sd = nullptr;
        if (shared)
        {
            const BitsK bs = bits_from_K((float) K_sh);
            k_sgu = select_kernel(1, bs.bits, bs.half, cb, false, mr);
            k_sd = select_kernel(1, bs.bits, bs.half, cb, true, mr);
            if (!k_sgu || !k_sd) return false;
        }
        if (!k_gu || !k_d) return false;

        const int groups_gu = I / 128, kb_gu = H / 128;
        const int groups_d = H / 128, kb_d = I / 128;
        if (groups_gu * 2 * P > EXL3_RDNA3_MAX_COUNTERS || groups_d * P > EXL3_RDNA3_MAX_COUNTERS) return false;

        if (!g_ws[device]) exl3_rdna3_prepare(device);
        g_prepared[device].A = nullptr;   // the input workspaces are overwritten here

        uint64_t* tb = (uint64_t*) tabs.data_ptr();
        uint64_t* b_gu = tb;
        uint64_t* s_gu = tb + 2 * PS;
        uint64_t* b_d = tb + 4 * PS;
        uint64_t* s_d = tb + 5 * PS;
        float* gate = shared ? (float*) (tb + 6 * PS) : nullptr;
        auto P64 = [] (const at::Tensor& t) { return (const uint64_t*) t.data_ptr(); };
        const uint64_t* sh = shared ? P64(sh_tab.value()) : nullptr;
        const half* shg = shared && sh_gate.has_value() ? (const half*) sh_gate.value().data_ptr() : nullptr;
        uint2* xh = g_xh[device];
        float* xcs = g_xcs[device];
        uint2* xh2 = g_xh2[device];
        float* xcs2 = g_xcs2[device];
        half* cgu = (half*) c_gu.data_ptr();
        float* cd = (float*) c_d.data_ptr();

        moe_prep_kernel<<<dim3((2 * kb_gu + 7) / 8, PS), 256, 0, stream>>>
        (
            (const half*) y.data_ptr(), (const int64_t*) sel.data_ptr(), topk, P, (int) num_local,
            P64(g_tr), P64(g_suh), P64(g_svh), P64(u_tr), P64(u_suh), P64(u_svh),
            sh, shg, gate, b_gu, s_gu, xh, xcs, H
        );

        auto launch = [&] (fp_exl3_rdna3_kernel k, int entries, int groups, int kblocks, int size_k, int size_n,
                           const uint64_t* b, const uint64_t* s, uint2* x, float* xs, void* C, bool fp32)
        {
            int splits, ks;
            choose_splits(groups * entries, kblocks, (size_t) entries * size_n * sizeof(float), device, splits, ks);
            Exl3Rdna3MTab mt {};
            mt.b = b;
            mt.svh = s;
            k<<<dim3(groups * splits, 1, entries), EXL3_RDNA3_THREADS, 0, stream>>>
            (
                x, nullptr, C, 1, size_k, size_n, g_counters[device], xs, g_ws[device], nullptr, splits, ks, mt
            );
            (void) fp32;
        };

        launch(k_gu, 2 * P, groups_gu, kb_gu, H, I, b_gu, s_gu, xh, xcs, cgu, false);
        if (shared)
            launch(k_sgu, 2 * Ts, groups_gu, kb_gu, H, I, b_gu + 2 * P, s_gu + 2 * P,
                   xh + (size_t) 2 * P * (H / 16) * 4, xcs + (size_t) 2 * P * kb_gu, cgu + (size_t) 2 * P * I, false);

        moe_act_kernel<<<dim3((kb_d + 7) / 8, PS), 256, 0, stream>>>
        (
            cgu, (const int64_t*) sel.data_ptr(), P, (int) num_local,
            P64(d_tr), P64(d_suh), P64(d_svh), sh, b_d, s_d, xh2, xcs2, I
        );

        launch(k_d, P, groups_d, kb_d, I, H, b_d, s_d, xh2, xcs2, cd, true);
        if (shared)
            launch(k_sd, Ts, groups_d, kb_d, I, H, b_d + P, s_d + P,
                   xh2 + (size_t) P * (I / 16) * 4, xcs2 + (size_t) P * kb_d, cd + (size_t) P * H, true);

        moe_combine_kernel<<<dim3((H / 4 + 255) / 256, T), 256, 0, stream>>>
        (
            cd, b_d, (const half*) w.data_ptr(), gate, topk, P, H, (float*) out.data_ptr()
        );
        cuda_check(cudaPeekAtLastError());
        return true;
    #else
        return false;
    #endif
}
