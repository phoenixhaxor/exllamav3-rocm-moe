#include <cuda_fp16.h>
#include "kv_stream.cuh"
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"

/*
KV streaming for the QSA attention layers (design after Strata's kv_stream, MIT,
github.com/Niko1221/Strata, include/strata/kernels/kv_stream.hpp).

The authoritative paged K/V of a streamed layer lives in pinned, device-mapped host memory
(kv_host_alloc), in exactly the layout of the resident cache, so every existing writer and the
dense-regime readers keep working on it through the mapping. The sparse decode reader instead
goes through a small VRAM pool of slots, one slot = one 4-token indexer block (all K/V heads,
codes and scales):

  page_table[pb]   physical block pb (= page * blocks_per_page + block in page) -> slot, or -1
  slot_block[s]    slot -> physical block, or -1
  slot_stamp[s]    the resolve epoch that last used the slot
  slot_ref[s]      clock reference bit

kv_stream_resolve (one block of RT threads) walks a call's selected cache positions, stamps the
hits, claims each missing block once (-1 -> -2), picks one victim per miss with a clock sweep
that never takes a slot this call uses, re-points the tables and rewrites the selection as slot
rows (slot * 4 + position % 4). kv_stream_copy then reads the missed blocks from the host copy.
Keys are PHYSICAL blocks, so pages shared between jobs share slots, and every write to the host
copy (new tokens, page copies) invalidates the blocks it touches (kv_stream_invalidate*): a
slot is never stale.
*/

#define RT 1024
#define BLOCK_TOKENS 4
#define CTL_EPOCH 0
#define CTL_HAND 1
#define CTL_PLACED 2
#define CTL_OVERFLOW 3

at::Tensor kv_host_alloc
(
    std::vector<int64_t> sizes,
    at::ScalarType dtype,
    int64_t device
)
{
    int64_t numel = 1;
    for (auto s : sizes) numel *= s;
    size_t bytes = (size_t) numel * at::elementSize(dtype);
    void* host = nullptr;
    cuda_check(cudaHostAlloc(&host, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
    memset(host, 0, bytes);
    void* dev = nullptr;
    cuda_check(cudaHostGetDevicePointer(&dev, host, 0));
    auto opts = at::TensorOptions().dtype(dtype).device(at::kCUDA, device);
    return at::from_blob(dev, sizes, [host](void*) { cudaFreeHost(host); }, opts);
}

// Physical block of a cache position through a sequence's block table row
__device__ __forceinline__ int phys_block(const int* bt_row, int pos, int page_size)
{
    int bpp = page_size / BLOCK_TOKENS;
    return bt_row[pos / page_size] * bpp + (pos % page_size) / BLOCK_TOKENS;
}

__device__ __forceinline__ int block_scan_excl(int v, int* warp_sums, int& total)
{
    const int lane = threadIdx.x & 31;
    const int w = threadIdx.x >> 5;
    int x = v;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1)
    {
        int y = __shfl_up_sync(EXL3_FULL_MASK, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_sums[w] = x;
    __syncthreads();
    if (w == 0)
    {
        int t = warp_sums[lane];
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1)
        {
            int y = __shfl_up_sync(EXL3_FULL_MASK, t, o);
            if (lane >= o) t += y;
        }
        warp_sums[lane] = t;
    }
    __syncthreads();
    total = warp_sums[RT / 32 - 1];
    int excl = x - v + (w > 0 ? warp_sums[w - 1] : 0);
    __syncthreads();
    return excl;
}

__global__ __launch_bounds__(RT)
void kv_stream_resolve_kernel
(
    int* __restrict__ page_table,
    int* __restrict__ slot_block,
    int* __restrict__ slot_stamp,
    int* __restrict__ slot_ref,
    int* __restrict__ ctl,
    int64_t* __restrict__ stats,
    int* __restrict__ miss_block,
    int* __restrict__ miss_slot,
    const int* __restrict__ indices,    // (R, K_pad) cache positions, -1 padded
    const int* __restrict__ block_table,// (bsz, npr)
    int* __restrict__ out,              // (R, K_pad) slot rows, -1 padded
    int R,
    int K_pad,
    int seq,
    int npr,
    int page_size,
    int n_slots
)
{
    __shared__ int s_nmiss, s_cut;
    __shared__ int warp_sums[RT / 32];
    const int epoch = ctl[CTL_EPOCH] + 1;
    if (threadIdx.x == 0) s_nmiss = 0;
    __syncthreads();

    // 1. Hits take this epoch and their reference bit; a missing block is claimed once (-1 -> -2)
    int lookups = 0;
    const int n = R * K_pad;
    for (int i = threadIdx.x; i < n; i += RT)
    {
        int pos = indices[i];
        if (pos < 0) continue;
        int r = i / K_pad;
        int pb = phys_block(block_table + (r / seq) * npr, pos, page_size);
        lookups++;
        int sl = page_table[pb];
        if (sl >= 0)
        {
            slot_stamp[sl] = epoch;
            slot_ref[sl] = 1;
        }
        else if (sl == -1 && atomicCAS(&page_table[pb], -1, -2) == -1)
        {
            int k = atomicAdd(&s_nmiss, 1);
            if (k < n_slots) miss_block[k] = pb;
            else page_table[pb] = -1;    // overflow, flagged below (the caller sizes n_slots)
        }
    }
    __syncthreads();

    // 2. One victim per miss, clock sweep from the hand. A slot this call uses (stamp == epoch) is never
    //    taken; a referenced one loses its bit as the hand passes and is taken on a later pass
    const int nmiss = s_nmiss;
    const int need = min(nmiss, n_slots);
    int hand = ctl[CTL_HAND];
    int got = 0;
    for (int scanned = 0; got < need && scanned < 3 * n_slots; scanned += RT)
    {
        int j = (int) (((int64_t) hand + threadIdx.x) % n_slots);
        bool active = threadIdx.x < n_slots;
        bool mine = active && slot_stamp[j] == epoch;
        bool cand = active && !mine && (slot_block[j] < 0 || slot_ref[j] == 0);
        int total = 0;
        int rank = block_scan_excl(cand ? 1 : 0, warp_sums, total);
        int want = need - got;
        if (threadIdx.x == 0) s_cut = min(RT, n_slots);
        __syncthreads();
        if (cand && rank == want - 1) s_cut = threadIdx.x + 1;   // the hand stops just past the last slot taken
        __syncthreads();
        int cut = s_cut;
        if (cand && rank < want)
        {
            miss_slot[got + rank] = j;
            slot_stamp[j] = epoch;   // taken: a sweep that wraps around must not take it twice
        }
        else if (active && threadIdx.x < cut && !mine)
        {
            slot_ref[j] = 0;
        }
        got += total < want ? total : want;
        hand = (int) (((int64_t) hand + cut) % n_slots);
        __syncthreads();
    }

    // 3. Re-point the tables; kv_stream_copy fills the slots
    const int placed = got < need ? got : need;
    for (int k = threadIdx.x; k < need; k += RT)
    {
        int pb = miss_block[k];
        if (k >= placed) { page_table[pb] = -1; continue; }
        int sl = miss_slot[k];
        int old = slot_block[sl];
        if (old >= 0) page_table[old] = -1;
        slot_block[sl] = pb;
        slot_stamp[sl] = epoch;
        slot_ref[sl] = 1;
        page_table[pb] = sl;
    }
    __syncthreads();

    // 4. The selection as slot rows
    for (int i = threadIdx.x; i < n; i += RT)
    {
        int pos = indices[i];
        int o = -1;
        if (pos >= 0)
        {
            int r = i / K_pad;
            int pb = phys_block(block_table + (r / seq) * npr, pos, page_size);
            int sl = page_table[pb];
            if (sl >= 0) o = sl * BLOCK_TOKENS + pos % BLOCK_TOKENS;
        }
        out[i] = o;
    }

    if (lookups) atomicAdd((unsigned long long*) &stats[1], (unsigned long long) lookups);
    if (threadIdx.x == 0)
    {
        ctl[CTL_EPOCH] = epoch;
        ctl[CTL_HAND] = hand;
        ctl[CTL_PLACED] = placed;
        if (placed < nmiss) ctl[CTL_OVERFLOW] = 1;
        stats[0] += placed;
        stats[2] += 1;
    }
}

void kv_stream_resolve
(
    at::Tensor page_table,
    at::Tensor slot_block,
    at::Tensor slot_stamp,
    at::Tensor slot_ref,
    at::Tensor ctl,
    at::Tensor stats,
    at::Tensor miss_block,
    at::Tensor miss_slot,
    const at::Tensor& indices,
    const at::Tensor& block_table,
    at::Tensor out,
    int64_t seq,
    int64_t page_size
)
{
    const at::cuda::OptionalCUDAGuard device_guard(indices.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(indices.dtype() == at::kInt && block_table.dtype() == at::kInt && out.dtype() == at::kInt);
    TORCH_CHECK(indices.is_contiguous() && block_table.is_contiguous() && out.is_contiguous());
    TORCH_CHECK(page_size % BLOCK_TOKENS == 0);
    int R = indices.size(0);
    int K_pad = indices.size(1);
    int n_slots = slot_block.size(0);
    kv_stream_resolve_kernel<<<1, RT, 0, stream>>>
    (
        (int*) page_table.data_ptr(),
        (int*) slot_block.data_ptr(),
        (int*) slot_stamp.data_ptr(),
        (int*) slot_ref.data_ptr(),
        (int*) ctl.data_ptr(),
        (int64_t*) stats.data_ptr(),
        (int*) miss_block.data_ptr(),
        (int*) miss_slot.data_ptr(),
        (const int*) indices.data_ptr(),
        (const int*) block_table.data_ptr(),
        (int*) out.data_ptr(),
        R, K_pad, (int) seq, (int) block_table.size(1), (int) page_size, n_slots
    );
    cuda_check(cudaPeekAtLastError());
}

#define MAX_ARRAYS 6

struct CopyRuns
{
    const uint4* src[MAX_ARRAYS];
    uint4* dst[MAX_ARRAYS];
    int len16[MAX_ARRAYS];   // 16-byte words per block
    int n;
};

// One workgroup per missed block (grid-stride): its runs from the host copy into its slot
__global__ __launch_bounds__(256)
void kv_stream_copy_kernel
(
    const int* __restrict__ ctl,
    const int* __restrict__ miss_block,
    const int* __restrict__ miss_slot,
    CopyRuns r
)
{
    const int placed = ctl[CTL_PLACED];
    for (int k = blockIdx.x; k < placed; k += gridDim.x)
    {
        int64_t pb = miss_block[k];
        int64_t sl = miss_slot[k];
        for (int a = 0; a < r.n; ++a)
        {
            const uint4* src = r.src[a] + pb * r.len16[a];
            uint4* dst = r.dst[a] + sl * r.len16[a];
            for (int i = threadIdx.x; i < r.len16[a]; i += blockDim.x) dst[i] = src[i];
        }
    }
}

void kv_stream_copy
(
    const at::Tensor& ctl,
    const at::Tensor& miss_block,
    const at::Tensor& miss_slot,
    std::vector<at::Tensor> src,
    std::vector<at::Tensor> dst,
    int64_t max_blocks
)
{
    const at::cuda::OptionalCUDAGuard device_guard(ctl.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(src.size() == dst.size() && src.size() <= MAX_ARRAYS);
    CopyRuns r = {};
    r.n = src.size();
    for (int a = 0; a < r.n; ++a)
    {
        // Both sides page-major with BLOCK_TOKENS-token blocks contiguous: block bytes = token row bytes * 4
        int64_t row_bytes = src[a].stride(-2) * src[a].element_size();
        TORCH_CHECK(dst[a].stride(-2) * dst[a].element_size() == row_bytes, "kv_stream_copy: row size mismatch");
        int64_t blk = row_bytes * BLOCK_TOKENS;
        TORCH_CHECK(blk % 16 == 0, "kv_stream_copy: block bytes must be a multiple of 16");
        r.src[a] = (const uint4*) src[a].data_ptr();
        r.dst[a] = (uint4*) dst[a].data_ptr();
        r.len16[a] = blk / 16;
    }
    int grid = (int) std::max((int64_t) 1, std::min(max_blocks, (int64_t) 1024));
    kv_stream_copy_kernel<<<grid, 256, 0, stream>>>
    (
        (const int*) ctl.data_ptr(),
        (const int*) miss_block.data_ptr(),
        (const int*) miss_slot.data_ptr(),
        r
    );
    cuda_check(cudaPeekAtLastError());
}

// Drop the slots of the blocks holding positions [seqlen, seqlen + length) of every sequence
__global__ void kv_stream_invalidate_kernel
(
    int* __restrict__ page_table,
    int* __restrict__ slot_block,
    const int* __restrict__ cache_seqlens,
    const int* __restrict__ block_table,
    int bsz,
    int length,
    int npr,
    int page_size
)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= bsz * length) return;
    int b = t / length;
    int pos = cache_seqlens[b] + t % length;
    if (pos / page_size >= npr) return;
    int pb = phys_block(block_table + b * npr, pos, page_size);
    int sl = page_table[pb];
    if (sl >= 0)
    {
        page_table[pb] = -1;
        slot_block[sl] = -1;
    }
}

void kv_stream_invalidate
(
    at::Tensor page_table,
    at::Tensor slot_block,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    int64_t length,
    int64_t page_size
)
{
    const at::cuda::OptionalCUDAGuard device_guard(page_table.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(cache_seqlens.dtype() == at::kInt && block_table.dtype() == at::kInt);
    TORCH_CHECK(block_table.is_contiguous());
    int bsz = block_table.size(0);
    int threads = bsz * (int) length;
    if (!threads) return;
    kv_stream_invalidate_kernel<<<(threads + 255) / 256, 256, 0, stream>>>
    (
        (int*) page_table.data_ptr(),
        (int*) slot_block.data_ptr(),
        (const int*) cache_seqlens.data_ptr(),
        (const int*) block_table.data_ptr(),
        bsz, (int) length, (int) block_table.size(1), (int) page_size
    );
    cuda_check(cudaPeekAtLastError());
}

// Drop the slots of physical blocks [b0, b1) (whole pages rewritten by a page copy)
__global__ void kv_stream_invalidate_range_kernel
(
    int* __restrict__ page_table,
    int* __restrict__ slot_block,
    int b0,
    int b1
)
{
    int pb = b0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (pb >= b1) return;
    int sl = page_table[pb];
    if (sl >= 0)
    {
        page_table[pb] = -1;
        slot_block[sl] = -1;
    }
}

void kv_stream_invalidate_range
(
    at::Tensor page_table,
    at::Tensor slot_block,
    int64_t b0,
    int64_t b1
)
{
    const at::cuda::OptionalCUDAGuard device_guard(page_table.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int n = (int) (b1 - b0);
    if (n <= 0) return;
    kv_stream_invalidate_range_kernel<<<(n + 255) / 256, 256, 0, stream>>>
    (
        (int*) page_table.data_ptr(),
        (int*) slot_block.data_ptr(),
        (int) b0, (int) b1
    );
    cuda_check(cudaPeekAtLastError());
}

// Copy whole pages pages[0..n) (same index on both sides) from the host copy into the staging layer image.
// blockIdx.y splits each page's runs so one page keeps several workgroups' loads in flight over PCIe
__global__ __launch_bounds__(256)
void kv_stage_pages_kernel
(
    const int* __restrict__ pages,
    int n_pages,
    CopyRuns r
)
{
    int i = blockIdx.x;
    if (i >= n_pages) return;
    int64_t page = pages[i];
    for (int a = 0; a < r.n; ++a)
    {
        int len = r.len16[a];
        int per = (len + gridDim.y - 1) / gridDim.y;
        int j0 = blockIdx.y * per;
        int j1 = min(j0 + per, len);
        const uint4* src = r.src[a] + page * len;
        uint4* dst = r.dst[a] + page * len;
        for (int j = j0 + threadIdx.x; j < j1; j += blockDim.x) dst[j] = src[j];
    }
}

void kv_stage_pages
(
    const at::Tensor& pages,
    std::vector<at::Tensor> src,
    std::vector<at::Tensor> dst
)
{
    const at::cuda::OptionalCUDAGuard device_guard(pages.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(pages.dtype() == at::kInt && pages.is_contiguous());
    TORCH_CHECK(src.size() == dst.size() && src.size() <= MAX_ARRAYS);
    int n_pages = pages.numel();
    if (!n_pages) return;
    CopyRuns r = {};
    r.n = src.size();
    for (int a = 0; a < r.n; ++a)
    {
        int64_t page_bytes = src[a].stride(0) * src[a].element_size();
        TORCH_CHECK(dst[a].stride(0) * dst[a].element_size() == page_bytes, "kv_stage_pages: page size mismatch");
        TORCH_CHECK(page_bytes % 16 == 0);
        r.src[a] = (const uint4*) src[a].data_ptr();
        r.dst[a] = (uint4*) dst[a].data_ptr();
        r.len16[a] = page_bytes / 16;
    }
    dim3 grid(n_pages, 8);
    kv_stage_pages_kernel<<<grid, 256, 0, stream>>>((const int*) pages.data_ptr(), n_pages, r);
    cuda_check(cudaPeekAtLastError());
}

// Copy page runs [p0, p1) of the host copy into the staging layer image with the DMA engine (the host copy is
// pinned, so these are true host-to-device copies at PCIe speed; GPU loads through the mapping are far slower
// for bulk data). runs: CPU int32 (n, 2)
void kv_stage_runs
(
    const at::Tensor& runs,
    std::vector<at::Tensor> src,
    std::vector<at::Tensor> dst
)
{
    TORCH_CHECK(runs.device().is_cpu() && runs.dtype() == at::kInt && runs.is_contiguous());
    TORCH_CHECK(src.size() == dst.size() && !src.empty());
    const at::cuda::OptionalCUDAGuard device_guard(dst[0].device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int* r = (const int*) runs.data_ptr();
    int n = runs.size(0);
    for (size_t a = 0; a < src.size(); ++a)
    {
        int64_t page_bytes = src[a].stride(0) * src[a].element_size();
        TORCH_CHECK(dst[a].stride(0) * dst[a].element_size() == page_bytes, "kv_stage_runs: page size mismatch");
        const char* s = (const char*) src[a].data_ptr();
        char* d = (char*) dst[a].data_ptr();
        for (int i = 0; i < n; ++i)
        {
            int64_t p0 = r[2 * i], p1 = r[2 * i + 1];
            cuda_check(cudaMemcpyAsync(d + p0 * page_bytes, s + p0 * page_bytes, (p1 - p0) * page_bytes,
                                       cudaMemcpyHostToDevice, stream));
        }
    }
}
