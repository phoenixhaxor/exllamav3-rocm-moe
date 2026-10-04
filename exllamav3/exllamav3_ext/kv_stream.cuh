#pragma once

#include <ATen/Tensor.h>
#include <vector>

at::Tensor kv_host_alloc
(
    std::vector<int64_t> sizes,
    at::ScalarType dtype,
    int64_t device
);

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
);

void kv_stream_copy
(
    const at::Tensor& ctl,
    const at::Tensor& miss_block,
    const at::Tensor& miss_slot,
    std::vector<at::Tensor> src,
    std::vector<at::Tensor> dst,
    int64_t max_blocks
);

void kv_stream_invalidate
(
    at::Tensor page_table,
    at::Tensor slot_block,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    int64_t length,
    int64_t page_size
);

void kv_stream_invalidate_range
(
    at::Tensor page_table,
    at::Tensor slot_block,
    int64_t b0,
    int64_t b1
);

void kv_stage_pages
(
    const at::Tensor& pages,
    std::vector<at::Tensor> src,
    std::vector<at::Tensor> dst
);

void kv_stage_runs
(
    const at::Tensor& runs,
    std::vector<at::Tensor> src,
    std::vector<at::Tensor> dst
);
