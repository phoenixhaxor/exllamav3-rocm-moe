#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>

// Routed MoE experts (plus an optional shared expert) for decode-sized batches on RDNA3 (see
// exl3_rdna3.cu). Returns false when the shapes are not covered (caller falls back to the generic paths)
bool exl3_rdna3_moe_decode
(
    const at::Tensor& y,
    const at::Tensor& sel,
    const at::Tensor& w,
    const at::Tensor& g_tr, const at::Tensor& g_suh, const at::Tensor& g_svh,
    const at::Tensor& u_tr, const at::Tensor& u_suh, const at::Tensor& u_svh,
    const at::Tensor& d_tr, const at::Tensor& d_suh, const at::Tensor& d_svh,
    double K_gu,
    double K_d,
    bool mcg,
    bool mul1,
    int64_t num_local,
    at::Tensor& tabs,
    at::Tensor& c_gu,
    at::Tensor& c_d,
    at::Tensor& out,
    const c10::optional<at::Tensor>& sh_tab,
    const c10::optional<at::Tensor>& sh_gate,
    double K_sh
);
