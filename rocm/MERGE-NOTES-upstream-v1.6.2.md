# Merge notes: upstream exllamav3 v1.6.2 into exllamav3-rocm-moe

Written for the fork maintainer and for any AI assistant that works with this repository.
Everything below is a *record of decisions*; the commits themselves carry the detail.

## 0. Status (updated while the work proceeds)

- Work branch: `merge-upstream-v1.6.2` (fork `main` @ `1c049a13`, 2026-10-15 + upstream tag `v1.6.2` @ `4c84273`).
- Hardware used for verification: AMD Radeon AI PRO R9700 (gfx1201, 32 GB), Ryzen 9 7945HX3D, 2x48 GB DDR5-5200.
  **Not tested on gfx1100 (RX 7900 XTX).** That is the maintainer's hardware; please test there.

## 1. How the merge base was found

The fork root commit `159edc61` is an import of upstream `6b84a21b` (2026-09-20, "MSVC fix (__builtin_popcount)"),
which sits in upstream history 30 commits before `v1.5.2`. The import tree differs from `6b84a21b` in 2 files
(527 deleted lines). Because the fork has no shared git history with upstream, `git merge` refuses or produces
add/add conflicts everywhere. Reproduce a real three-way merge with:

```
git remote add upstream https://github.com/turboderp-org/exllamav3.git
git fetch upstream --tags
git replace --graft 159edc61 6b84a21b6f1e5da3f291b9e1019061f0de788279   # local only, not pushed
git checkout -b merge-upstream-v1.6.2 main
git config merge.conflictstyle zdiff3
git merge --no-commit --no-ff v1.6.2
```

The graft exists only in the local clone (refs/replace). The final merge commit has the two normal parents
(fork `main`, upstream `v1.6.2`) and needs no graft afterwards. Upstream is 317 commits ahead of the base.

## 2. Size of the problem

Trial merge: 52 conflicted files, 152 conflict hunks (zdiff3 counting).
- 21 files were changed by the fork **only in its ROCm/RDNA3 port commits** (not by later feature commits).
- 31 files were changed by the fork's later **feature commits** (MoE CPU experts, MTP, concurrency, GDN replay, ...).

## 3. Principle used

Upstream (since v1.6.0) has its own, much larger ROCm port: a portable PTX layer (`ptx_portable.cuh`,
`arch.cuh`), a `rocm/` source tree (WMMA GEMM, RDNA GEMV, compat headers), a hipify-aware source list in
`exllamav3/util/cuda_flags.py` and HIP-aware dispatch in `exl3_gemm.cu`. It solves the same problems as the
fork's port (inline-PTX emulation, `lop3`, `ldmatrix`, `__dp4a`, build flags) in a different way.

**Rule A (port-only files): take upstream's file unchanged.** The fork's emulation layer is superseded.
**Rule B (feature files): merge by hand, hunk by hunk; where a hunk only touches ROCm portability, take
upstream's way; where it carries fork behaviour (MoE CPU experts, concurrency, MTP, GDN replay, KV streaming,
tail skip, ...), keep the behaviour and re-express it on top of upstream's code.**
Every deviation is listed in section 5.

## 4. Method in practice (what was actually done, in order)

1. Real three-way merge with the graft above, `zdiff3` conflict style (shows the base).
2. Conflicted files changed by the fork only in its four port commits: taken from upstream (rule A).
3. **Every other file the fork touched only through port or RDNA3 commits** (not only the conflicted ones) was also reset to
   upstream's version (`git checkout v1.6.2 -- <file>`). Reason: git merged these without conflicts but left the fork's port
   edits (e.g. `EXL3_FULL_MASK`, `rocm_ldsm4`, `exl3_rdna3_*` calls) inside upstream's code, which no longer exists in upstream.
4. Conflicted files that carry fork features were then decided per family (section 5). Where upstream implements the same
   idea differently, upstream won in phase 1.
5. The fork's RDNA3-only kernels were **moved, not deleted**: `rocm/rdna3-fork-kernels/` holds `exl3_rdna3*.cu/.cuh`,
   `rdna3_attn.cu/.h`, `ablate.cu/.cuh` (17 files, `git mv`, history kept). They are outside `exllamav3_ext/`, so
   `cuda_flags.extension_sources()` no longer compiles them. They are the starting point for phase 2 on gfx11.
6. Result: the tree differs from upstream v1.6.2 in a small, reviewable set of files (list in section 6).

## 5. Decisions per file

| File | Decision |
|---|---|
| `.gitignore` | upstream's file + `*.so`, `build/` kept from the fork |
| 21 port-only files (see section 3) | upstream's file unchanged (rule A) |
| `architecture/qwen4_exp_mtp.py` | upstream's `lm_head_argmax(...)` call. The fork's `draft_sample` hook (Gumbel-coupled MTP drafting, `Job.coupled_sample`) was moved into `Model.lm_head_argmax` in `model/model.py`; it is applied to the logits **before** the vocabulary crop so the noise tensor has the same width as the target's sampler sees |
| `cache/recurrent.py` | upstream's `put()` -> `_insert()` refactor kept; the fork's `EXL3_STASH_LADDER` eviction (replay-cost-based victim) is re-added inside `_insert()` between "stranded" and "LRU" eviction; `t_use` stamping moved to `_insert()` (after `state.stash()`); upstream's `is_content_hash` helper kept |
| `exllamav3_ext/norm.cu` | fork's extra kernel args (`tail`, `abl`) kept, upstream's added closing brace of the launch macro kept |
| `exllamav3_ext/reduction.cuh` | upstream (it contains the same lane-0 fix as the fork) |
| `exllamav3_ext/hc_mix.cu` | upstream (`hc_warp_sum_lane0`, dangling-else braces) |
| `exllamav3_ext/graph.cu`, `graph.cuh` | **upstream, fork changes dropped.** The fork's `EXL3_NOGRAPH=mlp,gdn,moe,attn` (per-module eager decode) and its graph re-instantiation after 2048 kernel-node argument updates (`EXL3_GRAPH_REINSTANTIATE`, fixes fork issue #4, VRAM growth) are workarounds for the same ROCm problem that upstream now solves globally: `EXL3_GRAPHS` defaults to **off on ROCm** (each instantiated exec holds a kernarg pool that grows with every parameter update). Consequence: `EXL3_NOGRAPH` no longer exists; setting `EXL3_GRAPHS=1` on ROCm brings the leak back unless the re-instantiation logic is ported again |
| `libtorch/mlp.cpp`, `libtorch/blocksparse_mlp.cpp` | removed the fork's `graph_disabled_for(...)` guard (upstream's `graph.disabled` covers it) |
| `libtorch/attention.cpp` | upstream's eager QSA top-k path (`dsa_topk_prepare`, `s.graph->disabled`); the fork's `std::max(t_scan, 1)` guard kept on the `narrow` |
| GDN family: `modules/gated_delta_net.py`, `modules/gated_delta_net_fn/{conv1d,gated_delta_rule}.py`, `exllamav3_ext/{gdn.cu,gdn.cuh,bindings.cpp}`, `libtorch/gated_delta_net.{cpp,h}` | **upstream, fork changes dropped for phase 1.** Upstream has its own answer to the fork's `EXL3_GDN_REPLAY` (commit `c62775ee`, "2 recurrent states per slot instead of 5"): `c1136a02` "gdn: 2-slot parity scan pool with staged-input replay for spec rewinds", exposed as `batched_scan_replay`. The fork's `batched_state_replay` / `StateReplayJob` bindings are gone. Also dropped here (see section 6): `24f82f0f` DeltaNet prefill conv in place, and the RDNA3-only GDN megakernel / register-resident recurrence (`ef5975ae`, `8e3dbc99`) |
| CPU-MoE stack: `model/moe_cpu_host.py`, `modules/block_sparse_mlp_cpu.py`, `exllamav3_ext/cpu/moe_handoff.cu`, `cpu/moe_mul1.cpp`, `modules/block_sparse_mlp.py` | **upstream, fork changes dropped for phase 1.** Upstream v1.6.x has its own hot/cold expert placement (stats file `EXL3_MOE_CPU_SPLIT_STATS` or the profile saved by a previous run, `save_expert_stats`), promote-on-miss (`EXL3_MOE_CPU_PROMOTE`, spare GPU slots, DMA through the pinned arena), a fused split-issue (`exl3_moe_split_issue_fused`), a host-core reservation (`EXL3_MOE_HOST_CORES`) and 64-bit HIP stream memops. This overlaps the fork's `862cc6a4` (frequency-guided placement), `7a18518b` (adaptive placement), `3598d9cd` (persisted placement), `2922b761` (HIP stream memops) and `c00b4722`. Re-added in phase 2: **tail skip** (`886b4e2d`, needs a change in upstream's fused issue kernel) and the **standby nap** (`e4a82826`, PR #3 by loxdude, `cpu/moe_idle.h` stays in the tree). Dropped: the two-micro-batch overlap (`501c8c38`, only active on the RDNA3 decode path), `install_expert_aux_batch` |
| `modules/ngram_embedding.py` | upstream (it restructured the prefetch staging; the fork's `EXL3_NGRAM_PROF`, `EXL3_NGRAM_FADV` and next-chunk staging, commits `a83bdef7`, `d726a3cf`, are phase 2) |
| `modules/hyperconnections.py`, `modules/attention_fn/bc_attn.py`, `exllamav3_ext/quant/exl3_gemm.cu`, `exllamav3_ext/bindings.cpp` | upstream. RDNA3-only fork paths dropped here: the HIP split-attention kernel (`rdna3_attn.cu`), `exl3_rdna3_gemm` dispatch, fp16-source release for the RDNA3 prefill GEMM, `EXL3_ATTN_BLOCK_N/WARPS/STAGES` knobs |
| `generator/pagetable.py` | both: the fork's prefix-cache diagnostic print and upstream's `persistable` flag loop |
| `tests/test_moe_cpu_tiers_.py` | deleted (upstream deleted it) |
| `generator/generator.py`, `generator/job.py` | **upstream, fork changes dropped for phase 1** (they are re-added one by one in phase 2, see section 6): MTP prompt lookup (`EXL3_MTP_LOOKUP*`, `draft_headroom`, batched lookup), decode between prompt chunks (`_prefill_debt`), grouped prefill (`PrefillGroup`, off in the fork's served profile), two micro-batch decode (`_forward_micro_batches`, RDNA3 decode path only), the stall watchdog (`EXL3_WATCHDOG_S`), `_log_cache_wait`. Upstream meanwhile has its own hybrid drafting (`hybrid_draft_tokens`, suffix automaton lookup), `Generator.close()`, disk-tier idle flush and `max_draft_tokens`. The Gumbel-coupled draft hook lives in `model/model.py` (see above) and waits for its generator side (`iterate_draftmodel_mtp_gen`) |
| `generator/async_generator.py` | **both.** Upstream's idle flush (`has_idle_work()/idle_work()`) and `Generator.close()` plus the fork's threaded step (`EXL3_ASYNC_THREAD=1`: generator steps run on one worker thread, loop-side changes are queued in `_calls`). Upstream's "a rejected job must not stay tracked" is implemented for both modes: in threaded mode the worker records the refusal in `_rejected` and the loop thread untracks the job and delivers the exception (`_deliver_rejected`) |
| `README.md` | fork's README kept; `README.exllamav3.md` refreshed to upstream v1.6.2's README |

## 6. What the merged tree still contains from the fork, and what is waiting for phase 2

**Kept (differs from upstream v1.6.2):**
`cache/recurrent.py` (checkpoint ladder eviction), `cache/kv_stream.py`, `exllamav3_ext/kv_stream.cu/.cuh` + bindings and
`cache/qsa.py`, `modules/attn.py`, `modules/qsa_indexer.py`, `modules/attention_fn/qsa_triton.py` (KV streaming for the QSA layers),
`generator/async_generator.py` (threaded step, `EXL3_ASYNC_THREAD=1`), `generator/pagetable.py` (diagnostic print),
`model/model.py` (`draft_sample` hook), `modules/moe_batch_recon.py` (fp16 down projection, `EXL3_MOE_RECON_DOWN_HALF`),
`modules/ple.py` (`prefetch_next` param), `loader/safetensors.py` (`POSIX_FADV_RANDOM` for n-gram row gathers),
`util/memory.py`, `architecture/{glm4v,qwen2_5_vl,qwen3_vl}.py` + `architecture/mm_processing/qwen2.py` (pad images smaller than one
merged patch instead of failing the request), `cpu/moe_idle.h` + `tests/test_moe_cpu_idle_.py` (files only; not wired yet),
`generator/adaptive_window.py`, `generator/prefill_group.py`, `util/ablation.py` (files only; not wired).
Plus fork-only trees: `rocm/` (scripts, TabbyAPI patches and configs, `rdna3-fork-kernels/`, `tools/`), `rocm_tests/`, `README.md`,
`README.exllamav3.md` (upstream's README).

**Phase 2 backlog** (each item = one commit with a measurement; fork commit hashes in brackets):
1. (covered by `EXL3_ASYNC_THREAD=1`, see section 8) Decode between prompt chunks, generator off the event loop [`f5a558d8`] - this is the part that keeps a short request
   responsive while a very long prompt is being read (measured on gfx1201 with two slots: first token of the short request
   16.5 s with the fork vs 53 s with plain upstream while a 142k prompt is read).
2. Gumbel-coupled MTP drafts and prompt lookup, batched [`4af70aa8`, `b3673605`, `bb926c1b`] (`Model.lm_head_argmax` hook is in place).
3. **DONE (branch `phase2`)** Tail skip for CPU-resident experts [`886b4e2d`, `960ed40e`], carried through upstream's fused split-issue kernel. See section 8 for the rank that is safe on gfx1201.
4. **DONE (branch `phase2`)** Standby nap for the CPU MoE workers [`e4a82826`, fork PR #3, original author kept].
5. Next-chunk n-gram staging and the n-gram prefetch switches [`d726a3cf`, `a83bdef7`, `92a0990d`] (`prefetch_next` in `ple.py` is in place).
6. **DONE (branch `phase2`)** Stall watchdog [`32f8418f`, `aef16ef3`] and the cache-wait log line [`1c049a13`].
7. DeltaNet prefill conv in place [`24f82f0f`], gated residual single kernel [`b247165c`], MoE prefill batched down projection [`1ab433ec`], QSA prefill single-stage pipelining [`3f086b8a`].
8. RDNA3 (gfx11) kernels in `rocm/rdna3-fork-kernels/` and the dispatch hooks for them: only for the maintainer's hardware; each needs
   a benchmark against upstream's `rocm/` kernels (`rocm_tests/test_rdna3_*.py`).
9. Graph handling: if CUDA-graph decode on ROCm is wanted back (`EXL3_GRAPHS=1`), port the re-instantiation after N argument updates [`6fd81152`].

## 7. Verification log

Branch `merge-upstream-v1.6.2`, merge commit `208fc1cb` (+ the unit test commit), tested 2026-10-16 on gfx1201.

| Check | Result |
|---|---|
| Build `setup.py build_ext` in `tcclaviger/vllm` (torch 2.11+rocm10.0, hip 7.15), `PYTORCH_ROCM_ARCH=gfx1201` | 174/174 units, return code 0, 0 `error:` lines, about 10 min with `MAX_JOBS=3` |
| `import exllamav3_ext` without GPU | 205 symbols, `kv_stream_copy` and `batched_scan_replay` present |
| Unit test `tests/generator/test_async_generator_threaded_.py` (stubbed generator, no GPU) | 4 passed: accepted job, rejected job in plain mode, rejected job in threaded mode reaches its consumer and is untracked |
| Load of Qwen3.8 Flash-Next 5.05 bpw (exl3) with 400 CPU experts, MTP draft, vision, 262144 context Q8, `max_batch_size: 2`, `EXL3_ASYNC_THREAD=1` | loaded in 91.6 s, version string `1.6.2` |
| Functional gate (36 stress prompts 800 to 70000 characters, 3 tool calls, 2 images, 1863-token generation, prefix cache 2816/3008, 8-round history, history edited at an early position) | all ok, 0 journal errors |
| Needle test 64k prose | 50/50 |
| Prompt of about 400k tokens | HTTP 400 from TabbyAPI, instance healthy afterwards |
| Client disconnect in the middle of a stream | next request fine, 0 tracebacks in the log |

Speed (same machine, same model and config, 1 to 2 runs per number - indicative, not a benchmark). "fork" = the fork's
`9528a5a` as served before (different model file: the non-abliterated one), "upstream" = plain v1.6.2 with the same config,
"merged" = this branch, warm Triton cache:

| | fork | upstream v1.6.2 | merged |
|---|---|---|---|
| one request, 800 tokens, code (tok/s) | 39.2 / 45.4 | 31.9 / 33.1 | 37.3 / 35.9 |
| one request, 800 tokens, prose | 26.0 / 33.8 | 23.9 / 25.6 | 25.8 / 27.5 |
| two at once, per request: code / prose | 24-27 / 20-22 | 16.7-18.5 / 14.2-15.2 | 19.0 / 15.6-15.8 |
| 142k-token prompt plus a short request 15 s later: first token of the short one | 16.5 s | 53.1 s | 18.7 s |
| same: first token of the long prompt | 205 s | 214 s | 207 s |
| 64k prose: read / generate | 892 T/s / 41.7 tok/s | 689-712 T/s / 24.1-30.4 | 730 T/s / 27.2 |

Readings (hypotheses, not proven): the threaded step (`EXL3_ASYNC_THREAD=1`) is what brings the short request back from 53 s to
under 19 s, so backlog item 1 of section 6 is already largely covered by it; the remaining gap to the fork in concurrent decode
is probably the fork's MoE placement/issue path (backlog item 3 and the fork's own learned placement), not measured.

**Cold Triton cache warning:** the first ten minutes on an empty Triton cache ran at 13 tok/s instead of 36 (the first
conc_test run of the merged branch). Reuse a persistent `TRITON_CACHE_DIR`.

**Not tested:** gfx1100 (the maintainer's card), the CUDA build, tensor parallel, DFlash drafters, any model other than Flash-Next,
three or more simultaneous slots, decoding beyond 64k context, KV streaming switched on, the checkpoint ladder under real memory
pressure, and the rejected-enqueue path through a real server (only the stubbed unit test; TabbyAPI refuses over-long prompts
before they reach the generator).

## 8. Phase 2 results and operating notes (gfx1201, 2026-10-16)

Branch `phase2` = phase 1 + four commits: standby nap, stall watchdog, cache-wait log line, tail skip. Built 174/174 units, no errors.

### 8.1 The most important finding: upstream's host-core reservation throttles this stack

Symptom: right after every start, decode ran at 13-15 tok/s for a long time and became fast (35-40) only after a very long prefill
had run, while the fork code started at 25-37 tok/s. What it was NOT (each tested, all still slow): Triton cache cold or warm,
hugepage collapse (`EXL3_MOE_ARENA_HUGEPAGE=0`), static versus dynamic placement (`EXL3_MOE_CPU_SWAP=0`), the learned profile
(`EXL3_MOE_CPU_SPLIT_STATS` file versus the profile saved by upstream), the 15 versus 16 worker thread mismatch, disk reads (vmstat `bi` 0),
swap, draft acceptance (83 percent in a slow request).

What it was: `EXL3_MOE_HOST_CORES` (default 1). Upstream plans the worker pool on all but one physical core and confines the whole
host process (the generator thread, CUDA driver threads, the API server) to that one core and its SMT sibling. The log line is
`CPU MoE host affinity: host on LPs [15, 31]`. `mpstat` showed exactly those two logical processors at 85-89 percent while every
worker thread idled at about 25 percent: the host side was the bottleneck. Our host is heavier than upstream's reference setup
(threaded generator step, TabbyAPI with MTP drafting, two slots).

Decode of one long code prompt (600 tokens, 400 CPU experts of 512, Ryzen 9 7945HX3D with 16 physical cores, `cpu_moe_threads` = 16 minus host cores):

| `EXL3_MOE_HOST_CORES` | 1 (default) | 2 | 3 | 4 | **0** |
|---|---|---|---|---|---|
| tok/s right after start | 15 | 20 | 23-24 | 25 | **31-34** |

`EXL3_MOE_HOST_CORES=0` (no reservation, no pinning, workers float) gives the fork's behaviour. Recommendation for this machine:
`EXL3_MOE_HOST_CORES=0`, `cpu_moe_threads: 16`. If the reservation is wanted on a lighter host: the placement is planned once from the
first worker that starts (here the 1-layer MTP head with 15 threads) and then kept for the 16-thread main worker
(`placement planned for 15 workers, kept for a 16-thread worker`); keep both counts equal.

### 8.2 Prerequisites of the upstream features that were not met in our setup (read from doc/env_vars.md)

- Promote-on-miss and `EXL3_MOE_CPU_PROMOTE` need the pinned arena (`EXL3_MOE_PINNED_ARENA=1`, config `moe_cpu_pinned_arena`).
  We did not set it, so promote-on-miss was never active in any measurement here; the periodic swap sweep ran instead.
  The pinned arena page-locks the expert memory (`cudaHostRegister`); on this 90 GB machine that is the setting that once locked
  about 73 GiB and made the machine unusable, so it was deliberately not tried. Open question.
- The learned expert profile is written to `$XDG_CACHE_HOME/exllamav3/expert_stats/<hash>.json` (default `~/.cache`, which is
  lost with a `--rm` container). `EXL3_MOE_CPU_SPLIT_STATS` takes precedence over it.
- `exllamav3/autotune/coop_autotune_v1.bin` is written to the same cache directory.

### 8.3 Measurements with `EXL3_MOE_HOST_CORES=0`, learned stats file, `EXL3_ASYNC_THREAD=1`, from a cold start

| | fork (11.10., other model file) | merged, tail skip off | tail rank 9 | tail rank 8 |
|---|---|---|---|---|
| one request, code, tok/s | 39-45 | 34-36 | 39.1 | **41.2** |
| one request, prose | 26-34 | 22-26 | 26.5 | **28.7** |
| two at once, code / prose | 24-27 / 20-22 | 19.0-19.6 / 15.0-15.2 | 21.1 / 16.9 | **22.5 / 17.6** |
| 142k prompt + short request: first token of the short one | 16.5 s | 16.8 s | - | - |
| same: first token of the long prompt | 205 s | 199.5 s | - | - |
| 64k prose read / generate | 892 T/s / 41.7 | 747 T/s / 25.1 | - | - |
| 64k needle test | 50/50 | 50/50 | - | - |

One to two runs each, indicative. The 64k generation figure is clearly below the fork's and is unexplained.

### 8.4 Tail skip on this setup: rank 7 breaks the output

`EXL3_MOE_TAIL_SKIP=7` (the rank the fork serves with) made Flash-Next 5.05 bpw (abliterated variant) derail on the coding prompt at
temperature 0: after the imports it emitted endless `import dict / import list / ...` lines (the MTP acceptance of such text is
high, which is why raw speed looked good while a long `min_tokens` request ran at 15 tok/s). Ranks 8, 9 and 10 gave normal code.
So on this model the safe rank is 8 or higher (+14-17 percent decode at rank 8), the default stays 0 (off), and a quality check
is needed before enabling it on another model. The fork's measured KL of 0.037 at rank 7 (gfx1100, 3.05 bpw model) does not carry
over. Only one prompt was checked for sanity; no task benchmark was run.

### 8.5 Not done in phase 2

Gumbel-coupled MTP drafts and prompt lookup (needs per-position seeds in upstream's sequential verification, no presample path in
upstream), n-gram next-chunk staging and prefill micro-optimisations (backlog items 5 and 7), the RDNA3 kernels (item 8), the graph
re-instantiation (item 9; not needed while `EXL3_GRAPHS` stays off on ROCm).

## 9. Phase 2b results: pinned arena, quality checks, fork control (gfx1201, 2026-10-16, afternoon)

All numbers: same machine, same model file (Qwen3.8 Flash-Next 5.05 bpw, abliterated, bf16 n-gram), same TabbyAPI config (400 CPU
experts of 512, MTP draft, vision, 262144 context Q8, `max_batch_size: 2`), learned stats file, `EXL3_ASYNC_THREAD=1`, one to two runs each.

### 9.1 Pinned arena works here (`EXL3_MOE_PINNED_ARENA=1`) and is the biggest single gain

Safety first: a guard script killed the container below 3.5 GB `MemAvailable` (never triggered), `EXL3_HOST_MEM_RESERVE_MB=6000`.
Observed: `MemAvailable` stayed at 17-20 GB with the 57 GB expert arena loaded, `Mlocked` 48 kB (the CUDA host registration is not counted
there), no stall, load 63 s. The log shows `promote-on-miss ... 4 spare slots`; hit rate on the GPU 51-67 percent.
Not changed: `/sys/kernel/mm/transparent_hugepage/shmem_enabled` is `never` on this host, so the shared arena runs on 4K pages
(upstream documents a few percent of decode lost on some hosts; `within_size` would be the setting).

| | fork control (today) | merged, unpinned, host cores 0 | **merged, pinned arena** | pinned + tail rank 8 |
|---|---|---|---|---|
| one request, code (tok/s) | 34.8 / 43.2 | 34.3 / 36.5 | 51.4 / 41.6 | **55.5 / 51.4** |
| one request, prose | 24.7 / 28.7 | 22.3 / 26.0 | 24.9 / 31.7 | **31.5 / 36.8** |
| two at once, code (per request) | 22.7 / 25.7 | 19.0 / 19.6 | 21.9 / 21.2 | **23.8 / 26.1** |
| two at once, prose | 18.6 / 19.8 | 15.0 / 15.2 | 18.1 / 17.6 | **20.7 / 21.9** |
| 142k prompt: first token of the long prompt | 217 s | 200 s | **136 s** | - |
| same: first token of a short request sent 15 s later | 18.5 s | 16.8 s | **13.4 s** | - |
| 64k prose: read / generate | 856 T/s / 34.4 | 747 / 25.1 | **1125 / 30.9** | - |
| 64k needle test | 50/50 | 50/50 | 50/50 | - |

(The fork control is the production code `9528a5a` with its production environment, run through the same scripts today; the
41.7 tok/s for 64k generation measured on 2026-10-11 was not reproduced today.)

### 9.2 Tail skip quality checks

- Local code check `rocm/tools/codecheck.py` (23 small Python tasks with assertions, greedy): rank 8: 23/23, rank 7: 23/23, off: 23/23.
- Degeneration probe `rocm/tools/degprobe.py` (5 long prompts, 500 tokens, greedy; repeated-4-gram share and compression ratio): rank 7 and rank 8
  are within noise of "off" (code 8.9 / 7.8 / 8.4 percent repeated, prose 0.7 / 0.7 / 0.5, sql 67.5 / 66.4 / 66.1 which is the SQL boilerplate itself).
  Text identical with "off" only for the first 50-250 characters (expected: different experts drop out, greedy paths diverge).
- The derailed output seen once with rank 7 (section 8.4) was **not reproducible**: six later fresh instances with the same flags (also after load and
  eight swap sweeps, also with and without the pinned arena) produced normal code. Cause unknown; one suspicion is the first-run autotune of the
  fused MoE kernels before `$XDG_CACHE_HOME/exllamav3/autotune` existed. Treat tail skip as experimental; default stays off; rank 8 was never seen to fail.

### 9.3 Gumbel-coupled MTP drafts: implemented, cannot engage through TabbyAPI, no effect measured

Branch `phase2-coupled` (not merged into `phase2`): job-side seeds per verification position, draft sampling through the `Model.lm_head_argmax` hook.
Debug trace (`EXL3_MTP_COUPLED_DEBUG=1`) shows why nothing happens: `Job.can_presample_window()` is false because `sampler.reqs_past_ids` is
true for every request that comes through TabbyAPI (even with a preset that has no penalties and explicit zero penalties), and the dynamic draft
calibrator disables it as well. With the coupling forced off or "on" the acceptance was identical (e.g. 302/392 and 303/388 drafted tokens accepted).
The same restriction (past-ids) existed in the fork. Left out of the merge.

### 9.4 Recommended production settings for this machine if the merge replaces the fork

`EXL3_MOE_PINNED_ARENA=1`, `EXL3_MOE_HOST_CORES=0`, `cpu_moe_threads: 16`, `EXL3_ASYNC_THREAD=1`, `EXL3_HOST_MEM_RESERVE_MB=6000`, a persistent
`XDG_CACHE_HOME` and `TRITON_CACHE_DIR`, the learned stats file; dropped (no longer exist): `EXL3_NOGRAPH`, `EXL3_GDN_REPLAY`, `EXL3_RDNA3_ATTN`. Optional:
`EXL3_MOE_TAIL_SKIP=8` (the conservative rank, see section 10). Keep the fork tree for rollback.

## 10. Tail skip: quality and speed measurements (gfx1201, 2026-10-16/17)

Tools used: `rocm/tools/tail_skip_check_merge.py` (KL / top-1 / perplexity against rank 0, speed; needs a text directory, `TEXT_DIR`),
`rocm/tools/iqbench_paired.py` (paired task comparison; the test server polls a rank file, `EXL3_MOE_TAIL_RANKFILE`; needs the data files written by
`rocm_tests/iqbench_data.py`), `rocm/tools/codecheck.py`, `rocm/tools/degprobe.py`. Tail skip stays **off by default**; rank 8 is the conservative choice
(rank 7 produced one derailed output on this setup, see sections 8.4 and 9.2; the maintainer serves rank 7 on a 3.05 bpw model without problems, so this is model dependent).

### 10.1 Quality

#### Distribution shift of tail skip (KL / top-1 against tail skip off, 1800 tokens of mixed prose and code)

| | KL mean | KL p99 | top-1 agreement | perplexity | CPU expert picks skipped |
|---|---|---|---|---|---|
| noise floor (off vs off, run-to-run) | 0.0144 | 0.152 | 94.7 % | x1.0066 | 0 % |
| rank 9 | 0.0367 | 0.390 | 92.2 % | x1.0057 | 20.9 % |
| rank 8 | 0.0472 | 0.461 | 90.4 % | x1.0067 | 30.9 % |
| rank 7 | 0.0578 | 0.550 | 90.3 % | x1.0087 | 40.7 % |

So rank 8 is clearly above the run-to-run noise in KL and top-1 (about 3x the KL), while perplexity is indistinguishable from the noise floor. The run-to-run noise itself is not zero because the CPU expert path is not bit-reproducible.

#### Paired task comparison (same questions, same seeds, one server, tail rank switched per request; off vs rank 8)

| set | questions | off | rank 8 | only off right / only rank 8 right |
|---|---|---|---|---|
| MMLU-Pro | 80 | 73 (91.2 %) | 71 (88.8 %) | 3 / 1 |
| MATH-500 | 30 | 27 (90.0 %) | 27 (90.0 %) | 0 / 0 |
| HumanEval | 60 | 58 (96.7 %) | 58 (96.7 %) | 0 / 0 |

170 questions, 4 flips in total, none statistically significant (3:1 on MMLU-Pro is well inside chance). This is a small sample and one run per rank, so it can only rule out large losses, not small ones. Average answer length on MMLU-Pro went from 324 to 383 tokens with rank 8 (MATH 229 to 239, HumanEval 34 to 34), which eats part of the speed gain on that set.

#### Other checks

Code check 23/23 with and without rank 8; degeneration probe (five 2k-token generations per rank, repeated-4-gram share and zlib ratio) shows no difference between off and rank 8 (the SQL prompt is repetitive in both, 64 % vs 66 %); no errors or GPU faults in the server log over 2 h 15 min. Speed in the same run: rank 7 gave +29 % (code) and +17 % (prose) over off.

**Conclusion:** keep tail skip off by default and document it as an opt-in speed/quality trade-off, with rank 8 or 9 as the suggested values. The merge itself (everything except tail skip) does not depend on it.

### 10.2 Speed (absolute numbers, 2026-10-17)

Absolute numbers (the relative figures above carry no prompt-processing (PP) values). Same machine and server as before (gfx1201, Flash-Next 5.05 bpw, pinned arena, `EXL3_MOE_HOST_CORES=0`, threaded step, 400 of 512 experts on the CPU). One server process, tail skip rank switched through a file between passes in the order off, 8, 8, off so that drift cancels out. Greedy decoding, 400 generated tokens per request, a unique marker at the start of every prompt so the prefix cache cannot help. Method as in LLMSpeedTest: PP = new prompt tokens / (time to first response minus base latency), decode = (tokens - 1) / (last minus first token).

**Decode speed, short context (3 runs per pass, 6 per rank, median and range)**

| | tail skip off | tail skip rank 8 | change |
|---|---|---|---|
| code | 46.2 tok/s (32.8 - 51.3) | 56.1 tok/s (48.1 - 59.8) | +21 % |
| prose | 43.2 tok/s (26.8 - 46.0) | 46.4 tok/s (39.7 - 50.7) | +7 % |

**64k context (about 62k prompt tokens, 2 passes per rank)**

| | tail skip off | tail skip rank 8 |
|---|---|---|
| PP, prose | 1104 / 1138 tok/s | 1130 / 1093 tok/s |
| PP, code text | 1012 / 939 tok/s | 1119 / 1091 tok/s |
| decode at 62k context, prose | 32.9 / 32.7 tok/s | 39.3 / 39.1 tok/s |
| decode at 62k context, code text | 34.3 / 35.4 tok/s | 42.6 / 43.4 tok/s |

Reading: tail skip speeds up **decode** (+7 % to +23 % here), and **PP is not affected** (the differences in the PP rows are within the run-to-run spread of about 5 to 10 %, and go in both directions). The first request after a rank switch is often the slowest of its series, so the ranges are wide; the medians are the safer figure. Two passes and six runs per rank is a small sample, indicative only. These absolute numbers differ from the afternoon figures in section 8.3 (for example prose 25 / 32 there with tail skip off, 43 here); those were one or two runs each right after a server start, and I did not isolate the reason for the difference, so compare the off / rank 8 ratios within this table rather than absolutes across runs.

Not measured: two concurrent requests with rank 8 in this run (the section 8.3 figures still stand), and PP at other context sizes.
