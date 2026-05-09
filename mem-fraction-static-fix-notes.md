# Fix `--mem-fraction-static` for EAGLE draft model KV cache

## Problem

When EAGLE speculative decoding is enabled, `--mem-fraction-static` only controls the
target model's KV cache pool.  The draft model's weights and KV cache are not accounted
for, causing OOM at launch:

1. Target KV pool is over-allocated because memory profiling runs before draft weights
   are loaded.
2. Draft KV pool (same `max_total_num_tokens`) is allocated on top without being
   budgeted.

Previously this was mitigated by a flat 4 GB / 6 GB heuristic reservation in
`server_args`, which is crude and insufficient for large models or user-specified
`--mem-fraction-static`.

## Change

Refactor model worker initialization into three explicit phases orchestrated by the
scheduler (`scheduler.py:init_model_worker`):

1. **Load weights** — target and draft weights load without allocating pools.
2. **Allocate pools** — memory profiling now sees all loaded weights; `cell_size` is
   scaled to include draft KV overhead (mirroring the existing DFLASH pattern).
3. **Init backends** — attention backends and CUDA graphs for target and draft
   separately.

Remove the 4 GB / 6 GB heuristic reservation for EAGLE/STANDALONE from `server_args`
since proper accounting makes it unnecessary.

---

## Errors encountered during development and their fixes

### 1. `vectorized_gather_kernel` OOB assertion during CUDA graph replay

**Symptom**: EAGLE draft decode CUDA graph replay crashes with
`cudaErrorLaunchFailure`.  Eager mode works; CUDA graph mode crashes.  Regression from
the init-reorder changes.

**Root cause**: `EAGLEWorker.init_backends()` did not call the draft model runner's
standard `ModelRunner.init_backends()`, so FlashInfer workspace buffers needed by the
draft attention backend during CUDA graph capture were never initialized.

**Fix** (`eagle_worker.py`): Call
`self.draft_model_runner.init_backends(disable_cuda_graph=True)` before the
EAGLE-specific `init_attention_backend()` / `init_cuda_graphs()`.

---

### 2. Missing standard backend init in all speculative workers

**Symptom**: Same class of crash (missing cublas handles, attention backends, warmup)
for MultiLayerEagle, Standalone, and v2 workers.

**Root cause**: On `main`, every speculative worker's `__init__` set
`server_args.disable_cuda_graph = True` before `super().__init__()`.  This caused
`ModelRunner.__init__()` to run standard backend init (cublas, attention, warmup) but
skip CUDA graph capture.  Our branch removed those `__init__` hacks (correctly, since
`__init__` no longer inits backends), but the new `init_backends()` methods only called
draft-specific init — standard backends were never initialized.

**Fix**: Added `disable_cuda_graph` parameter to `ModelRunner.init_backends()` and
`TpModelWorker.init_backends()`.  Each speculative worker now calls
`init_backends(disable_cuda_graph=True)` for standard backend init, then runs its own
draft-specific attention and CUDA graph capture.  Common logic lives in
`BaseDraftWorker.init_backends()` (`base_spec_worker.py`); subclasses only wrap with
their context managers.

**Files**:
- `model_runner.py` — `init_backends(disable_cuda_graph=False)` parameter
- `tp_worker.py` — pass-through `disable_cuda_graph` parameter
- `base_spec_worker.py` — `BaseDraftWorker.init_backends()` with common pattern
- `eagle_worker.py`, `eagle_worker_v2.py`, `multi_layer_eagle_worker.py`,
  `multi_layer_eagle_worker_v2.py`, `standalone_worker.py`, `standalone_worker_v2.py`

---

### 3. `DFlashWorker` `AssertionError: Draft worker requires memory_pool_config`

**Symptom**: CI failure — `DFlashWorker.__init__` accessed
`target_worker.model_runner.memory_pool_config` before `init_memory_pools()` ran.

**Root cause**: `DFlashWorker.__init__` passed `memory_pool_config`, pool, and
allocator directly to the inner `TpModelWorker` constructor.  After the refactor,
`memory_pool_config` is `None` at construction time because pool allocation is deferred.

**Fix** (`dflash_worker.py`): Store shared pool references as instance attributes;
defer allocation to `alloc_memory_pool()` and backend init to `init_backends()`, both
called by the scheduler.

---

### 4. `'EagleVerifyInput' object has no attribute 'hidden_states'`

**Symptom**: Crash during kernel warmup (`_flashinfer_autotune` → `_dummy_run`) or
CUDA graph capture for models whose forward pass reads
`forward_batch.spec_info.hidden_states` (e.g. `deepseek_nextn` / MTP models).

**Root cause**: `EagleVerifyInput` does not declare a `hidden_states` field.  During
warmup and CUDA graph capture, `_dummy_run()` / `get_spec_info()` creates an
`EagleVerifyInput` without it.  MTP target models (DeepSeek NextN) unconditionally
access `spec_info.hidden_states` in their forward pass.

**Fix**: Set a dummy `hidden_states` tensor on the `EagleVerifyInput` in both
`_dummy_run()` (`model_runner.py`) and `CudaGraphRunner.get_spec_info()`
(`cuda_graph_runner.py`).

---

### 5. `BreakableCUDAGraph has no attribute 'pool'` (debug scaffolding, reverted)

**Symptom**: Crash when breakable CUDA graph debug support was added to the EAGLE draft
CUDA graph runner.

**Root cause**: `BreakableCUDAGraph` does not implement `pool()`.  Guard
`set_global_graph_memory_pool(graph.pool())` was missing.

**Resolution**: All breakable CUDA graph debug scaffolding was reverted — not needed
for the final fix.

---

### 6. `cudaErrorStreamCaptureUnsupported` from `.item()` calls (debug scaffolding, reverted)

**Symptom**: OOM / capture failure after adding NaN/OOB debug checks with `.item()`.

**Root cause**: `.item()` forces GPU→CPU synchronization, which is illegal during CUDA
graph capture.

**Resolution**: Debug checks removed; not needed for the final fix.

---

## Key design decisions

- **`disable_cuda_graph` as a parameter vs. mutating `server_args`**: Passing a flag
  through `init_backends()` is cleaner than the old pattern of temporarily mutating
  `server_args.disable_cuda_graph` and restoring it.  No shared-state side effects.

- **`BaseDraftWorker.init_backends()` base class method**: All `BaseDraftWorker`
  subclasses follow the same three-step pattern: (1) init standard backends without
  CUDA graphs, (2) init draft attention backend, (3) capture draft CUDA graphs.
  Putting this in the base class means subclasses only wrap with context managers.

- **`cell_size` scaling for EAGLE**: Uses `cell_size * (1 + draft_layers / target_layers)`,
  assuming draft and target share the same per-layer KV size (head_dim, num_kv_heads,
  dtype).  This holds for EAGLE/MTP draft models that reuse the target architecture's
  attention config.  Mirrors the existing DFLASH pattern.
