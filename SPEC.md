# Project Specification: llama-shmem (OpenSHMEM Distributed Inference Engine)

## Objective
Extend the verified 2-PE `llama-shmem` greedy pipeline into a batched, speculative-decoding OpenSHMEM inference engine using one-sided RMA puts (`shmem_putmem`, `shmem_ulong_put`) and zero-network local spinlocks (`shmem_ulong_wait_until`).

## Environment & Execution Context
- Cluster Topology: 2-node LXD system container cluster (`pe0` and `pe1`) sharing `/workspace` (`~/llama-shmem-workspace` on host).
- Build Command:
  lxc exec pe0 -- bash -c "cd /workspace/build && cmake .. && make -j$(nproc)"
- Run Command (Always wrap binary in `setarch x86_64 -R`):
  lxc exec pe0 -- oshrun --allow-run-as-root -np 2 --host pe0:1,pe1:1 -x SHMEM_SYMMETRIC_SIZE=512M -x UCX_TLS=tcp,self -x LD_LIBRARY_PATH=/usr/local/lib setarch x86_64 -R /workspace/build/llama_shmem /workspace/qwen2.5-0.5b-instruct-q4_k_m.gguf "The capital of France is" 64

## Critical Engineering Rules (Do Not Violate)
1. Same-Lane RMA Ordering (`shmem_ulong_put` + `shmem_quiet`):
   - Over `UCX_TLS=tcp`, mixing `shmem_putmem` (RMA lane) with `shmem_ulong_atomic_set` (AMO active-message lane) causes TCP socket backpressure deadlocks after ~20 steps.
   - Always use `unsigned long` for sync flags, and signal completion using ordered one-sided RMA puts on the same lane:
     shmem_putmem(remote_buf, local_src, n_bytes, target_pe);
     shmem_quiet();
     shmem_ulong_put(&sym_slot->step_flag, &local_step, 1, target_pe);
     shmem_quiet();
   - Wait on the receiver using `shmem_ulong_wait_until(&sym_slot->step_flag, SHMEM_CMP_EQ, local_step);`.
   - Never use `shmem_uint64_*` (unsupported in OpenSHMEM 1.4) or `shmem_ulong_atomic_set`.
2. ASLR & Segment Registration (`-fno-pie -no-pie` + `setarch x86_64 -R`):
   - Keep `-fno-pie` and `-no-pie` in `CMakeLists.txt`.
   - Always invoke the binary under `oshrun` via `setarch x86_64 -R /workspace/build/llama_shmem ...` so all 33+ linked shared libraries map to identical virtual addresses across `pe0` and `pe1`.
   - `memheap_base_max_segments = 128` is configured in `/etc/openmpi/openmpi-mca-params.conf` on both containers.
3. Explicit Symmetric Broadcasts (No Local Write Assumptions):
   - Writing to a symmetric struct locally on PE 0 (`sym_slot->pos = val`) ONLY modifies PE 0's local memory. Moreover, `oshrun` over SSH splits quoted CLI strings on remote nodes.
   - Any runtime parameters parsed or tokenized on PE 0 (`n_prompt`, `n_predict`, batch control headers) MUST be explicitly pushed to remote PEs via `shmem_int32_put` / `shmem_putmem` + `shmem_quiet()`.
4. SPMD & Thread-0 Guard:
   - Inside any `ggml` custom operator callback (`ith`, `nth`), guard all OpenSHMEM calls with `if (ith != 0) return;` so only thread 0 performs inter-PE transfers.
5. Symmetric vs. Private Memory:
   - Model weights (`[il_start, il_end)`) and local KV caches live strictly in private RAM. Only activation handoff buffers, control metadata headers, and synchronization flags live in the symmetric heap (`shmem_align`).

---

## Staged Implementation Plan (TODOs)

### Phase 1: Environment & Symmetric Ping-Pong Sanity Check [COMPLETED]
- [x] 1.1 Build System Setup: `CMakeLists.txt` compiling with `oshcxx` (`-fno-pie -no-pie`) and linking against `libggml` and `libllama`.
- [x] 1.2 Symmetric Allocation: `shmem_align(64, ...)` allocation verified across `pe0` and `pe1`.
- [x] 1.3 Custom Op Test Harness: `src/test_shmem_pipe.cpp` verified (`out[0] = 30.0`).

### Phase 2: Minimal Single-Token Greedy Pipeline (B = 1) [COMPLETED]
- [x] 2.1 Layer Sharding & Selective GGUF Loading: Implemented in `src/llama_shmem.cpp` (`[0..12)` on `pe0`, `[12..24)` on `pe1`).
- [x] 2.2 Subgraph Boundary Wiring & Zero-Copy Input Binding: Implemented via `shmem_op_pipeline_send` and `shmem_op_pipeline_recv`.
- [x] 2.3 Greedy Tail Sampling & Verification Gate: Verified 64-token greedy generation across 2 PEs on `qwen2.5-0.5b-instruct-q4_k_m.gguf` with clean exit.

### Phase 3: Batched Forward Pass, Linear Speculative Decoding & Lazy KV Rollback (B = K + 1) [NEXT TASK]
- [ ] 3.1 Upgrade Symmetric Structs to Batch Size `B_VERIFY = 8`:
  Replace `shmem_stage_slot_t` and `shmem_feedback_slot_t` with:
  - Constants: `K_DRAFT = 7`, `B_VERIFY = 8`, `MAX_D_MODEL = 8192`
  - `shmem_spec_packet_t` (64-byte aligned):
    - `uint32_t n_tokens;` (Active tokens in this step: `1 .. B_VERIFY`)
    - `uint32_t kv_commit_idx;` (Authoritative KV cache start slot for this batch)
    - `int32_t draft_ids[B_VERIFY];` (`[t_root, d_1, ..., d_K]`)
    - `int32_t pos[B_VERIFY];` (RoPE positions: `[kv_commit_idx + 0 ... kv_commit_idx + n_tokens - 1]`)
    - `uint16_t parent_mask[B_VERIFY];` (Causal/tree ancestor bitmask for batch tokens)
    - `float act_buf[B_VERIFY * MAX_D_MODEL];`
    - `unsigned long step_flag;`
  - `shmem_verify_feedback_t` (64-byte aligned):
    - `uint32_t n_accepted;` (`0 .. K_DRAFT`)
    - `int32_t committed_ids[B_VERIFY];` (Accepted draft tokens + 1 bonus/correction token)
    - `unsigned long verify_flag;`
  - Note: In `shmem_op_pipeline_send`, PE `i` must transfer both the control header fields (`n_tokens`, `kv_commit_idx`, `draft_ids`, `pos`, `parent_mask`) and the active slice of `act_buf` (`n_tokens * n_embd * sizeof(float)`) to `next_pe` via `shmem_putmem`, call `shmem_quiet()`, and then set `step_flag` via `shmem_ulong_put` + `shmem_quiet()`.
- [ ] 3.2 Upgrade `evaluate_sharded_step()` to Multi-Token Batching (`n_tokens >= 1`):
  - On PE > 0, wait for `step_flag == local_step` BEFORE building the `ggml` graph for the step so `n_tokens`, `kv_commit_idx`, `pos`, and `parent_mask` are already populated in `sym_slot`.
  - Shape `inp_act` as `ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens)` bound to `sym_slot->act_buf`.
  - Shape `pos_t` as `ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens)` populated from `sym_slot->pos`.
  - Reshape `Qcur` to `[n_embd_head, n_head, n_tokens]` and `Kcur` to `[n_embd_head, n_head_kv, n_tokens]`.
  - Construct a 2D causal/parent attention mask tensor `kq_mask` of shape `[n_kv, n_tokens]` (`GGML_TYPE_F32`) where `n_kv = kv_commit_idx + n_tokens`:
    - For each batch token `b` (`0 .. n_tokens - 1`), set `kq_mask[c + b * n_kv] = 0.0f` for all confirmed history `c < kv_commit_idx`.
    - For in-flight batch indices `s` (`0 .. n_tokens - 1`), set `kq_mask[(kv_commit_idx + s) + b * n_kv] = (sym_slot->parent_mask[b] & (1U << s)) ? 0.0f : -INFINITY`.
    - Pass `kq_mask` into `ggml_soft_max_ext(ctx, kq, kq_mask, 1.0f / sqrtf((float)n_embd_head), 0.0f)`.
  - Store `k_cache` and `v_cache` in `[n_embd_k_gqa * MAX_CTX]` layout; write `n_tokens * n_embd_k_gqa` floats starting at `kv_offset = kv_commit_idx * n_embd_k_gqa * sizeof(float)`.
- [ ] 3.3 Lazy KV Cache Rollback:
  - By writing incoming `Kcur` and `Vcur` slices starting at `kv_commit_idx`, any unaccepted speculative KV slots written beyond `kv_commit_idx + n_accepted + 1` during step `s - 1` are automatically overwritten in-place on step `s` with zero extra network round-trips.
- [ ] 3.4 Chunked Prompt Prefill + Draft Proposer on PE 0:
  - Process the input prompt in chunks of up to `B_VERIFY` tokens per step instead of 1 token per step (with standard causal `parent_mask[b] = (1U << (b + 1)) - 1`).
  - During autoregressive decoding, use an N-gram / prompt-history lookup drafter (or optional draft GGUF on PE 0) to propose up to `K_DRAFT` candidate tokens `[t_root, d_1, ..., d_k]`. If no N-gram match exists in context, fall back cleanly to `n_tokens = 1` (or short speculative lookahead).
- [ ] 3.5 Tail Verification (`tail_pe_verify_and_feedback`):
  - On PE `n_pes - 1`, compute `[n_vocab, n_tokens]` logits.
  - For each position `i = 0 .. n_tokens - 1`, compute greedy `argmax` token `t_greedy[i]`.
  - During speculative decode steps (`n_tokens > 1`), compare `t_greedy[i]` against `draft_ids[i + 1]` for `i = 0 .. n_tokens - 2`:
    - Always commit `committed_ids[i] = t_greedy[i]`.
    - If `t_greedy[i] == draft_ids[i + 1]`, increment `n_accepted` and continue; otherwise stop and break (the mismatching `t_greedy[i]` acts as the free correction token!).
    - If all `n_tokens - 1` draft tokens match, also append the bonus token `committed_ids[n_accepted] = t_greedy[n_tokens - 1]`.
  - Send `shmem_verify_feedback_t` back to PE 0 via `shmem_putmem` + `shmem_quiet()` + `shmem_ulong_put` + `shmem_quiet()`.
  - Note on Loop Termination: Since each speculative step advances output generation by `1 + n_accepted` tokens, PE 0 and PE 1 must stay synchronized on when generation finishes (e.g., PE 0 sets a `done` flag or `n_tokens = 0` sentinel packet to cleanly exit the step loop on all PEs).
- [ ] 3.6 Verification Gate:
  - Confirm that batched prefill + speculative decoding produces the exact same greedy token sequence as Phase 2 single-token decoding while reporting total pipeline steps, accepted draft tokens, and tokens/sec speedup.

### Phase 4: Exact Stochastic Speculative Sampling (T > 0)
- [ ] 4.1 Top-S Sparse Draft Support (`TOP_S_SUPPORT = 32`): Min-heap + Min-P filter on PE 0 populating `q_sparse[K_DRAFT][32]` and `q_sampled[K_DRAFT]`.
- [ ] 4.2 SplitMix64 PRNG & O(S) Sparse Residual Resampling on Tail PE.

### Phase 5: Multi-Branch Tree Speculation & In-Place KV Compaction
- [ ] 5.1 Tree Topology (`parent_idx`, `ancestor_mask`) & Without-Replacement Branch Sampling on PE 0.
- [ ] 5.2 Iterative Multi-Sibling Residual Verification on Tail PE.
- [ ] 5.3 Lazy Non-Contiguous KV Cache Compaction (`prev_accepted_nodes[]`) on All PEs.
