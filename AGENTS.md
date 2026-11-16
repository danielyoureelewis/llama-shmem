# Agent Instructions: `llama-shmem`

## Environment & Execution Model
- **Host vs. Cluster:** You are editing files in `~/llama-shmem-workspace` on the host, which is bind-mounted at `/workspace` inside two Ubuntu 24.04 LXD system containers (`pe0` and `pe1`).
- **Build & Test Strictly Inside `pe0`:** Never compile on the host. Run all builds and `oshrun` commands via `lxc exec pe0 -- ...`.

## Canonical Build & Run Commands
- **Build:**
  lxc exec pe0 -- bash -c "cd /workspace/build && cmake .. && make -j$(nproc)"
- **Run Phase 1 Sanity Test (`test_shmem_pipe`):**
  lxc exec pe0 -- oshrun --allow-run-as-root -np 2 --host pe0:1,pe1:1 -x SHMEM_SYMMETRIC_SIZE=512M -x UCX_TLS=tcp,self -x LD_LIBRARY_PATH=/usr/local/lib setarch x86_64 -R /workspace/build/test_shmem_pipe
- **Run Distributed Inference (`llama_shmem`):**
  lxc exec pe0 -- oshrun --allow-run-as-root -np 2 --host pe0:1,pe1:1 -x SHMEM_SYMMETRIC_SIZE=512M -x UCX_TLS=tcp,self -x LD_LIBRARY_PATH=/usr/local/lib setarch x86_64 -R /workspace/build/llama_shmem /workspace/qwen2.5-0.5b-instruct-q4_k_m.gguf "The capital of France is" 64
- **Single-Node Greedy Baseline (`llama-cli`):**
  lxc exec pe0 -- /usr/local/bin/llama-cli -m /workspace/qwen2.5-0.5b-instruct-q4_k_m.gguf -p "The capital of France is" -n 64 --temp 0 -no-cnv

## Mandatory OpenSHMEM 1.4 & UCX Rules
1. **Never use `shmem_ulong_atomic_set` or `shmem_uint64_*`:**
   - OpenMPI 4.1.x on Ubuntu 24.04 is OpenSHMEM 1.4 (no `uint64` wrappers).
   - Over `UCX_TLS=tcp`, mixing `shmem_putmem` (RMA) with `shmem_ulong_atomic_set` (AMO active message) deadlocks after ~20 steps.
   - Always use `unsigned long` flags with `shmem_ulong_put(&flag, &val, 1, pe)` preceded and followed by `shmem_quiet()`.
2. **Always launch with `setarch x86_64 -R` and `-fno-pie -no-pie`:**
   - Keeps all 33+ linked shared libraries (`libggml.so`, `libllama.so`, etc.) mapped at identical virtual addresses across `pe0` and `pe1` so `mca_memheap_modex_recv_all` never segfaults during `shmem_init_thread`.
3. **Never assume local symmetric writes broadcast:**
   - Each PE has its own symmetric heap instance. Writing `sym_slot->field = x` on PE 0 only updates PE 0. You must call `shmem_putmem` / `shmem_int32_put` + `shmem_quiet()` to push data to remote PEs.
   - Do not rely on `argv` parsing on PE > 0 (`oshrun` over SSH splits quoted strings containing spaces). Parse CLI arguments on PE 0 and broadcast via OpenSHMEM, or drive termination via an explicit `n_tokens == 0` sentinel packet from PE 0.
4. **Thread-0 Guard:** Inside `ggml` custom operators, always start with `if (ith != 0) return;`.

## Current Task
Phases 1 and 2 are complete and verified. Implement **Phase 3** in `src/llama_shmem.cpp` per `SPEC.md`.
