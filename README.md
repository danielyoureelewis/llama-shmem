# llama-shmem

A bare-metal C++ distributed LLM inference engine built on ggml / llama.cpp and OpenSHMEM. It runs Megatron-style Tensor Parallelism (TP) and neural speculative decoding across ARM64 clusters over AWS EFA RDMA or standard Ethernet TCP/IP with zero Python or container dependencies.

## How It Works

* **Lockless Butterfly AllReduce:** Synchronizes partial activations across P nodes in log2(P) network hops using one-sided shmem_putmem_nbi and sequence-monitored shmem_ulong_wait_until doorbells.
* **Column & Row Sharding:** Splits attention heads (wq, wk, wv) and FFN intermediates (ffn_gate, ffn_up) column-wise, and wo / ffn_down row-wise. Automatically repacks sub-256 row shards to Q8_0 at load time to preserve block alignment and fast ARM64 SDOT execution.
* **Neural Speculative Decoding:** PE 0 drafts candidate tokens using Qwen2.5-0.5B-Instruct and verifies them in a single batched forward pass across the sharded target model.

## Measured Performance (ARM64 Graviton3)

Benchmarked with Q4_K_M weights, temp=0.0, 4 threads/PE, and Qwen2.5-0.5B-Instruct (k_draft=1) as the drafter:

* **2x c7gn.16xlarge (200 Gbps EFA RDMA) — Qwen2.5-7B (TP=2):**
  * Direct TP (k_draft=0): **3.83 tok/s** (260.93 ms/tok)
  * Speculative TP (k_draft=1): **4.06 tok/s** (246.45 ms/tok, 68.4% acceptance)
* **8x c7gn.4xlarge (50 Gbps ENA TCP) — Qwen2.5-32B (TP=8):**
  * Direct TP (k_draft=0): **4.51 tok/s** (221.57 ms/tok)
  * Speculative TP (k_draft=1): **5.79 tok/s** (172.57 ms/tok, 72.0% acceptance)

## Setup & Build

### 1. Install Dependencies (Ubuntu 24.04)

    sudo apt-get update && sudo apt-get install -y \
      build-essential cmake git openmpi-bin libopenmpi-dev libfabric-dev

### 2. Build llama.cpp Shared Libraries

    git clone https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp
    cd /opt/llama.cpp
    cmake -B build -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON
    cmake --build build --config Release -j $(nproc)
    sudo cmake --install build
    sudo ldconfig

### 3. Build llama-shmem

    git clone https://github.com/danielyoureelewis/llama-shmem.git
    cd llama-shmem
    mkdir -p build && cd build
    cmake ..
    make -j $(nproc)

## Run

**Standard TCP Cluster (8 Nodes running Qwen2.5-32B):**

    oshrun -np 8 --host IP0:1,IP1:1,IP2:1,IP3:1,IP4:1,IP5:1,IP6:1,IP7:1 \
      --mca btl tcp,self \
      -x LD_LIBRARY_PATH=/usr/local/lib \
      setarch aarch64 -R ./build/llama_shmem \
      ./qwen2.5-32b-instruct-q4_k_m.gguf \
      "Explain speculative decoding." 128 0.0 0.0 42 4 1 \
      ./qwen2.5-0.5b-instruct-q4_k_m.gguf

**AWS EFA RDMA Cluster (2 Nodes running Qwen2.5-7B):**

    oshrun -np 2 --host IP0:1,IP1:1 \
      -x FI_PROVIDER=efa \
      -x FI_EFA_USE_DEVICE_RDMA=1 \
      -x LD_LIBRARY_PATH=/usr/local/lib:/opt/amazon/efa/lib \
      setarch aarch64 -R ./build/llama_shmem \
      ./qwen2.5-7b-instruct-q4_k_m.gguf \
      "Explain speculative decoding." 128 0.0 0.0 42 4 1 \
      ./qwen2.5-0.5b-instruct-q4_k_m.gguf
