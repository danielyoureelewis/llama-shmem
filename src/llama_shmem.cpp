#include <shmem.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <fcntl.h>
#include <unistd.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"

#define MAX_K_DRAFT   7
#define MAX_B_VERIFY  (MAX_K_DRAFT + 1)
#define TOP_S_SUPPORT 32
#define MAX_D_MODEL   8192
#define MAX_CTX       2048
#define MAX_LOG2_PES  6   // Supports up to 2^6 = 64 PEs in Butterfly AllReduce

struct sparse_prob_t {
    int32_t token_id;
    float   prob;
};

struct __attribute__((aligned(64))) shmem_spec_packet_t {
    uint32_t      n_tokens;
    uint32_t      kv_commit_idx;
    uint32_t      is_decode;
    float         temperature;
    float         min_p;
    uint64_t      rng_seed;

    uint32_t      prev_kv_base;
    uint32_t      prev_n_compact;
    uint8_t       prev_accepted_slots[MAX_B_VERIFY];

    int32_t       draft_ids[MAX_B_VERIFY];
    int8_t        parent_idx[MAX_B_VERIFY];
    int32_t       pos[MAX_B_VERIFY];
    uint16_t      parent_mask[MAX_B_VERIFY];

    uint8_t       q_support_len[MAX_K_DRAFT];
    float         q_sampled[MAX_K_DRAFT];
    sparse_prob_t q_sparse[MAX_K_DRAFT][TOP_S_SUPPORT];

    unsigned long step_flag;
};

// Multi-Stage Butterfly (Recursive Doubling) RDMA Exchange Buffer for N_PES = 2, 4, 8, 16, 32, 64
struct logit_cand_t {
    int32_t token_id;
    float   logit;
};

struct __attribute__((aligned(64))) shmem_tp_sync_t {
    float         stage_peer_buf[2][MAX_LOG2_PES][MAX_B_VERIFY * MAX_D_MODEL];
    unsigned long stage_ready_seq[2][MAX_LOG2_PES];
    logit_cand_t  peer_top_logits[8][MAX_B_VERIFY][TOP_S_SUPPORT];
    unsigned long logits_ready_seq[8];
};

static shmem_spec_packet_t * sym_slot = nullptr;
static shmem_tp_sync_t     * sym_tp   = nullptr;
static unsigned long local_step       = 1;
static unsigned long tp_op_seq[2]     = { 0, 0 };

struct splitmix64_t {
    uint64_t state;
    explicit splitmix64_t(uint64_t seed = 0x9E3779B97F4A7C15ULL) : state(seed) {}

    uint64_t next_u64() {
        uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    float next_f32() {
        return (float)(next_u64() >> 40) * (1.0f / 16777216.0f);
    }
};

struct tp_allreduce_ctx_t {
    int my_pe;
    int n_pes;
    int buf_slot;
};

// O(log2 P) Lockless Butterfly (Recursive Doubling) Non-Blocking RDMA AllReduce
// Completes 8-node AllReduce in 3 RDMA hops with zero global barriers.
void shmem_op_tp_allreduce(struct ggml_tensor * dst, const struct ggml_tensor * src, int ith, int nth, void * userdata) {
    (void)nth;
    if (ith != 0) return;

    tp_allreduce_ctx_t * tp = (tp_allreduce_ctx_t *)userdata;
    const size_t total_floats = ggml_nelements(src);
    const size_t nbytes = total_floats * sizeof(float);
    const float * src_ptr = (const float *)src->data;
    float * dst_ptr = (float *)dst->data;

    if (dst_ptr != src_ptr) {
        memcpy(dst_ptr, src_ptr, nbytes);
    }
    if (tp->n_pes <= 1) return;

    const int slot = tp->buf_slot;
    const unsigned long seq = ++tp_op_seq[slot];

    int s = 0;
    for (int dist = 1; dist < tp->n_pes; dist <<= 1, ++s) {
        const int peer_pe = tp->my_pe ^ dist;

        // Send current accumulated buffer to XOR partner for stage s
        shmem_putmem_nbi(sym_tp->stage_peer_buf[slot][s], dst_ptr, nbytes, peer_pe);
        shmem_fence();
        shmem_ulong_put_nbi(&sym_tp->stage_ready_seq[slot][s], &seq, 1, peer_pe);

        // Wait until XOR partner's stage s buffer arrives in local symmetric memory
        shmem_ulong_wait_until(&sym_tp->stage_ready_seq[slot][s], SHMEM_CMP_GE, seq);

        const float * __restrict p_buf = sym_tp->stage_peer_buf[slot][s];
        float       * __restrict out   = dst_ptr;
        for (size_t i = 0; i < total_floats; ++i) {
            out[i] += p_buf[i];
        }
    }

    shmem_quiet();
}

struct sharded_layer_t {
    struct ggml_tensor * attn_norm = nullptr;
    struct ggml_tensor * wq        = nullptr;
    struct ggml_tensor * wk        = nullptr;
    struct ggml_tensor * wv        = nullptr;
    struct ggml_tensor * wo        = nullptr;
    struct ggml_tensor * bq        = nullptr;
    struct ggml_tensor * bk        = nullptr;
    struct ggml_tensor * bv        = nullptr;
    struct ggml_tensor * ffn_norm  = nullptr;
    struct ggml_tensor * ffn_gate  = nullptr;
    struct ggml_tensor * ffn_up    = nullptr;
    struct ggml_tensor * ffn_down  = nullptr;
    struct ggml_tensor * k_cache   = nullptr;
    struct ggml_tensor * v_cache   = nullptr;
};

struct sharded_model_t {
    int my_pe;
    int n_pes;

    uint32_t n_vocab;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_layer;
    uint32_t n_ff;

    uint32_t n_head_local;
    uint32_t n_head_kv_local;
    uint32_t n_ff_local;
    uint32_t vocab_offset;
    uint32_t vocab_count;

    float    rms_eps;
    float    rope_freq_base;
    int32_t  rope_type;

    struct ggml_context * weight_ctx = nullptr;
    struct ggml_context * kv_ctx     = nullptr;

    struct ggml_tensor * tok_embd    = nullptr;
    struct ggml_tensor * output_norm = nullptr;
    struct ggml_tensor * output      = nullptr;

    std::vector<sharded_layer_t> layers;
};

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint32_t gguf_get_u32_suffix(struct gguf_context * ctx, const char * suffix, uint32_t def_val = 0) {
    int n_kv = gguf_get_n_kv(ctx);
    size_t s_len = strlen(suffix);
    for (int i = 0; i < n_kv; ++i) {
        const char * key = gguf_get_key(ctx, i);
        size_t k_len = strlen(key);
        if (k_len >= s_len && strcmp(key + k_len - s_len, suffix) == 0) {
            return gguf_get_val_u32(ctx, i);
        }
    }
    return def_val;
}

static float gguf_get_f32_suffix(struct gguf_context * ctx, const char * suffix, float def_val) {
    int n_kv = gguf_get_n_kv(ctx);
    size_t s_len = strlen(suffix);
    for (int i = 0; i < n_kv; ++i) {
        const char * key = gguf_get_key(ctx, i);
        size_t k_len = strlen(key);
        if (k_len >= s_len && strcmp(key + k_len - s_len, suffix) == 0) {
            return gguf_get_val_f32(ctx, i);
        }
    }
    return def_val;
}

static struct ggml_tensor * load_gguf_tensor_full(
    int fd,
    struct gguf_context * gguf_ctx,
    struct ggml_context * meta_ctx,
    struct ggml_context * dst_ctx,
    const std::string & name,
    size_t & total_bytes_loaded
) {
    int idx = gguf_find_tensor(gguf_ctx, name.c_str());
    if (idx < 0) return nullptr;

    struct ggml_tensor * meta = ggml_get_tensor(meta_ctx, name.c_str());
    struct ggml_tensor * t    = ggml_dup_tensor(dst_ctx, meta);
    ggml_set_name(t, name.c_str());

    size_t offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, idx);
    size_t nbytes = ggml_nbytes(t);

    ssize_t r = pread(fd, t->data, nbytes, (off_t)offset);
    if (r != (ssize_t)nbytes) {
        fprintf(stderr, "[PE %d] Failed reading tensor %s\n", shmem_my_pe(), name.c_str());
        exit(1);
    }
    total_bytes_loaded += nbytes;
    return t;
}

static struct ggml_tensor * load_gguf_tensor_col_shard(
    int fd,
    struct gguf_context * gguf_ctx,
    struct ggml_context * meta_ctx,
    struct ggml_context * dst_ctx,
    const std::string & name,
    int my_pe,
    int n_pes,
    size_t & total_bytes_loaded
) {
    int idx = gguf_find_tensor(gguf_ctx, name.c_str());
    if (idx < 0) return nullptr;

    struct ggml_tensor * meta = ggml_get_tensor(meta_ctx, name.c_str());
    size_t base_offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, idx);

    if (ggml_n_dims(meta) == 1) {
        int64_t ne0_local = meta->ne[0] / n_pes;
        struct ggml_tensor * t = ggml_new_tensor_1d(dst_ctx, meta->type, ne0_local);
        ggml_set_name(t, name.c_str());
        size_t nbytes = ggml_nbytes(t);
        size_t shard_offset = base_offset + (size_t)my_pe * nbytes;
        if (pread(fd, t->data, nbytes, (off_t)shard_offset) != (ssize_t)nbytes) {
            fprintf(stderr, "[PE %d] Failed reading 1D col shard %s\n", my_pe, name.c_str());
            exit(1);
        }
        total_bytes_loaded += nbytes;
        return t;
    }

    int64_t ne0 = meta->ne[0];
    int64_t base_rows = meta->ne[1] / n_pes;
    int64_t row_start = (int64_t)my_pe * base_rows;
    int64_t ne1_local = (my_pe == n_pes - 1) ? (meta->ne[1] - row_start) : base_rows;
    struct ggml_tensor * t = ggml_new_tensor_2d(dst_ctx, meta->type, ne0, ne1_local);
    ggml_set_name(t, name.c_str());

    size_t row_bytes    = ggml_row_size(meta->type, ne0);
    size_t nbytes       = (size_t)ne1_local * row_bytes;
    size_t shard_offset = base_offset + (size_t)row_start * row_bytes;
    if (pread(fd, t->data, nbytes, (off_t)shard_offset) != (ssize_t)nbytes) {
        fprintf(stderr, "[PE %d] Failed reading 2D col shard %s\n", my_pe, name.c_str());
        exit(1);
    }
    total_bytes_loaded += nbytes;
    return t;
}

// Row-Parallel Loader with Automatic Q8_0 Repacking when ne0_local is not a multiple of 256 (e.g. 32B at TP=8)
static struct ggml_tensor * load_gguf_tensor_row_shard(
    int fd,
    struct gguf_context * gguf_ctx,
    struct ggml_context * meta_ctx,
    struct ggml_context * dst_ctx,
    const std::string & name,
    int my_pe,
    int n_pes,
    size_t & total_bytes_loaded
) {
    int idx = gguf_find_tensor(gguf_ctx, name.c_str());
    if (idx < 0) return nullptr;

    struct ggml_tensor * meta = ggml_get_tensor(meta_ctx, name.c_str());
    int64_t ne0_full  = meta->ne[0];
    int64_t ne0_local = ne0_full / n_pes;
    int64_t ne1       = meta->ne[1];

    size_t full_row_bytes = ggml_row_size(meta->type, ne0_full);
    size_t base_offset    = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, idx);

    std::vector<uint8_t> tmp(full_row_bytes * ne1);
    if (pread(fd, tmp.data(), tmp.size(), (off_t)base_offset) != (ssize_t)tmp.size()) {
        fprintf(stderr, "[PE %d] Failed reading row shard %s\n", my_pe, name.c_str());
        exit(1);
    }

    int64_t blck_size = ggml_blck_size(meta->type);
    if (ne0_local % blck_size == 0) {
        // Direct zero-conversion quantized block slicing
        struct ggml_tensor * t = ggml_new_tensor_2d(dst_ctx, meta->type, ne0_local, ne1);
        ggml_set_name(t, name.c_str());
        size_t local_row_bytes = ggml_row_size(meta->type, ne0_local);
        uint8_t * dst_ptr = (uint8_t *)t->data;
        size_t col_byte_offset = (size_t)my_pe * local_row_bytes;

        for (int64_t r = 0; r < ne1; ++r) {
            memcpy(dst_ptr + r * local_row_bytes,
                   tmp.data() + r * full_row_bytes + col_byte_offset,
                   local_row_bytes);
        }
        total_bytes_loaded += ggml_nbytes(t);
        return t;
    }

    // Fallback for sub-256 block boundaries (e.g., Qwen2.5-32B on 8 nodes where ne0_local = 640 or 3456):
    // Dequantize full row to F32, slice local columns [my_pe * ne0_local ..], and quantize to Q8_0 (block_size = 32)
    if (ne0_local % 32 != 0) {
        fprintf(stderr, "[PE %d] Fatal: Tensor %s ne0_local (%lld) not divisible by 32!\n",
                my_pe, name.c_str(), (long long)ne0_local);
        exit(1);
    }

    struct ggml_tensor * t = ggml_new_tensor_2d(dst_ctx, GGML_TYPE_Q8_0, ne0_local, ne1);
    ggml_set_name(t, name.c_str());

    const struct ggml_type_traits * qtraits = ggml_get_type_traits(meta->type);
    const struct ggml_type_traits_cpu * q8_cpu = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);

    size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_0, ne0_local);
    uint8_t * dst_ptr = (uint8_t *)t->data;
    std::vector<float> f32_row(ne0_full);
    size_t col_elem_offset = (size_t)my_pe * ne0_local;

    for (int64_t r = 0; r < ne1; ++r) {
        qtraits->to_float(tmp.data() + r * full_row_bytes, f32_row.data(), ne0_full);
        q8_cpu->from_float(f32_row.data() + col_elem_offset, dst_ptr + r * q8_row_bytes, ne0_local);
    }

    (void)qtraits;
    total_bytes_loaded += ggml_nbytes(t);
    return t;
}

static void load_tp_sharded_gguf(const char * path, sharded_model_t & m) {
    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params params = { true, &meta_ctx };
    struct gguf_context * gguf_ctx = gguf_init_from_file(path, params);
    if (!gguf_ctx) {
        fprintf(stderr, "[PE %d] Failed to open GGUF: %s\n", m.my_pe, path);
        exit(1);
    }

    const char * arch = "llama";
    int arch_id = gguf_find_key(gguf_ctx, "general.architecture");
    if (arch_id >= 0) arch = gguf_get_val_str(gguf_ctx, arch_id);

    m.n_embd         = gguf_get_u32_suffix(gguf_ctx, ".embedding_length");
    m.n_head         = gguf_get_u32_suffix(gguf_ctx, ".attention.head_count");
    m.n_head_kv      = gguf_get_u32_suffix(gguf_ctx, ".attention.head_count_kv", m.n_head);
    m.n_layer        = gguf_get_u32_suffix(gguf_ctx, ".block_count");
    m.n_ff           = gguf_get_u32_suffix(gguf_ctx, ".feed_forward_length");
    m.rms_eps        = gguf_get_f32_suffix(gguf_ctx, ".attention.layer_norm_rms_epsilon", 1e-5f);
    m.rope_freq_base = gguf_get_f32_suffix(gguf_ctx, ".rope.freq_base", 10000.0f);
    m.rope_type      = (strcmp(arch, "qwen2") == 0) ? 2 : 0;

    if (m.n_head % m.n_pes != 0 || m.n_head_kv % m.n_pes != 0) {
        fprintf(stderr, "[PE %d] Error: n_head (%u) and n_head_kv (%u) must be divisible by n_pes (%d)\n",
                m.my_pe, m.n_head, m.n_head_kv, m.n_pes);
        exit(1);
    }

    m.n_head_local    = m.n_head / m.n_pes;
    m.n_head_kv_local = m.n_head_kv / m.n_pes;
    m.n_ff_local      = m.n_ff / m.n_pes;

    struct ggml_tensor * embd_meta = ggml_get_tensor(meta_ctx, "token_embd.weight");
    m.n_vocab = embd_meta->ne[1];

    // Reserve extra headroom in case Q4_K row-shards are repacked to Q8_0
    size_t shard_bytes = 256 * 1024 * 1024;
    int n_tensors = gguf_get_n_tensors(gguf_ctx);
    for (int i = 0; i < n_tensors; ++i) {
        const char * tname = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor * tm = ggml_get_tensor(meta_ctx, tname);
        int blk_id = -1;
        if (sscanf(tname, "blk.%d.", &blk_id) == 1) {
            shard_bytes += ((ggml_nbytes(tm) * 2) / m.n_pes) + 4096;
        } else {
            shard_bytes += ggml_nbytes(tm) + 4096;
        }
    }

    struct ggml_init_params w_params = { shard_bytes, nullptr, false };
    m.weight_ctx = ggml_init(w_params);

    int fd = open(path, O_RDONLY);
    size_t bytes_loaded = 0;

    uint32_t base_v = m.n_vocab / (uint32_t)m.n_pes;
    m.vocab_offset  = (uint32_t)m.my_pe * base_v;
    m.vocab_count   = (m.my_pe == m.n_pes - 1) ? (m.n_vocab - m.vocab_offset) : base_v;

    m.tok_embd    = load_gguf_tensor_full(fd, gguf_ctx, meta_ctx, m.weight_ctx, "token_embd.weight", bytes_loaded);
    m.output_norm = load_gguf_tensor_full(fd, gguf_ctx, meta_ctx, m.weight_ctx, "output_norm.weight", bytes_loaded);
    m.output      = load_gguf_tensor_col_shard(fd, gguf_ctx, meta_ctx, m.weight_ctx, "output.weight", m.my_pe, m.n_pes, bytes_loaded);
    if (!m.output) {
        size_t row_bytes = ggml_row_size(m.tok_embd->type, m.n_embd);
        m.output = ggml_view_2d(m.weight_ctx, m.tok_embd, m.n_embd, m.vocab_count, row_bytes, (size_t)m.vocab_offset * row_bytes);
    }

    m.layers.resize(m.n_layer);
    for (uint32_t il = 0; il < m.n_layer; ++il) {
        sharded_layer_t & l = m.layers[il];
        std::string pfx = "blk." + std::to_string(il) + ".";

        l.attn_norm = load_gguf_tensor_full(fd, gguf_ctx, meta_ctx, m.weight_ctx, pfx + "attn_norm.weight", bytes_loaded);
        l.wq        = load_gguf_tensor_col_shard(fd, gguf_ctx, meta_ctx, m.weight_ctx, pfx + "attn_q.weight", m.my_pe, m.n_pes, bytes_loaded);
        l.wk        = load_gguf_tensor_col_shard(fd, gguf_ctx, meta_ctx, m.weight_ctx, pfx + "attn_k.weight", m.my_pe, m.n_pes, bytes_loaded);
        l.wv        = load_gguf_tensor_col_shard(fd, gguf_ctx, meta_ctx, m.weight_ctx, pfx + "attn_v.weight", m.my_pe, m.n_pes, bytes_loaded);
        l.wo        = load_gguf_tensor_row_shard(fd, gguf_ctx, meta_ctx, m.weight_ctx, pfx + "attn_output.weight", m.my_pe, m.n_pes, bytes_loaded);

        l.bq        = load_gguf_tensor_col_shard(fd, gguf_ctx, meta_ctx, m.weight_ctx, pfx + "attn_q.bias", m.my_pe, m.n_pes, bytes_loaded);
        l.bk        = load_gguf_tensor_col_shard(fd, gguf_ctx, meta_ctx, m.weight_ctx, pfx + "attn_k.bias", m.my_pe, m.n_pes, bytes_loaded);
        l.bv        = load_gguf_tensor_col_shard(fd, gguf_ctx, meta_ctx, m.weight_ctx, pfx + "attn_v.bias", m.my_pe, m.n_pes, bytes_loaded);

        l.ffn_norm  = load_gguf_tensor_full(fd, gguf_ctx, meta_ctx, m.weight_ctx, pfx + "ffn_norm.weight", bytes_loaded);
        l.ffn_gate  = load_gguf_tensor_col_shard(fd, gguf_ctx, meta_ctx, m.weight_ctx, pfx + "ffn_gate.weight", m.my_pe, m.n_pes, bytes_loaded);
        l.ffn_up    = load_gguf_tensor_col_shard(fd, gguf_ctx, meta_ctx, m.weight_ctx, pfx + "ffn_up.weight", m.my_pe, m.n_pes, bytes_loaded);
        l.ffn_down  = load_gguf_tensor_row_shard(fd, gguf_ctx, meta_ctx, m.weight_ctx, pfx + "ffn_down.weight", m.my_pe, m.n_pes, bytes_loaded);
    }

    close(fd);
    gguf_free(gguf_ctx);
    ggml_free(meta_ctx);

    uint32_t n_embd_k_gqa_local = (m.n_embd / m.n_head) * m.n_head_kv_local;
    size_t kv_bytes = (size_t)m.n_layer * 2 * MAX_CTX * n_embd_k_gqa_local * sizeof(float) + 16 * 1024 * 1024;
    struct ggml_init_params kv_params = { kv_bytes, nullptr, false };
    m.kv_ctx = ggml_init(kv_params);

    for (uint32_t i = 0; i < m.n_layer; ++i) {
        m.layers[i].k_cache = ggml_new_tensor_1d(m.kv_ctx, GGML_TYPE_F32, (size_t)MAX_CTX * n_embd_k_gqa_local);
        m.layers[i].v_cache = ggml_new_tensor_1d(m.kv_ctx, GGML_TYPE_F32, (size_t)MAX_CTX * n_embd_k_gqa_local);
    }

    printf("[PE %d/%d] Tensor-Parallel across all %u layers (heads: %u/%u, kv_heads: %u/%u) | Weights: %.2f MB | Local KV: %.2f MB\n",
           m.my_pe, m.n_pes, m.n_layer, m.n_head_local, m.n_head, m.n_head_kv_local, m.n_head_kv,
           bytes_loaded / (1024.0 * 1024.0), kv_bytes / (1024.0 * 1024.0));
}

static int32_t sample_from_logits(
    const float * logits,
    uint32_t n_vocab,
    float temperature,
    float min_p,
    splitmix64_t & rng,
    sparse_prob_t out_top_s[TOP_S_SUPPORT],
    uint8_t & out_len
) {
    if (temperature <= 0.0f) {
        int32_t best_id = 0;
        float best_val  = logits[0];
        for (uint32_t v = 1; v < n_vocab; ++v) {
            if (logits[v] > best_val) {
                best_val = logits[v];
                best_id  = (int32_t)v;
            }
        }
        out_len = 1;
        out_top_s[0] = { best_id, 1.0f };
        return best_id;
    }

    sparse_prob_t heap[TOP_S_SUPPORT];
    uint32_t h_sz = 0;
    auto cmp_min = [](const sparse_prob_t & a, const sparse_prob_t & b) {
        return a.prob > b.prob;
    };

    for (uint32_t v = 0; v < n_vocab; ++v) {
        float val = logits[v];
        if (h_sz < TOP_S_SUPPORT) {
            heap[h_sz++] = { (int32_t)v, val };
            if (h_sz == TOP_S_SUPPORT) std::make_heap(heap, heap + TOP_S_SUPPORT, cmp_min);
        } else if (val > heap[0].prob) {
            std::pop_heap(heap, heap + TOP_S_SUPPORT, cmp_min);
            heap[TOP_S_SUPPORT - 1] = { (int32_t)v, val };
            std::push_heap(heap, heap + TOP_S_SUPPORT, cmp_min);
        }
    }
    std::sort(heap, heap + h_sz, cmp_min);

    float max_l    = heap[0].prob;
    float inv_temp = 1.0f / temperature;
    float sum_exp  = 0.0f;
    for (uint32_t i = 0; i < h_sz; ++i) {
        heap[i].prob = expf((heap[i].prob - max_l) * inv_temp);
        sum_exp += heap[i].prob;
    }
    for (uint32_t i = 0; i < h_sz; ++i) {
        heap[i].prob /= sum_exp;
    }

    float p_max  = heap[0].prob;
    float thresh = min_p * p_max;
    uint8_t p_len = 0;
    float renorm = 0.0f;
    for (uint32_t i = 0; i < h_sz; ++i) {
        if (i == 0 || heap[i].prob >= thresh) {
            out_top_s[p_len++] = heap[i];
            renorm += heap[i].prob;
        }
    }
    for (uint8_t i = 0; i < p_len; ++i) {
        out_top_s[i].prob /= renorm;
    }
    out_len = p_len;

    float r = rng.next_f32();
    float cdf = 0.0f;
    for (uint8_t i = 0; i < p_len; ++i) {
        cdf += out_top_s[i].prob;
        if (r <= cdf || i + 1 == p_len) {
            return out_top_s[i].token_id;
        }
    }
    return out_top_s[0].token_id;
}

// Helper: Sample up to 2 distinct child branches (top-1 and top-2) from draft logits
static void sample_branch_children(
    const float * logits,
    uint32_t n_vocab,
    float temperature,
    float min_p,
    splitmix64_t & rng,
    sparse_prob_t out_top_s[TOP_S_SUPPORT],
    uint8_t & out_len,
    int32_t & tok1, float & q1,
    int32_t & tok2, float & q2
) {
    if (temperature <= 0.0f) {
        int32_t best1 = 0, best2 = 1;
        if (logits[best2] > logits[best1]) std::swap(best1, best2);
        for (uint32_t v = 2; v < n_vocab; ++v) {
            if (logits[v] > logits[best1]) {
                best2 = best1;
                best1 = (int32_t)v;
            } else if (logits[v] > logits[best2]) {
                best2 = (int32_t)v;
            }
        }
        out_len = 2;
        out_top_s[0] = { best1, 1.0f };
        out_top_s[1] = { best2, 0.0f };
        tok1 = best1; q1 = 1.0f;
        tok2 = best2; q2 = 1.0f;
        return;
    }

    tok1 = sample_from_logits(logits, n_vocab, temperature, min_p, rng, out_top_s, out_len);
    q1 = 1.0f;
    for (uint8_t i = 0; i < out_len; ++i) {
        if (out_top_s[i].token_id == tok1) {
            q1 = out_top_s[i].prob;
            break;
        }
    }

    tok2 = (out_top_s[0].token_id != tok1) ? out_top_s[0].token_id : ((out_len > 1) ? out_top_s[1].token_id : (tok1 ^ 1));
    q2   = (out_top_s[0].token_id != tok1) ? out_top_s[0].prob     : ((out_len > 1) ? out_top_s[1].prob     : 1e-4f);
}

// Single-Sequence (seq_id = 0) Dynamic Branching Tree Drafter
// Uses fast O(1) tail KV trimming (llama_memory_seq_rm) on seq 0 to evaluate branches without seq_cp assertions.
static uint32_t propose_neural_draft(
    struct llama_context * draft_ctx,
    const std::vector<int32_t> & history,
    uint32_t & draft_kv_len,
    int32_t root_token,
    uint32_t kv_commit_idx,
    float temperature,
    float min_p,
    int k_draft,
    uint32_t n_vocab,
    splitmix64_t & rng,
    shmem_spec_packet_t * pkt
) {
    pkt->draft_ids[0]   = root_token;
    pkt->parent_idx[0]  = -1;
    pkt->pos[0]         = (int32_t)kv_commit_idx;
    pkt->parent_mask[0] = 1U;

    if (k_draft <= 0 || !draft_ctx) {
        return 1;
    }

    const uint32_t max_nodes = (uint32_t)(1 + k_draft);
    llama_memory_t mem = llama_get_memory(draft_ctx);

    // Track exact token IDs materialized in draft_ctx KV cache (including speculative forward tokens)
    // so accepted primary-branch tokens are reused in-place without calling seq_rm + re-decoding them.
    static std::vector<int32_t> draft_kv_tokens;

    uint32_t target_len = (uint32_t)history.size();
    if (draft_kv_len == 0 && !draft_kv_tokens.empty()) {
        draft_kv_tokens.clear();
    }

    uint32_t match_len = 0;
    uint32_t max_check = std::min((uint32_t)draft_kv_tokens.size(), target_len);
    while (match_len < max_check && draft_kv_tokens[match_len] == history[match_len]) {
        match_len++;
    }

    // Ensure at least the final token (target_len - 1) is decoded so llama_get_logits_ith(draft_ctx, -1) has Root's logits
    if (match_len == target_len && match_len > 0) {
        match_len--;
    }

    if (match_len < (uint32_t)draft_kv_tokens.size()) {
        llama_memory_seq_rm(mem, 0, (llama_pos)match_len, -1);
        draft_kv_tokens.resize(match_len);
    }
    draft_kv_len = match_len;

    // Sync only the uncached tail of history into seq 0
    if (target_len > draft_kv_len) {
        uint32_t n_new = target_len - draft_kv_len;
        llama_batch batch = llama_batch_init((int32_t)n_new, 0, 1);
        batch.n_tokens = (int32_t)n_new;
        for (uint32_t i = 0; i < n_new; ++i) {
            batch.token[i]     = history[draft_kv_len + i];
            batch.pos[i]       = (llama_pos)(draft_kv_len + i);
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (i == n_new - 1) ? 1 : 0;
        }
        if (llama_decode(draft_ctx, batch) != 0) {
            llama_batch_free(batch);
            return 1;
        }
        llama_batch_free(batch);
        for (uint32_t i = 0; i < n_new; ++i) {
            draft_kv_tokens.push_back(history[draft_kv_len + i]);
        }
        draft_kv_len = target_len;
    }

    uint32_t n_nodes = 1;
    auto add_node = [&](int8_t p, int32_t tok, float q_val, const sparse_prob_t * q_sup, uint8_t q_len) -> uint32_t {
        uint32_t u = n_nodes++;
        uint32_t d_idx = u - 1;
        pkt->draft_ids[u]         = tok;
        pkt->parent_idx[u]        = p;
        pkt->pos[u]               = pkt->pos[p] + 1;
        pkt->parent_mask[u]       = (uint16_t)(pkt->parent_mask[p] | (1U << u));
        pkt->q_support_len[d_idx] = q_len;
        pkt->q_sampled[d_idx]     = q_val;
        memcpy(pkt->q_sparse[d_idx], q_sup, q_len * sizeof(sparse_prob_t));
        return u;
    };

    llama_batch one_tok = llama_batch_init(1, 0, 1);
    auto decode_forward = [&](int32_t tok, uint32_t kv_pos) -> bool {
        one_tok.n_tokens     = 1;
        one_tok.token[0]     = tok;
        one_tok.pos[0]       = (llama_pos)kv_pos;
        one_tok.n_seq_id[0]  = 1;
        one_tok.seq_id[0][0] = 0;
        one_tok.logits[0]    = 1;
        return llama_decode(draft_ctx, one_tok) == 0;
    };

    // Depth 1 (0 extra decodes): Sample top-1 and top-2 from Root logits
    const float * cur_logits = llama_get_logits_ith(draft_ctx, -1);
    sparse_prob_t q_sup[TOP_S_SUPPORT];
    uint8_t q_len = 0;
    int32_t t1 = 0, t2 = 0;
    float q1 = 1.0f, q2 = 1.0f;
    sample_branch_children(cur_logits, n_vocab, temperature, min_p, rng, q_sup, q_len, t1, q1, t2, q2);

    uint32_t cur_primary = add_node(0, t1, q1, q_sup, q_len); // Node 1: Depth-1 top-1
    if (n_nodes < max_nodes && k_draft >= 2) {
        add_node(0, t2, q2, q_sup, q_len);                    // Node 2: Depth-1 top-2
    }

    // Forward-Only Heavy-Path Expansion: at each depth, decode only the primary top-1 token
    // and attach both its top-1 and top-2 children (2 tree nodes per single-token decode, 0 rewinds!)
    uint32_t cur_kv_pos = draft_kv_len;
    while (n_nodes < max_nodes) {
        int32_t fwd_tok = pkt->draft_ids[cur_primary];
        if (!decode_forward(fwd_tok, cur_kv_pos)) {
            break;
        }
        draft_kv_tokens.push_back(fwd_tok);
        cur_kv_pos++;
        cur_logits = llama_get_logits_ith(draft_ctx, -1);
        sample_branch_children(cur_logits, n_vocab, temperature, min_p, rng, q_sup, q_len, t1, q1, t2, q2);

        uint32_t next_primary = add_node((int8_t)cur_primary, t1, q1, q_sup, q_len);
        if (n_nodes < max_nodes) {
            add_node((int8_t)cur_primary, t2, q2, q_sup, q_len);
        }
        cur_primary = next_primary;
    }

    llama_batch_free(one_tok);
    return n_nodes;
}

static int32_t sample_from_top_candidates(
    uint32_t tok_slot,
    int n_pes,
    float temperature,
    float min_p,
    splitmix64_t & rng,
    sparse_prob_t out_top_s[TOP_S_SUPPORT],
    uint8_t & out_len
) {
    if (temperature <= 0.0f) {
        int32_t best_id  = sym_tp->peer_top_logits[0][tok_slot][0].token_id;
        float   best_val = sym_tp->peer_top_logits[0][tok_slot][0].logit;
        for (int pe = 1; pe < n_pes; ++pe) {
            const logit_cand_t & c = sym_tp->peer_top_logits[pe][tok_slot][0];
            if (c.logit > best_val) {
                best_val = c.logit;
                best_id  = c.token_id;
            }
        }
        out_len = 1;
        out_top_s[0] = { best_id, 1.0f };
        return best_id;
    }

    sparse_prob_t heap[TOP_S_SUPPORT];
    uint32_t h_sz = 0;
    auto cmp_min = [](const sparse_prob_t & a, const sparse_prob_t & b) {
        return a.prob > b.prob;
    };
    for (int pe = 0; pe < n_pes; ++pe) {
        const logit_cand_t * cands = sym_tp->peer_top_logits[pe][tok_slot];
        for (int k = 0; k < TOP_S_SUPPORT; ++k) {
            if (cands[k].token_id < 0) continue;
            float val = cands[k].logit;
            if (h_sz < TOP_S_SUPPORT) {
                heap[h_sz++] = { cands[k].token_id, val };
                if (h_sz == TOP_S_SUPPORT) std::make_heap(heap, heap + TOP_S_SUPPORT, cmp_min);
            } else if (val > heap[0].prob) {
                std::pop_heap(heap, heap + TOP_S_SUPPORT, cmp_min);
                heap[TOP_S_SUPPORT - 1] = { cands[k].token_id, val };
                std::push_heap(heap, heap + TOP_S_SUPPORT, cmp_min);
            }
        }
    }
    std::sort(heap, heap + h_sz, cmp_min);

    float max_l    = heap[0].prob;
    float inv_temp = 1.0f / temperature;
    float sum_exp  = 0.0f;
    for (uint32_t i = 0; i < h_sz; ++i) {
        heap[i].prob = expf((heap[i].prob - max_l) * inv_temp);
        sum_exp += heap[i].prob;
    }
    for (uint32_t i = 0; i < h_sz; ++i) {
        heap[i].prob /= sum_exp;
    }

    float p_max  = heap[0].prob;
    float thresh = min_p * p_max;
    uint8_t p_len = 0;
    float renorm = 0.0f;
    for (uint32_t i = 0; i < h_sz; ++i) {
        if (i == 0 || heap[i].prob >= thresh) {
            out_top_s[p_len++] = heap[i];
            renorm += heap[i].prob;
        }
    }
    for (uint8_t i = 0; i < p_len; ++i) {
        out_top_s[i].prob /= renorm;
    }
    out_len = p_len;

    float r = rng.next_f32();
    float cdf = 0.0f;
    for (uint8_t i = 0; i < p_len; ++i) {
        cdf += out_top_s[i].prob;
        if (r <= cdf || i + 1 == p_len) {
            return out_top_s[i].token_id;
        }
    }
    return out_top_s[0].token_id;
}

static void verify_tree_local(
    int n_pes,
    uint32_t n_tokens,
    float temp,
    float min_p,
    splitmix64_t & rng,
    uint32_t & out_n_accepted,
    uint32_t & out_n_committed,
    int32_t committed_ids[MAX_B_VERIFY],
    uint8_t accepted_slots[MAX_B_VERIFY]
) {
    uint32_t cur_node = 0;
    uint32_t n_acc    = 0;
    accepted_slots[0] = 0;

    while (true) {
        sparse_prob_t p_sup[TOP_S_SUPPORT];
        uint8_t p_len = 0;
        int32_t greedy_or_init = sample_from_top_candidates(cur_node, n_pes, temp, min_p, rng, p_sup, p_len);

        uint32_t children[MAX_B_VERIFY];
        uint32_t n_children = 0;
        for (uint32_t c = 1; c < n_tokens; ++c) {
            if ((uint32_t)sym_slot->parent_idx[c] == cur_node) {
                children[n_children++] = c;
            }
        }

        if (n_children == 0) {
            committed_ids[n_acc] = greedy_or_init;
            break;
        }

        if (temp <= 0.0f) {
            int32_t matched_child = -1;
            for (uint32_t i = 0; i < n_children; ++i) {
                uint32_t c = children[i];
                if (sym_slot->draft_ids[c] == greedy_or_init) {
                    matched_child = (int32_t)c;
                    break;
                }
            }
            committed_ids[n_acc] = greedy_or_init;
            if (matched_child >= 0) {
                n_acc++;
                accepted_slots[n_acc] = (uint8_t)matched_child;
                cur_node = (uint32_t)matched_child;
                continue;
            } else {
                break;
            }
        } else {
            bool child_accepted = false;
            for (uint32_t i = 0; i < n_children; ++i) {
                uint32_t c     = children[i];
                uint32_t d_idx = c - 1;
                int32_t  d_tok = sym_slot->draft_ids[c];
                float    q_val = sym_slot->q_sampled[d_idx];

                float p_val = 0.0f;
                for (uint8_t j = 0; j < p_len; ++j) {
                    if (p_sup[j].token_id == d_tok) {
                        p_val = p_sup[j].prob;
                        break;
                    }
                }

                float r_acc = rng.next_f32();
                float ratio = (q_val > 1e-9f) ? (p_val / q_val) : 0.0f;
                if (r_acc < std::min(1.0f, ratio)) {
                    committed_ids[n_acc] = d_tok;
                    n_acc++;
                    accepted_slots[n_acc] = (uint8_t)c;
                    cur_node = c;
                    child_accepted = true;
                    break;
                } else {
                    const sparse_prob_t * q_sup = sym_slot->q_sparse[d_idx];
                    uint8_t q_len               = sym_slot->q_support_len[d_idx];
                    float res_sum = 0.0f;
                    for (uint8_t j = 0; j < p_len; ++j) {
                        int32_t tid = p_sup[j].token_id;
                        float q_j = 0.0f;
                        for (uint8_t m = 0; m < q_len; ++m) {
                            if (q_sup[m].token_id == tid) {
                                q_j = q_sup[m].prob;
                                break;
                            }
                        }
                        p_sup[j].prob = std::max(0.0f, p_sup[j].prob - q_j);
                        res_sum += p_sup[j].prob;
                    }
                    if (res_sum > 1e-7f) {
                        for (uint8_t j = 0; j < p_len; ++j) {
                            p_sup[j].prob /= res_sum;
                        }
                    }
                }
            }

            if (child_accepted) continue;

            float sum_p = 0.0f;
            for (uint8_t j = 0; j < p_len; ++j) sum_p += p_sup[j].prob;
            int32_t resampled = p_sup[0].token_id;
            if (sum_p > 1e-7f) {
                float r = rng.next_f32() * sum_p;
                float cdf = 0.0f;
                for (uint8_t j = 0; j < p_len; ++j) {
                    cdf += p_sup[j].prob;
                    if (r <= cdf || j + 1 == p_len) {
                        resampled = p_sup[j].token_id;
                        break;
                    }
                }
            }
            committed_ids[n_acc] = resampled;
            break;
        }
    }

    out_n_accepted  = n_acc;
    out_n_committed = n_acc + 1;
}

static bool evaluate_tp_step(
    sharded_model_t & m,
    int n_threads,
    std::vector<uint8_t> & compute_buf,
    uint32_t & out_n_accepted,
    uint32_t & out_n_committed,
    int32_t committed_ids[MAX_B_VERIFY],
    uint8_t accepted_slots[MAX_B_VERIFY]
) {
    if (m.my_pe == 0) {
        for (int pe = 1; pe < m.n_pes; ++pe) {
            shmem_putmem_nbi(sym_slot, sym_slot, offsetof(shmem_spec_packet_t, step_flag), pe);
        }
        shmem_fence();
        for (int pe = 1; pe < m.n_pes; ++pe) {
            shmem_ulong_put_nbi(&sym_slot->step_flag, &local_step, 1, pe);
        }
        shmem_quiet();
    } else {
        shmem_ulong_wait_until(&sym_slot->step_flag, SHMEM_CMP_EQ, local_step);
    }

    const uint32_t n_tokens      = sym_slot->n_tokens;
    const uint32_t kv_commit_idx = sym_slot->kv_commit_idx;
    if (n_tokens == 0) {
        return false;
    }

    const uint32_t n_embd              = m.n_embd;
    const uint32_t n_head_local        = m.n_head_local;
    const uint32_t n_head_kv_local     = m.n_head_kv_local;
    const uint32_t n_embd_head         = n_embd / m.n_head;
    const uint32_t n_embd_q_local      = n_embd_head * n_head_local;
    const uint32_t n_embd_k_gqa_local  = n_embd_head * n_head_kv_local;
    const uint32_t n_kv                = kv_commit_idx + n_tokens;

    if (sym_slot->prev_n_compact > 1) {
        uint32_t base = sym_slot->prev_kv_base;
        size_t slot_bytes = (size_t)n_embd_k_gqa_local * sizeof(float);
        for (size_t il = 0; il < m.layers.size(); ++il) {
            float * k_ptr = (float *)m.layers[il].k_cache->data;
            float * v_ptr = (float *)m.layers[il].v_cache->data;
            for (uint32_t d = 0; d < sym_slot->prev_n_compact; ++d) {
                uint32_t src_slot = base + sym_slot->prev_accepted_slots[d];
                uint32_t dst_slot = base + d;
                if (src_slot != dst_slot) {
                    memmove(k_ptr + (size_t)dst_slot * n_embd_k_gqa_local,
                            k_ptr + (size_t)src_slot * n_embd_k_gqa_local,
                            slot_bytes);
                    for (uint32_t e = 0; e < n_embd_k_gqa_local; ++e) {
                        v_ptr[(size_t)e * MAX_CTX + dst_slot] = v_ptr[(size_t)e * MAX_CTX + src_slot];
                    }
                }
            }
        }
    }

    struct ggml_init_params c_params = { compute_buf.size(), compute_buf.data(), false };
    struct ggml_context * ctx = ggml_init(c_params);
    struct ggml_cgraph  * gf  = ggml_new_graph_custom(ctx, 16384, false);

    struct ggml_tensor * pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    memcpy(pos_t->data, sym_slot->pos, n_tokens * sizeof(int32_t));

    struct ggml_tensor * kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_tokens);
    float * mask_data = (float *)kq_mask->data;
    for (uint32_t b = 0; b < n_tokens; ++b) {
        float * row = mask_data + (size_t)b * n_kv;
        for (uint32_t c = 0; c < kv_commit_idx; ++c) {
            row[c] = 0.0f;
        }
        uint16_t pmask = sym_slot->parent_mask[b];
        for (uint32_t s = 0; s < n_tokens; ++s) {
            row[kv_commit_idx + s] = (pmask & (1U << s)) ? 0.0f : -INFINITY;
        }
    }

    struct ggml_tensor * inp_tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    memcpy(inp_tok->data, sym_slot->draft_ids, n_tokens * sizeof(int32_t));
    struct ggml_tensor * cur = ggml_get_rows(ctx, m.tok_embd, inp_tok);

    static tp_allreduce_ctx_t tp_wo_ctx;
    static tp_allreduce_ctx_t tp_ffn_ctx;
    tp_wo_ctx  = { m.my_pe, m.n_pes, 0 };
    tp_ffn_ctx = { m.my_pe, m.n_pes, 1 };

    for (uint32_t il = 0; il < m.n_layer; ++il) {
        sharded_layer_t & l = m.layers[il];
        struct ggml_tensor * residual = cur;

        cur = ggml_rms_norm(ctx, cur, m.rms_eps);
        cur = ggml_mul(ctx, cur, l.attn_norm);

        struct ggml_tensor * Qcur = ggml_mul_mat(ctx, l.wq, cur);
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx, l.wk, cur);
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx, l.wv, cur);

        if (l.bq) Qcur = ggml_add(ctx, Qcur, l.bq);
        if (l.bk) Kcur = ggml_add(ctx, Kcur, l.bk);
        if (l.bv) Vcur = ggml_add(ctx, Vcur, l.bv);

        Qcur = ggml_reshape_3d(ctx, Qcur, n_embd_head, n_head_local,    n_tokens);
        Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv_local, n_tokens);

        Qcur = ggml_rope_ext(ctx, Qcur, pos_t, nullptr, n_embd_head, m.rope_type, 0, m.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        Kcur = ggml_rope_ext(ctx, Kcur, pos_t, nullptr, n_embd_head, m.rope_type, 0, m.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        size_t kv_offset = (size_t)kv_commit_idx * n_embd_k_gqa_local * sizeof(float);
        size_t kv_len    = (size_t)n_tokens * n_embd_k_gqa_local;
        struct ggml_tensor * k_updated = ggml_set_1d_inplace(ctx, l.k_cache, ggml_reshape_1d(ctx, Kcur, kv_len), kv_offset);
        struct ggml_tensor * v_cur_2d  = ggml_reshape_2d(ctx, Vcur, n_embd_k_gqa_local, n_tokens);
        struct ggml_tensor * v_updated = ggml_set_2d_inplace(
            ctx, l.v_cache,
            ggml_cont(ctx, ggml_transpose(ctx, v_cur_2d)),
            MAX_CTX * sizeof(float),
            (size_t)kv_commit_idx * sizeof(float)
        );
        ggml_build_forward_expand(gf, k_updated);
        ggml_build_forward_expand(gf, v_updated);

        struct ggml_tensor * K_mem = ggml_view_3d(ctx, k_updated,
            n_embd_head, n_head_kv_local, n_kv,
            n_embd_head * sizeof(float),
            n_embd_k_gqa_local * sizeof(float), 0);

        struct ggml_tensor * v = ggml_view_3d(ctx, v_updated,
            n_kv, n_embd_head, n_head_kv_local,
            MAX_CTX * sizeof(float),
            MAX_CTX * n_embd_head * sizeof(float), 0);

        struct ggml_tensor * q  = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
        struct ggml_tensor * k  = ggml_permute(ctx, K_mem, 0, 2, 1, 3);
        struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
        kq = ggml_soft_max_ext(ctx, kq, kq_mask, 1.0f / sqrtf((float)n_embd_head), 0.0f);

        struct ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);
        kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx, kqv, n_embd_q_local, n_tokens);

        cur = ggml_mul_mat(ctx, l.wo, cur);
        cur = ggml_map_custom1_inplace(ctx, cur, shmem_op_tp_allreduce, 1, &tp_wo_ctx);
        cur = ggml_add(ctx, cur, residual);

        residual = cur;
        cur = ggml_rms_norm(ctx, cur, m.rms_eps);
        cur = ggml_mul(ctx, cur, l.ffn_norm);

        struct ggml_tensor * gate = ggml_silu(ctx, ggml_mul_mat(ctx, l.ffn_gate, cur));
        struct ggml_tensor * up   = ggml_mul_mat(ctx, l.ffn_up, cur);
        cur = ggml_mul(ctx, gate, up);

        cur = ggml_mul_mat(ctx, l.ffn_down, cur);
        cur = ggml_map_custom1_inplace(ctx, cur, shmem_op_tp_allreduce, 1, &tp_ffn_ctx);
        cur = ggml_add(ctx, cur, residual);
    }

    // During prefill (!is_decode), only the last token's logits are needed
    if (!sym_slot->is_decode && n_tokens > 1) {
        cur = ggml_view_2d(ctx, cur, n_embd, 1, cur->nb[1], (size_t)(n_tokens - 1) * cur->nb[1]);
    }
    cur = ggml_rms_norm(ctx, cur, m.rms_eps);
    cur = ggml_mul(ctx, cur, m.output_norm);
    cur = ggml_mul_mat(ctx, m.output, cur);

    ggml_build_forward_expand(gf, cur);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    const uint32_t n_eval_toks  = sym_slot->is_decode ? n_tokens : 1;
    const float *  local_logits = (const float *)cur->data;
    const bool     is_greedy    = (sym_slot->temperature <= 0.0f);
    const uint32_t n_cands_send = is_greedy ? 1 : TOP_S_SUPPORT;

    for (uint32_t t = 0; t < n_eval_toks; ++t) {
        const float * row = local_logits + (size_t)t * m.vocab_count;
        logit_cand_t * top_k = sym_tp->peer_top_logits[m.my_pe][t];
        if (is_greedy) {
            float best_val = row[0];
            uint32_t best_v = 0;
            for (uint32_t v = 1; v < m.vocab_count; ++v) {
                if (row[v] > best_val) {
                    best_val = row[v];
                    best_v   = v;
                }
            }
            top_k[0] = { (int32_t)(m.vocab_offset + best_v), best_val };
        } else {
            for (int k = 0; k < TOP_S_SUPPORT; ++k) top_k[k] = { -1, -INFINITY };
            float min_top = -INFINITY;
            int   min_idx = 0;
            for (uint32_t v = 0; v < m.vocab_count; ++v) {
                float val = row[v];
                if (val > min_top) {
                    top_k[min_idx] = { (int32_t)(m.vocab_offset + v), val };
                    min_top = top_k[0].logit;
                    min_idx = 0;
                    for (int k = 1; k < TOP_S_SUPPORT; ++k) {
                        if (top_k[k].logit < min_top) {
                            min_top = top_k[k].logit;
                            min_idx = k;
                        }
                    }
                }
            }
        }
    }

    static unsigned long logits_seq = 0;
    const unsigned long l_seq = ++logits_seq;

    if (m.my_pe != 0) {
        size_t total_put_bytes = is_greedy
            ? ((size_t)(n_eval_toks - 1) * TOP_S_SUPPORT + 1) * sizeof(logit_cand_t)
            : (size_t)n_eval_toks * TOP_S_SUPPORT * sizeof(logit_cand_t);
        shmem_putmem_nbi(
            sym_tp->peer_top_logits[m.my_pe][0],
            sym_tp->peer_top_logits[m.my_pe][0],
            total_put_bytes,
            0
        );
        shmem_fence();
        shmem_ulong_put_nbi(&sym_tp->logits_ready_seq[m.my_pe], &l_seq, 1, 0);
        shmem_quiet();
    } else {
        for (int pe = 1; pe < m.n_pes; ++pe) {
            shmem_ulong_wait_until(&sym_tp->logits_ready_seq[pe], SHMEM_CMP_GE, l_seq);
        }

        const float temp  = sym_slot->temperature;
        const float min_p = sym_slot->min_p;
        splitmix64_t tail_rng(sym_slot->rng_seed);

        if (!sym_slot->is_decode) {
            sparse_prob_t dummy_sup[TOP_S_SUPPORT];
            uint8_t dummy_len = 0;
            out_n_accepted    = 0;
            out_n_committed   = 1;
            accepted_slots[0] = 0;
            committed_ids[0]  = sample_from_top_candidates(0, m.n_pes, temp, min_p, tail_rng, dummy_sup, dummy_len);
        } else {
            verify_tree_local(m.n_pes, n_tokens, temp, min_p, tail_rng,
                              out_n_accepted, out_n_committed, committed_ids, accepted_slots);
        }
    }

    ggml_free(ctx);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: oshrun -np 8 ./llama_shmem <model.gguf> [prompt] [n_predict] [temp] [min_p] [seed] [n_threads] [k_draft] [draft_model.gguf]\n");
        return 1;
    }
    const char * model_path       = argv[1];
    const char * prompt_str       = (argc >= 3) ? argv[2] : "The capital of France is";
    int n_predict                 = (argc >= 4) ? atoi(argv[3]) : 64;
    float temperature             = (argc >= 5) ? strtof(argv[4], nullptr) : 0.0f;
    float min_p                   = (argc >= 6) ? strtof(argv[5], nullptr) : 0.05f;
    uint64_t seed                 = (argc >= 7) ? strtoull(argv[6], nullptr, 10) : 42ULL;
    int n_threads                 = (argc >= 8) ? atoi(argv[7]) : 4;
    int k_draft                   = (argc >= 9) ? std::min(MAX_K_DRAFT, std::max(0, atoi(argv[8]))) : 1;
    const char * draft_model_path = (argc >= 10) ? argv[9] : nullptr;

    int provided;
    shmem_init_thread(SHMEM_THREAD_FUNNELED, &provided);

    sharded_model_t model;
    model.my_pe = shmem_my_pe();
    model.n_pes = shmem_n_pes();

    if ((model.n_pes & (model.n_pes - 1)) != 0) {
        if (model.my_pe == 0) {
            fprintf(stderr, "Error: n_pes (%d) must be a power of 2 for Butterfly AllReduce\n", model.n_pes);
        }
        shmem_finalize();
        return 1;
    }

    sym_slot = (shmem_spec_packet_t *)shmem_align(64, sizeof(shmem_spec_packet_t));
    sym_tp   = (shmem_tp_sync_t *)shmem_align(64, sizeof(shmem_tp_sync_t));
    memset(sym_slot, 0, sizeof(shmem_spec_packet_t));
    memset(sym_tp,   0, sizeof(shmem_tp_sync_t));
    shmem_barrier_all();

    load_tp_sharded_gguf(model_path, model);

    struct llama_model   * vocab_model = nullptr;
    struct llama_model   * draft_model = nullptr;
    struct llama_context * draft_ctx   = nullptr;
    const struct llama_vocab * vocab   = nullptr;
    std::vector<llama_token> prompt_tokens;

    if (model.my_pe == 0) {
        llama_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);
        llama_backend_init();

        if (draft_model_path && k_draft > 0) {
            struct llama_model_params d_mparams = llama_model_default_params();
            draft_model = llama_model_load_from_file(draft_model_path, d_mparams);
            if (!draft_model) {
                fprintf(stderr, "[PE 0] Failed to load neural draft model: %s\n", draft_model_path);
                exit(1);
            }
            struct llama_context_params d_cparams = llama_context_default_params();
            d_cparams.n_ctx           = MAX_CTX;
            d_cparams.n_batch         = MAX_CTX;
            d_cparams.n_threads       = n_threads;
            d_cparams.n_threads_batch = n_threads;
            draft_ctx = llama_init_from_model(draft_model, d_cparams);
            vocab     = llama_model_get_vocab(draft_model);
            printf("[PE 0] Loaded Neural Drafter (%s) with K_DRAFT=%d\n", draft_model_path, k_draft);
        } else {
            struct llama_model_params lparams = llama_model_default_params();
            lparams.vocab_only = true;
            vocab_model = llama_model_load_from_file(model_path, lparams);
            vocab = llama_model_get_vocab(vocab_model);
        }

        prompt_tokens.resize(strlen(prompt_str) + 16);
        int n_tok = llama_tokenize(vocab, prompt_str, strlen(prompt_str),
                                   prompt_tokens.data(), prompt_tokens.size(), true, true);
        prompt_tokens.resize(n_tok);

        printf("\nPrompt (%d tokens): \"%s\"\nGenerating %d tokens across %d PEs (Butterfly TP RDMA + %s: B_VERIFY=%d, threads=%d, temp=%.2f, min_p=%.2f, seed=%llu):\n---\n%s",
               n_tok, prompt_str, n_predict, model.n_pes,
               draft_ctx ? "0.5B Neural Drafter" : "Direct TP",
               (draft_ctx ? k_draft : 0) + 1, n_threads,
               temperature, min_p, (unsigned long long)seed, prompt_str);
        fflush(stdout);
    }
    shmem_barrier_all();

    std::vector<uint8_t> compute_buf(512 * 1024 * 1024);
    uint32_t dummy_acc = 0, dummy_comm = 0;
    int32_t  committed_ids[MAX_B_VERIFY]  = { 0 };
    uint8_t  accepted_slots[MAX_B_VERIFY] = { 0 };

    if (model.my_pe > 0) {
        while (evaluate_tp_step(model, n_threads, compute_buf, dummy_acc, dummy_comm, committed_ids, accepted_slots)) {
            local_step++;
        }
    } else {
        double t_start = now_sec();
        splitmix64_t draft_rng(seed);
        std::vector<int32_t> history(prompt_tokens.begin(), prompt_tokens.end());
        uint32_t draft_kv_len = 0;

        uint32_t kv_commit_idx = 0;
        uint32_t n_prompt      = (uint32_t)prompt_tokens.size();
        int32_t  next_root_tok = prompt_tokens.back();

        uint32_t prefill_steps = 0;
        uint32_t decode_steps  = 0;
        uint32_t total_drafted = 0;
        uint32_t total_accept  = 0;
        uint32_t branch_hits   = 0;

        uint32_t prefill_end = (n_prompt > 1) ? (n_prompt - 1) : 0;
        while (kv_commit_idx < prefill_end) {
            uint32_t chunk = std::min((uint32_t)MAX_B_VERIFY, prefill_end - kv_commit_idx);
            sym_slot->n_tokens       = chunk;
            sym_slot->kv_commit_idx  = kv_commit_idx;
            sym_slot->is_decode      = 0;
            sym_slot->temperature    = temperature;
            sym_slot->min_p          = min_p;
            sym_slot->rng_seed       = draft_rng.next_u64();
            sym_slot->prev_n_compact = 0;

            for (uint32_t b = 0; b < chunk; ++b) {
                sym_slot->draft_ids[b]   = prompt_tokens[kv_commit_idx + b];
                sym_slot->parent_idx[b]  = (int8_t)b - 1;
                sym_slot->pos[b]         = (int8_t)(kv_commit_idx + b);
                sym_slot->parent_mask[b] = (uint16_t)((1U << (b + 1)) - 1U);
            }

            evaluate_tp_step(model, n_threads, compute_buf, dummy_acc, dummy_comm, committed_ids, accepted_slots);

            kv_commit_idx += chunk;
            prefill_steps++;
            local_step++;
        }

        uint32_t prev_kv_base   = 0;
        uint32_t prev_n_compact = 0;
        uint8_t  prev_slots[MAX_B_VERIFY] = { 0 };

        int generated = 0;
        while (generated < n_predict) {
            uint32_t draft_n_vocab = vocab ? std::min(model.n_vocab, (uint32_t)llama_vocab_n_tokens(vocab)) : model.n_vocab;
            uint32_t n_batch = propose_neural_draft(
                draft_ctx, history, draft_kv_len, next_root_tok, kv_commit_idx,
                temperature, min_p, k_draft, draft_n_vocab, draft_rng, sym_slot
            );
            if (kv_commit_idx + n_batch >= MAX_CTX) {
                n_batch = 1;
            }

            sym_slot->n_tokens       = n_batch;
            sym_slot->kv_commit_idx  = kv_commit_idx;
            sym_slot->is_decode      = 1;
            sym_slot->temperature    = temperature;
            sym_slot->min_p          = min_p;
            sym_slot->rng_seed       = draft_rng.next_u64();
            sym_slot->prev_kv_base   = prev_kv_base;
            sym_slot->prev_n_compact = prev_n_compact;
            memcpy(sym_slot->prev_accepted_slots, prev_slots, MAX_B_VERIFY);

            uint32_t n_acc = 0, n_comm = 0;
            evaluate_tp_step(model, n_threads, compute_buf, n_acc, n_comm, committed_ids, accepted_slots);

            total_drafted += (n_batch - 1);
            total_accept  += n_acc;
            decode_steps++;

            prev_kv_base   = kv_commit_idx;
            prev_n_compact = n_comm;
            for (uint32_t d = 0; d < n_comm; ++d) {
                prev_slots[d] = accepted_slots[d];
                if (d > 0 && prev_slots[d] != d) {
                    branch_hits++;
                }
            }

            uint32_t used = 0;
            for (uint32_t i = 0; i < n_comm && generated < n_predict; ++i, ++used) {
                int32_t tok = committed_ids[i];
                history.push_back(tok);
                next_root_tok = tok;
                generated++;

                char piece[128];
                int n = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, true);
                if (n > 0) {
                    fwrite(piece, 1, n, stdout);
                    fflush(stdout);
                }
            }
            kv_commit_idx += used;
            local_step++;
        }

        sym_slot->n_tokens = 0;
        for (int pe = 1; pe < model.n_pes; ++pe) {
            shmem_putmem_nbi(sym_slot, sym_slot, offsetof(shmem_spec_packet_t, step_flag), pe);
        }
        shmem_fence();
        for (int pe = 1; pe < model.n_pes; ++pe) {
            shmem_ulong_put_nbi(&sym_slot->step_flag, &local_step, 1, pe);
        }
        shmem_quiet();

        double elapsed = now_sec() - t_start;
        double tok_per_s = (double)generated / elapsed;
        double avg_tok_per_step = (double)generated / (double)decode_steps;

        printf("\n---\n[TP Phase 5 Stats] Generated %d tokens in %u decode steps (+ %u prefill steps) | %.2f ms/tok (%.2f tok/s)\n",
               generated, decode_steps, prefill_steps, (elapsed * 1000.0) / generated, tok_per_s);
        printf("[Speculative Stats] Accepted %u/%u drafted tokens ( Alt-Branch Hits: %u ) | Avg %.2f tokens/step | Final kv_commit_idx = %u\n",
               total_accept, total_drafted, branch_hits, avg_tok_per_step, kv_commit_idx);
    }

    shmem_barrier_all();
    if (draft_ctx)   llama_free(draft_ctx);
    if (draft_model) llama_model_free(draft_model);
    if (vocab_model) llama_model_free(vocab_model);
    ggml_free(model.kv_ctx);
    ggml_free(model.weight_ctx);
    shmem_free(sym_slot);
    shmem_free(sym_tp);
    shmem_finalize();
    return 0;
}
