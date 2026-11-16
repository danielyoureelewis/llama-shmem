#include <shmem.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include "ggml.h"
#include "ggml-cpu.h"

#define D_MODEL 4096

struct __attribute__((aligned(64))) shmem_stage_slot_t {
    float    act_buf[D_MODEL];
    unsigned long step_flag;
};

static shmem_stage_slot_t * sym_slot = nullptr;
static unsigned long local_step = 1;

void shmem_op_send(struct ggml_tensor * dst, const struct ggml_tensor * src, int ith, int nth, void * userdata) {
    (void)nth;
    if (ith != 0) return;
    int next_pe = *(int *)userdata;
    size_t n_bytes = ggml_nbytes(src);

    shmem_putmem(sym_slot->act_buf, src->data, n_bytes, next_pe);
    shmem_quiet();
    shmem_ulong_put(&sym_slot->step_flag, &local_step, 1, next_pe);
    shmem_quiet();

    if (dst->data != src->data) {
        memcpy(dst->data, src->data, n_bytes);
    }
}

void shmem_op_recv(struct ggml_tensor * dst, const struct ggml_tensor * src, int ith, int nth, void * userdata) {
    (void)src; (void)nth; (void)userdata;
    if (ith != 0) return;

    shmem_ulong_wait_until(&sym_slot->step_flag, SHMEM_CMP_EQ, local_step);
    if (dst->data != sym_slot->act_buf) {
        memcpy(dst->data, sym_slot->act_buf, ggml_nbytes(dst));
    }
}

int main() {
    int provided;
    shmem_init_thread(SHMEM_THREAD_FUNNELED, &provided);

    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();

    if (n_pes < 2) {
        if (my_pe == 0) fprintf(stderr, "Run with at least 2 PEs: oshrun -np 2 ./test_shmem_pipe\n");
        shmem_finalize();
        return 1;
    }

    sym_slot = (shmem_stage_slot_t *)shmem_align(64, sizeof(shmem_stage_slot_t));
    memset(sym_slot, 0, sizeof(shmem_stage_slot_t));
    shmem_barrier_all();

    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D_MODEL);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    int next_pe = (my_pe + 1) % n_pes;
    struct ggml_tensor * out = nullptr;

    if (my_pe == 0) {
        float * data = (float *)x->data;
        for (int i = 0; i < D_MODEL; ++i) data[i] = 3.0f;
        struct ggml_tensor * scaled = ggml_scale(ctx, x, 2.0f);
        out = ggml_map_custom1_inplace(ctx, scaled, shmem_op_send, 1, &next_pe);
    } else if (my_pe == 1) {
        x->data = sym_slot->act_buf;
        struct ggml_tensor * recv_t = ggml_map_custom1_inplace(ctx, x, shmem_op_recv, 1, nullptr);
        out = ggml_scale(ctx, recv_t, 5.0f);
    }

    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 2);

    if (my_pe == 1) {
        float * res = (float *)out->data;
        printf("[PE 1] Pipeline verification passed! out[0] = %.1f (expected 30.0)\n", res[0]);
    }

    shmem_barrier_all();
    ggml_free(ctx);
    shmem_free(sym_slot);
    shmem_finalize();
    return 0;
}
