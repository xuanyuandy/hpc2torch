#include <stdio.h>
#include "cpu/common_cpu.h"
#include <cstring>
#include "omp.h"

typedef uint16_t fp16_t;

void PackQuantizedWeight(const fp16_t *B, const fp16_t *b_scale, const fp16_t *zero,
                         int32_t *packed_B, int K, int N, int group_size)
{
    int num_groups = (group_size == -1) ? 1 : K / group_size;
    int blocks_per_group = (group_size == -1) ? K / 8 : group_size / 8;

    for (int n = 0; n < N; ++n)
    {
        for (int g = 0; g < num_groups; ++g)
        {
            float scale = f16_to_f32(b_scale[n * num_groups + g]);
            float zero_f = f16_to_f32(zero[n * num_groups + g]);

            for (int b = 0; b < blocks_per_group; ++b)
            {
                int row_base = (group_size == -1) ? b * 8 : g * group_size + b * 8;
                int row_block_idx = row_base / 8;

                int32_t packed = 0;
                for (int i = 0; i < 8; ++i)
                {
                    int k = row_base + i;
                    float val = f16_to_f32(B[n * K + k]); // B: [N, K]
                    int q = static_cast<int>(std::roundf(val / scale + zero_f));
                    q = std::max(0, std::min(15, q));
                    packed |= (q & 0xF) << (i * 4);
                }

                packed_B[n * (K / 8) + row_block_idx] = packed;
            }
        }
    }
}

void MatmulPackedWeight(fp16_t *C, const fp16_t *A, const int32_t *packed_B,
                        const fp16_t *b_scale, const fp16_t *zero,
                        int M, int K, int N, int group_size)
{
    int num_groups = (group_size == -1) ? 1 : K / group_size;
    int blocks_per_group = (group_size == -1) ? K / 8 : group_size / 8;

    for (int n = 0; n < N; ++n)
    {
        for (int m = 0; m < M; ++m)
        {
            float acc = 0.0f;

            for (int g = 0; g < num_groups; ++g)
            {
                float scale = f16_to_f32(b_scale[n * num_groups + g]);
                float zero_f = f16_to_f32(zero[n * num_groups + g]);

                for (int b = 0; b < blocks_per_group; ++b)
                {
                    int row_base = (group_size == -1) ? b * 8 : g * group_size + b * 8;
                    int row_block_idx = row_base / 8;
                    int32_t packed = packed_B[n * (K / 8) + row_block_idx];

                    for (int i = 0; i < 8; ++i)
                    {
                        int k = row_base + i;
                        int q = (packed >> (i * 4)) & 0xF;
                        float w = (q - zero_f) * scale;

                        float a_val = f16_to_f32(A[k * M + m]); // A: [K, M]
                        acc += w * a_val;
                    }
                }
            }

            C[n * M + m] = f32_to_f16(acc);
        }
    }
}
void gptqDevice(fp16_t *C, const fp16_t *A, const fp16_t *B, const fp16_t *b_scale, const fp16_t *zero,
                int M, int K, int N, int group_size)
{
    int k_blocks = K / 16;
    int n_blocks = N * 2;

    int32_t *packed_weights = (int32_t *)malloc(sizeof(int32_t) * k_blocks * n_blocks);
    memset(packed_weights, 0, sizeof(int32_t) * k_blocks * n_blocks);
    PackQuantizedWeight(B, b_scale, zero, packed_weights, K, N, group_size);
    MatmulPackedWeight(C, A, packed_weights, b_scale, zero, M, K, N, group_size);
}
extern "C" void gptq_cpu(fp16_t *C, const fp16_t *A, const fp16_t *B, const fp16_t *b_scale, const fp16_t *zero,
                         int M, int K, int N, int group_size)
{
    gptqDevice(C, A, B, b_scale, zero,
               M, K, N, group_size);
}