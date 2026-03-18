#include "hicma_parsec_gpu.h"
#include <math.h>
#include <stdint.h>

static inline uint64_t hicma_parsec_splitmix64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static inline float hicma_parsec_u01_from_coords(int i, int j, int ld)
{
    uint64_t seed = (uint64_t)(uint32_t)i;
    seed ^= ((uint64_t)(uint32_t)j << 21);
    seed ^= ((uint64_t)(uint32_t)ld << 42);
    uint64_t rnd = hicma_parsec_splitmix64(seed);
    return (float)((rnd >> 11) * (1.0 / 9007199254740992.0)); /* 53-bit mantissa to [0,1) */
}

static inline float hicma_parsec_next_float(float x)
{
    union { float f; uint32_t u; } v = { .f = x };
    v.u += 1U;
    return v.f;
}

static inline float hicma_parsec_prev_float(float x)
{
    union { float f; uint32_t u; } v = { .f = x };
    v.u -= 1U;
    return v.f;
}

static inline float hicma_parsec_stochastic_round_fp32(double value, float random)
{
    float nearest = (float)value;
    if (!isfinite(value) || !isfinite((double)nearest)) {
        return nearest;
    }

    double nearest_value = (double)nearest;
    float lower, upper;
    if (value > nearest_value) {
        lower = nearest;
        upper = hicma_parsec_next_float(nearest);
    } else if (value < nearest_value) {
        lower = hicma_parsec_prev_float(nearest);
        upper = nearest;
    } else {
        return nearest;
    }

    double d_lower = fabs(value - (double)lower);
    double d_upper = fabs((double)upper - value);
    double p_lower = d_upper / (d_lower + d_upper);
    return (random < p_lower) ? lower : upper;
}

void double2float_round_CPU(int nrows, int ncols,
                            const double *D, int ldd,
                            float *F, int ldf)
{
    for (int j = 0; j < ncols; ++j) {
        for (int i = 0; i < nrows; ++i) {
            float random = hicma_parsec_u01_from_coords(i, j, ldf);
            F[j * ldf + i] = hicma_parsec_stochastic_round_fp32(D[j * ldd + i], random);
        }
    }
}

