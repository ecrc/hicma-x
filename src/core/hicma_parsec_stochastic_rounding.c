#include "hicma_parsec_gpu.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

static inline uint64_t hicma_parsec_splitmix64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

/*
 * Stochastic rounding (SR): x is rounded to one of its two neighbours in FP32,
 * lower <= |x| <= upper, choosing lower with probability
 * (upper - |x|) / (upper - lower), so that E[SR(x)] = x.  Rounding is done on
 * |x| and the sign put back, so negative values use their own neighbours.
 * The uniform random number of element (i, j) is a hash of (i, j), of the
 * seed of its tile (hicma_parsec_sr_seed(m, n), from the tile coordinates)
 * and of the bits of the value being rounded, the same as in the CUDA and HIP
 * versions: results are reproducible and identical on CPU and GPU, and the
 * elements of different tiles get independent random numbers, even where
 * their values are equal.
 */

/* Uniform in [0, 1), exactly representable as float, from the position
 * (i, j), the seed of the tile and the value to round */
static inline float hicma_parsec_sr_u01(int i, int j, uint64_t seed, double value)
{
    uint64_t bits, pos = (uint64_t)(uint32_t)i | ((uint64_t)(uint32_t)j << 32);
    memcpy(&bits, &value, sizeof(bits));
    uint64_t rnd = hicma_parsec_splitmix64(bits ^ hicma_parsec_splitmix64(pos ^ seed));
    return (float)(rnd >> 40) * (1.0f / 16777216.0f);
}

static inline float hicma_parsec_stochastic_round_fp32(double value, float random)
{
    float nearest = (float)value;
    if (!isfinite(value) || !isfinite(nearest)) {
        return nearest;
    }

    double a = fabs(value);
    float n = fabsf(nearest);
    float lower, upper;
    if (a > (double)n) {
        lower = n;
        upper = nextafterf(n, INFINITY);
    } else if (a < (double)n) {
        lower = nextafterf(n, 0.0f);
        upper = n;
    } else {
        return nearest;
    }
    if (isinf(upper)) {
        return nearest;
    }

    double p_lower = ((double)upper - a) / ((double)upper - (double)lower);
    float r = (random < p_lower) ? lower : upper;
    return (value < 0.0) ? -r : r;
}

void double2float_round_CPU(int nrows, int ncols,
                            const double *D, int ldd,
                            float *F, int ldf, uint64_t seed)
{
    for (int j = 0; j < ncols; ++j) {
        for (int i = 0; i < nrows; ++i) {
            double value = D[j * ldd + i];
            F[j * ldf + i] = hicma_parsec_stochastic_round_fp32(value, hicma_parsec_sr_u01(i, j, seed, value));
        }
    }
}

