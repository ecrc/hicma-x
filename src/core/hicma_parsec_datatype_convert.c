/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 */

#include "hicma_parsec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

#ifndef ON_FUGAKU
//#include <emmintrin.h>
#endif

/* ============================================================================
   Half Precision Conversion Utilities
   ============================================================================ */

/* Reference 
 * https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/
 * https://gist.github.com/rygorous/2156668
 */

typedef unsigned int uint;

typedef union FP32
{
    uint u;
    float f;
    struct
    {
        uint Mantissa : 23;
        uint Exponent : 8;
        uint Sign : 1;
    };
} FP32;

typedef union FP16
{
    unsigned short u;
    struct
    {
        uint Mantissa : 10;
        uint Exponent : 5;
        uint Sign : 1;
    };
} FP16;

// Original ISPC reference version; this always rounds ties up.
static FP16 float_to_half(FP32 f)
{
    FP16 o = { 0 };

    // Based on ISPC reference code (with minor modifications)
    if (f.Exponent == 0) // Signed zero/denormal (which will underflow)
        o.Exponent = 0;
    else if (f.Exponent == 255) // Inf or NaN (all exponent bits set)
    {
        o.Exponent = 31;
        o.Mantissa = f.Mantissa ? 0x200 : 0; // NaN->qNaN and Inf->Inf
    }
    else // Normalized number
    {
        // Exponent unbias the single, then bias the halfp
        int newexp = f.Exponent - 127 + 15;
        if (newexp >= 31) // Overflow, return signed infinity
            o.Exponent = 31;
        else if (newexp <= 0) // Underflow
        {
            if ((14 - newexp) <= 24) // Mantissa might be non-zero
            {
                uint mant = f.Mantissa | 0x800000; // Hidden 1 bit
                o.Mantissa = mant >> (14 - newexp);
                if ((mant >> (13 - newexp)) & 1) // Check for rounding
                    o.u++; // Round, might overflow into exp bit, but this is OK
            }
        }
        else
        {
            o.Exponent = newexp;
            o.Mantissa = f.Mantissa >> 13;
            if (f.Mantissa & 0x1000) // Check for rounding
                o.u++; // Round, might overflow to inf, this is OK
        }
    }

    o.Sign = f.Sign;
    return o;
}

// Same as above, but with full round-to-nearest-even.
static FP16 float_to_half_rtne(FP32 f)
{
    FP16 o = { 0 };

    // Based on ISPC reference code (with minor modifications)
    if (f.Exponent == 0) // Signed zero/denormal (which will underflow)
        o.Exponent = 0;
    else if (f.Exponent == 255) // Inf or NaN (all exponent bits set)
    {
        o.Exponent = 31;
        o.Mantissa = f.Mantissa ? 0x200 : 0; // NaN->qNaN and Inf->Inf
    }
    else // Normalized number
    {
        // Exponent unbias the single, then bias the halfp
        int newexp = f.Exponent - 127 + 15;
        if (newexp >= 31) // Overflow, return signed infinity
            o.Exponent = 31;
        else if (newexp <= 0) // Underflow
        {
            if ((14 - newexp) <= 24) // Mantissa might be non-zero
            {
                uint mant = f.Mantissa | 0x800000; // Hidden 1 bit
                uint shift = 14 - newexp;
                o.Mantissa = mant >> shift;

                uint lowmant = mant & ((1 << shift) - 1);
                uint halfway = 1 << (shift - 1);

                if (lowmant >= halfway) // Check for rounding
                {
                    if (lowmant > halfway || (o.Mantissa & 1)) // if above halfway point or unrounded result is odd
                        o.u++; // Round, might overflow into exp bit, but this is OK
                }
            }
        }
        else
        {
            o.Exponent = newexp;
            o.Mantissa = f.Mantissa >> 13;
            if (f.Mantissa & 0x1000) // Check for rounding
            {
                if (((f.Mantissa & 0x1fff) > 0x1000) || (o.Mantissa & 1)) // if above halfway point or unrounded result is odd
                    o.u++; // Round, might overflow to inf, this is OK
            }
        }
    }

    o.Sign = f.Sign;
    return o;
}

// from half->float code - just for verification.
static FP32 half_to_float(FP16 h)
{
    static const FP32 magic = { 113 << 23 };
    static const uint shifted_exp = 0x7c00 << 13; // exponent mask after shift
    FP32 o;

    o.u = (h.u & 0x7fff) << 13;     // exponent/mantissa bits
    uint exp = shifted_exp & o.u;   // just the exponent
    o.u += (127 - 15) << 23;        // exponent adjust

    // handle exponent special cases
    if (exp == shifted_exp) // Inf/NaN?
        o.u += (128 - 16) << 23;    // extra exp adjust
    else if (exp == 0) // Zero/Denormal?
    {
        o.u += 1 << 23;             // extra exp adjust
        o.f -= magic.f;             // renormalize
    }

    o.u |= (h.u & 0x8000) << 16;    // sign bit
    return o;
}

static FP32 half_to_float_fast(FP16 h)
{
    static const FP32 magic = { (254 - 15) << 23 };
    static const FP32 was_infnan = { (127 + 16) << 23 };
    FP32 o;

    o.u = (h.u & 0x7fff) << 13;     // exponent/mantissa bits
    o.f *= magic.f;                 // exponent adjust
    if (o.f >= was_infnan.f)        // make sure Inf/NaN survive
        o.u |= 255 << 23;
    o.u |= (h.u & 0x8000) << 16;    // sign bit
    return o;
}

// Helper function to convert FP32 to FP16
static uint16_t float32_to_float16(float value) {
    uint32_t float32 = *((uint32_t *)&value);
    uint32_t sign = (float32 >> 16) & 0x8000;
    uint32_t exponent = (float32 >> 23) & 0xFF;
    uint32_t mantissa = float32 & 0x007FFFFF;
    uint16_t float16;

    if (exponent == 0) { // Zero or denormal
        // Handle denormalized numbers with rounding
        mantissa >>= (13 - 1);
        float16 = sign | (mantissa + (mantissa == 0x3FF));
    } else if (exponent == 0xFF) { // Infinity or NaN
        float16 = sign | 0x7C00 | ((mantissa != 0) ? 0x0200 : 0);
    } else {
        int16_t new_exponent = exponent - 127 + 15;
        if (new_exponent >= 0x1F) { // Overflow, set to infinity
            float16 = sign | 0x7C00;
        } else if (new_exponent <= 0) { // Underflow or denormalized
            mantissa = (mantissa | 0x00800000) >> (1 - new_exponent);
            // Handle denormalized numbers with rounding
            mantissa >>= (13 - 1);
            float16 = sign | (mantissa + (mantissa == 0x3FF));
        } else { // Normal case
            float16 = sign | (new_exponent << 10) | (mantissa >> 13);
        }
    }

    return float16;
}

/* ============================================================================
   Unified Conversion Functions
   ============================================================================ */

/**
 * @brief Unified datatype conversion function (unary version)
 * 
 * This function performs datatype conversions based on the type string parameter.
 * It calls the appropriate conversion function and sets the output size.
 * 
 * **Supported Conversion Types:**
 * - "d2s": Double precision to single precision
 * - "s2d": Single precision to double precision
 * - "d2i8": Double precision to 8-bit integer
 * - "d2i16": Double precision to 16-bit integer
 * - "d2i": Double precision to 32-bit integer
 * - "s2i8": Single precision to 8-bit integer
 * - "s2i16": Single precision to 16-bit integer
 * - "s2i": Single precision to 32-bit integer
 * - "i82s": 8-bit integer to Single precision
 * - "i82d": 8-bit integer to Double precision
 * - "i162s": 16-bit integer to Single precision
 * - "i162d": 16-bit integer to Double precision
 * - "i2s": 32-bit integer to Single precision
 * - "i2d": 32-bit integer to Double precision
 * 
 * **Error Handling:**
 * - Returns -1 on error (NULL pointers, unsupported conversion type)
 * - Returns 0 on successful conversion
 * - Updates size parameter with output buffer size in bytes
 * 
 * @param[in,out] A Pointer to the data to be converted (in-place conversion)
 * @param[in] mb Number of rows in the matrix
 * @param[in] nb Number of columns in the matrix
 * @param[in] lda Leading dimension of the matrix
 * @param[in] type Conversion type string (e.g., "d2s", "d2i8", "s2i16", etc.)
 * @param[out] size Pointer to store the output size in bytes
 * 
 * @return 0 on success, -1 on error
 * 
 * @note The conversion is performed in-place, modifying the input buffer
 * @note The size parameter is updated to reflect the new datatype size
 * @note Case-insensitive conversion type matching is supported
 */
int convert_datatype_unary_CPU(void *A, int mb, int nb, int lda, char *type, size_t *size) {
    /* Validate input parameters */
    if (!A || !type || !size) {
        return -1;
    }
    
    /* Convert type string to lowercase for case-insensitive comparison */
    char type_lower[10];
    strncpy(type_lower, type, sizeof(type_lower) - 1);
    type_lower[sizeof(type_lower) - 1] = '\0';
    
    /* Convert to lowercase for consistent comparison */
    for (int i = 0; type_lower[i]; i++) {
        if (type_lower[i] >= 'A' && type_lower[i] <= 'Z') {
            type_lower[i] = type_lower[i] - 'A' + 'a';
        }
    }
    
    /* Handle different conversion types with appropriate function calls */
    if (strcmp(type_lower, "d2s") == 0) {
        /* Double to Single precision conversion */
        convert_d2s_unary_CPU((double*)A, mb, nb);
        *size = mb * nb * sizeof(float);
        return 0;
    }
    else if (strcmp(type_lower, "s2d") == 0) {
        /* Single to Double precision conversion */
        convert_s2d_unary_CPU((float*)A, mb, nb);
        *size = mb * nb * sizeof(double);
        return 0;
    }
    else if (strcmp(type_lower, "d2i8") == 0) {
        /* Double to 8-bit integer conversion */
        convert_d2i8_unary_CPU((double*)A, mb, nb, lda);
        *size = mb * nb * sizeof(int8_t);
        return 0;
    }
    else if (strcmp(type_lower, "d2i16") == 0) {
        /* Double to 16-bit integer conversion */
        convert_d2i16_unary_CPU((double*)A, mb, nb, lda);
        *size = mb * nb * sizeof(int16_t);
        return 0;
    }
    else if (strcmp(type_lower, "d2i") == 0) {
        /* Double to 32-bit integer conversion */
        convert_d2i_unary_CPU((double*)A, mb, nb, lda);
        *size = mb * nb * sizeof(int);
        return 0;
    }
    else if (strcmp(type_lower, "s2i8") == 0) {
        /* Single to 8-bit integer conversion */
        convert_s2i8_unary_CPU((float*)A, mb, nb, lda);
        *size = mb * nb * sizeof(int8_t);
        return 0;
    }
    else if (strcmp(type_lower, "s2i16") == 0) {
        /* Single to 16-bit integer conversion */
        convert_s2i16_unary_CPU((float*)A, mb, nb, lda);
        *size = mb * nb * sizeof(int16_t);
        return 0;
    }
    else if (strcmp(type_lower, "s2i") == 0) {
        /* Single to 32-bit integer conversion */
        convert_s2i_unary_CPU((float*)A, mb, nb, lda);
        *size = mb * nb * sizeof(int);
        return 0;
    }
    else if (strcmp(type_lower, "i82s") == 0) {
        /* 8-bit integer to Single precision conversion */
        convert_8i2s_unary_CPU((int8_t*)A, mb, nb, lda);
        *size = mb * nb * sizeof(float);
        return 0;
    }
    else if (strcmp(type_lower, "i82d") == 0) {
        /* 8-bit integer to Double precision conversion */
        convert_8i2d_unary_CPU((int8_t*)A, mb, nb, lda);
        *size = mb * nb * sizeof(double);
        return 0;
    }
    else if (strcmp(type_lower, "i162s") == 0) {
        /* 16-bit integer to Single precision conversion */
        convert_16i2s_unary_CPU((int16_t*)A, mb, nb, lda);
        *size = mb * nb * sizeof(float);
        return 0;
    }
    else if (strcmp(type_lower, "i162d") == 0) {
        /* 16-bit integer to Double precision conversion */
        convert_16i2d_unary_CPU((int16_t*)A, mb, nb, lda);
        *size = mb * nb * sizeof(double);
        return 0;
    }
    else if (strcmp(type_lower, "i2s") == 0) {
        /* 32-bit integer to Single precision conversion */
        convert_i2s_unary_CPU((int*)A, mb, nb, lda);
        *size = mb * nb * sizeof(float);
        return 0;
    }
    else if (strcmp(type_lower, "i2d") == 0) {
        /* 32-bit integer to Double precision conversion */
        convert_i2d_unary_CPU((int*)A, mb, nb, lda);
        *size = mb * nb * sizeof(double);
        return 0;
    }
#if HAVE_HP_CPU
    else if (strcmp(type_lower, "s2h") == 0) {
        /* Single to Half precision conversion */
        convert_s2h_unary_CPU((float*)A, mb, nb);
        *size = mb * nb * sizeof(__fp16);
        return 0;
    }
    else if (strcmp(type_lower, "h2s") == 0) {
        /* Half to Single precision conversion */
        convert_h2s_unary_CPU((__fp16*)A, mb, nb);
        *size = mb * nb * sizeof(float);
        return 0;
    }
    else if (strcmp(type_lower, "d2h") == 0) {
        /* Double to Half precision conversion */
        convert_d2h_unary_CPU((double*)A, mb, nb);
        *size = mb * nb * sizeof(__fp16);
        return 0;
    }
    else if (strcmp(type_lower, "h2d") == 0) {
        /* Half to Double precision conversion */
        convert_h2d_unary_CPU((__fp16*)A, mb, nb);
        *size = mb * nb * sizeof(double);
        return 0;
    }
#endif
    else {
        /* Unknown conversion type */
        return -1;
    }
}

/**
 * @brief Binary version of the unified conversion function
 * 
 * This function performs datatype conversions with separate source and target buffers.
 * 
 * @param[out] target Pointer to the target buffer
 * @param[in] source Pointer to the source buffer
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns  
 * @param[in] lda Leading dimension
 * @param[in] type Conversion type string (e.g., "d2s", "d2i8", "s2i16", etc.)
 * @param[out] size Pointer to store the output size in bytes
 * 
 * @return 0 on success, -1 on error
 */
int convert_datatype_binary_CPU(void *target, void *source, int mb, int nb, int lda, char *type) {
    if (!target || !source || !type) {
        return -1;
    }
    
    // Convert type string to lowercase for case-insensitive comparison
    char type_lower[10];
    strncpy(type_lower, type, sizeof(type_lower) - 1);
    type_lower[sizeof(type_lower) - 1] = '\0';
    
    // Convert to lowercase
    for (int i = 0; type_lower[i]; i++) {
        if (type_lower[i] >= 'A' && type_lower[i] <= 'Z') {
            type_lower[i] = type_lower[i] - 'A' + 'a';
        }
    }
    
    // Handle different conversion types
    if (strcmp(type_lower, "d2s") == 0) {
        // Double to Single precision
        convert_d2s_binary_CPU((float*)target, (double*)source, mb, nb);
        return 0;
    }
    else if (strcmp(type_lower, "s2d") == 0) {
        // Single to Double precision
        convert_s2d_binary_CPU((double*)target, (float*)source, mb, nb);
        return 0;
    }
    else if (strcmp(type_lower, "d2i8") == 0) {
        // Double to 8-bit integer
        convert_d2i8_binary_CPU((int8_t*)target, (double*)source, mb, nb, lda);
        return 0;
    }
    else if (strcmp(type_lower, "d2i16") == 0) {
        // Double to 16-bit integer
        convert_d2i16_binary_CPU((int16_t*)target, (double*)source, mb, nb, lda);
        return 0;
    }
    else if (strcmp(type_lower, "s2i8") == 0) {
        // Single to 8-bit integer
        convert_s2i8_binary_CPU((int8_t*)target, (float*)source, mb, nb, lda);
        return 0;
    }
    else if (strcmp(type_lower, "s2i16") == 0) {
        // Single to 16-bit integer
        convert_s2i16_binary_CPU((int16_t*)target, (float*)source, mb, nb, lda);
        return 0;
    }
    else if (strcmp(type_lower, "i82s") == 0) {
        // 8-bit integer to Single precision
        convert_8i2s_binary_CPU((float*)target, (int8_t*)source, mb, nb, lda);
        return 0;
    }
    else if (strcmp(type_lower, "i82d") == 0) {
        // 8-bit integer to Double precision
        convert_8i2d_binary_CPU((double*)target, (int8_t*)source, mb, nb, lda);
        return 0;
    }
    else if (strcmp(type_lower, "i162s") == 0) {
        // 16-bit integer to Single precision
        convert_16i2s_binary_CPU((float*)target, (int16_t*)source, mb, nb, lda);
        return 0;
    }
    else if (strcmp(type_lower, "i162d") == 0) {
        // 16-bit integer to Double precision
        convert_16i2d_binary_CPU((double*)target, (int16_t*)source, mb, nb, lda);
        return 0;
    }
    else if (strcmp(type_lower, "i2s") == 0) {
        // 32-bit integer to Single precision
        convert_i2s_binary_CPU((float*)target, (int*)source, mb, nb, lda);
        return 0;
    }
    else if (strcmp(type_lower, "i2d") == 0) {
        // 32-bit integer to Double precision
        convert_i2d_binary_CPU((double*)target, (int*)source, mb, nb, lda);
        return 0;
    }
    else if (strcmp(type_lower, "s2i") == 0) {
        // Single to 32-bit integer
        convert_s2i_binary_CPU((int*)target, (float*)source, mb, nb, lda);
        return 0;
    }
    else if (strcmp(type_lower, "d2i") == 0) {
        // Double to 32-bit integer
        convert_d2i_binary_CPU((int*)target, (double*)source, mb, nb, lda);
        return 0;
    }
#if HAVE_HP_CPU
    else if (strcmp(type_lower, "s2h") == 0) {
        // Single to Half precision
        convert_s2h_binary_CPU((__fp16*)target, (float*)source, mb, nb);
        return 0;
    }
    else if (strcmp(type_lower, "h2s") == 0) {
        // Half to Single precision
        convert_h2s_binary_CPU((float*)target, (__fp16*)source, mb, nb);
        return 0;
    }
    else if (strcmp(type_lower, "d2h") == 0) {
        // Double to Half precision
        convert_d2h_binary_CPU((__fp16*)target, (double*)source, mb, nb);
        return 0;
    }
    else if (strcmp(type_lower, "h2d") == 0) {
        // Half to Double precision
        convert_h2d_binary_CPU((double*)target, (__fp16*)source, mb, nb);
        return 0;
    }
#endif
    else {
        // Unknown conversion type
        return -1;
    }
}

/* ============================================================================
   Individual Conversion Functions - Floating Point
   ============================================================================ */

/**
 * @brief Converts single precision to double precision in-place
 * 
 * This function converts a matrix from single precision (float) to double precision (double)
 * in-place, overwriting the original data. The conversion is performed element-wise
 * for the entire matrix tile.
 * 
 * @param[in,out] data Pointer to the matrix data (converted in-place)
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 */
void convert_s2d_unary_CPU(float *data, int mb, int nb) {
    double *data_d = (double *)data;
    for( int j = nb-1; j >= 0; j-- )
        for( int i = mb-1; i >= 0; i-- )
            data_d[j*mb+i] = (double)data[j*mb+i];
}

/**
 * @brief Converts double precision to single precision in-place
 * 
 * This function converts a matrix from double precision (double) to single precision (float)
 * in-place, overwriting the original data. The conversion is performed element-wise
 * for the entire matrix tile.
 * 
 * @param[in,out] data Pointer to the matrix data (converted in-place)
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 */
void convert_d2s_unary_CPU(double *data, int mb, int nb) {
    float *data_s = (float *)data;
    for( int j = 0; j < nb; j++ )
        for( int i = 0; i < mb; i++ )
            data_s[j*mb+i] = (float)data[j*mb+i];
}

/**
 * @brief Converts single precision to double precision (non-destructive)
 * 
 * This function converts a matrix from single precision (float) to double precision (double)
 * without modifying the source data. The conversion is performed element-wise
 * from source to target matrix.
 * 
 * @param[out] _target Pointer to the target double precision matrix
 * @param[in] _source Pointer to the source single precision matrix
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 */
void convert_s2d_binary_CPU(double *_target, float *_source, int mb, int nb) {
    double *target = (double *)_target;
    float *source = (float *)_source;
    for( int j = 0; j < nb; j++ )
        for( int i = 0; i < mb; i++ )
            target[j*mb+i] = (double)source[j*mb+i];
}

/**
 * @brief Converts double precision to single precision (non-destructive)
 * 
 * This function converts a matrix from double precision (double) to single precision (float)
 * without modifying the source data. The conversion is performed element-wise
 * from source to target matrix.
 * 
 * @param[out] _target Pointer to the target single precision matrix
 * @param[in] _source Pointer to the source double precision matrix
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 */
void convert_d2s_binary_CPU(float *_target, double *_source, int mb, int nb) {
    float *target = (float *)_target;
    double *source = (double *)_source;
    for( int j = 0; j < nb; j++ )
        for( int i = 0; i < mb; i++ )
            target[j*mb+i] = (float)source[j*mb+i];
}

/* ============================================================================
   Individual Conversion Functions - Half Precision (if available)
   ============================================================================ */

#if HAVE_HP_CPU 
/**
 * @brief Converts double precision to half precision (non-destructive)
 * 
 * This function converts a matrix from double precision (double) to half precision (__fp16)
 * without modifying the source data. The conversion is performed element-wise
 * from source to target matrix.
 * 
 * @param[out] _target Pointer to the target half precision matrix
 * @param[in] _source Pointer to the source double precision matrix
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 */
void convert_d2h_binary_CPU(__fp16 *_target, double *_source, int mb, int nb) {
     __fp16 *target = (__fp16 *)_target;
     double *source = (double *)_source;
     for( int j = 0; j < nb; j++ )
         for( int i = 0; i < mb; i++ )
             target[j*mb+i] = (__fp16)source[j*mb+i];
}

/**
 * @brief Converts half precision to double precision (non-destructive)
 * 
 * This function converts a matrix from half precision (__fp16) to double precision (double)
 * without modifying the source data. The conversion is performed element-wise
 * from source to target matrix.
 * 
 * @param[out] _target Pointer to the target double precision matrix
 * @param[in] _source Pointer to the source half precision matrix
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 */
void convert_h2d_binary_CPU(double *_target, __fp16 *_source, int mb, int nb) {
     double *target = (double *)_target;
     __fp16 *source = (__fp16 *)_source;
     for( int j = 0; j < nb; j++ )
         for( int i = 0; i < mb; i++ )
             target[j*mb+i] = (double)source[j*mb+i];
}

/**
 * @brief Converts double precision to half precision in-place
 *
 * The conversion is performed in forward order so that writing each smaller
 * half-precision value does not overwrite an unread double-precision value.
 *
 * @param[in,out] data Pointer to the matrix data (converted in-place)
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 */
void convert_d2h_unary_CPU(double *data, int mb, int nb) {
     __fp16 *data_h = (__fp16 *)data;
     for( int j = 0; j < nb; j++ )
         for( int i = 0; i < mb; i++ )
             data_h[j*mb+i] = (__fp16)data[j*mb+i];
}

/**
 * @brief Converts half precision to double precision in-place
 *
 * The conversion is performed in reverse order so that expanding each value
 * does not overwrite half-precision values that have not yet been read.
 *
 * @param[in,out] data Pointer to the matrix data (converted in-place)
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 */
void convert_h2d_unary_CPU(__fp16 *data, int mb, int nb) {
     double *data_d = (double *)data;
     for( int j = nb-1; j >= 0; j-- )
         for( int i = mb-1; i >= 0; i-- )
             data_d[j*mb+i] = (double)data[j*mb+i];
}

/**
 * @brief Converts single precision to half precision (non-destructive)
 * 
 * This function converts a matrix from single precision (float) to half precision (__fp16)
 * without modifying the source data. The conversion is performed element-wise
 * from source to target matrix.
 * 
 * @param[out] _target Pointer to the target half precision matrix
 * @param[in] _source Pointer to the source single precision matrix
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 */
void convert_s2h_binary_CPU(__fp16 *_target, float *_source, int mb, int nb) {
     __fp16 *target = (__fp16 *)_target;
     float *source = (float *)_source;
     for( int j = 0; j < nb; j++ )
         for( int i = 0; i < mb; i++ )
             target[j*mb+i] = (__fp16)source[j*mb+i];
 }

/**
 * @brief Converts half precision to single precision (non-destructive)
 * 
 * This function converts a matrix from half precision (__fp16) to single precision (float)
 * without modifying the source data. The conversion is performed element-wise
 * from source to target matrix.
 * 
 * @param[out] _target Pointer to the target single precision matrix
 * @param[in] _source Pointer to the source half precision matrix
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 */
void convert_h2s_binary_CPU(float *_target, __fp16 *_source, int mb, int nb) {
     float *target = (float *)_target;
     __fp16 *source = (__fp16 *)_source;
     for( int j = 0; j < nb; j++ )
         for( int i = 0; i < mb; i++ )
             target[j*mb+i] = (float)source[j*mb+i];
}

/**
 * @brief Converts single precision to half precision in-place
 * 
 * This function converts a matrix from single precision (float) to half precision (__fp16)
 * in-place, overwriting the original data. The conversion is performed element-wise
 * for the entire matrix tile.
 * 
 * @param[in,out] data Pointer to the matrix data (converted in-place)
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 */
void convert_s2h_unary_CPU(float *data, int mb, int nb) {
     __fp16 *data_h = (__fp16 *)data;
     for( int j = 0; j < nb; j++ )
         for( int i = 0; i < mb; i++ )
             data_h[j*mb+i] = (__fp16)data[j*mb+i];
 }

/**
 * @brief Converts half precision to single precision in-place
 * 
 * This function converts a matrix from half precision (__fp16) to single precision (float)
 * in-place, overwriting the original data. The conversion is performed element-wise
 * for the entire matrix tile.
 * 
 * @param[in,out] data Pointer to the matrix data (converted in-place)
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 */
void convert_h2s_unary_CPU(__fp16 *data, int mb, int nb) {
     float *data_s = (__fp16 *)data;
     for( int j = nb-1; j >= 0; j-- )
         for( int i = mb-1; i >= 0; i-- )
             data_s[j*mb+i] = (float)data[j*mb+i];
}
#endif

/* ============================================================================
   Individual Conversion Functions - 8-bit Integer
   ============================================================================ */

/**
 * @brief Convert single precision to 8-bit integer (unary version)
 * 
 * Converts a single precision matrix to 8-bit integer in-place.
 * This function modifies the input data buffer directly.
 * 
 * **Algorithm Details:**
 * - Iterates through matrix elements in forward order
 * - Performs explicit cast from float to int8_t
 * - Memory layout: column-major (j*lda+i indexing)
 * - Data truncation occurs for values outside int8_t range
 * 
 * **Memory Management:**
 * - The input buffer is modified to store 8-bit integer data
 * - Memory may be reclaimed after conversion if needed
 * - The function assumes the buffer will be used for int8_t data
 * 
 * @param[in,out] data Pointer to the matrix data to convert (modified in-place)
 * @param[in] mb Number of rows in the matrix
 * @param[in] nb Number of columns in the matrix
 * @param[in] lda Leading dimension of the matrix
 * 
 * @note This function modifies the input data in-place
 * @note Memory reallocation occurs to accommodate smaller datatype
 * @warning Data truncation occurs for values outside int8_t range (-128 to 127)
 * @note The function assumes column-major memory layout
 * @note Forward iteration order is used for memory safety
 * 
 * @see convert_s2i8_binary_CPU() for non-destructive conversion
 * @see convert_8i2s_unary_CPU() for reverse conversion
 */
void convert_s2i8_unary_CPU(float *data, int mb, int nb, int lda) {
    int8_t *data_i = (int8_t *)data;
    /* Convert data in-place from single precision to 8-bit integer */
    for( int j = 0; j < nb; j++ )
        for( int i = 0; i < mb; i++ )
            data_i[j*lda+i] = (int8_t)data[j*lda+i];
}

/**
 * @brief Convert double precision to 8-bit integer (unary version)
 * 
 * Converts a double precision matrix to 8-bit integer in-place.
 * This function modifies the input data buffer directly.
 * 
 * **Algorithm Details:**
 * - Iterates through matrix elements in forward order
 * - Performs explicit cast from double to int8_t
 * - Memory layout: column-major (j*lda+i indexing)
 * - Data truncation occurs for values outside int8_t range
 * 
 * **Memory Management:**
 * - The input buffer is modified to store 8-bit integer data
 * - Memory may be reclaimed after conversion if needed
 * - The function assumes the buffer will be used for int8_t data
 * 
 * @param[in,out] data Pointer to the matrix data to convert (modified in-place)
 * @param[in] mb Number of rows in the matrix
 * @param[in] nb Number of columns in the matrix
 * @param[in] lda Leading dimension of the matrix
 * 
 * @note This function modifies the input data in-place
 * @note Memory reallocation occurs to accommodate smaller datatype
 * @warning Data truncation occurs for values outside int8_t range (-128 to 127)
 * @note The function assumes column-major memory layout
 * @note Forward iteration order is used for memory safety
 * 
 * @see convert_d2i8_binary_CPU() for non-destructive conversion
 * @see convert_8i2s_unary_CPU() for reverse conversion
 */
void convert_d2i8_unary_CPU(double *data, int mb, int nb, int lda) {
    int8_t *data_i = (int8_t *)data;
    /* Convert data in-place from double precision to 8-bit integer */
    for( int j = 0; j < nb; j++ )
        for( int i = 0; i < mb; i++ )
            data_i[j*lda+i] = (int8_t)data[j*lda+i];
}

/**
 * @brief Convert single precision to 8-bit integer (binary version)
 * 
 * Converts a single precision matrix to 8-bit integer with separate
 * source and target buffers. The source data remains unchanged.
 * 
 * **Algorithm Details:**
 * - Iterates through matrix elements in forward order
 * - Performs explicit cast from float to int8_t
 * - Memory layout: column-major (j*lda+i indexing)
 * - Data truncation occurs for values outside int8_t range
 * 
 * **Memory Management:**
 * - The source buffer remains unchanged
 * - The target buffer must be pre-allocated with sufficient memory
 * - Memory layout is preserved during conversion
 * 
 * @param[out] _target Pointer to the target 8-bit integer buffer
 * @param[in] _source Pointer to the source single precision buffer
 * @param[in] mb Number of rows in the matrix
 * @param[in] nb Number of columns in the matrix
 * @param[in] lda Leading dimension of the matrix
 * 
 * @note This function does not modify the source data
 * @note The target buffer must be pre-allocated with sufficient memory
 * @warning Data truncation occurs for values outside int8_t range (-128 to 127)
 * @note The function assumes column-major memory layout
 * @note Memory layout is preserved during conversion
 * 
 * @see convert_s2i8_unary_CPU() for in-place conversion
 * @see convert_8i2s_binary_CPU() for reverse conversion
 */
void convert_s2i8_binary_CPU(int8_t *_target, float *_source, int mb, int nb, int lda) {
    int8_t *target = (int8_t *)_target;
    float *source = (float *)_source;
    /* Copy and convert data from source to target buffer */
    for( int j = 0; j < nb; j++ )
        for( int i = 0; i < mb; i++ )
            target[j*lda+i] = (int8_t)source[j*lda+i];
}

/**
 * @brief Convert double precision to 8-bit integer (binary version)
 * 
 * Converts a double precision matrix to 8-bit integer with separate
 * source and target buffers. The source data remains unchanged.
 * 
 * **Algorithm Details:**
 * - Iterates through matrix elements in forward order
 * - Performs explicit cast from double to int8_t
 * - Memory layout: column-major (j*lda+i indexing)
 * - Data truncation occurs for values outside int8_t range
 * 
 * **Memory Management:**
 * - The source buffer remains unchanged
 * - The target buffer must be pre-allocated with sufficient memory
 * - Memory layout is preserved during conversion
 * 
 * @param[out] _target Pointer to the target 8-bit integer buffer
 * @param[in] _source Pointer to the source double precision buffer
 * @param[in] mb Number of rows in the matrix
 * @param[in] nb Number of columns in the matrix
 * @param[in] lda Leading dimension of the matrix
 * 
 * @note This function does not modify the source data
 * @note The target buffer must be pre-allocated with sufficient memory
 * @warning Data truncation occurs for values outside int8_t range (-128 to 127)
 * @note The function assumes column-major memory layout
 * @note Memory layout is preserved during conversion
 * 
 * @see convert_d2i8_unary_CPU() for in-place conversion
 * @see convert_8i2s_binary_CPU() for reverse conversion
 */
void convert_d2i8_binary_CPU(int8_t *_target, double *_source, int mb, int nb, int lda) {
    int8_t *target = (int8_t *)_target;
    double *source = (double *)_source;
    /* Copy and convert data from source to target buffer */
    for( int j = 0; j < nb; j++ )
        for( int i = 0; i < mb; i++ )
            target[j*lda+i] = (int8_t)source[j*lda+i];
}

/**
 * @brief Convert 8-bit integer to single precision (unary version)
 * 
 * Converts an 8-bit integer matrix to single precision in-place.
 * This function modifies the input data buffer directly.
 * 
 * **Algorithm Details:**
 * - Iterates through matrix elements in reverse order (bottom-up, right-to-left)
 * - Performs explicit cast from int8_t to float
 * - Memory layout: column-major (j*lda+i indexing)
 * - No precision loss occurs during conversion
 * 
 * **Memory Management:**
 * - The input buffer must have sufficient space for single precision data
 * - Memory reallocation may be required before calling this function
 * - The function assumes the buffer has been expanded to accommodate floats
 * 
 * @param[in,out] data Pointer to the matrix data to convert (modified in-place)
 * @param[in] mb Number of rows in the matrix
 * @param[in] nb Number of columns in the matrix
 * @param[in] lda Leading dimension of the matrix
 * 
 * @note This function modifies the input data in-place
 * @note Memory reallocation occurs to accommodate larger datatype
 * @note No precision loss occurs during conversion to single precision
 * @note The function assumes column-major memory layout
 * @note Reverse iteration order is used for memory safety
 * 
 * @see convert_8i2s_binary_CPU() for non-destructive conversion
 * @see convert_s2i8_unary_CPU() for reverse conversion
 */
void convert_8i2s_unary_CPU(int8_t *data, int mb, int nb, int lda) {
    float *data_s = (float *)data;
    /* Iterate in reverse order for memory safety during in-place conversion */
    for( int j = nb-1; j >= 0; j-- )
        for( int i = mb-1; i >= 0; i-- )
            data_s[j*lda+i] = (float)data[j*lda+i];
}

/**
 * @brief Convert 8-bit integer to double precision (unary version)
 * 
 * Converts an 8-bit integer matrix to double precision in-place.
 * This function modifies the input data buffer directly.
 * 
 * **Algorithm Details:**
 * - Iterates through matrix elements in reverse order (bottom-up, right-to-left)
 * - Performs explicit cast from int8_t to double
 * - Memory layout: column-major (j*lda+i indexing)
 * - No precision loss occurs during conversion
 * 
 * **Memory Management:**
 * - The input buffer must have sufficient space for double precision data
 * - Memory reallocation may be required before calling this function
 * - The function assumes the buffer has been expanded to accommodate doubles
 * 
 * @param[in,out] data Pointer to the matrix data to convert (modified in-place)
 * @param[in] mb Number of rows in the matrix
 * @param[in] nb Number of columns in the matrix
 * @param[in] lda Leading dimension of the matrix
 * 
 * @note This function modifies the input data in-place
 * @note Memory reallocation occurs to accommodate larger datatype
 * @note No precision loss occurs during conversion to double precision
 * @note The function assumes column-major memory layout
 * @note Reverse iteration order is used for memory safety
 * 
 * @see convert_8i2d_binary_CPU() for non-destructive conversion
 * @see convert_d2i8_unary_CPU() for reverse conversion
 */
void convert_8i2d_unary_CPU(int8_t *data, int mb, int nb, int lda) {
    double *data_s = (double *)data;
    /* Iterate in reverse order for memory safety during in-place conversion */
    for( int j = nb-1; j >= 0; j-- )
        for( int i = mb-1; i >= 0; i-- )
            data_s[j*lda+i] = (double)data[j*lda+i];
}

/**
 * @brief Convert 8-bit integer to single precision (binary version)
 * 
 * Converts an 8-bit integer matrix to single precision with separate
 * source and target buffers. The source data remains unchanged.
 * 
 * **Algorithm Details:**
 * - Iterates through matrix elements in reverse order (bottom-up, right-to-left)
 * - Performs explicit cast from int8_t to float
 * - Memory layout: column-major (j*lda+i indexing)
 * - No precision loss occurs during conversion
 * 
 * **Memory Management:**
 * - The source buffer remains unchanged
 * - The target buffer must be pre-allocated with sufficient memory
 * - Memory layout is preserved during conversion
 * 
 * @param[out] _target Pointer to the target single precision buffer
 * @param[in] _source Pointer to the source 8-bit integer buffer
 * @param[in] mb Number of rows in the matrix
 * @param[in] nb Number of columns in the matrix
 * @param[in] lda Leading dimension of the matrix
 * 
 * @note This function does not modify the source data
 * @note The target buffer must be pre-allocated with sufficient memory
 * @note No precision loss occurs during conversion to single precision
 * @note The function assumes column-major memory layout
 * @note Memory layout is preserved during conversion
 * 
 * @see convert_8i2s_unary_CPU() for in-place conversion
 * @see convert_s2i8_binary_CPU() for reverse conversion
 */
void convert_8i2s_binary_CPU(float *_target, int8_t *_source, int mb, int nb, int lda) {
    float *target = (float *)_target;
    int8_t *source = (int8_t *)_source;
    /* Copy and convert data from source to target buffer */
    for( int j = nb-1; j >= 0; j-- )
        for( int i = mb-1; i >= 0; i-- )
            target[j*lda+i] = (float)source[j*lda+i];
}

/**
 * @brief Convert 8-bit integer to double precision (binary version)
 * 
 * Converts an 8-bit integer matrix to double precision with separate
 * source and target buffers. The source data remains unchanged.
 * 
 * **Algorithm Details:**
 * - Iterates through matrix elements in reverse order (bottom-up, right-to-left)
 * - Performs explicit cast from int8_t to double
 * - Memory layout: column-major (j*lda+i indexing)
 * - No precision loss occurs during conversion
 * 
 * **Memory Management:**
 * - The source buffer remains unchanged
 * - The target buffer must be pre-allocated with sufficient memory
 * - Memory layout is preserved during conversion
 * 
 * @param[out] _target Pointer to the target double precision buffer
 * @param[in] _source Pointer to the source 8-bit integer buffer
 * @param[in] mb Number of rows in the matrix
 * @param[in] nb Number of columns in the matrix
 * @param[in] lda Leading dimension of the matrix
 * 
 * @note This function does not modify the source data
 * @note The target buffer must be pre-allocated with sufficient memory
 * @note No precision loss occurs during conversion to double precision
 * @note The function assumes column-major memory layout
 * @note Memory layout is preserved during conversion
 * 
 * @see convert_8i2d_unary_CPU() for in-place conversion
 * @see convert_d2i8_binary_CPU() for reverse conversion
 */
void convert_8i2d_binary_CPU(double *_target, int8_t *_source, int mb, int nb, int lda) {
    double *target = (double *)_target;
    int8_t *source = (int8_t *)_source;
    /* Copy and convert data from source to target buffer */
    for( int j = nb-1; j >= 0; j-- )
        for( int i = mb-1; i >= 0; i-- )
            target[j*lda+i] = (double)source[j*lda+i];
}

/* ============================================================================
   Individual Conversion Functions - 16-bit Integer
   ============================================================================ */

/**
 * @brief Converts single precision to 16-bit integer in-place
 * 
 * This function converts a matrix from single precision (float) to 16-bit integer (int16_t)
 * in-place, overwriting the original data. The conversion is performed element-wise
 * for the entire matrix tile.
 * 
 * @param[in,out] data Pointer to the matrix data (converted in-place)
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 * @param[in] lda Leading dimension of the matrix
 */
void convert_s2i16_unary_CPU(float *data, int mb, int nb, int lda) {
    int16_t *data_i = (int16_t *)data;
    for( int j = 0; j < nb; j++ )
        for( int i = 0; i < mb; i++ )
            data_i[j*lda+i] = (int16_t)data[j*lda+i];
}

/**
 * @brief Converts double precision to 16-bit integer in-place
 * 
 * This function converts a matrix from double precision (double) to 16-bit integer (int16_t)
 * in-place, overwriting the original data. The conversion is performed element-wise
 * for the entire matrix tile.
 * 
 * @param[in,out] data Pointer to the matrix data (converted in-place)
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 * @param[in] lda Leading dimension of the matrix
 */
void convert_d2i16_unary_CPU(double *data, int mb, int nb, int lda) {
    int16_t *data_i = (int16_t *)data;
    for( int j = 0; j < nb; j++ )
        for( int i = 0; i < mb; i++ )
            data_i[j*lda+i] = (int16_t)data[j*lda+i];
}

/**
 * @brief Converts single precision to 16-bit integer (non-destructive)
 * 
 * This function converts a matrix from single precision (float) to 16-bit integer (int16_t)
 * without modifying the source data. The conversion is performed element-wise
 * from source to target matrix.
 * 
 * @param[out] _target Pointer to the target 16-bit integer matrix
 * @param[in] _source Pointer to the source single precision matrix
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 * @param[in] lda Leading dimension of the matrix
 */
void convert_s2i16_binary_CPU(int16_t *_target, float *_source, int mb, int nb, int lda) {
    int16_t *target = (int16_t *)_target;
    float *source = (float *)_source;
    for( int j = 0; j < nb; j++ )
        for( int i = 0; i < mb; i++ )
            target[j*lda+i] = (int16_t)source[j*lda+i];
}

/**
 * @brief Converts double precision to 16-bit integer (non-destructive)
 * 
 * This function converts a matrix from double precision (double) to 16-bit integer (int16_t)
 * without modifying the source data. The conversion is performed element-wise
 * from source to target matrix.
 * 
 * @param[out] _target Pointer to the target 16-bit integer matrix
 * @param[in] _source Pointer to the source double precision matrix
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 * @param[in] lda Leading dimension of the matrix
 */
void convert_d2i16_binary_CPU(int16_t *_target, double *_source, int mb, int nb, int lda) {
    int16_t *target = (int16_t *)_target;
    double *source = (double *)_source;
    for( int j = 0; j < nb; j++ )
        for( int i = 0; i < mb; i++ )
            target[j*lda+i] = (int16_t)source[j*lda+i];
}

/**
 * @brief Converts 16-bit integer to single precision in-place
 * 
 * This function converts a matrix from 16-bit integer (int16_t) to single precision (float)
 * in-place, overwriting the original data. The conversion is performed element-wise
 * for the entire matrix tile.
 * 
 * @param[in,out] data Pointer to the matrix data (converted in-place)
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 * @param[in] lda Leading dimension of the matrix
 */
void convert_16i2s_unary_CPU(int16_t *data, int mb, int nb, int lda) {
    float *data_s = (float *)data;
    for( int j = nb-1; j >= 0; j-- )
        for( int i = mb-1; i >= 0; i-- )
            data_s[j*lda+i] = (float)data[j*lda+i];
}

/**
 * @brief Converts 16-bit integer to double precision in-place
 * 
 * This function converts a matrix from 16-bit integer (int16_t) to double precision (double)
 * in-place, overwriting the original data. The conversion is performed element-wise
 * for the entire matrix tile.
 * 
 * @param[in,out] data Pointer to the matrix data (converted in-place)
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 * @param[in] lda Leading dimension of the matrix
 */
void convert_16i2d_unary_CPU(int16_t *data, int mb, int nb, int lda) {
    double *data_s = (double *)data;
    for( int j = nb-1; j >= 0; j-- )
        for( int i = mb-1; i >= 0; i-- )
            data_s[j*lda+i] = (double)data[j*lda+i];
}

/**
 * @brief Converts 16-bit integer to single precision (non-destructive)
 * 
 * This function converts a matrix from 16-bit integer (int16_t) to single precision (float)
 * without modifying the source data. The conversion is performed element-wise
 * from source to target matrix.
 * 
 * @param[out] _target Pointer to the target single precision matrix
 * @param[in] _source Pointer to the source 16-bit integer matrix
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 * @param[in] lda Leading dimension of the matrix
 */
void convert_16i2s_binary_CPU(float *_target, int16_t *_source, int mb, int nb, int lda) {
    float *target = (float *)_target;
    int16_t *source = (int16_t *)_source;
    for( int j = nb-1; j >= 0; j-- )
        for( int i = mb-1; i >= 0; i-- )
            target[j*lda+i] = (float)source[j*lda+i];
}

/**
 * @brief Converts 16-bit integer to double precision (non-destructive)
 * 
 * This function converts a matrix from 16-bit integer (int16_t) to double precision (double)
 * without modifying the source data. The conversion is performed element-wise
 * from source to target matrix.
 * 
 * @param[out] _target Pointer to the target double precision matrix
 * @param[in] _source Pointer to the source 16-bit integer matrix
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 * @param[in] lda Leading dimension of the matrix
 */
void convert_16i2d_binary_CPU(double *_target, int16_t *_source, int mb, int nb, int lda) {
    double *target = (double *)_target;
    int16_t *source = (int16_t *)_source;
    for( int j = nb-1; j >= 0; j-- )
        for( int i = mb-1; i >= 0; i-- )
            target[j*lda+i] = (double)source[j*lda+i];
}

/* ============================================================================
   Individual Conversion Functions - 32-bit Integer
   ============================================================================ */

/**
 * @brief Converts 32-bit integer to single precision in-place
 * 
 * This function converts a matrix from 32-bit integer (int) to single precision (float)
 * in-place, overwriting the original data. The conversion is performed element-wise
 * for the entire matrix tile.
 * 
 * @param[in,out] data Pointer to the matrix data (converted in-place)
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 * @param[in] lda Leading dimension of the matrix
 */
void convert_i2s_unary_CPU(int *data, int mb, int nb, int lda) {
    float *data_s = (float *)data;
    for( int j = nb-1; j >= 0; j-- )
        for( int i = mb-1; i >= 0; i-- )
            data_s[j*lda+i] = (float)data[j*lda+i];
}

/**
 * @brief Converts 32-bit integer to double precision in-place
 * 
 * This function converts a matrix from 32-bit integer (int) to double precision (double)
 * in-place, overwriting the original data. The conversion is performed element-wise
 * for the entire matrix tile.
 * 
 * @param[in,out] data Pointer to the matrix data (converted in-place)
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 * @param[in] lda Leading dimension of the matrix
 */
void convert_i2d_unary_CPU(int *data, int mb, int nb, int lda) {
    double *data_s = (double *)data;
    for( int j = nb-1; j >= 0; j-- )
        for( int i = mb-1; i >= 0; i-- )
            data_s[j*lda+i] = (double)data[j*lda+i];
}

/**
 * @brief Converts 32-bit integer to single precision (non-destructive)
 * 
 * This function converts a matrix from 32-bit integer (int) to single precision (float)
 * without modifying the source data. The conversion is performed element-wise
 * from source to target matrix.
 * 
 * @param[out] _target Pointer to the target single precision matrix
 * @param[in] _source Pointer to the source 32-bit integer matrix
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 * @param[in] lda Leading dimension of the matrix
 */
void convert_i2s_binary_CPU(float *_target, int *_source, int mb, int nb, int lda) {
    float *target = (float *)_target;
    int *source = (int *)_source;
    for( int j = nb-1; j >= 0; j-- )
        for( int i = mb-1; i >= 0; i-- )
            target[j*lda+i] = (float)source[j*lda+i];
}

/**
 * @brief Converts 32-bit integer to double precision (non-destructive)
 * 
 * This function converts a matrix from 32-bit integer (int) to double precision (double)
 * without modifying the source data. The conversion is performed element-wise
 * from source to target matrix.
 * 
 * @param[out] _target Pointer to the target double precision matrix
 * @param[in] _source Pointer to the source 32-bit integer matrix
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 * @param[in] lda Leading dimension of the matrix
 */
void convert_i2d_binary_CPU(double *_target, int *_source, int mb, int nb, int lda) {
    double *target = (double *)_target;
    int *source = (int *)_source;
    for( int j = nb-1; j >= 0; j-- )
        for( int i = mb-1; i >= 0; i-- )
            target[j*lda+i] = (double)source[j*lda+i];
}

/**
 * @brief Converts single precision to 32-bit integer in-place
 * 
 * This function converts a matrix from single precision (float) to 32-bit integer (int)
 * in-place, overwriting the original data. The conversion is performed element-wise
 * for the entire matrix tile.
 * 
 * @param[in,out] data Pointer to the matrix data (converted in-place)
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 * @param[in] lda Leading dimension of the matrix
 */
void convert_s2i_unary_CPU(float *data, int mb, int nb, int lda) {
    int *data_i = (int *)data;
    for( int j = 0; j < nb; j++ )
        for( int i = 0; i < mb; i++ )
            data_i[j*lda+i] = (int)data[j*lda+i];
}

/**
 * @brief Converts double precision to 32-bit integer in-place
 * 
 * This function converts a matrix from double precision (double) to 32-bit integer (int)
 * in-place, overwriting the original data. The conversion is performed element-wise
 * for the entire matrix tile.
 * 
 * @param[in,out] data Pointer to the matrix data (converted in-place)
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 * @param[in] lda Leading dimension of the matrix
 */
void convert_d2i_unary_CPU(double *data, int mb, int nb, int lda) {
    int *data_i = (int *)data;
    for( int j = 0; j < nb; j++ )
        for( int i = 0; i < mb; i++ )
            data_i[j*lda+i] = (int)data[j*lda+i];
}

/**
 * @brief Converts single precision to 32-bit integer (non-destructive)
 * 
 * This function converts a matrix from single precision (float) to 32-bit integer (int)
 * without modifying the source data. The conversion is performed element-wise
 * from source to target matrix.
 * 
 * @param[out] _target Pointer to the target 32-bit integer matrix
 * @param[in] _source Pointer to the source single precision matrix
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 * @param[in] lda Leading dimension of the matrix
 */
void convert_s2i_binary_CPU(int *_target, float *_source, int mb, int nb, int lda) {
    int *target = (int *)_target;
    float *source = (float *)_source;
    for( int j = 0; j < nb; j++ )
        for( int i = 0; i < mb; i++ )
            target[j*lda+i] = (int)source[j*lda+i];
}

/**
 * @brief Converts double precision to 32-bit integer (non-destructive)
 * 
 * This function converts a matrix from double precision (double) to 32-bit integer (int)
 * without modifying the source data. The conversion is performed element-wise
 * from source to target matrix.
 * 
 * @param[out] _target Pointer to the target 32-bit integer matrix
 * @param[in] _source Pointer to the source double precision matrix
 * @param[in] mb Number of rows in the matrix tile
 * @param[in] nb Number of columns in the matrix tile
 * @param[in] lda Leading dimension of the matrix
 */
void convert_d2i_binary_CPU(int *_target, double *_source, int mb, int nb, int lda) {
    int *target = (int *)_target;
    double *source = (double *)_source;
    for( int j = 0; j < nb; j++ )
        for( int i = 0; i < mb; i++ )
            target[j*lda+i] = (int)source[j*lda+i];
}

/* ============================================================================
   Utility Functions
   ============================================================================ */

/**
 * @brief Get the byte size of a datatype based on input string
 * 
 * This function returns the size in bytes of various datatypes based on the input string.
 * 
 * @param[in] datatype_str String representation of the datatype
 * 
 * @return Size in bytes of the datatype, or 0 if datatype is not recognized
 */
size_t get_datatype_size(const char *datatype_str) {
    if (datatype_str == NULL) {
        return 0;
    }
    
    // Floating point types
    if (strcmp(datatype_str, "double") == 0 || strcmp(datatype_str, "d") == 0) {
        return sizeof(double);
    }
    if (strcmp(datatype_str, "float") == 0 || strcmp(datatype_str, "single") == 0 || strcmp(datatype_str, "s") == 0) {
        return sizeof(float);
    }
    
    // Signed integer types
    if (strcmp(datatype_str, "int8") == 0 || strcmp(datatype_str, "i8") == 0) {
        return sizeof(int8_t);
    }
    if (strcmp(datatype_str, "int16") == 0 || strcmp(datatype_str, "i16") == 0) {
        return sizeof(int16_t);
    }
    if (strcmp(datatype_str, "int32") == 0 || strcmp(datatype_str, "int") == 0 || 
        strcmp(datatype_str, "i32") == 0 || strcmp(datatype_str, "i") == 0) {
        return sizeof(int);
    }
    if (strcmp(datatype_str, "int64") == 0 || strcmp(datatype_str, "i64") == 0) {
        return sizeof(int64_t);
    }
    
    // Unsigned integer types
    if (strcmp(datatype_str, "uint8") == 0 || strcmp(datatype_str, "u8") == 0) {
        return sizeof(uint8_t);
    }
    if (strcmp(datatype_str, "uint16") == 0 || strcmp(datatype_str, "u16") == 0) {
        return sizeof(uint16_t);
    }
    if (strcmp(datatype_str, "uint32") == 0 || strcmp(datatype_str, "uint") == 0 || 
        strcmp(datatype_str, "u32") == 0 || strcmp(datatype_str, "u") == 0) {
        return sizeof(unsigned int);
    }
    if (strcmp(datatype_str, "uint64") == 0 || strcmp(datatype_str, "u64") == 0) {
        return sizeof(uint64_t);
    }
    
    // Half precision
    if (strcmp(datatype_str, "half") == 0 || strcmp(datatype_str, "fp16") == 0 || strcmp(datatype_str, "h") == 0) {
        return 2;
    }
    
    // FP8 representation (stored as uint8_t)
    if (strcmp(datatype_str, "fp8") == 0) {
        return sizeof(uint8_t);
    }
    
    // FP4 representation (stored as uint8_t, 2 values per byte)
    if (strcmp(datatype_str, "fp4") == 0) {
        return sizeof(uint8_t)/2;
    }
    
    // INT4 representation (stored as uint8_t, 2 values per byte)
    if (strcmp(datatype_str, "int4") == 0) {
        return sizeof(uint8_t)/2;
    }
    
    // 1-bit representation (stored as uint8_t, 8 values per byte)
    if (strcmp(datatype_str, "1bit") == 0) {
        return sizeof(uint8_t)/8;
    }
    
    // Unknown datatype
    fprintf(stderr, "Error: Unknown datatype '%s' in get_datatype_size(). Exiting.\n", datatype_str);
    exit(1);
}

/**
 * @brief Initialize an array to zero values based on datatype
 * 
 * This function initializes a 2D array of mb * nb elements with leading dimension lda
 * to zero values based on the specified datatype. The function supports both row-major
 * and column-major memory layouts.
 * 
 * @param[out] array Pointer to the array to initialize
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension
 * @param[in] datatype_str String representation of the datatype
 * @param[in] is_row_major 1 for row-major layout, 0 for column-major layout
 * 
 * @return 0 on success, -1 on error
 */
int hicma_parsec_init_array_to_zero(void *array, int mb, int nb, int lda, const char *datatype_str, int is_row_major) {
    if (array == NULL || datatype_str == NULL) {
        fprintf(stderr, "Error: NULL pointer passed to hicma_parsec_init_array_to_zero\n");
        return -1;
    }
    
    if (mb <= 0 || nb <= 0 || lda <= 0) {
        fprintf(stderr, "Error: Invalid dimensions in hicma_parsec_init_array_to_zero: mb=%d, nb=%d, lda=%d\n", mb, nb, lda);
        return -1;
    }
    
    // Floating point types
    if (strcmp(datatype_str, "double") == 0 || strcmp(datatype_str, "d") == 0) {
        double *data = (double *)array;
        if (is_row_major) {
            // Row-major: data[i * lda + j]
            for (int i = 0; i < mb; i++) {
                for (int j = 0; j < nb; j++) {
                    data[i * lda + j] = 0.0;
                }
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int j = 0; j < nb; j++) {
                for (int i = 0; i < mb; i++) {
                    data[j * lda + i] = 0.0;
                }
            }
        }
    }
    else if (strcmp(datatype_str, "float") == 0 || strcmp(datatype_str, "single") == 0 || strcmp(datatype_str, "s") == 0) {
        float *data = (float *)array;
        if (is_row_major) {
            for (int i = 0; i < mb; i++) {
                for (int j = 0; j < nb; j++) {
                    data[i * lda + j] = 0.0f;
                }
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int j = 0; j < nb; j++) {
                for (int i = 0; i < mb; i++) {
                    data[j * lda + i] = 0.0f;
                }
            }
        }
    }
    // Signed integer types
    else if (strcmp(datatype_str, "int8") == 0 || strcmp(datatype_str, "i8") == 0) {
        int8_t *data = (int8_t *)array;
        if (is_row_major) {
            for (int i = 0; i < mb; i++) {
                for (int j = 0; j < nb; j++) {
                    data[i * lda + j] = 0;
                }
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int j = 0; j < nb; j++) {
                for (int i = 0; i < mb; i++) {
                    data[j * lda + i] = 0;
                }
            }
        }
    }
    else if (strcmp(datatype_str, "int16") == 0 || strcmp(datatype_str, "i16") == 0) {
        int16_t *data = (int16_t *)array;
        if (is_row_major) {
            for (int i = 0; i < mb; i++) {
                for (int j = 0; j < nb; j++) {
                    data[i * lda + j] = 0;
                }
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int j = 0; j < nb; j++) {
                for (int i = 0; i < mb; i++) {
                    data[j * lda + i] = 0;
                }
            }
        }
    }
    else if (strcmp(datatype_str, "int32") == 0 || strcmp(datatype_str, "int") == 0 || 
             strcmp(datatype_str, "i32") == 0 || strcmp(datatype_str, "i") == 0) {
        int *data = (int *)array;
        if (is_row_major) {
            for (int i = 0; i < mb; i++) {
                for (int j = 0; j < nb; j++) {
                    data[i * lda + j] = 0;
                }
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int j = 0; j < nb; j++) {
                for (int i = 0; i < mb; i++) {
                    data[j * lda + i] = 0;
                }
            }
        }
    }
    else if (strcmp(datatype_str, "int64") == 0 || strcmp(datatype_str, "i64") == 0) {
        int64_t *data = (int64_t *)array;
        if (is_row_major) {
            for (int i = 0; i < mb; i++) {
                for (int j = 0; j < nb; j++) {
                    data[i * lda + j] = 0;
                }
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int j = 0; j < nb; j++) {
                for (int i = 0; i < mb; i++) {
                    data[j * lda + i] = 0;
                }
            }
        }
    }
    // Unsigned integer types
    else if (strcmp(datatype_str, "uint8") == 0 || strcmp(datatype_str, "u8") == 0) {
        uint8_t *data = (uint8_t *)array;
        if (is_row_major) {
            for (int i = 0; i < mb; i++) {
                for (int j = 0; j < nb; j++) {
                    data[i * lda + j] = 0;
                }
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int j = 0; j < nb; j++) {
                for (int i = 0; i < mb; i++) {
                    data[j * lda + i] = 0;
                }
            }
        }
    }
    else if (strcmp(datatype_str, "uint16") == 0 || strcmp(datatype_str, "u16") == 0) {
        uint16_t *data = (uint16_t *)array;
        if (is_row_major) {
            for (int i = 0; i < mb; i++) {
                for (int j = 0; j < nb; j++) {
                    data[i * lda + j] = 0;
                }
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int j = 0; j < nb; j++) {
                for (int i = 0; i < mb; i++) {
                    data[j * lda + i] = 0;
                }
            }
        }
    }
    else if (strcmp(datatype_str, "uint32") == 0 || strcmp(datatype_str, "uint") == 0 || 
             strcmp(datatype_str, "u32") == 0 || strcmp(datatype_str, "u") == 0) {
        unsigned int *data = (unsigned int *)array;
        if (is_row_major) {
            for (int i = 0; i < mb; i++) {
                for (int j = 0; j < nb; j++) {
                    data[i * lda + j] = 0;
                }
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int j = 0; j < nb; j++) {
                for (int i = 0; i < mb; i++) {
                    data[j * lda + i] = 0;
                }
            }
        }
    }
    else if (strcmp(datatype_str, "uint64") == 0 || strcmp(datatype_str, "u64") == 0) {
        uint64_t *data = (uint64_t *)array;
        if (is_row_major) {
            for (int i = 0; i < mb; i++) {
                for (int j = 0; j < nb; j++) {
                    data[i * lda + j] = 0;
                }
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int j = 0; j < nb; j++) {
                for (int i = 0; i < mb; i++) {
                    data[j * lda + i] = 0;
                }
            }
        }
    }
    // Half precision
    else if (strcmp(datatype_str, "half") == 0 || strcmp(datatype_str, "fp16") == 0 || strcmp(datatype_str, "h") == 0) {
        uint16_t *data = (uint16_t *)array;
        if (is_row_major) {
            for (int i = 0; i < mb; i++) {
                for (int j = 0; j < nb; j++) {
                    data[i * lda + j] = 0;
                }
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int j = 0; j < nb; j++) {
                for (int i = 0; i < mb; i++) {
                    data[j * lda + i] = 0;
                }
            }
        }
    }
    // FP8, FP4, INT4, 1bit representations
    else if (strcmp(datatype_str, "fp8") == 0 || strcmp(datatype_str, "fp4") == 0 || 
             strcmp(datatype_str, "int4") == 0 || strcmp(datatype_str, "1bit") == 0) {
        uint8_t *data = (uint8_t *)array;
        if (is_row_major) {
            for (int i = 0; i < mb; i++) {
                for (int j = 0; j < nb; j++) {
                    data[i * lda + j] = 0;
                }
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int j = 0; j < nb; j++) {
                for (int i = 0; i < mb; i++) {
                    data[j * lda + i] = 0;
                }
            }
        }
    }
    else {
        fprintf(stderr, "Error: Unknown datatype '%s' in hicma_parsec_init_array_to_zero\n", datatype_str);
        return -1;
    }
    
    return 0;
}

/**
 * @brief Print values in a 2D array based on datatype
 * 
 * This function prints the values of a 2D array of mb * nb elements with leading dimension lda
 * in a formatted matrix layout (mb rows and nb columns) based on the specified datatype.
 * The function supports both row-major and column-major memory layouts.
 * 
 * @param[in] array Pointer to the array to print
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension
 * @param[in] datatype_str String representation of the datatype
 * @param[in] array_name Optional name for the array (can be NULL)
 * @param[in] is_row_major 1 for row-major layout, 0 for column-major layout
 * 
 * @return 0 on success, -1 on error
 */
int hicma_parsec_print_array_values(const void *array, int mb, int nb, int lda, const char *datatype_str, const char *array_name, int is_row_major) {
    if (array == NULL || datatype_str == NULL) {
        fprintf(stderr, "Error: NULL pointer passed to hicma_parsec_print_array_values\n");
        return -1;
    }
    
    if (mb <= 0 || nb <= 0 || lda <= 0) {
        fprintf(stderr, "Error: Invalid dimensions in hicma_parsec_print_array_values: mb=%d, nb=%d, lda=%d\n", mb, nb, lda);
        return -1;
    }
    
    // Print header
    if (array_name != NULL) {
        printf("%s (%s):\n", array_name, datatype_str);
    } else {
        printf("Array (%s):\n", datatype_str);
    }
    
    // Floating point types
    if (strcmp(datatype_str, "double") == 0 || strcmp(datatype_str, "d") == 0) {
        const double *data = (const double *)array;
        if (is_row_major) {
            // Row-major: data[i * lda + j]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%8.4f", data[i * lda + j]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%8.4f", data[j * lda + i]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        }
    }
    else if (strcmp(datatype_str, "float") == 0 || strcmp(datatype_str, "single") == 0 || strcmp(datatype_str, "s") == 0) {
        const float *data = (const float *)array;
        if (is_row_major) {
            // Row-major: data[i * lda + j]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%8.4f", data[i * lda + j]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%8.4f", data[j * lda + i]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        }
    }
    // Signed integer types
    else if (strcmp(datatype_str, "int8") == 0 || strcmp(datatype_str, "i8") == 0) {
        const int8_t *data = (const int8_t *)array;
        if (is_row_major) {
            // Row-major: data[i * lda + j]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%4d", data[i * lda + j]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%4d", data[j * lda + i]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        }
    }
    else if (strcmp(datatype_str, "int16") == 0 || strcmp(datatype_str, "i16") == 0) {
        const int16_t *data = (const int16_t *)array;
        if (is_row_major) {
            // Row-major: data[i * lda + j]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%6d", data[i * lda + j]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%6d", data[j * lda + i]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        }
    }
    else if (strcmp(datatype_str, "int32") == 0 || strcmp(datatype_str, "int") == 0 || 
             strcmp(datatype_str, "i32") == 0 || strcmp(datatype_str, "i") == 0) {
        const int *data = (const int *)array;
        if (is_row_major) {
            // Row-major: data[i * lda + j]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%8d", data[i * lda + j]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%8d", data[j * lda + i]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        }
    }
    else if (strcmp(datatype_str, "int64") == 0 || strcmp(datatype_str, "i64") == 0) {
        const int64_t *data = (const int64_t *)array;
        if (is_row_major) {
            // Row-major: data[i * lda + j]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%12ld", data[i * lda + j]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%12ld", data[j * lda + i]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        }
    }
    // Unsigned integer types
    else if (strcmp(datatype_str, "uint8") == 0 || strcmp(datatype_str, "u8") == 0) {
        const uint8_t *data = (const uint8_t *)array;
        if (is_row_major) {
            // Row-major: data[i * lda + j]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%4u", data[i * lda + j]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%4u", data[j * lda + i]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        }
    }
    else if (strcmp(datatype_str, "uint16") == 0 || strcmp(datatype_str, "u16") == 0) {
        const uint16_t *data = (const uint16_t *)array;
        if (is_row_major) {
            // Row-major: data[i * lda + j]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%6u", data[i * lda + j]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%6u", data[j * lda + i]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        }
    }
    else if (strcmp(datatype_str, "uint32") == 0 || strcmp(datatype_str, "uint") == 0 || 
             strcmp(datatype_str, "u32") == 0 || strcmp(datatype_str, "u") == 0) {
        const unsigned int *data = (const unsigned int *)array;
        if (is_row_major) {
            // Row-major: data[i * lda + j]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%8u", data[i * lda + j]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%8u", data[j * lda + i]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        }
    }
    else if (strcmp(datatype_str, "uint64") == 0 || strcmp(datatype_str, "u64") == 0) {
        const uint64_t *data = (const uint64_t *)array;
        if (is_row_major) {
            // Row-major: data[i * lda + j]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%12lu", data[i * lda + j]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%12lu", data[j * lda + i]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        }
    }
    // Half precision
    else if (strcmp(datatype_str, "half") == 0 || strcmp(datatype_str, "fp16") == 0 || strcmp(datatype_str, "h") == 0) {
        const uint16_t *data = (const uint16_t *)array;
        if (is_row_major) {
            // Row-major: data[i * lda + j]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%6u", data[i * lda + j]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%6u", data[j * lda + i]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        }
    }
    // FP8, FP4, INT4, 1bit representations
    else if (strcmp(datatype_str, "fp8") == 0 || strcmp(datatype_str, "fp4") == 0 || 
             strcmp(datatype_str, "int4") == 0 || strcmp(datatype_str, "1bit") == 0) {
        const uint8_t *data = (const uint8_t *)array;
        if (is_row_major) {
            // Row-major: data[i * lda + j]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%4u", data[i * lda + j]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        } else {
            // Column-major (default): data[j * lda + i]
            for (int i = 0; i < mb; i++) {
                printf("  [");
                for (int j = 0; j < nb; j++) {
                    printf("%4u", data[j * lda + i]);
                    if (j < nb - 1) printf(" ");
                }
                printf("]\n");
            }
        }
    }
    else {
        fprintf(stderr, "Error: Unknown datatype '%s' in hicma_parsec_print_array_values\n", datatype_str);
        return -1;
    }
    
    printf("\n");
    return 0;
}

/**
 * @brief Manually convert double/single precision to half precision using bit manipulation
 * 
 * This function performs manual conversion from double precision (DP) or single precision (SP)
 * to half precision (HP) using bit-level manipulation for optimal performance. It handles
 * the conversion based on the matrix tile's decision type and performs the conversion
 * element-wise using custom bit manipulation routines.
 * 
 * @param[in] params_tlr HICMA parameters containing decision matrix and tile information
 * @param[in,out] A Matrix data to convert (type determined by decision matrix)
 * @param[in] A_use Single precision matrix for intermediate conversion
 * @param[in] m Row index of the tile
 * @param[in] n Column index of the tile
 * @param[in] mb Number of rows in the tile
 * @param[in] nb Number of columns in the tile
 * 
 * @note This function modifies the input matrix A in-place
 * @note Uses bit manipulation for efficient half precision conversion
 * @note Handles both DENSE_DP and LOW_RANK_DP tile types
 * @note Conversion is performed element-wise using float_to_half_rtne function
 */
void hicma_parsec_convert_2h_bit(void *A, float *A_use, int mb, int nb, uint16_t decision) {
    FP32 f;
    FP16 h;

    /* If A is DP */ 
    if( DENSE_DP == decision || LOW_RANK_DP == decision ) {

        double *A_d = (double *)A;
        for( int j = 0; j < nb; j++ ) {
            for( int i = 0; i < mb; i++ ) {
                f.f = (float)(A_d[j*mb+i]);
                h = float_to_half_rtne( f ); 
                f = half_to_float( h );
                A_use[j*mb+i] = f.f;
            } 
        } 

    /* If A is SP */
    } else {

        float *A_s = (float *)A;
        for( int j = 0; j < nb; j++ ) {
            for( int i = 0; i < mb; i++ ) {
                f.f = (float)(A_s[j*mb+i]);
                h = float_to_half_rtne( f );
                f = half_to_float( h );
                A_use[j*mb+i] = f.f;
            }
        }
    }
} 

void float2half_CPU( int nrows, int ncols,
                const float *_source, int ld_s,
                void *_target, int ld_t ) {

     uint16_t *target = (uint16_t *)_target;
     float *source = (float *)_source;
     for( int j = 0; j < ncols; j++ )
         for( int i = 0; i < nrows; i++ )
             target[j*ld_t+i] = float32_to_float16( source[j*ld_s+i] ); 
}

/**
 * @brief Checks whether to convert datatype in TRSM operation
 * 
 * This function determines whether datatype conversion should be performed
 * during TRSM (Triangular Solve with Multiple right-hand sides) operations
 * based on the current precision settings and matrix location.
 * 
 * @param[in] params_tlr HICMA PaRSEC parameters including decision matrix
 * @param[in] m Row index of the current tile
 * @param[in] n Column index of the current tile
 * @return true if datatype conversion should be performed, false otherwise
 */
bool hicma_parsec_convert_in_trsm( hicma_parsec_params_t *params_tlr, int m, int n) {
    return ( !params_tlr->adaptive_decision_runtime
            && ( (DENSE_SP == params_tlr->decisions_send[n*params_tlr->NT+m]
                    && DENSE_DP == params_tlr->decisions[n*params_tlr->NT+m])
                || DENSE_HP == params_tlr->decisions_send[n*params_tlr->NT+m]
                || DENSE_FP8 == params_tlr->decisions_send[n*params_tlr->NT+m]
               )
           );
} 
