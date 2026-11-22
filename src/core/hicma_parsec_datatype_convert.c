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
#include <strings.h>
#include <stdint.h>
#include <stddef.h>

/* ============================================================================
   Element conversions are done by OmniCast (omnicast/omnicast.h). This file
   keeps the HiCMA-X-level helpers: tile precision -> OmniCast type, the
   string-typed tile conversions, the precision-decision conversions and the
   FP8 operand of the cuBLASLt matmul.
   ============================================================================ */

omc_dtype_t hicma_parsec_omc_dtype(uint16_t precision)
{
    switch( precision ) {
    case DENSE_DP:
    case LOW_RANK_DP: return OMC_FP64;
    case DENSE_SP:
    case LOW_RANK_SP: return OMC_FP32;
    case DENSE_HP:    return OMC_FP16;
    case DENSE_FP8:   return OMC_FP8_E4M3;
    }
    fprintf(stderr, "hicma_parsec_omc_dtype: no element type for precision %u\n", (unsigned)precision);
    exit(1);
}

/* ============================================================================
   Unified Conversion Functions
   ============================================================================ */

/* Conversion type strings. Float conversions treat the tile as contiguous
 * (mb * nb elements, as before), integer ones use lda. */
typedef struct {
    const char *name;
    omc_dtype_t src, dst;
    int use_lda;
} hicma_convert_type_t;

static const hicma_convert_type_t hicma_convert_types[] = {
    {"d2s",   OMC_FP64,  OMC_FP32,  0}, {"s2d",   OMC_FP32,  OMC_FP64,  0},
    {"d2h",   OMC_FP64,  OMC_FP16,  0}, {"h2d",   OMC_FP16,  OMC_FP64,  0},
    {"s2h",   OMC_FP32,  OMC_FP16,  0}, {"h2s",   OMC_FP16,  OMC_FP32,  0},
    {"d2i8",  OMC_FP64,  OMC_INT8,  1}, {"d2i16", OMC_FP64,  OMC_INT16, 1}, {"d2i", OMC_FP64, OMC_INT32, 1},
    {"s2i8",  OMC_FP32,  OMC_INT8,  1}, {"s2i16", OMC_FP32,  OMC_INT16, 1}, {"s2i", OMC_FP32, OMC_INT32, 1},
    {"i82s",  OMC_INT8,  OMC_FP32,  1}, {"i82d",  OMC_INT8,  OMC_FP64,  1},
    {"i162s", OMC_INT16, OMC_FP32,  1}, {"i162d", OMC_INT16, OMC_FP64,  1},
    {"i2s",   OMC_INT32, OMC_FP32,  1}, {"i2d",   OMC_INT32, OMC_FP64,  1},
};

static const hicma_convert_type_t *hicma_find_convert_type(const char *type)
{
    for( size_t k = 0; k < sizeof(hicma_convert_types) / sizeof(hicma_convert_types[0]); k++ )
        if( 0 == strcasecmp(type, hicma_convert_types[k].name) ) return &hicma_convert_types[k];
    return NULL;
}

int convert_datatype_unary_CPU(void *A, int mb, int nb, int lda, char *type, size_t *size)
{
    if( !A || !type || !size ) return -1;
    const hicma_convert_type_t *c = hicma_find_convert_type(type);
    if( !c ) return -1;
    if( c->use_lda ) {
        HICMA_OMC( omc_convert_inplace_2d(c->src, c->dst, A, lda, mb, nb, NULL) );
    } else {
        HICMA_OMC( omc_convert_inplace(c->src, c->dst, A, (size_t)mb * nb, NULL) );
    }
    /* as before, only "d2s" reports the new size */
    if( OMC_FP64 == c->src && OMC_FP32 == c->dst ) *size = (size_t)mb * nb * sizeof(float);
    return 0;
}

int convert_datatype_binary_CPU(void *target, void *source, int mb, int nb, int lda, char *type)
{
    if( !target || !source || !type ) return -1;
    const hicma_convert_type_t *c = hicma_find_convert_type(type);
    if( !c ) return -1;
    if( c->use_lda ) {
        HICMA_OMC( omc_convert_2d(c->src, source, lda, c->dst, target, lda, mb, nb, NULL) );
    } else {
        HICMA_OMC( omc_convert(c->src, source, c->dst, target, (size_t)mb * nb, NULL) );
    }
    return 0;
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

void hicma_parsec_convert_2h_bit(void *A, float *A_use, int mb, int nb, uint16_t decision)
{
    const omc_dtype_t src = (DENSE_DP == decision || LOW_RANK_DP == decision) ? OMC_FP64 : OMC_FP32;
    const size_t count = (size_t)mb * nb;
    /* the FP16 values in the first half of A_use, then widened in place */
    HICMA_OMC( omc_convert(src, A, OMC_FP16, A_use, count, NULL) );
    HICMA_OMC( omc_convert_inplace(OMC_FP16, OMC_FP32, A_use, count, NULL) );
}

int convert_datatype_adaptive_unary_CPU(void *A, int mb, int nb, int lda, uint16_t decision, int convert_direction, int enable_stochastic_rounding, uint64_t sr_seed, size_t *size)
{
    (void)lda;
    if( !A || !size ) return -1;
    if( DENSE_DP == decision ) return 0;    /* already in double precision */
    if( DENSE_SP != decision && DENSE_HP != decision ) {
        /* Leaving the tile in double precision would let its readers take it
         * for the decided precision */
        fprintf(stderr, "convert_datatype_adaptive_unary_CPU: no CPU conversion for precision %u\n",
                (unsigned)decision);
        exit(1);
    }
    const omc_dtype_t low = hicma_parsec_omc_dtype(decision);
    if( 0 == convert_direction ) {
        /* stochastic rounding for FP32, as before (contiguous tile, positions (i, j)) */
        const int sr = enable_stochastic_rounding && DENSE_SP == decision;
        HICMA_OMC( omc_convert_inplace_2d(OMC_FP64, low, A, mb, mb, nb, HICMA_OMC_SR(sr, sr_seed)) );
    } else {
        HICMA_OMC( omc_convert_inplace(low, OMC_FP64, A, (size_t)mb * nb, NULL) );
    }
    return 0;
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


#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
int convert_datatype_adaptive_unary_GPU(void *A, int mb, int nb, int lda,
        uint16_t decision, int convert_direction, int enable_stochastic_rounding, uint64_t sr_seed,
        size_t *size, cudaStream_t stream)
{
    (void)lda;
    if( !A || !size ) return -1;
    if( DENSE_SP != decision && DENSE_HP != decision ) return 0;
    const omc_dtype_t low = hicma_parsec_omc_dtype(decision);
    if( 0 == convert_direction ) {
        HICMA_OMC( omc_gpu_convert_inplace_2d(OMC_FP64, low, A, mb, mb, nb,
                                              HICMA_OMC_SR(enable_stochastic_rounding, sr_seed), stream) );
    } else {
        HICMA_OMC( omc_gpu_convert_inplace(low, OMC_FP64, A, (size_t)mb * nb, NULL, stream) );
    }
    return 0;
}

/* Per-tile scaled FP8 (E4M3). A plain cast to E4M3 only represents magnitudes
 * in [2^-9, 448]. The tiles the runtime decision sends to FP8 have small
 * norms, so most of their entries would flush to zero and the GEMM would drop
 * the update. Scale each tile by s, the largest power of two with
 * s * amax <= 448, convert s * S, and let cuBLASLt multiply the product by
 * 1/s_A and 1/s_B through its A/B scale pointers. Powers of two make both
 * s * S (before rounding to FP8) and 1/s exact. The scale is computed in the
 * source precision, so FP64 entries below FLT_MIN survive. */
void hicma_parsec_fp8_operand_GPU(omc_dtype_t src_type, int nrows, int ncols,
        const void *S, int lds, void *T, int ldt, int padded_rows, int padded_cols,
        float *scale, int scaling, cudaStream_t stream)
{
    /* cuBLASLt takes only TN for FP8: T is the transpose, padded_cols rows by
     * padded_rows columns, zero outside the tile */
    omc_options_t o = {0};
    o.flags = OMC_FLAG_SATURATE | OMC_FLAG_TRANSPOSE;
    o.dst_rows = padded_cols;
    o.dst_cols = padded_rows;
    if( scaling ) {
        o.scale_mode = OMC_SCALE_DYNAMIC_POW2;   /* s * max|S| <= 448, writes {amax, s, 1/s} */
        o.scale_ptr = scale;
    } else {
        static const float unscaled[3] = {0.0f, 1.0f, 1.0f};
        if( cudaSuccess != cudaMemcpyAsync(scale, unscaled, sizeof(unscaled), cudaMemcpyHostToDevice, stream) ) {
            fprintf(stderr, "hicma_parsec_fp8_operand_GPU: cudaMemcpyAsync failed\n");
            abort();
        }
    }
    HICMA_OMC( omc_gpu_convert_2d(src_type, S, lds, OMC_FP8_E4M3, T, ldt, nrows, ncols, &o, stream) );
}

static void hicma_operand_cache_check(cudaError_t err, const char *what)
{
    if( cudaSuccess != err ) {
        fprintf(stderr, "GPU operand cache: %s failed: %s\n", what, cudaGetErrorString(err));
        abort();
    }
}

void hicma_operand_convert(const hicma_operand_conv_t *cv, void *T, int mb, cudaStream_t stream)
{
#if HAVE_FP8
    if( OMC_FP8_E4M3 == cv->dst_type ) {
        int ld = hicma_fp8_ld(mb);
        hicma_parsec_fp8_operand_GPU(cv->src_type, cv->nrows, cv->ncols, cv->S, cv->lds, T, ld, ld, ld,
                hicma_fp8_tile_scale(T, mb), cv->fp8_scaling, stream);
        return;
    }
#endif
    omc_options_t o = {0};
    if( cv->sr ) { o.rounding = OMC_ROUND_STOCHASTIC; o.seed = cv->seed; }
    if( 1.0f != cv->scale ) { o.scale_mode = OMC_SCALE_HOST; o.scale = cv->scale; }
    HICMA_OMC( omc_gpu_convert_2d(cv->src_type, cv->S, cv->lds, cv->dst_type, T, mb, cv->nrows, cv->ncols,
                                  (cv->sr || 1.0f != cv->scale) ? &o : NULL, stream) );
}

/* Take one more slot from the device memory zone; NULL (and no more tries)
 * when it has no room */
static hicma_operand_cache_slot_t *hicma_operand_cache_grow(hicma_operand_cache_t *c)
{
    hicma_operand_cache_slot_t *e = &c->slots[c->nslots];
    void *buf = zone_malloc(c->zone, c->slot_bytes);
    if( NULL == buf ) {
        c->grow = 0;
        return NULL;
    }
    e->done = (cudaEvent_t *)calloc((size_t)c->nstreams, sizeof(cudaEvent_t));
    int ok = (NULL != e->done) && cudaSuccess == cudaEventCreateWithFlags(&e->ready, cudaEventDisableTiming);
    int j = 0;
    for( ; ok && j < c->nstreams; j++ ) {
        ok = cudaSuccess == cudaEventCreateWithFlags(&e->done[j], cudaEventDisableTiming);
    }
    if( !ok ) {
        fprintf(stderr, "GPU operand cache: event creation failed\n");
        abort();
    }
    e->buf = buf;
    e->producer = -1;
    e->valid = 0;
    e->stamp = 0;
    e->used = 0;
    e->pins = 0;
    c->nslots++;
    return e;
}

void *hicma_operand_cache_get(hicma_operand_cache_t *c, int sidx, cudaStream_t stream,
        const void *desc, int epoch, size_t tile, const hicma_operand_conv_t *cv, int *converted)
{
    *converted = 0;
    if( NULL == c || sidx < 0 || sidx >= c->nstreams ) return NULL;

    pthread_mutex_lock(&c->lock);
    hicma_operand_cache_slot_t *hit = NULL, *victim = NULL;
    for( int i = 0; i < c->nslots; i++ ) {
        hicma_operand_cache_slot_t *e = &c->slots[i];
        if( e->valid && e->tile == tile && e->desc == desc && e->epoch == epoch
                && e->src_type == (int)cv->src_type && e->dst_type == (int)cv->dst_type
                && e->sr == cv->sr && e->scale == cv->scale ) {
            hit = e;
            break;
        }
        if( 0 == e->pins && (NULL == victim || e->stamp < victim->stamp) ) {
            victim = e;
        }
    }

    if( NULL != hit ) {
        if( hit->producer != sidx ) {
            hicma_operand_cache_check(cudaStreamWaitEvent(stream, hit->ready, 0), "cudaStreamWaitEvent");
        }
        hit->pins++;
        hit->stamp = ++c->clock;
        c->hits++;
        pthread_mutex_unlock(&c->lock);
        return hit->buf;
    }

    /* A stale or empty slot first, then a new slot, then the least recently used one */
    hicma_operand_cache_slot_t *e = NULL;
    if( NULL != victim && (!victim->valid || victim->epoch != epoch || victim->desc != desc) ) {
        e = victim;
    } else if( c->grow && c->nslots < c->max_slots ) {
        e = hicma_operand_cache_grow(c);
    }
    if( NULL == e ) e = victim;
    if( NULL == e ) {
        c->fallbacks++;
        pthread_mutex_unlock(&c->lock);
        return NULL;
    }

    /* Overwrite the slot only after its conversion and every GEMM that read
     * it; those of this stream are ordered before already */
    for( int j = 0; j < c->nstreams; j++ ) {
        if( j != sidx && ((e->used >> j) & 1) ) {
            hicma_operand_cache_check(cudaStreamWaitEvent(stream, e->done[j], 0), "cudaStreamWaitEvent");
        }
    }
    if( e->producer >= 0 && e->producer != sidx ) {
        hicma_operand_cache_check(cudaStreamWaitEvent(stream, e->ready, 0), "cudaStreamWaitEvent");
    }

    hicma_operand_convert(cv, e->buf, c->mb, stream);
    hicma_operand_cache_check(cudaEventRecord(e->ready, stream), "cudaEventRecord");

    e->desc = desc;
    e->tile = tile;
    e->epoch = epoch;
    e->src_type = (int)cv->src_type;
    e->dst_type = (int)cv->dst_type;
    e->sr = cv->sr;
    e->scale = cv->scale;
    e->valid = 1;
    e->producer = sidx;
    e->used = 0;
    e->pins = 1;
    e->stamp = ++c->clock;
    c->misses++;
    *converted = 1;
    pthread_mutex_unlock(&c->lock);
    return e->buf;
}

void hicma_operand_cache_put(hicma_operand_cache_t *c, void *buf, int sidx, cudaStream_t stream)
{
    pthread_mutex_lock(&c->lock);
    hicma_operand_cache_slot_t *e = NULL;
    for( int i = 0; i < c->nslots; i++ ) {
        if( c->slots[i].buf == buf ) { e = &c->slots[i]; break; }
    }
    assert(NULL != e && e->pins > 0);
    hicma_operand_cache_check(cudaEventRecord(e->done[sidx], stream), "cudaEventRecord");
    e->used |= (uint64_t)1 << sidx;
    e->pins--;
    pthread_mutex_unlock(&c->lock);
}
#endif
