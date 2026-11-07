/**
 * @file hicma_parsec_flat_file.h
 * @brief HICMA PaRSEC flat file operations header file
 * 
 * This header file contains declarations for flat file operations including
 * location file reading, observation file reading, and time file reading
 * for the HICMA library. These functions provide utilities for reading
 * spatial data, observational data, and temporal data from text files
 * in various formats (CSV, space-separated, etc.).
 * 
 * The file operations support:
 * - 2D and 3D spatial coordinate reading
 * - Observation data reading for spatial modeling
 * - Time series data reading for space-time modeling
 * - File line counting for data validation
 * 
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 * 
 * @version 1.0.0
 */

#ifndef HICMA_PARSEC_FLAT_FILE_H
#define HICMA_PARSEC_FLAT_FILE_H

/* ============================================================================
 * Feature test macros
 * ============================================================================ */
#define _GNU_SOURCE

/* ============================================================================
 * HICMA PaRSEC includes
 * ============================================================================ */
#include "hicma_parsec_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Data structures
 * ============================================================================ */

/**
 * @brief Location structure for storing spatial coordinates
 * 
 * This structure holds arrays of coordinates in 3D space for location data.
 * The coordinates are stored as separate arrays for efficient memory access
 * and vectorized operations. All arrays should have the same length 'n'.
 * 
 * @note Memory for the coordinate arrays is dynamically allocated by the
 *       reading functions and should be freed by the caller when no longer needed.
 */
typedef struct {
    double *x;                ///< Array of X-coordinate values
    double *y;                ///< Array of Y-coordinate values  
    double *z;                ///< Array of Z-coordinate values (for 3D locations)
} location;

/* ============================================================================
 * File utility functions
 * ============================================================================ */

/**
 * @brief Count the number of lines in a file
 * 
 * Counts the total number of lines in a text file. This function
 * is useful for determining the size of data files before reading them
 * and for validating file integrity. The function reads the file
 * character by character to count newline characters.
 * 
 * @param[in] filename Name of the file to count lines in
 * @return Number of lines in the file, or 0 on error
 * @note This function is designed for text files and may not work
 *       correctly with binary files or files with different line endings.
 */
int countlines(char *filename);

/* ============================================================================
 * Location file reading functions
 * ============================================================================ */

/**
 * @brief Read 2D location file
 * 
 * Reads a file containing 2D location coordinates and returns a location structure.
 * The file is expected to contain comma-separated values with X and Y coordinates
 * on each line. Memory is dynamically allocated for the coordinate arrays.
 * 
 * @param[in] locs_file Filename of the location file
 * @param[in] n Number of locations to read
 * @return Pointer to location structure with x and y arrays populated, or NULL on error
 * @note The caller is responsible for freeing the returned structure and its arrays
 * @warning The function assumes the file contains at least 'n' lines of valid data
 */
location* readLocsFile(char* locs_file, int n);

/**
 * @brief Read 3D location file
 * 
 * Reads a file containing 3D location coordinates and returns a location structure.
 * The file is expected to contain comma-separated values with X, Y, and Z coordinates
 * on each line. Memory is dynamically allocated for the coordinate arrays.
 * 
 * @param[in] locs_file Filename of the location file
 * @param[in] n Number of locations to read
 * @return Pointer to location structure with x, y, and z arrays populated, or NULL on error
 * @note The caller is responsible for freeing the returned structure and its arrays
 * @warning The function assumes the file contains at least 'n' lines of valid data
 */
location* readLocsFile3d(char* locs_file, int n);

/* ============================================================================
 * Data file reading functions
 * ============================================================================ */

#ifdef __cplusplus
}
#endif

#endif /* HICMA_PARSEC_FLAT_FILE_H */
