# @copyright (c) 2023-2025     Saint Louis University (SLU)
# @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
# @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
# @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
#                              All rights reserved.

cmake_minimum_required (VERSION 3.16)

# Fix for CMake regeneration issues with Ninja
# This prevents system includes from being treated as dependencies
if (CMAKE_GENERATOR MATCHES "Ninja")
    file(
        WRITE "${CMAKE_BINARY_DIR}/GNUMakeRulesOverwrite.cmake" 
            "STRING(REPLACE \"-MD\" \"-MMD\" CMAKE_DEPFILE_FLAGS_C \"\${CMAKE_DEPFILE_FLAGS_C}\")\n"
            "STRING(REPLACE \"-MD\" \"-MMD\" CMAKE_DEPFILE_FLAGS_CXX \"\${CMAKE_DEPFILE_FLAGS_CXX}\")\n"
    )
    set(CMAKE_USER_MAKE_RULES_OVERRIDE "${CMAKE_BINARY_DIR}/GNUMakeRulesOverwrite.cmake" CACHE INTERNAL "")
endif()

project (HICMA_PARSEC LANGUAGES C CXX Fortran)

# Set CMake policies early
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0094 NEW)
set(CMAKE_NO_SYSTEM_FROM_IMPORTED True)

# Include required CMake modules
include(CMakeDependentOption)
include(CMakePushCheckState)
include(CheckSymbolExists)
include(CheckCSourceCompiles)
include(CheckIncludeFiles)
include(GNUInstallDirs)

# Version management - read from VERSION.txt if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION.txt")
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION.txt" VERSION_CONTENT)
    string(STRIP "${VERSION_CONTENT}" VERSION_CONTENT)
    if(VERSION_CONTENT MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)")
        set(HICMA_PARSEC_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(HICMA_PARSEC_VERSION_MINOR ${CMAKE_MATCH_2})
        set(HICMA_PARSEC_VERSION_PATCH ${CMAKE_MATCH_3})
        set(HICMA_PARSEC_VERSION "${HICMA_PARSEC_VERSION_MAJOR}.${HICMA_PARSEC_VERSION_MINOR}.${HICMA_PARSEC_VERSION_PATCH}")
    else()
        message(WARNING "Invalid version format in VERSION.txt: ${VERSION_CONTENT}")
        set(HICMA_PARSEC_VERSION_MAJOR 2)
        set(HICMA_PARSEC_VERSION_MINOR 0)
        set(HICMA_PARSEC_VERSION_PATCH 0)
        set(HICMA_PARSEC_VERSION "${HICMA_PARSEC_VERSION_MAJOR}.${HICMA_PARSEC_VERSION_MINOR}.${HICMA_PARSEC_VERSION_PATCH}")
    endif()
else()
    # Fallback version
    set(HICMA_PARSEC_VERSION_MAJOR 2)
    set(HICMA_PARSEC_VERSION_MINOR 0)
    set(HICMA_PARSEC_VERSION_PATCH 0)
    set(HICMA_PARSEC_VERSION "${HICMA_PARSEC_VERSION_MAJOR}.${HICMA_PARSEC_VERSION_MINOR}.${HICMA_PARSEC_VERSION_PATCH}")
endif()

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build, options are None, Debug, Release, RelWithDebInfo and MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# Project configuration options
option(BUILD_SHARED_LIBS "Enable building shared DPLASMA lib (default ON)" ON)
option(SUPPORT_C11 "Enable support for C11 capabilities. Might not work with OpenMP support due to an OpenMP compilers restriction (default ON)." ON)
option(HICMA_PARSEC_HAVE_CUDA "Support CUDA (default OFF)" OFF)
option(HICMA_PARSEC_HAVE_HIP "Support HIP (default OFF)" OFF)
option(ON_FUGAKU "Running on Fugaku Supercomputer (default OFF)" OFF)
option(ENABLE_SANITIZER "Enable AddressSanitizer (default OFF)" OFF)
option(ENABLE_CUTLASS "Enable CUTLASS (default OFF)" OFF)
option(GIT_SUBMODULE "Check submodules during build (default ON)" ON)

# C standard configuration
if(SUPPORT_C11)
  set(CMAKE_C_STANDARD 11)
else(SUPPORT_C11)
  set(CMAKE_C_STANDARD 99)
endif(SUPPORT_C11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# GPU support configuration
if(HICMA_PARSEC_HAVE_CUDA AND HICMA_PARSEC_HAVE_HIP)
    message(FATAL_ERROR "HICMA_PARSEC_HAVE_CUDA and HICMA_PARSEC_HAVE_HIP cannot be supported at the same time!")
endif(HICMA_PARSEC_HAVE_CUDA AND HICMA_PARSEC_HAVE_HIP)

if(HICMA_PARSEC_HAVE_CUDA)
    add_definitions(-DHICMA_PARSEC_HAVE_CUDA)
    set(DPLASMA_HAVE_CUDA ON)
    set(DPLASMA_GPU_WITH_CUDA ON)
    set(PARSEC_HAVE_CUDA ON)
    set(PARSEC_GPU_WITH_CUDA ON)
    set(DPLASMA_HAVE_HIP OFF)
    set(DPLASMA_GPU_WITH_HIP OFF)
    set(PARSEC_HAVE_HIP OFF)
    set(PARSEC_GPU_WITH_HIP OFF)
    enable_language(CUDA)
    # Set CUDA architectures - can be overridden by user
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 80 90 90a)
    endif()
    message(STATUS "CUDA support enabled")

    if(ENABLE_CUTLASS)
        if(NOT DEFINED CUTLASS_ROOT)
            message(FATAL_ERROR "CUTLASS_ROOT must be set when ENABLE_CUTLASS is ON")
        endif()
        if(NOT EXISTS "${CUTLASS_ROOT}/include")
            message(FATAL_ERROR "CUTLASS_ROOT does not contain include directory: ${CUTLASS_ROOT}")
        endif()
        include_directories(${CUTLASS_ROOT}/include)
        message(STATUS "CUTLASS support enabled")
    endif(ENABLE_CUTLASS)
endif(HICMA_PARSEC_HAVE_CUDA)

if(HICMA_PARSEC_HAVE_HIP)
    add_definitions(-DHICMA_PARSEC_HAVE_HIP)
    set(DPLASMA_HAVE_CUDA OFF)
    set(DPLASMA_GPU_WITH_CUDA OFF)
    set(PARSEC_HAVE_CUDA OFF)
    set(PARSEC_GPU_WITH_CUDA OFF)
    set(DPLASMA_HAVE_HIP ON)
    set(DPLASMA_GPU_WITH_HIP ON)
    set(PARSEC_HAVE_HIP ON)
    set(PARSEC_GPU_WITH_HIP ON)
    enable_language(HIP)
    message(STATUS "HIP support enabled")
endif(HICMA_PARSEC_HAVE_HIP)

# Fugaku-specific configuration
if(ON_FUGAKU)
    add_definitions(-DON_FUGAKU)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    message(STATUS "Fugaku configuration enabled")
endif(ON_FUGAKU)

# Check for in-place build
STRING(COMPARE EQUAL ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR} HICMA_PARSEC_BUILD_INPLACE)
if(HICMA_PARSEC_BUILD_INPLACE)
    message(WARNING "Building in source directory is not recommended. Consider using a separate build directory.")
endif()

# Add CMake module paths
# ECRC modules should come first to avoid conflicts with DPLASMA modules
LIST(APPEND CMAKE_MODULE_PATH 
     "${CMAKE_CURRENT_SOURCE_DIR}/stars-h/cmake_modules/ecrc/modules"
     "${CMAKE_CURRENT_SOURCE_DIR}/hcore/cmake_modules/ecrc/modules"
     "${CMAKE_CURRENT_SOURCE_DIR}/stars-h/cmake_modules/"
     "${CMAKE_CURRENT_SOURCE_DIR}/hcore/cmake_modules/"
     "${CMAKE_CURRENT_SOURCE_DIR}/dplasma/cmake_modules/"
     "${CMAKE_CURRENT_SOURCE_DIR}/dplasma/parsec/cmake_modules/"
     "${CMAKE_CURRENT_SOURCE_DIR}/dplasma/tools/PrecisionGenerator/"
     "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/"
     "${CMAKE_CURRENT_SOURCE_DIR}")

# Pull submodules if not exist
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/dplasma/CMakeLists.txt" OR NOT EXISTS "${PROJECT_SOURCE_DIR}/stars-h/CMakeLists.txt" OR NOT EXISTS "${PROJECT_SOURCE_DIR}/hcore/CMakeLists.txt" OR NOT EXISTS "${PROJECT_SOURCE_DIR}/stars-h/cmake_modules/ecrc/modules" OR NOT EXISTS "${PROJECT_SOURCE_DIR}/hcore/cmake_modules/ecrc/modules")
    find_package(Git QUIET)
    if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
        if(GIT_SUBMODULE)
            message(STATUS "Updating submodules (including nested submodules)...")
            execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --force --init --recursive --remote
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
            if(NOT GIT_SUBMOD_RESULT EQUAL "0")
                message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
            endif()
            message(STATUS "Submodules updated successfully")
        endif()
    else()
        message(FATAL_ERROR "Required submodules not found and git is not available. Please clone with --recursive or run 'git submodule update --init --recursive'")
    endif()
endif()

# Find MPI C++ if not already found
if(NOT MPI_CXX_FOUND)
    find_package(MPI COMPONENTS CXX)
endif()

# Add subdirectories
add_subdirectory(dplasma)
add_subdirectory(stars-h)
add_subdirectory(hcore)

# Include package configuration generation
include(GenPkgConfig)

# Configure HCORE from submodule
# HCORE is now built as part of the project via submodule
# The hcore library target is available after add_subdirectory(hcore)

# Add HCORE library to our dependency list
list(APPEND HCORE_STARSH_LIB hcore)
message(STATUS "HCORE configured from submodule")

# Configure STARSH from submodule
# STARSH is now built as part of the project via submodule
# The starsh library target is available after add_subdirectory(stars-h)

# STARSH exposes its GSL option through the CMake cache. Use that option
# directly instead of inspecting a previously-built archive: on a clean build
# (or when GSL is newly enabled) the archive does not exist yet or is stale.
if(GSL)
    message(STATUS "STARSH depends on GSL. Adding it to dependency list")
    find_package(GSL REQUIRED)
    add_definitions(-DGSL)
    include_directories(${GSL_INCLUDE_DIRS})
    message(STATUS "GSL support enabled")
    list(APPEND HCORE_STARSH_LIB ${GSL_LIBRARIES})
endif()

# Add STARSH library to our dependency list
list(APPEND HCORE_STARSH_LIB starsh)
message(STATUS "STARSH configured from submodule")

# Sanitizer configuration.
if(ENABLE_SANITIZER)
    # Keep compile-time ASan instrumentation enabled.
    add_definitions(-DENABLE_SANITIZER)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer -Wall)

    # Portable path: prefer compiler-driven sanitizer linking.
    set(HICMA_ASAN_LINKS_WITH_COMPILER_FLAGS FALSE)
    cmake_push_check_state()
    list(APPEND CMAKE_REQUIRED_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
    list(APPEND CMAKE_REQUIRED_LINK_OPTIONS "-fsanitize=address")
    check_c_source_compiles("int main(void) { return 0; }" HICMA_ASAN_LINKS_WITH_COMPILER_FLAGS)
    cmake_pop_check_state()

    if(HICMA_ASAN_LINKS_WITH_COMPILER_FLAGS)
        add_link_options(-fsanitize=address)
        message(STATUS "AddressSanitizer runtime: compiler-managed (-fsanitize=address)")
    else()
        # Fallback path for environments with broken/missing compiler runtime resolution.
        set(HICMA_ASAN_RUNTIME_LIBRARY "" CACHE FILEPATH
            "Optional absolute path to libasan runtime (e.g. /path/to/libasan.so.8)")
        if(HICMA_ASAN_RUNTIME_LIBRARY)
            if(EXISTS "${HICMA_ASAN_RUNTIME_LIBRARY}")
                add_link_options("-Wl,--no-as-needed" "${HICMA_ASAN_RUNTIME_LIBRARY}")
                message(STATUS "AddressSanitizer runtime: ${HICMA_ASAN_RUNTIME_LIBRARY}")
            else()
                message(FATAL_ERROR
                    "HICMA_ASAN_RUNTIME_LIBRARY is set but does not exist: "
                    "${HICMA_ASAN_RUNTIME_LIBRARY}")
            endif()
        else()
            message(FATAL_ERROR
                "ENABLE_SANITIZER is ON, but linker probe with -fsanitize=address failed.\n"
                "Provide -DHICMA_ASAN_RUNTIME_LIBRARY=/absolute/path/to/libasan.so "
                "or configure with -DENABLE_SANITIZER=OFF.")
        endif()
    endif()

    message(STATUS "AddressSanitizer enabled")
endif()

# Add include and link directories
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/dplasma
    ${PROJECT_SOURCE_DIR}/dplasma/src
    ${PROJECT_SOURCE_DIR}/dplasma/src/include
    ${PROJECT_SOURCE_DIR}/dplasma/parsec
    ${PROJECT_SOURCE_DIR}/dplasma/parsec/parsec
    ${PROJECT_SOURCE_DIR}/dplasma/parsec/parsec/include
    ${PROJECT_BINARY_DIR}/dplasma
    ${PROJECT_BINARY_DIR}
    ${PROJECT_BINARY_DIR}/dplasma
    ${PROJECT_BINARY_DIR}/dplasma/src
    ${PROJECT_BINARY_DIR}/dplasma/src/include
    ${PROJECT_BINARY_DIR}/dplasma/parsec
    ${PROJECT_BINARY_DIR}/dplasma/parsec/parsec
    ${PROJECT_BINARY_DIR}/dplasma/parsec/parsec/include
    ${PROJECT_SOURCE_DIR}/stars-h/include
    ${PROJECT_BINARY_DIR}/stars-h/include
    ${PROJECT_SOURCE_DIR}/hcore/src/include
    ${PROJECT_SOURCE_DIR}/hcore/src/misc/include
    ${PROJECT_BINARY_DIR}/hcore/src/include
    ${PROJECT_BINARY_DIR}/hcore/src/misc/include
    ${PROJECT_SOURCE_DIR}/src/core
    ${PROJECT_BINARY_DIR}/src/core
    ${PROJECT_SOURCE_DIR}/src/kernels
    ${PROJECT_BINARY_DIR}/src/kernels
    ${PROJECT_SOURCE_DIR}/src/climate_emulator
    ${PROJECT_BINARY_DIR}/src/climate_emulator
)

include(ParsecCompilePTG)

link_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_BINARY_DIR}/src
    ${PROJECT_BINARY_DIR}/stars-h/src
    ${PROJECT_BINARY_DIR}/hcore/src
)

# Set library dependencies
set(LD_LIB_BASIC
    parsec
    dplasma
    stdc++
)

# Add MPI C++ libraries if MPI is found and C++ is used
if(MPI_CXX_FOUND)
    list(APPEND LD_LIB_BASIC ${MPI_CXX_LIBRARIES})
    message(STATUS "MPI C++ libraries added: ${MPI_CXX_LIBRARIES}")
endif()

set(LD_LIB_CUDA
    cublas
    cublasLt
    cudart
    cusolver
)

set(LD_LIB_HIP
    hipblas
    rocsolver
)

# Configure CUDA dependencies
if(HICMA_PARSEC_HAVE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    # Use CUDA targets instead of library names
    set(LD_LIB_CUDA_TARGETS
        CUDA::cublas
        CUDA::cublasLt
        CUDA::cudart
        CUDA::cusolver
    )
    message(STATUS "CUDA dependencies configured")
endif(HICMA_PARSEC_HAVE_CUDA)

# Configure HIP dependencies
if(HICMA_PARSEC_HAVE_HIP)
    set(LD_LIB_BASIC ${LD_LIB_HIP} ${LD_LIB_BASIC})
    # Use CMake's find mechanism for HIP instead of hardcoded paths
    find_package(hip REQUIRED)
    message(STATUS "HIP dependencies configured")
endif(HICMA_PARSEC_HAVE_HIP)

# Configure BLAS/LAPACK dependencies
# First, try to find BLAS if not already found
if(NOT BLAS_FOUND)
    find_package(BLAS REQUIRED)
endif()

# Set up BLAS libraries - handle both cases where BLAS_LIBRARIES is set or not
if(BLAS_LIBRARIES)
    set(LD_LIB_HICMA_PARSEC
        ${BLAS_LIBRARIES}
        m
    )
    # Add CBLAS if available
    if(CBLAS_cblas_LIBRARY)
        list(APPEND LD_LIB_HICMA_PARSEC ${CBLAS_cblas_LIBRARY})
    endif()
else()
    # Fallback: if BLAS_LIBRARIES is not set, try to construct it from found libraries
    if(BLAS_openblas_LIBRARY)
        set(LD_LIB_HICMA_PARSEC
            ${BLAS_openblas_LIBRARY}
            m
        )
        message(STATUS "Using OpenBLAS library: ${BLAS_openblas_LIBRARY}")
    else()
        # Last resort: use generic openblas
        set(LD_LIB_HICMA_PARSEC
            openblas
            m
        )
        message(WARNING "BLAS_LIBRARIES not found, using generic openblas")
    endif()
endif()

# Add CUDA targets if available
if(HICMA_PARSEC_HAVE_CUDA)
    list(APPEND LD_LIB_HICMA_PARSEC ${LD_LIB_CUDA_TARGETS})
endif(HICMA_PARSEC_HAVE_CUDA)

# MKL-specific configuration
string(FIND "${CBLAS_DIR_FOUND}" "mkl" _match_mkl)
if(NOT "-1" STREQUAL "${_match_mkl}")
    set(LD_LIB_HICMA_PARSEC
        mkl_intel_lp64
        mkl_sequential
        mkl_core
        ${LD_LIB_HICMA_PARSEC}
    )
    message(STATUS "MKL configuration detected")
endif(NOT "-1" STREQUAL "${_match_mkl}")

# Print configuration summary
message(STATUS "=== HICMA PARSEC Configuration Summary ===")
message(STATUS "Version: ${HICMA_PARSEC_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Shared libraries: ${BUILD_SHARED_LIBS}")
message(STATUS "C11 support: ${SUPPORT_C11}")
message(STATUS "CUDA support: ${HICMA_PARSEC_HAVE_CUDA}")
message(STATUS "HIP support: ${HICMA_PARSEC_HAVE_HIP}")
message(STATUS "Fugaku mode: ${ON_FUGAKU}")
message(STATUS "AddressSanitizer: ${ENABLE_SANITIZER}")
message(STATUS "CUTLASS support: ${ENABLE_CUTLASS}")
message(STATUS "Dependencies:")
message(STATUS "  - HCORE: Built from submodule")
message(STATUS "  - DPLASMA: Built from submodule")
message(STATUS "  - STARS-H: Built from submodule")
message(STATUS "HCORE_STARSH_LIB: ${HCORE_STARSH_LIB}")
message(STATUS "BLAS_LIBRARIES: ${BLAS_LIBRARIES}")
message(STATUS "LD_LIB_HICMA_PARSEC: ${LD_LIB_HICMA_PARSEC}")
message(STATUS "==========================================")

# Add subdirectories
add_subdirectory(src)

# Testing configuration
include(CTest)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
    message(STATUS "Testing enabled")
endif()

# Installation configuration
install(FILES
    src/core/hicma_parsec.h
    src/core/hicma_parsec_internal.h
    src/core/hicma_parsec_flat_file.h
    src/core/hicma_parsec_datatype_convert.h
    src/core/hicma_parsec_decision.h
    src/core/hicma_parsec_core.h
    src/core/hicma_parsec_sparse_analysis.h
    src/core/hicma_parsec_rank_statistics.h
    src/core/hicma_parsec_io.h
    src/core/hicma_parsec_gpu.h
    src/core/hicma_parsec_distribution.h
    src/core/hicma_parsec_check.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS hicma_parsec
    EXPORT HICMAParsecTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Generate and install CMake config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/HICMAParsecConfigVersion.cmake"
    VERSION ${HICMA_PARSEC_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/HICMAParsecConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/HICMAParsecConfig.cmake"
    @ONLY
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/HICMAParsecConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/HICMAParsecConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HICMAParsec
)

install(EXPORT HICMAParsecTargets
    FILE HICMAParsecTargets.cmake
    NAMESPACE HICMAParsec::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HICMAParsec
)
