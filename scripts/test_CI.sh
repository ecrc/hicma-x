#!/bin/bash

# HiCMA-PaRSEC Test Script
# Based on commands from TESTS.md
# This script runs various test scenarios for HiCMA-PaRSEC

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if we're in the build directory
check_build_dir() {
    if [ ! -d "tests" ] || [ ! -f "tests/testing_potrf_tlr" ]; then
        print_error "Not in build directory or test executables not found!"
        print_status "Please run this script from the build directory: cd build"
        exit 1
    fi
}

# Function to check if test executable exists
check_executable() {
    local exe="$1"
    if [ ! -f "$exe" ]; then
        print_error "Test executable not found: $exe"
        return 1
    fi
    return 0
}

# Function to run a test with error handling
run_test() {
    local test_name="$1"
    local test_cmd="$2"
    local expected_exit_code="${3:-0}"
    
    print_status "Running test: $test_name"
    print_status "Command: $test_cmd"
    
    if eval "$test_cmd"; then
        if [ $? -eq $expected_exit_code ]; then
            print_success "Test '$test_name' completed successfully"
        else
            print_warning "Test '$test_name' completed with unexpected exit code: $?"
        fi
    else
        print_error "Test '$test_name' failed with exit code: $?"
        return 1
    fi
    echo "----------------------------------------"
}

# Function to detect build system and return appropriate compile command
detect_build_system() {
    if [ -f "build.ninja" ]; then
        echo "ninja"
    elif [ -f "Makefile" ]; then
        echo "make"
    else
        echo "unknown"
    fi
}

# Function to get compile command based on build system
get_compile_command() {
    local build_system=$(detect_build_system)
    case $build_system in
        "ninja")
            echo "ninja"
            ;;
        "make")
            echo "make -j$(nproc)"
            ;;
        *)
            print_error "Unknown build system detected"
            return 1
            ;;
    esac
}

# Function to handle genomics test with GENOMICS compilation
run_genomics_test() {
    local hicma_header="../src/core/hicma_parsec_internal.h"
    local gene_backup=""
    
    print_status "Setting up Genomics test (requires GENOMICS compilation)"
    
    # Check if header file exists
    if [ ! -f "$hicma_header" ]; then
        print_error "HiCMA header file not found: $hicma_header"
        return 1
    fi
    
    # Backup original GENOMICS setting
    gene_backup=$(grep "#define GENOMICS" "$hicma_header" | head -1)
    print_status "Original GENOMICS setting: $gene_backup"
    
    # Enable GENOMICS compilation
    print_status "Enabling GENOMICS compilation..."
    if sed -i 's/#define GENOMICS[[:space:]]*0/#define GENOMICS                    1/' "$hicma_header"; then
        print_success "GENOMICS compilation enabled"
    else
        print_error "Failed to enable GENOMICS compilation"
        return 1
    fi
    
    # Detect build system and compile with GENOMICS enabled
    local compile_cmd=$(get_compile_command)
    if [ $? -ne 0 ]; then
        return 1
    fi
    
    print_status "Detected build system: $(detect_build_system)"
    print_status "Compiling with GENOMICS enabled using: $compile_cmd"
    
    # Show some debugging info for ninja builds
    if [ "$(detect_build_system)" = "ninja" ]; then
        print_status "Note: CMake may re-run when source files change (this is normal)"
        print_status "Using ninja target-specific build to minimize rebuild time"
    fi
    
    # For ninja, we can try to build only the specific target to avoid full regeneration
    if [ "$(detect_build_system)" = "ninja" ]; then
        print_status "Using ninja to build testing_KRR target specifically"
        if ninja; then
            print_success "Compilation with GENOMICS enabled completed (ninja target-specific build)"
        else
            print_warning "Target-specific build failed, trying full build"
            if eval "$compile_cmd"; then
                print_success "Compilation with GENOMICS enabled completed (full build)"
            else
                print_error "Compilation with GENOMICS enabled failed"
                # Restore original setting before returning
                sed -i "s/#define GENOMICS[[:space:]]*1/$gene_backup/" "$hicma_header"
                return 1
            fi
        fi
    else
        if eval "$compile_cmd"; then
            print_success "Compilation with GENOMICS enabled completed"
        else
            print_error "Compilation with GENOMICS enabled failed"
            # Restore original setting before returning
            sed -i "s/#define GENOMICS[[:space:]]*1/$gene_backup/" "$hicma_header"
            return 1
        fi
    fi
    
    # Run genomics test
    if check_executable "tests/testing_KRR"; then
        run_test "Genomics" \
            "./tests/testing_KRR --N 6000 --nsnp 6000 --NB 200 --band_dense_dp 1000 --band_dense_sp 1000 --band_dense 10000 --gpus 1 --kind_of_problem 15 --radius=1.0e-7 --adddiag=1000 --adaptive_decision 1 --kind_of_cholesky 9"
    else
        print_warning "testing_KRR executable not found after GENOMICS compilation"
    fi
    
    # Disable GENOMICS compilation
    print_status "Disabling GENOMICS compilation..."
    if sed -i 's/#define GENOMICS[[:space:]]*1/#define GENOMICS                    0/' "$hicma_header"; then
        print_success "GENOMICS compilation disabled"
    else
        print_error "Failed to disable GENOMICS compilation"
        return 1
    fi
    
    # Recompile with GENOMICS disabled
    print_status "Recompiling with GENOMICS disabled using: $compile_cmd"
    
    # For ninja, try target-specific build first
    if [ "$(detect_build_system)" = "ninja" ]; then
        print_status "Using ninja to rebuild after GENOMICS disabled"
        if ninja; then
            print_success "Recompilation with GENOMICS disabled completed (ninja target-specific build)"
        else
            print_warning "Target-specific rebuild failed, trying full build"
            if eval "$compile_cmd"; then
                print_success "Recompilation with GENOMICS disabled completed (full build)"
            else
                print_error "Recompilation with GENOMICS disabled failed"
                return 1
            fi
        fi
    else
        if eval "$compile_cmd"; then
            print_success "Recompilation with GENOMICS disabled completed"
        else
            print_error "Recompilation with GENOMICS disabled failed"
            return 1
        fi
    fi
    
    print_success "Genomics test completed and GENOMICS setting restored"
}

# Main execution
main() {
    print_status "Starting HiCMA-PaRSEC Test Suite"
    print_status "=================================="
    
    # Check if we're in the right directory
    check_build_dir
    
    # Show detected build system
    local build_system=$(detect_build_system)
    print_status "Detected build system: $build_system"
    
    # Test 1: Statistics-2d-sqexp
    if check_executable "tests/testing_potrf_tlr"; then
        run_test "Statistics-2d-sqexp" \
            "./tests/testing_potrf_tlr --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2"

        run_test "Statistics-2d-sqexp" \
            "./tests/testing_potrf_tlr --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2 --verbose 3 --band_dense 1 --adaptive_decision 1 --adaptive_memory 0 --kind_of_cholesky 1"

        run_test "Statistics-2d-sqexp" \
            "./tests/testing_potrf_tlr --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2 --verbose 3 --band_dense 1 --adaptive_decision 0 --adaptive_memory 0 --kind_of_cholesky 1"

        run_test "Statistics-2d-sqexp" \
            "./tests/testing_potrf_tlr --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2 --verbose 3 --band_dense 1000 --adaptive_decision 1 --adaptive_memory 0 --kind_of_cholesky 1"

        run_test "Statistics-2d-sqexp" \
            "./tests/testing_potrf_tlr --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2 --verbose 3 --band_dense 1000 --adaptive_decision 1 --adaptive_memory 1 --kind_of_cholesky 1"

        #run_test "Statistics-2d-sqexp" \
        #    "./tests/testing_potrf_tlr --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2 --verbose 3 --band_dense 1000 --adaptive_decision 1 --adaptive_memory 1 --kind_of_cholesky 5 --gpus 1"
    fi

    # Mixed-precision TRMM, checked against DPLASMA's FP64 TRMM: FP64 and
    # adaptive precisions, with partial tiles (N and K not multiples of NB)
    if check_executable "tests/testing_trmm_mp"; then
        run_test "TRMM-FP64" \
            "./tests/testing_trmm_mp --N 2000 --K 1500 --NB 256 --check 1 --adaptive_decision 0"
        run_test "TRMM-adaptive" \
            "./tests/testing_trmm_mp --N 2000 --K 1500 --NB 256 --check 1 --adaptive_decision 1 --fixedacc 1e-4"
        run_test "TRMM-adaptive-stochastic-rounding" \
            "./tests/testing_trmm_mp --N 2000 --K 1500 --NB 256 --check 1 --adaptive_decision 1 --fixedacc 1e-4 --enable_stochastic_rounding 1"
    fi

    # Test 2: Statistics-3d-exp
    if check_executable "tests/testing_potrf_tlr"; then
        run_test "Statistics-3d-exp" \
            "./tests/testing_potrf_tlr --N 5400 --NB 540 --fixedacc 1e-8 --maxrank 540 --kind_of_problem 4 --auto_band 1 -- -mca runtime_comm_coll_bcast 0"
    fi
    
    # Test 3: 3D-RBF-Mesh-Coordinates-Virus
    if check_executable "tests/testing_potrf_tlr"; then
        # Check if mesh file exists
        mesh_file="../stars-h/SARS-CoV-2-meshes/singleviursdata/SortVirus10370.txt"
        if [ -f "$mesh_file" ]; then
            run_test "3D-RBF-Mesh-Coordinates-Virus" \
                "./tests/testing_potrf_tlr --N 10370 --NB 1037 --fixedacc 1e-4 --maxrank 50 --adddiag 0.1 --kind_of_problem 6 --band_dense 1 --send_full_tile 0 --lookahead 1 --auto_band 0 --mesh_file $mesh_file --numobj 1 --rbf_kernel 0 --radius 4.63e-04 --order 2 --density -1 --grid_rows 2 --sparse 1 --verbose 2"
        else
            print_warning "Mesh file not found: $mesh_file"
            print_status "Skipping 3D-RBF-Mesh-Coordinates-Virus test"
        fi
    fi
    
    # Test 4: Hamming Distance
    #if check_executable "tests/testing_hamming"; then
    #    run_test "Hamming Distance" \
    #        "./tests/testing_hamming --N 10240 --nsnp=10240 --NB 1024 --gpus 0"
    #else
    #    print_warning "testing_hamming executable not found, skipping Hamming Distance test"
    #fi
    
    # Test 5: Climate-emulator
    if check_executable "tests/testing_climate_emulator"; then
        run_test "Climate-emulator" \
            "./tests/testing_climate_emulator --latitude 120 --NB 1440 --N 14400 --verbose 1 --gpus 0 --verbose 2 --adaptive_decision 1 --mesh_file /home/qcao3/data"
    else
        print_warning "testing_climate_emulator executable not found, skipping Climate-emulator test"
    fi
    
    # Test 6: Genomics (requires GENOMICS compilation)
    #run_genomics_test
    
    print_success "All available tests completed!"
    print_status "=================================="
}

# Function to show help
show_help() {
    echo "HiCMA-PaRSEC Test Script"
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -v, --verbose  Enable verbose output"
    echo "  -q, --quiet    Suppress non-error output"
    echo ""
    echo "This script runs various test scenarios for HiCMA-PaRSEC based on TESTS.md"
    echo "Make sure to run this script from the build directory."
    echo ""
    echo "Available tests:"
    echo "  1. Statistics-2d-sqexp"
    echo "  2. Statistics-3d-exp"
    echo "  3. 3D-RBF-Mesh-Coordinates-Virus (requires mesh file)"
    #echo "  4. Hamming Distance"
    echo "  5. Climate-emulator"
    #echo "  6. Genomics (automatically enables GENOMICS compilation, runs test, then disables GENOMICS)"
}

# Parse command line arguments
VERBOSE=false
QUIET=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Adjust output based on quiet mode
if [ "$QUIET" = true ]; then
    exec 1>/dev/null
fi

# Run main function
main "$@"
