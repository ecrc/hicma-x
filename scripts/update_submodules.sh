#!/bin/bash

# @copyright (c) 2023-2025     Saint Louis University (SLU)
# @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
# @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
# @copyright (c) 2023-2025     The University of Tennessee and The University of Tennessee Research Foundation
#                              All rights reserved.

# HICMA PARSEC Submodule Management Script
# This script helps manage Git submodules for the project

set -e

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

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Manage HICMA PARSEC Git submodules

COMMANDS:
    init              Initialize submodules (git submodule update --init --recursive)
    update            Update submodules to latest commits (git submodule update --remote)
    status            Show submodule status
    clean             Clean and reset submodules
    help              Show this help message

OPTIONS:
    -h, --help        Show this help message

EXAMPLES:
    $0 init           # Initialize submodules
    $0 update         # Update submodules to latest
    $0 status         # Check submodule status

EOF
}

# Function to check if git is available
check_git() {
    if ! command -v git >/dev/null 2>&1; then
        print_error "Git is not installed or not in PATH"
        exit 1
    fi
}

# Function to check if we're in a git repository
check_git_repo() {
    if [[ ! -d ".git" ]]; then
        print_error "Not in a Git repository. Please run this script from the project root."
        exit 1
    fi
}

# Function to initialize submodules
init_submodules() {
    print_status "Initializing submodules (including nested submodules)..."
    
    if git submodule update --init --recursive; then
        print_success "Submodules initialized successfully"
        
        print_status "Submodule information:"
        echo "  - dplasma: $(cd dplasma && git rev-parse --short HEAD) ($(cd dplasma && git branch --show-current))"
        echo "  - stars-h: $(cd stars-h && git rev-parse --short HEAD) ($(cd stars-h && git branch --show-current))"
        echo "    - ecrc: $(cd stars-h/cmake_modules/ecrc && git rev-parse --short HEAD) ($(cd stars-h/cmake_modules/ecrc && git branch --show-current))"
        echo "  - hcore: $(cd hcore && git rev-parse --short HEAD) ($(cd hcore && git branch --show-current))"
        echo "    - ecrc: $(cd hcore/cmake_modules/ecrc && git rev-parse --short HEAD) ($(cd hcore/cmake_modules/ecrc && git branch --show-current))"
        echo "  - omnicast: $(cd omnicast && git rev-parse --short HEAD) ($(cd omnicast && git branch --show-current))"
    else
        print_error "Failed to initialize submodules"
        exit 1
    fi
}

# Function to update submodules
update_submodules() {
    print_status "Updating submodules to latest commits..."
    
    if git submodule update --remote --recursive; then
        print_success "Submodules updated successfully"
        
        print_status "Updated submodule information:"
        echo "  - dplasma: $(cd dplasma && git rev-parse --short HEAD) ($(cd dplasma && git branch --show-current))"
        echo "  - stars-h: $(cd stars-h && git rev-parse --short HEAD) ($(cd stars-h && git branch --show-current))"
        echo "    - ecrc: $(cd stars-h/cmake_modules/ecrc && git rev-parse --short HEAD) ($(cd stars-h/cmake_modules/ecrc && git branch --show-current))"
        echo "  - hcore: $(cd hcore && git rev-parse --short HEAD) ($(cd hcore && git branch --show-current))"
        echo "    - ecrc: $(cd hcore/cmake_modules/ecrc && git rev-parse --short HEAD) ($(cd hcore/cmake_modules/ecrc && git branch --show-current))"
        echo "  - omnicast: $(cd omnicast && git rev-parse --short HEAD) ($(cd omnicast && git branch --show-current))"
    else
        print_error "Failed to update submodules"
        exit 1
    fi
}

# Function to show submodule status
show_status() {
    print_status "Submodule status:"
    echo ""
    git submodule status --recursive
    echo ""
    
    print_status "Submodule details:"
    for submodule in dplasma stars-h hcore; do
        if [[ -d "$submodule" ]]; then
            echo "  $submodule:"
            echo "    - Branch: $(cd $submodule && git branch --show-current 2>/dev/null || echo "detached HEAD")"
            echo "    - Commit: $(cd $submodule && git rev-parse --short HEAD 2>/dev/null || echo "unknown")"
            echo "    - Status: $(cd $submodule && git status --porcelain 2>/dev/null | wc -l) changes"
            
            # Show nested submodules for stars-h
            if [[ "$submodule" == "stars-h" ]] && [[ -d "stars-h/cmake_modules/ecrc" ]]; then
                echo "    - ecrc:"
                echo "      - Branch: $(cd stars-h/cmake_modules/ecrc && git branch --show-current 2>/dev/null || echo "detached HEAD")"
                echo "      - Commit: $(cd stars-h/cmake_modules/ecrc && git rev-parse --short HEAD 2>/dev/null || echo "unknown")"
                echo "      - Status: $(cd stars-h/cmake_modules/ecrc && git status --porcelain 2>/dev/null | wc -l) changes"
            fi
            
            # Show nested submodules for hcore
            if [[ "$submodule" == "hcore" ]] && [[ -d "hcore/cmake_modules/ecrc" ]]; then
                echo "    - ecrc:"
                echo "      - Branch: $(cd hcore/cmake_modules/ecrc && git branch --show-current 2>/dev/null || echo "detached HEAD")"
                echo "      - Commit: $(cd hcore/cmake_modules/ecrc && git rev-parse --short HEAD 2>/dev/null || echo "unknown")"
                echo "      - Status: $(cd hcore/cmake_modules/ecrc && git status --porcelain 2>/dev/null | wc -l) changes"
            fi
        else
            echo "  $submodule: Not initialized"
        fi
    done
}

# Function to clean submodules
clean_submodules() {
    print_warning "This will clean and reset all submodules. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_status "Cleaning submodules..."
        
        for submodule in dplasma stars-h hcore; do
            if [[ -d "$submodule" ]]; then
                print_status "Cleaning $submodule..."
                cd "$submodule"
                git clean -fdx
                git reset --hard HEAD
                cd ..
            fi
        done
        
        print_success "Submodules cleaned successfully"
    else
        print_status "Clean operation cancelled"
    fi
}

# Main function
main() {
    # Get script directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
    
    # Change to project directory
    cd "$PROJECT_DIR"
    
    # Check prerequisites
    check_git
    check_git_repo
    
    # Parse command line arguments
    case "${1:-help}" in
        init)
            init_submodules
            ;;
        update)
            update_submodules
            ;;
        status)
            show_status
            ;;
        clean)
            clean_submodules
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 