#!/bin/bash
# Debug build on the ICL machines: install_icl.sh with CMAKE_BUILD_TYPE=Debug
# and PaRSEC's memory-address checks, in build-debug/ by default.
#   scripts/machines/install_icl_debug.sh [build-dir] [extra CMake options]
DIR=$(cd "$(dirname "$0")" && pwd)
BUILD=$DIR/../../build-debug
case "${1:-}" in ""|-*) ;; *) BUILD=$1; shift ;; esac
exec "$DIR/install_icl.sh" "$BUILD" -DCMAKE_BUILD_TYPE=Debug -DPARSEC_DEBUG_MEM_ADDR=ON "$@"
