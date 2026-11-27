#!/bin/bash
# install_leonardo.sh without GSL (STARS-H's spatial-statistics kernels are left out).
#   scripts/machines/install_leonardo_no_gsl.sh [build-dir] [extra CMake options]
DIR=$(cd "$(dirname "$0")" && pwd)
BUILD=$DIR/../../build
case "${1:-}" in ""|-*) ;; *) BUILD=$1; shift ;; esac
exec "$DIR/install_leonardo.sh" "$BUILD" -DGSL=OFF "$@"
