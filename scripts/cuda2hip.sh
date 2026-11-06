#!/usr/bin/env bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

input_file="$1"

if [ ! -f "$input_file" ]; then
    echo "Error: file '$input_file' not found."
    exit 1
fi

# Rewrite the CUDA thread/block built-ins to their HIP names in place
# (sed -i keeps the file's permissions)
sed -i \
    -e 's/threadIdx\.\([xyz]\)/hipThreadIdx_\1/g' \
    -e 's/blockIdx\.\([xyz]\)/hipBlockIdx_\1/g' \
    -e 's/blockDim\.\([xyz]\)/hipBlockDim_\1/g' \
    -e 's/gridDim\.\([xyz]\)/hipGridDim_\1/g' \
    "$input_file"

echo "Updated: $input_file"
