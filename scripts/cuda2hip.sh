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

tmp_file="$(mktemp)"

sed \
    -e 's/threadIdx\.x/hipThreadIdx_x/g' \
    -e 's/threadIdx\.y/hipThreadIdx_y/g' \
    -e 's/blockIdx\.x/hipBlockIdx_x/g' \
    -e 's/blockDim\.x/hipBlockDim_x/g' \
    -e 's/blockIdx\.y/hipBlockIdx_y/g' \
    -e 's/blockDim\.y/hipBlockDim_y/g' \
    "$input_file" > "$tmp_file"

mv "$tmp_file" "$input_file"

echo "Updated: $input_file"
