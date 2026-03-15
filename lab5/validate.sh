#!/bin/bash

set -euo pipefail

# build the program
make

# https://www.hugin.com.au/prime/twin.php
declare -A expected=(
    [2]=0
    [3]=0
    [4]=0
    [5]=1
    [6]=1
    [10]=2
    [100]=8
    [1000]=35
    [10000]=205
    [100000]=1224
    [500000]=4565
    [1000000]=8169
    [5000000]=32463
    [10000000]=58980
    [8272]=179
    [15456]=277
    [17612]=306
    [33433]=516
    [58916]=800
    [61899]=826
    [64938]=854
    [74607]=954
    [85406]=1069
    [99741]=1223
)

for n in "${!expected[@]}"; do
    echo "testing range up to $n..."
    # run with 4 processes by default
    raw=$(make run $n Test 16 2>/dev/null)
    # parse the trailing number
    got=$(echo "$raw" | grep -oP 'Twin primes up to .*: \K[0-9]+' || true)
    if [[ -z "$got" ]]; then
        echo "could not parse output from program for n=$n"
        echo "program output:"
        echo "$raw"
        exit 2
    fi
    if [[ "$got" -eq "${expected[$n]}" ]]; then
        echo "  PASS: $got"
    else
        echo "  FAIL: got $got expected ${expected[$n]}"
        echo "  program output:"
        echo "$raw"
        exit 1
    fi
done

echo "All validation tests passed."
