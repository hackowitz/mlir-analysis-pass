#! /usr/bin/env bash
# Run my analysis pass and check for known bits
mlir-translate --import-llvm $1 \
    | mlir-opt \
        --load-pass-plugin='./build/KnownBitsAnalysis.so' \
        --pass-pipeline='builtin.module(known-bits-analysis)' \
        2>&1 1>/dev/null \
    | grep -P 'llvm\.icmp .* // known bits: '
