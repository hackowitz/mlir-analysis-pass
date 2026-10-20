ANALYSIS="${ANALYSIS:-known-bits-analysis}"
SRC_DIR="${SRC_DIR:-regex/src}"
BUILD_DIR="${BUILD_DIR:-build}"
mkdir -p "${BUILD_DIR}"

# lower .c -> .ll -> .mlir
clang-22 $(python3-config --cflags) -g -S -emit-llvm -fPIC \
    "${SRC_DIR}"/_regex.c \
    -o "${BUILD_DIR}"/_regex.ll
clang-22 $(python3-config --cflags) -g -S -emit-llvm -fPIC \
    "${SRC_DIR}"/_regex_unicode.c \
    -o "${BUILD_DIR}"/_regex_unicode.ll
clang-22 -g -S -emit-llvm src/assertions.c \
    -o "${BUILD_DIR}/_regex_assertions.ll"
llvm-link-22 "${BUILD_DIR}"/_regex*.ll -S \
    -o "${BUILD_DIR}"/combined.ll
mlir-translate-22 --import-llvm "${BUILD_DIR}"/combined.ll \
    -o "${BUILD_DIR}"/combined.mlir

# analysis pipeline
mlir-opt-22 "${BUILD_DIR}"/combined.mlir \
    --load-pass-plugin='./build/KnownBitsAnalysis.so' \
    --pass-pipeline='builtin.module('$ANALYSIS')' \
    2>"${BUILD_DIR}"/${ANALYSIS}.mlir 1>/dev/null

# .mlir -> .llvm -> .so
 mlir-translate-22 --mlir-to-llvmir "${BUILD_DIR}/${ANALYSIS}.mlir" \
    -o "${BUILD_DIR}/${ANALYSIS}.ll"
 clang-22 "${BUILD_DIR}/${ANALYSIS}.ll" -shared -fPIC \
    -o "${BUILD_DIR}/${ANALYSIS}.so"
