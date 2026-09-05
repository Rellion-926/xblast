#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
NDK_ROOT="${NDK_ROOT:-/export/home/caoxuewei.7/ndk}"
QAIRT_SDK_ROOT="${QAIRT_SDK_ROOT:-/export/home/caoxuewei.7/hexagon/qairt}"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build-android-arm64}"

cmake -S "${ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_TOOLCHAIN_FILE="${NDK_ROOT}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DQAIRT_SDK_ROOT="${QAIRT_SDK_ROOT}" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "${BUILD_DIR}" --target dflash-runtime -j"${JOBS:-4}"

STRIP="${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip"
if [[ -x "${STRIP}" ]]; then
  "${STRIP}" "${BUILD_DIR}/dflash-runtime"
fi

file "${BUILD_DIR}/dflash-runtime"
