# Build

## Host Probe Build

Set QAIRT path:

```sh
export QAIRT_SDK_ROOT=/export/home/caoxuewei.7/hexagon/qairt
```

Build:

```sh
cmake -S . -B build-host -DQAIRT_SDK_ROOT="${QAIRT_SDK_ROOT}"
cmake --build build-host -j
```

This builds `qnn_context_probe` for validating QNN context metadata loading.

## Android Runtime Build

The Android build should use the Android NDK toolchain and the same
`QAIRT_SDK_ROOT` include path. The runtime still loads QNN libraries with
`dlopen`, so the deploy directory must include the matching QNN HTP libraries.

For the local server layout:

```sh
./scripts/build_android.sh
```

Equivalent manual command:

```sh
cmake -S . -B build-android-arm64 \
  -DCMAKE_TOOLCHAIN_FILE=/export/home/caoxuewei.7/ndk/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DQAIRT_SDK_ROOT=/export/home/caoxuewei.7/hexagon/qairt \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-android-arm64 -j
```

The AOT context binaries are still produced by the existing QNN AOT compiler
flow in this pass.
