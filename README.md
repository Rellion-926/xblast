# DFlash NPU

Minimal standalone DFlash runtime for Qualcomm QNN/HTP deployment.

This repository contains the runtime code, Android build script, and deploy
scripts needed to run dataset prompts on device. QNN AOT context
binaries are generated outside this repository and copied into deploy packages.
The runtime can consume either Qwen3 text prompts with `tokenizer.json` or
pre-tokenized `.ids` fixtures for exact regression checks.

## Layout

- `include/dflash`: runtime headers
- `src`: QNN runtime, tensor I/O, KV-cache, and graph session implementation
- `runtime/dflash_runtime.cpp`: DFlash decode runner
- `scripts`: Android build and deploy packaging scripts
- `deploy`: local deploy packages generated for devices
- `prompts`: small text prompt fixtures plus selected `.ids` regression inputs

## Build

Set `QAIRT_SDK_ROOT` to the Qualcomm AI Runtime SDK path, then build the
Android arm64 runtime:

```sh
export QAIRT_SDK_ROOT=/path/to/qairt
./scripts/build_android.sh
```

The device binary is written to:

```sh
build-android-arm64/dflash-runtime
```

## Package 8Gen5 Deploy

Place generated 8Gen5 context packages under:

```text
artifacts/8gen5/0p6b
artifacts/8gen5/1p7b
artifacts/8gen5/4b
```

Each model directory must contain its target context, draft context,
`embedding.fp16.bin`, `tokenizer.json`, and the required `libQnn*.so` files.
Then run:

```sh
./scripts/package_8gen5.sh
```

or point to another artifact directory:

```sh
SRC_ROOT=/path/to/8gen5/artifacts ./scripts/package_8gen5.sh
```

This creates:

```text
deploy/8gen5/0p6b
deploy/8gen5/1p7b
deploy/8gen5/4b
```

## Device Test

Push one model package to the device, then run:

```sh
cd /data/local/tmp/dflash-npu/4b
chmod +x *
./run_single.sh . ./prompts/gsm8k/gsm8k_0000.txt 128 16
```

Dataset loop:

```sh
./run_dataset.sh . gsm8k 128 16 16
```

`run_dataset.sh` writes per-sample logs under `bench_logs/`. The runtime uses
`.txt` prompts by default and falls back to `.ids` if no text prompts are found.
