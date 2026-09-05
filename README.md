# DFlash NPU

Minimal standalone DFlash runtime for Qualcomm QNN/HTP deployment.

This repository contains the runtime code and packaging scripts needed to run
dataset prompts on device. Prebuilt device packages are published in GitHub
Releases. The runtime can consume either Qwen3 text prompts with
`tokenizer.json` or pre-tokenized `.ids` fixtures for exact regression checks.

## Layout

- `include/dflash`: runtime headers
- `src`: QNN runtime, tensor I/O, KV-cache, and graph session implementation
- `runtime/dflash_runtime.cpp`: DFlash decode runner
- `scripts`: packaging scripts for device bundles
- `deploy`: local deploy packages generated for devices
- `prompts`: small text prompt fixtures plus selected `.ids` regression inputs

## Release

The recommended way to use this project is to download the published release
package:

- [v0.1.0 release](https://github.com/Rellion-926/dflash-npu/releases/tag/v0.1.0)
- `dflash-npu-8gen5-0p6b.zip`
- `dflash-npu-8gen5-0p6b.zip.sha256`

The archive contains the Android runtime binary, QNN libraries, the 0.6B target
and draft context binaries, `embedding.fp16.bin`, `tokenizer.json`, and sample
prompts.

## Package 8Gen5 Deploy

If you need to regenerate device packages locally, place generated 8Gen5
context packages under:

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

This creates local deploy folders such as:

```text
deploy/8gen5/0p6b
deploy/8gen5/1p7b
deploy/8gen5/4b
```

## Device Test

Unzip the release package, push one model package to the device, then run:

```sh
cd /data/local/tmp/dflash-npu/0p6b
chmod +x *
./run_single.sh . ./prompts/gsm8k/gsm8k_0000.txt 128 16
```

Dataset loop:

```sh
./run_dataset.sh . gsm8k 128 16 16
```

`run_dataset.sh` writes per-sample logs under `bench_logs/`. The runtime uses
`.txt` prompts by default and falls back to `.ids` if no text prompts are found.
