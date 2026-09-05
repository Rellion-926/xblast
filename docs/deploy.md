# Deploy

The deploy package exposes only dataset-oriented scripts:

- `run_single.sh . <prompt.txt|prompt.ids> [max_new] [block]`
- `run_dataset.sh . <dataset> [max_new] [block] [limit]`

The default runtime policy is fixed by the package script:

- DFlash mode
- top-k 5 acceptance
- target output top-1
- no per-round diagnostic tracing CLI

The runtime consumes `.txt` prompts by default using the packaged
`tokenizer.json`. `.ids` files are still accepted for exact token-level
regression checks.

## Package

Build the Android runtime:

```sh
export QAIRT_SDK_ROOT=/path/to/qairt
./scripts/build_android.sh
```

Create 8Gen5 deploy directories from generated artifacts:

```sh
SRC_ROOT=/path/to/8gen5/artifacts ./scripts/package_8gen5.sh
```

The expected source artifact layout is:

```text
0p6b/
  qwen3-0p6B-target.bin
  qwen3-0p6B-draft.bin
  embedding.fp16.bin
  tokenizer.json
  libQnn*.so
1p7b/
  qwen3-1p7B-target.bin
  qwen3-1p7B-draft.bin
  embedding.fp16.bin
  tokenizer.json
  libQnn*.so
4b/
  qwen3-4B-target.bin
  qwen3-4B-draft.bin
  embedding.fp16.bin
  tokenizer.json
  libQnn*.so
```

Generated deploy directories:

```text
deploy/8gen5/0p6b
deploy/8gen5/1p7b
deploy/8gen5/4b
```

## Run

On device:

```sh
cd /data/local/tmp/dflash-npu/4b
chmod +x *
./run_single.sh . ./prompts/gsm8k/gsm8k_0000.txt 128 16
./run_dataset.sh . gsm8k 128 16 16
```

The default output prints generated text in a delimited block and keeps the
benchmark summary lines stable. To also print generated token ids:

```sh
DFLASH_PRINT_TOKENS=1 ./run_single.sh . ./prompts/gsm8k/gsm8k_0000.txt 128 16
```

`limit=0` or `limit=all` runs all prompts under the dataset directory. The
dataset script prefers `.txt` and falls back to `.ids` only when no text prompts
exist.
