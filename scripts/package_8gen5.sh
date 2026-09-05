#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SRC_ROOT="${SRC_ROOT:-${ROOT}/artifacts/8gen5}"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build-android-arm64}"
DST_ROOT="${DST_ROOT:-${ROOT}/deploy/8gen5}"
PROMPTS_ROOT="${PROMPTS_ROOT:-${ROOT}/prompts}"

if [[ ! -x "${BUILD_DIR}/dflash-runtime" ]]; then
  echo "missing Android binary: ${BUILD_DIR}/dflash-runtime" >&2
  echo "run ./scripts/build_android.sh first" >&2
  exit 2
fi
if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "missing artifact root: ${SRC_ROOT}" >&2
  echo "set SRC_ROOT to a directory containing 0p6b/, 1p7b/, and 4b/ context packages" >&2
  exit 2
fi

copy_file() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "${src}" ]]; then
    echo "missing file: ${src}" >&2
    exit 2
  fi
  cp -a "${src}" "${dst}"
}

copy_qnn_libs() {
  local src="$1"
  local dst="$2"
  local found=0
  for f in "${src}"/libQnn*.so; do
    [[ -f "${f}" ]] || continue
    cp -a "${f}" "${dst}/"
    found=1
  done
  if [[ "${found}" -eq 0 ]]; then
    echo "missing QNN libs under ${src}" >&2
    exit 2
  fi
}

copy_prompts() {
  local dst="$1"
  mkdir -p "${dst}/prompts"
  if [[ ! -d "${PROMPTS_ROOT}" ]]; then
    echo "warning: prompt root not found: ${PROMPTS_ROOT}" >&2
    return
  fi
  while IFS= read -r -d '' file; do
    local rel="${file#${PROMPTS_ROOT}/}"
    mkdir -p "${dst}/prompts/$(dirname "${rel}")"
    cp -a "${file}" "${dst}/prompts/${rel}"
  done < <(find "${PROMPTS_ROOT}" -type f \( -name '*.ids' -o -name '*.txt' \) -print0 | sort -z)
}

write_single_script() {
  local dst="$1"
  shift
  cat > "${dst}/run_single.sh" <<'SH'
#!/system/bin/sh
set -e
ROOT="${1:-.}"
PROMPT="${2:-./prompts/gsm8k/gsm8k_0000.txt}"
MAX_NEW="${3:-128}"
BLOCK="${4:-16}"
ROOT="$(cd "${ROOT}" && pwd -P)"
cd "${ROOT}"
export LD_LIBRARY_PATH="${ROOT}:${LD_LIBRARY_PATH}"
INPUT_ARGS="--prompt_file ${PROMPT} --tokenizer ./tokenizer.json"
case "${PROMPT}" in
  *.ids) INPUT_ARGS="--tokens ${PROMPT}" ;;
esac
SH
  {
    printf 'exec ./dflash-runtime'
    for arg in "$@"; do
      printf ' %s' "${arg}"
    done
    printf ' ${INPUT_ARGS} --max_new "${MAX_NEW}" --block "${BLOCK}"\n'
  } >> "${dst}/run_single.sh"
  chmod +x "${dst}/run_single.sh"
}

write_dataset_script() {
  local dst="$1"
  cat > "${dst}/run_dataset.sh" <<'SH'
#!/system/bin/sh
set -e
ROOT="${1:-.}"
DATASET="${2:-gsm8k}"
MAX_NEW="${3:-128}"
BLOCK="${4:-16}"
LIMIT="${5:-16}"

ROOT="$(cd "${ROOT}" && pwd -P)"
cd "${ROOT}"
PROMPT_DIR="./prompts/${DATASET}"
if [ ! -d "${PROMPT_DIR}" ]; then
  echo "missing prompt dir: ${PROMPT_DIR}" >&2
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT="./bench_logs/${TS}_dflash_${DATASET}_b${BLOCK}_n${LIMIT}_m${MAX_NEW}_topk5"
mkdir -p "${OUT}"
echo "[dflash-dataset] dataset=${DATASET} prompt_dir=${PROMPT_DIR} max_new=${MAX_NEW} block=${BLOCK} limit=${LIMIT} out=${OUT}"

COUNT=0
FOUND=0
PROMPTS="$(find "${PROMPT_DIR}" -type f -name '*.txt' | sort)"
if [ -z "${PROMPTS}" ]; then
  PROMPTS="$(find "${PROMPT_DIR}" -type f -name '*.ids' | sort)"
fi
for sample in ${PROMPTS}; do
  FOUND=1
  if [ "${LIMIT}" != "0" ] && [ "${LIMIT}" != "all" ] && [ "${COUNT}" -ge "${LIMIT}" ]; then
    break
  fi
  COUNT=$((COUNT + 1))
  base="$(basename "${sample}")"
  base="${base%.*}"
  log="${OUT}/${base}.log"
  echo "[dflash-dataset] start ${COUNT} sample=${sample}"
  ./run_single.sh . "${sample}" "${MAX_NEW}" "${BLOCK}" 2>&1 | tee "${log}"
done

if [ "${FOUND}" -eq 0 ]; then
  echo "no .txt or .ids prompts found under ${PROMPT_DIR}" >&2
  exit 2
fi
echo "[dflash-dataset] done samples=${COUNT} out=${OUT}"
SH
  chmod +x "${dst}/run_dataset.sh"
}

prepare_common() {
  local model="$1"
  local src="${SRC_ROOT}/${model}"
  local dst="${DST_ROOT}/${model}"
  if [[ ! -d "${src}" ]]; then
    echo "missing source package: ${src}" >&2
    exit 2
  fi
  rm -rf "${dst}"
  mkdir -p "${dst}"
  cp -a "${BUILD_DIR}/dflash-runtime" "${dst}/"
  copy_qnn_libs "${src}" "${dst}"
  copy_file "${src}/embedding.fp16.bin" "${dst}/"
  copy_file "${src}/tokenizer.json" "${dst}/"
  copy_prompts "${dst}"
  chmod +x "${dst}/dflash-runtime"
}

package_0p6b() {
  local src="${SRC_ROOT}/0p6b"
  local dst="${DST_ROOT}/0p6b"
  prepare_common 0p6b
  copy_file "${src}/qwen3-0p6B-target.bin" "${dst}/qwen3-0p6B-target.bin"
  copy_file "${src}/qwen3-0p6B-draft.bin" "${dst}/qwen3-0p6B-draft.bin"
  write_single_script "${dst}" \
    --target ./qwen3-0p6B-target.bin \
    --draft ./qwen3-0p6B-draft.bin \
    --embedding ./embedding.fp16.bin \
    --target_kv_dtype uint16 \
    --target_layers 28 \
    --draft_layers 5 \
    --draft_hidden 1024 \
    --target_hidden 5120 \
    --prefill_seq 32 \
    --verify_seq 16 \
    --accept_topk 5
  write_dataset_script "${dst}"
}

package_1p7b() {
  local src="${SRC_ROOT}/1p7b"
  local dst="${DST_ROOT}/1p7b"
  prepare_common 1p7b
  copy_file "${src}/qwen3-1p7B-target.bin" "${dst}/qwen3-1p7B-target.bin"
  copy_file "${src}/qwen3-1p7B-draft.bin" "${dst}/qwen3-1p7B-draft.bin"
  write_single_script "${dst}" \
    --target ./qwen3-1p7B-target.bin \
    --draft ./qwen3-1p7B-draft.bin \
    --embedding ./embedding.fp16.bin \
    --target_kv_dtype uint16 \
    --target_layers 28 \
    --draft_layers 5 \
    --draft_hidden 2048 \
    --target_hidden 10240 \
    --prefill_seq 32 \
    --verify_seq 16 \
    --accept_topk 5
  write_dataset_script "${dst}"
}

package_4b() {
  local src="${SRC_ROOT}/4b"
  local dst="${DST_ROOT}/4b"
  prepare_common 4b
  copy_file "${src}/qwen3-4B-target.bin" "${dst}/qwen3-4B-target.bin"
  copy_file "${src}/qwen3-4B-draft.bin" "${dst}/qwen3-4B-draft.bin"
  write_single_script "${dst}" \
    --target ./qwen3-4B-target.bin \
    --draft ./qwen3-4B-draft.bin \
    --embedding ./embedding.fp16.bin \
    --target_kv_dtype uint8 \
    --target_layers 36 \
    --draft_layers 5 \
    --draft_hidden 2560 \
    --target_hidden 12800 \
    --prefill_seq 32 \
    --verify_seq 16 \
    --accept_topk 5
  write_dataset_script "${dst}"
}

mkdir -p "${DST_ROOT}"
package_0p6b
package_1p7b
package_4b

find "${DST_ROOT}" -maxdepth 2 -type f \( -name 'dflash-runtime' -o -name 'run_single.sh' -o -name 'run_dataset.sh' \) -printf '%P\n' | sort
