#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "dflash/kv_cache.hpp"
#include "dflash/qnn_runtime.hpp"

namespace dflash {

constexpr size_t kNoTensor = std::numeric_limits<size_t>::max();

struct TargetTensorGeometry {
  int ar_len = 32;
  int context_len = 1024;
  int num_layers = 36;
  int num_heads = 8;
  int head_dim = 128;
  int vocab_size = 151936;
};

struct DraftTensorGeometry {
  int block = 16;
  int context_len = 128;
  int target_hidden = 12800;
  int draft_hidden = 2560;
  int num_layers = 5;
  int num_heads = 8;
  int head_dim = 128;
  int cache_delta = 16;
  int cache_ar_len = 32;
  int cache_context = 128;
  float rope_theta = 1000000.0f;
};

struct TargetTensorBinding {
  size_t input_ids = kNoTensor;
  size_t position_ids = kNoTensor;
  size_t attention_mask = kNoTensor;
  std::vector<size_t> past_key_inputs;
  std::vector<size_t> past_value_inputs;
  size_t logits_output = kNoTensor;
  std::vector<size_t> present_key_outputs;
  std::vector<size_t> present_value_outputs;
  std::vector<size_t> extra_outputs;
};

struct DraftTensorBinding {
  size_t output0 = kNoTensor;
  size_t hidden_output = kNoTensor;
  size_t noise_input = kNoTensor;
  size_t target_hidden_input = kNoTensor;
  size_t position_input = kNoTensor;
  std::vector<size_t> full_rope_inputs;
  std::vector<size_t> q_rope_inputs;
  std::vector<size_t> mask_scale_inputs;
  std::vector<size_t> tap_outputs;
  bool cache_aware = false;
};

TargetTensorBinding buildTargetTensorBinding(const Graph& graph, int num_layers);
DraftTensorBinding buildDraftTensorBinding(const Graph& graph, const DraftTensorGeometry& geom);

std::vector<int32_t> makePrefillAttentionMap(int ar_len);
std::vector<int32_t> makeVerifyAttentionMap(int ar_len);

void fillInt32Buffer(TensorBuffer& dst, const std::vector<int32_t>& values);
void fillFloatBuffer(TensorBuffer& dst, const std::vector<float>& values);
void fillUInt16Buffer(TensorBuffer& dst, const std::vector<uint16_t>& values);

void fillBlockInputs(std::vector<TensorBuffer>& inputs,
                     size_t input_ids_index,
                     size_t position_ids_index,
                     const std::vector<int64_t>& block,
                     int64_t start_pos,
                     int ar_len);

void fillPositionIdsInput(std::vector<TensorBuffer>& inputs,
                          size_t position_index,
                          int64_t position_base,
                          int positions);

void fillAttentionMaskBuffer(TensorBuffer& dst,
                             const std::vector<int32_t>& attention_map,
                             int ar_len,
                             int n_past);

template <typename T>
void fillTargetCacheInputs(std::vector<TensorBuffer>& inputs,
                           const TargetTensorBinding& binding,
                           const KvCacheManager<T>& cache) {
  const auto& layers = cache.layers();
  const size_t n = std::min(layers.size(), std::min(binding.past_key_inputs.size(), binding.past_value_inputs.size()));
  for (size_t i = 0; i < n; ++i) {
    if (binding.past_key_inputs[i] != kNoTensor) {
      auto& dst = inputs[binding.past_key_inputs[i]];
      if (layers[i].key_in.aliases(dst.host.data())) {
        continue;
      }
      const size_t bytes = std::min(dst.host.size(), layers[i].key_in.size() * sizeof(T));
      std::memcpy(dst.host.data(), layers[i].key_in.data(), bytes);
      if (bytes < dst.host.size()) {
        std::fill(dst.host.begin() + static_cast<std::ptrdiff_t>(bytes), dst.host.end(), 0);
      }
    }
    if (binding.past_value_inputs[i] != kNoTensor) {
      auto& dst = inputs[binding.past_value_inputs[i]];
      if (layers[i].value_in.aliases(dst.host.data())) {
        continue;
      }
      const size_t bytes = std::min(dst.host.size(), layers[i].value_in.size() * sizeof(T));
      std::memcpy(dst.host.data(), layers[i].value_in.data(), bytes);
      if (bytes < dst.host.size()) {
        std::fill(dst.host.begin() + static_cast<std::ptrdiff_t>(bytes), dst.host.end(), 0);
      }
    }
  }
}

template <typename T>
void fillTargetCacheOutputs(const std::vector<TensorBuffer>& outputs,
                            const TargetTensorBinding& binding,
                            KvCacheManager<T>& cache) {
  auto& layers = cache.layers();
  const size_t n = std::min(layers.size(), std::min(binding.present_key_outputs.size(), binding.present_value_outputs.size()));
  for (size_t i = 0; i < n; ++i) {
    if (binding.present_key_outputs[i] != kNoTensor) {
      const auto& src = outputs[binding.present_key_outputs[i]];
      if (layers[i].key_out.aliases(src.host.data())) {
        continue;
      }
      const size_t bytes = std::min(src.host.size(), layers[i].key_out.size() * sizeof(T));
      std::memcpy(layers[i].key_out.data(), src.host.data(), bytes);
    }
    if (binding.present_value_outputs[i] != kNoTensor) {
      const auto& src = outputs[binding.present_value_outputs[i]];
      if (layers[i].value_out.aliases(src.host.data())) {
        continue;
      }
      const size_t bytes = std::min(src.host.size(), layers[i].value_out.size() * sizeof(T));
      std::memcpy(layers[i].value_out.data(), src.host.data(), bytes);
    }
  }
}

template <typename T>
void fillDraftCacheInputs(std::vector<TensorBuffer>& inputs,
                          const DraftTensorBinding& binding,
                          const KvCacheManager<T>& cache) {
  const auto& layers = cache.layers();
  if (!binding.cache_aware) return;
  constexpr size_t kCacheInputBase = 8;
  constexpr size_t kHeads = 8;
  constexpr size_t kPerLayerInputs = kHeads * 2;
  for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
    const auto& layer = layers[layer_idx];
    const size_t key_head_elems = layer.key_in.size() / kHeads;
    const size_t value_head_elems = layer.value_in.size() / kHeads;
    for (size_t head = 0; head < kHeads; ++head) {
      const size_t key_idx = kCacheInputBase + layer_idx * kPerLayerInputs + head;
      const size_t value_idx = kCacheInputBase + layer_idx * kPerLayerInputs + kHeads + head;
      if (key_idx < inputs.size() && key_head_elems > 0) {
        const T* src = layer.key_in.data() + head * key_head_elems;
        auto& dst = inputs[key_idx];
        const size_t bytes = std::min(dst.host.size(), key_head_elems * sizeof(T));
        std::memcpy(dst.host.data(), src, bytes);
        if (bytes < dst.host.size()) {
          std::fill(dst.host.begin() + static_cast<std::ptrdiff_t>(bytes), dst.host.end(), 0);
        }
      }
      if (value_idx < inputs.size() && value_head_elems > 0) {
        const T* src = layer.value_in.data() + head * value_head_elems;
        auto& dst = inputs[value_idx];
        const size_t bytes = std::min(dst.host.size(), value_head_elems * sizeof(T));
        std::memcpy(dst.host.data(), src, bytes);
        if (bytes < dst.host.size()) {
          std::fill(dst.host.begin() + static_cast<std::ptrdiff_t>(bytes), dst.host.end(), 0);
        }
      }
    }
  }
}

template <typename T>
void fillDraftCacheOutputs(const std::vector<TensorBuffer>& outputs,
                           const DraftTensorBinding& binding,
                           KvCacheManager<T>& cache) {
  auto& layers = cache.layers();
  if (!binding.cache_aware) return;
  constexpr size_t kCacheOutputBase = 1;
  constexpr size_t kHeads = 8;
  constexpr size_t kPerLayerOutputs = kHeads * 2;
  for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
    auto& layer = layers[layer_idx];
    const size_t key_head_elems = layer.key_out.size() / kHeads;
    const size_t value_head_elems = layer.value_out.size() / kHeads;
    for (size_t head = 0; head < kHeads; ++head) {
      const size_t key_idx = kCacheOutputBase + layer_idx * kPerLayerOutputs + head;
      const size_t value_idx = kCacheOutputBase + layer_idx * kPerLayerOutputs + kHeads + head;
      if (key_idx < outputs.size() && key_head_elems > 0) {
        const auto& src = outputs[key_idx];
        T* dst = layer.key_out.data() + head * key_head_elems;
        const size_t bytes = std::min(src.host.size(), key_head_elems * sizeof(T));
        std::memcpy(dst, src.host.data(), bytes);
      }
      if (value_idx < outputs.size() && value_head_elems > 0) {
        const auto& src = outputs[value_idx];
        T* dst = layer.value_out.data() + head * value_head_elems;
        const size_t bytes = std::min(src.host.size(), value_head_elems * sizeof(T));
        std::memcpy(dst, src.host.data(), bytes);
      }
    }
  }
}

void fillDraftNoiseInput(std::vector<TensorBuffer>& inputs, size_t noise_index, const std::vector<float>& noise);
void fillDraftTargetHiddenInput(std::vector<TensorBuffer>& inputs,
                               size_t target_hidden_index,
                               const std::vector<float>& history,
                               int64_t start_token,
                               int count,
                               int target_hidden,
                               int cache_delta);
void fillDraftStandardRopeInputs(std::vector<TensorBuffer>& inputs,
                                 const DraftTensorBinding& binding,
                                 int64_t position_base,
                                 const DraftTensorGeometry& geom);
void fillDraftStandardMaskScaleInputs(std::vector<TensorBuffer>& inputs,
                                     const DraftTensorBinding& binding,
                                     const DraftTensorGeometry& geom);
std::vector<float> buildRopeForPositions(int64_t position_base,
                                         int positions,
                                         int head_dim,
                                         bool want_sin,
                                         float rope_theta);
std::vector<float> buildRopeForCacheCurrent(int64_t position_base,
                                            int delta_count,
                                            int head_dim,
                                            int cache_ar_len,
                                            int cache_delta,
                                            bool want_sin,
                                            float rope_theta);
void fillDraftRopeInputs(std::vector<TensorBuffer>& inputs,
                         const DraftTensorBinding& binding,
                         int64_t position_base,
                         int delta_count,
                         const DraftTensorGeometry& geom);
void fillDraftMaskScaleInputs(std::vector<TensorBuffer>& inputs,
                              const DraftTensorBinding& binding,
                              int delta_count,
                              bool include_noise,
                              const DraftTensorGeometry& geom);

int64_t tensorElementCount(const TensorBuffer& buffer);
std::vector<float> tensorBufferToFloat(const TensorBuffer& buffer);
float readTensorValueAtRaw(const TensorBuffer& buffer, int64_t flat_index);
void copyTensorRowToFloat(const TensorBuffer& buffer, int64_t row_base, float* dst, int count);
void argmaxRowsAtRaw(const TensorBuffer& buffer,
                     int first_seq_index,
                     int row_count,
                     int vocab_size,
                     int64_t* dst);
int64_t readTokenIdAtRaw(const TensorBuffer& buffer, int seq_index);
int64_t argmaxTensorAtRaw(const TensorBuffer& buffer, int seq_index, int vocab_size);

struct RankAcceptResult {
  int64_t argmax = 0;
  int greater_count = 0;
  bool accepted = false;
};

RankAcceptResult rankAcceptTensorAtRaw(const TensorBuffer& buffer,
                                       int seq_index,
                                       int vocab_size,
                                       int token,
                                       int top_k);
bool tokenInTopKAtRaw(const TensorBuffer& buffer, int seq_index, int vocab_size, int token, int top_k);

}  // namespace dflash
