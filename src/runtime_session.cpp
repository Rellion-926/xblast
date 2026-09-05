#include "dflash/runtime_session.hpp"

namespace dflash {

bool TargetGraphSession::init(QnnRuntime& runtime, const std::string& graph_name, int num_layers) {
  runtime_ = &runtime;
  graph_name_ = graph_name;
  num_layers_ = num_layers;
  binding_ = buildTargetTensorBinding(runtime.graph(graph_name), num_layers);
  inputs_ = runtime.makeInputBuffers(graph_name);
  outputs_ = runtime.makeOutputBuffers(graph_name);
  return !inputs_.empty() || !outputs_.empty();
}

void TargetGraphSession::fillBlock(const std::vector<int64_t>& block, int64_t start_pos, int ar_len) {
  fillBlockInputs(inputs_, binding_.input_ids, binding_.position_ids, block, start_pos, ar_len);
}

void TargetGraphSession::fillPrefillAttention(int ar_len, int n_past) {
  if (binding_.attention_mask == kNoTensor || binding_.attention_mask >= inputs_.size()) return;
  fillAttentionMaskBuffer(inputs_[binding_.attention_mask], makePrefillAttentionMap(ar_len), ar_len, n_past);
}

void TargetGraphSession::fillVerifyAttention(int ar_len, int n_past) {
  if (binding_.attention_mask == kNoTensor || binding_.attention_mask >= inputs_.size()) return;
  fillAttentionMaskBuffer(inputs_[binding_.attention_mask], makeVerifyAttentionMap(ar_len), ar_len, n_past);
}

bool TargetGraphSession::execute() {
  return runtime_ && runtime_->execute(graph_name_, inputs_, outputs_);
}

bool DraftGraphSession::init(QnnRuntime& runtime, const std::string& graph_name, const DraftTensorGeometry& geom) {
  runtime_ = &runtime;
  graph_name_ = graph_name;
  geom_ = geom;
  binding_ = buildDraftTensorBinding(runtime.graph(graph_name), geom);
  inputs_ = runtime.makeInputBuffers(graph_name);
  outputs_ = runtime.makeOutputBuffers(graph_name);
  return !inputs_.empty() || !outputs_.empty();
}

void DraftGraphSession::fillStandardInputs(int64_t position_base,
                                           const std::vector<float>& noise,
                                           const std::vector<float>& target_hidden) {
  if (binding_.cache_aware) return;
  fillDraftNoiseInput(inputs_, binding_.noise_input, noise);
  fillDraftTargetHiddenInput(inputs_,
                             binding_.target_hidden_input,
                             target_hidden,
                             0,
                             geom_.cache_delta,
                             geom_.target_hidden,
                             geom_.cache_delta);
  fillDraftStandardRopeInputs(inputs_, binding_, position_base, geom_);
  fillDraftStandardMaskScaleInputs(inputs_, binding_, geom_);
  fillPositionIdsInput(inputs_, binding_.position_input, position_base, geom_.context_len + geom_.block);
}

void DraftGraphSession::fillStandardRope(int64_t position_base) {
  if (binding_.cache_aware) return;
  fillDraftStandardRopeInputs(inputs_, binding_, position_base, geom_);
}

void DraftGraphSession::fillStandardMaskScale() {
  if (binding_.cache_aware) return;
  fillDraftStandardMaskScaleInputs(inputs_, binding_, geom_);
}

void DraftGraphSession::fillCacheAwareInputs(int64_t position_base,
                                             int delta_count,
                                             const std::vector<float>& noise,
                                             const std::vector<float>& target_hidden,
                                             bool include_noise) {
  if (!binding_.cache_aware) return;
  fillDraftNoiseInput(inputs_, binding_.noise_input, noise);
  fillDraftTargetHiddenInput(inputs_,
                             binding_.target_hidden_input,
                             target_hidden,
                             0,
                             delta_count,
                             geom_.target_hidden,
                             geom_.cache_delta);
  fillDraftRopeInputs(inputs_, binding_, position_base, delta_count, geom_);
  fillDraftMaskScaleInputs(inputs_, binding_, delta_count, include_noise, geom_);
}

bool DraftGraphSession::execute() {
  return runtime_ && runtime_->execute(graph_name_, inputs_, outputs_);
}

}  // namespace dflash

