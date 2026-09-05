#pragma once

#include <string>
#include <vector>

#include "dflash/runtime_io.hpp"

namespace dflash {

class TargetGraphSession {
 public:
  bool init(QnnRuntime& runtime, const std::string& graph_name, int num_layers);

  const TargetTensorBinding& binding() const { return binding_; }
  std::vector<TensorBuffer>& inputs() { return inputs_; }
  std::vector<TensorBuffer>& outputs() { return outputs_; }
  const std::vector<TensorBuffer>& inputs() const { return inputs_; }
  const std::vector<TensorBuffer>& outputs() const { return outputs_; }

  void fillBlock(const std::vector<int64_t>& block, int64_t start_pos, int ar_len);
  void fillPrefillAttention(int ar_len, int n_past);
  void fillVerifyAttention(int ar_len, int n_past);

  template <typename T>
  void fillCacheInputs(const KvCacheManager<T>& cache) {
    fillTargetCacheInputs(inputs_, binding_, cache);
  }

  template <typename T>
  void fillCacheOutputs(KvCacheManager<T>& cache) const {
    fillTargetCacheOutputs(outputs_, binding_, cache);
  }

  bool execute();

 private:
  QnnRuntime* runtime_ = nullptr;
  std::string graph_name_;
  TargetTensorBinding binding_;
  std::vector<TensorBuffer> inputs_;
  std::vector<TensorBuffer> outputs_;
  int num_layers_ = 0;
};

class DraftGraphSession {
 public:
  bool init(QnnRuntime& runtime, const std::string& graph_name, const DraftTensorGeometry& geom);

  const DraftTensorBinding& binding() const { return binding_; }
  const DraftTensorGeometry& geometry() const { return geom_; }
  std::vector<TensorBuffer>& inputs() { return inputs_; }
  std::vector<TensorBuffer>& outputs() { return outputs_; }
  const std::vector<TensorBuffer>& inputs() const { return inputs_; }
  const std::vector<TensorBuffer>& outputs() const { return outputs_; }

  template <typename T>
  void fillCacheInputs(const KvCacheManager<T>& cache) {
    fillDraftCacheInputs(inputs_, binding_, cache);
  }

  template <typename T>
  void fillCacheOutputs(KvCacheManager<T>& cache) const {
    fillDraftCacheOutputs(outputs_, binding_, cache);
  }

  void fillStandardInputs(int64_t position_base, const std::vector<float>& noise, const std::vector<float>& target_hidden);
  void fillStandardRope(int64_t position_base);
  void fillStandardMaskScale();

  void fillCacheAwareInputs(int64_t position_base,
                            int delta_count,
                            const std::vector<float>& noise,
                            const std::vector<float>& target_hidden,
                            bool include_noise);

  bool execute();

 private:
  QnnRuntime* runtime_ = nullptr;
  std::string graph_name_;
  DraftTensorBinding binding_;
  DraftTensorGeometry geom_;
  std::vector<TensorBuffer> inputs_;
  std::vector<TensorBuffer> outputs_;
};

}  // namespace dflash
