#pragma once

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <vector>

namespace dflash {

struct AotModelConfig {
  int context_len = 1024;
  int max_cache_len = 1008;
  int max_ar_len = 16;
  int num_layers = 0;
  int num_heads = 0;
  int head_dim = 128;
};

template <typename T>
class KvStorage {
 public:
  void assign(size_t n, T value) {
    if (external_) {
      const size_t fill_n = std::min(n, external_size_);
      std::fill(external_, external_ + fill_n, value);
      if (fill_n < external_size_) std::fill(external_ + fill_n, external_ + external_size_, T{});
      size_ = external_size_;
      return;
    }
    owned_.assign(n, value);
    size_ = owned_.size();
  }

  void bindExternal(T* ptr, size_t n, bool preserve_existing = true) {
    if (preserve_existing && ptr && n > 0) {
      const size_t bytes = std::min(n, size_) * sizeof(T);
      if (bytes > 0 && data()) std::memcpy(ptr, data(), bytes);
      if (n > size_) std::fill(ptr + size_, ptr + n, T{});
    }
    owned_.clear();
    owned_.shrink_to_fit();
    external_ = ptr;
    external_size_ = n;
    size_ = n;
  }

  T* data() { return external_ ? external_ : owned_.data(); }
  const T* data() const { return external_ ? external_ : owned_.data(); }
  size_t size() const { return size_; }
  bool aliases(const void* ptr) const { return ptr && data() == ptr; }

 private:
  std::vector<T> owned_;
  T* external_ = nullptr;
  size_t external_size_ = 0;
  size_t size_ = 0;
};

template <typename T>
struct KvLayerCache {
  KvStorage<T> key_in;
  KvStorage<T> key_out;
  KvStorage<T> value_in;
  KvStorage<T> value_out;
};

template <typename T>
class KvCacheManager {
 public:
  explicit KvCacheManager(AotModelConfig config);

  void init(int ar_len);
  void rearrange(int ar_len_dst);
  void update(int ar_len, int n_past, int n_update, const std::vector<bool>& selected = {});

  void initAttentionMask(uint16_t* attention_mask,
                         const std::vector<int32_t>& attention_map,
                         int ar_len,
                         int n_past) const;
  void updateAttentionMask(uint16_t* attention_mask, int ar_len, int n_past, int n_update) const;

  const std::vector<KvLayerCache<T>>& layers() const { return layers_; }
  std::vector<KvLayerCache<T>>& layers() { return layers_; }
  size_t totalBytes() const;

 private:
  void rearrangeKey(KvLayerCache<T>& layer, int ar_len_dst);
  void rearrangeValue(KvLayerCache<T>& layer, int ar_len_dst);
  void updateKey(KvLayerCache<T>& layer, int n_past, int n_update, const std::vector<bool>& selected);
  void updateValue(KvLayerCache<T>& layer, int n_past, int n_update, const std::vector<bool>& selected);

  AotModelConfig config_;
  int cur_ar_len_ = 0;
  std::vector<KvLayerCache<T>> layers_;
};

extern template class KvCacheManager<uint8_t>;
extern template class KvCacheManager<uint16_t>;

}  // namespace dflash
