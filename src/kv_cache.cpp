#include "dflash/kv_cache.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace dflash {

template <typename T>
KvCacheManager<T>::KvCacheManager(AotModelConfig config) : config_(config) {
  layers_.resize(config_.num_layers);
}

template <typename T>
void KvCacheManager<T>::init(int ar_len) {
  cur_ar_len_ = ar_len;
  const size_t cache_in_elems =
      static_cast<size_t>(config_.num_heads) * config_.head_dim * config_.max_cache_len;
  const size_t cache_out_elems =
      static_cast<size_t>(config_.num_heads) * config_.head_dim * config_.max_ar_len;
  for (auto& layer : layers_) {
    layer.key_in.assign(cache_in_elems, T{});
    layer.key_out.assign(cache_out_elems, T{});
    layer.value_in.assign(cache_in_elems, T{});
    layer.value_out.assign(cache_out_elems, T{});
  }
}

template <typename T>
size_t KvCacheManager<T>::totalBytes() const {
  size_t elems = 0;
  for (const auto& layer : layers_) {
    elems += layer.key_in.size() + layer.key_out.size() + layer.value_in.size() + layer.value_out.size();
  }
  return elems * sizeof(T);
}

template <typename T>
void KvCacheManager<T>::initAttentionMask(uint16_t* attention_mask,
                                          const std::vector<int32_t>& attention_map,
                                          int ar_len,
                                          int n_past) const {
  if (static_cast<int>(attention_map.size()) > ar_len) {
    throw std::runtime_error("attention_map is larger than ar_len");
  }
  constexpr uint16_t neg_val = 0;
  constexpr uint16_t pos_val = 65535;
  std::fill_n(attention_mask, static_cast<size_t>(ar_len) * config_.context_len, neg_val);

  uint16_t* past_ptr = attention_mask;
  uint16_t* new_ptr = attention_mask + (config_.context_len - ar_len);
  for (int i = 0; i < ar_len; ++i) {
    if (attention_map[i] < 0) {
      std::fill_n(past_ptr, n_past, pos_val);
    } else {
      const int pidx = attention_map[i];
      const uint16_t* parent_ptr = attention_mask + static_cast<size_t>(pidx) * config_.context_len;
      std::memcpy(past_ptr, parent_ptr, static_cast<size_t>(config_.context_len) * sizeof(uint16_t));
    }
    new_ptr[i] = pos_val;
    past_ptr += config_.context_len;
    new_ptr += config_.context_len;
  }
}

template <typename T>
void KvCacheManager<T>::updateAttentionMask(uint16_t* attention_mask, int ar_len, int n_past, int n_update) const {
  constexpr uint16_t pos_val = 65535;
  uint16_t* cur_ptr = attention_mask + n_past;
  for (int i = 0; i < ar_len; ++i) {
    std::fill_n(cur_ptr, n_update, pos_val);
    cur_ptr += config_.context_len;
  }
}

template <typename T>
void KvCacheManager<T>::rearrange(int ar_len_dst) {
  if (cur_ar_len_ == ar_len_dst) return;
  for (auto& layer : layers_) {
    rearrangeKey(layer, ar_len_dst);
    rearrangeValue(layer, ar_len_dst);
  }
  cur_ar_len_ = ar_len_dst;
}

template <typename T>
void KvCacheManager<T>::rearrangeKey(KvLayerCache<T>& layer, int ar_len_dst) {
  const int src_cache_num = (cur_ar_len_ == config_.context_len) ? config_.context_len : config_.context_len - cur_ar_len_;
  const int dst_cache_num = config_.context_len - ar_len_dst;
  T* read_ptr = layer.key_in.data();
  T* write_ptr = layer.key_in.data();
  const int n_iter = config_.head_dim * config_.num_heads;

  if (src_cache_num > dst_cache_num) {
    for (int i = 0; i < n_iter; ++i) {
      std::memmove(write_ptr, read_ptr, static_cast<size_t>(dst_cache_num) * sizeof(T));
      read_ptr += src_cache_num;
      write_ptr += dst_cache_num;
    }
  } else {
    read_ptr += static_cast<size_t>(n_iter - 1) * src_cache_num;
    write_ptr += static_cast<size_t>(n_iter - 1) * dst_cache_num;
    for (int i = 0; i < n_iter; ++i) {
      std::memmove(write_ptr, read_ptr, static_cast<size_t>(src_cache_num) * sizeof(T));
      read_ptr -= src_cache_num;
      write_ptr -= dst_cache_num;
    }
  }
}

template <typename T>
void KvCacheManager<T>::rearrangeValue(KvLayerCache<T>& layer, int ar_len_dst) {
  const int src_cache_num = (cur_ar_len_ == config_.context_len) ? config_.context_len : config_.context_len - cur_ar_len_;
  const int dst_cache_num = config_.context_len - ar_len_dst;
  T* read_ptr = layer.value_in.data();
  T* write_ptr = layer.value_in.data();

  if (src_cache_num > dst_cache_num) {
    for (int i = 0; i < config_.num_heads; ++i) {
      std::memmove(write_ptr, read_ptr, static_cast<size_t>(dst_cache_num) * config_.head_dim * sizeof(T));
      read_ptr += src_cache_num * config_.head_dim;
      write_ptr += dst_cache_num * config_.head_dim;
    }
  } else {
    read_ptr += static_cast<size_t>(config_.head_dim) * (config_.num_heads - 1) * src_cache_num;
    write_ptr += static_cast<size_t>(config_.head_dim) * (config_.num_heads - 1) * dst_cache_num;
    for (int i = 0; i < config_.num_heads; ++i) {
      std::memmove(write_ptr, read_ptr, static_cast<size_t>(src_cache_num) * config_.head_dim * sizeof(T));
      read_ptr -= src_cache_num * config_.head_dim;
      write_ptr -= dst_cache_num * config_.head_dim;
    }
  }
}

template <typename T>
void KvCacheManager<T>::update(int ar_len, int n_past, int n_update, const std::vector<bool>& selected) {
  if (cur_ar_len_ != ar_len) {
    throw std::runtime_error("KV cache ar_len mismatch; call rearrange first");
  }
  for (auto& layer : layers_) {
    updateKey(layer, n_past, n_update, selected);
    updateValue(layer, n_past, n_update, selected);
  }
}

template <typename T>
void KvCacheManager<T>::updateKey(KvLayerCache<T>& layer, int n_past, int n_update, const std::vector<bool>& selected) {
  T* write_ptr = layer.key_in.data() + n_past;
  T* read_ptr = layer.key_out.data();
  const int iter_size = (cur_ar_len_ == config_.context_len) ? config_.context_len : config_.context_len - cur_ar_len_;
  const int out_size = cur_ar_len_;
  const int n_iter = config_.head_dim * config_.num_heads;

  if (selected.empty()) {
    const size_t copy_bytes = static_cast<size_t>(n_update) * sizeof(T);
    for (int i = 0; i < n_iter; ++i) {
      std::memcpy(write_ptr, read_ptr, copy_bytes);
      write_ptr += iter_size;
      read_ptr += out_size;
    }
    return;
  }

  std::vector<int> true_indices;
  true_indices.reserve(n_update);
  for (int i = 0; i < static_cast<int>(selected.size()) && static_cast<int>(true_indices.size()) < n_update; ++i) {
    if (selected[i]) true_indices.push_back(i);
  }
  for (int i = 0; i < n_iter; ++i) {
    for (int j = 0; j < n_update; ++j) write_ptr[j] = read_ptr[true_indices[j]];
    write_ptr += iter_size;
    read_ptr += out_size;
  }
}

template <typename T>
void KvCacheManager<T>::updateValue(KvLayerCache<T>& layer, int n_past, int n_update, const std::vector<bool>& selected) {
  T* write_ptr = layer.value_in.data() + static_cast<size_t>(n_past) * config_.head_dim;
  T* read_ptr = layer.value_out.data();
  const int iter_size = (cur_ar_len_ == config_.context_len) ? config_.context_len * config_.head_dim
                                                            : (config_.context_len - cur_ar_len_) * config_.head_dim;
  const int out_size = cur_ar_len_ * config_.head_dim;

  if (selected.empty()) {
    const size_t copy_bytes = static_cast<size_t>(n_update) * config_.head_dim * sizeof(T);
    for (int i = 0; i < config_.num_heads; ++i) {
      std::memcpy(write_ptr, read_ptr, copy_bytes);
      write_ptr += iter_size;
      read_ptr += out_size;
    }
    return;
  }

  for (int i = 0; i < config_.num_heads; ++i) {
    T* wp = write_ptr;
    T* rp = read_ptr;
    int copied = 0;
    for (bool sel : selected) {
      if (sel) {
        std::memcpy(wp, rp, static_cast<size_t>(config_.head_dim) * sizeof(T));
        wp += config_.head_dim;
        ++copied;
      }
      rp += config_.head_dim;
      if (copied == n_update) break;
    }
    write_ptr += iter_size;
    read_ptr += out_size;
  }
}

template class KvCacheManager<uint8_t>;
template class KvCacheManager<uint16_t>;

}  // namespace dflash
