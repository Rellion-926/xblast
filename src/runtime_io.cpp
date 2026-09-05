#include "dflash/runtime_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace dflash {
namespace {

bool startsWith(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

int trailingIndex(const std::string& name) {
  int value = -1;
  int mul = 1;
  bool found = false;
  for (int i = static_cast<int>(name.size()) - 1; i >= 0; --i) {
    if (name[i] < '0' || name[i] > '9') break;
    found = true;
    value = (name[i] - '0') * mul + value;
    mul *= 10;
  }
  return found ? value : -1;
}

template <typename T>
void copyVectorToBuffer(TensorBuffer& dst, const std::vector<T>& values) {
  const size_t bytes = std::min(dst.host.size(), values.size() * sizeof(T));
  std::memcpy(dst.host.data(), values.data(), bytes);
  if (bytes < dst.host.size()) {
    std::fill(dst.host.begin() + static_cast<std::ptrdiff_t>(bytes), dst.host.end(), 0);
  }
}

std::vector<size_t> collectByPrefix(const std::vector<TensorDesc>& descs, const std::string& prefix) {
  std::vector<std::pair<int, size_t>> tmp;
  for (size_t i = 0; i < descs.size(); ++i) {
    if (startsWith(descs[i].name, prefix)) {
      tmp.emplace_back(trailingIndex(descs[i].name), i);
    }
  }
  std::sort(tmp.begin(), tmp.end(), [](const auto& a, const auto& b) {
    if (a.first < 0 && b.first < 0) return a.second < b.second;
    if (a.first < 0) return false;
    if (b.first < 0) return true;
    if (a.first != b.first) return a.first < b.first;
    return a.second < b.second;
  });
  std::vector<size_t> out;
  out.reserve(tmp.size());
  for (auto& item : tmp) out.push_back(item.second);
  return out;
}

size_t findExact(const std::vector<TensorDesc>& descs, const std::string& name) {
  for (size_t i = 0; i < descs.size(); ++i) {
    if (descs[i].name == name) return i;
  }
  return kNoTensor;
}

int64_t numel(const std::vector<uint32_t>& dims) {
  int64_t n = 1;
  for (auto d : dims) n *= static_cast<int64_t>(d);
  return n;
}

bool hasScaleOffset(const Qnn_QuantizeParams_t& q) {
  return q.encodingDefinition == QNN_DEFINITION_DEFINED &&
         q.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
}

uint16_t floatToFp16Bits(float f) {
  uint32_t x = 0;
  std::memcpy(&x, &f, sizeof(x));
  uint32_t sign = (x >> 31) & 0x1;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127;
  uint32_t mant = x & 0x7fffff;
  if (exp == 128) {
    uint16_t h_mant = static_cast<uint16_t>(mant >> 13);
    if (mant != 0 && h_mant == 0) { h_mant = 1; }
    return static_cast<uint16_t>((sign << 15) | (0x1f << 10) | h_mant);
  }
  if (exp < -14) {
    int32_t shift = -14 - exp;
    if (shift > 24) { return static_cast<uint16_t>(sign << 15); }
    uint32_t mant_implicit = mant | 0x800000;
    uint16_t h_mant = static_cast<uint16_t>(mant_implicit >> (shift + 13));
    return static_cast<uint16_t>((sign << 15) | h_mant);
  }
  if (exp > 15) { return static_cast<uint16_t>((sign << 15) | (0x1f << 10)); }
  uint16_t h_exp = static_cast<uint16_t>((exp + 15) & 0x1f);
  uint16_t h_mant = static_cast<uint16_t>(mant >> 13);
  return static_cast<uint16_t>((sign << 15) | (h_exp << 10) | h_mant);
}

uint16_t floatToBf16Bits(float f) {
  uint32_t bits = 0;
  std::memcpy(&bits, &f, sizeof(bits));
  const uint32_t round_bit = (bits >> 16) & 1u;
  bits += 0x7FFFu + round_bit;
  return static_cast<uint16_t>(bits >> 16);
}

float bf16ToFloat(uint16_t v) {
  uint32_t bits = static_cast<uint32_t>(v) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

float fp16ToFloat(uint16_t h) {
  int sign = (h & 0x8000) ? -1 : 1;
  int exp = (h >> 10) & 0x1f;
  int mant = h & 0x03ff;
  if (exp == 0) { return sign * std::ldexp(static_cast<float>(mant), -24); }
  if (exp == 31) {
    return mant ? std::numeric_limits<float>::quiet_NaN()
                : sign * std::numeric_limits<float>::infinity();
  }
  return sign * std::ldexp(static_cast<float>(1024 + mant), exp - 25);
}

float dequantScaleOffset(double raw, const Qnn_QuantizeParams_t& q) {
  if (!hasScaleOffset(q)) return static_cast<float>(raw);
  return static_cast<float>((raw + static_cast<double>(q.scaleOffsetEncoding.offset)) *
                            static_cast<double>(q.scaleOffsetEncoding.scale));
}

template <typename RawT>
float dequantRaw(RawT raw, Qnn_DataType_t dtype, const Qnn_QuantizeParams_t& quant) {
  if (dtype == QNN_DATATYPE_FLOAT_16) {
    return fp16ToFloat(static_cast<uint16_t>(raw));
  }
  if (dtype == QNN_DATATYPE_BFLOAT_16) {
    return bf16ToFloat(static_cast<uint16_t>(raw));
  }
  return dequantScaleOffset(static_cast<double>(raw), quant);
}

#if defined(__aarch64__)
uint16_t maxU16Neon(const uint16_t* ptr, int count) {
  if (count <= 0) return 0;
  int i = 0;
  uint16x8_t max_v = vdupq_n_u16(0);
  for (; i + 8 <= count; i += 8) {
    max_v = vmaxq_u16(max_v, vld1q_u16(ptr + i));
  }
  uint16_t best = vmaxvq_u16(max_v);
  for (; i < count; ++i) {
    if (ptr[i] > best) best = ptr[i];
  }
  return best;
}

int64_t argmaxU16Neon(const uint16_t* ptr, int count) {
  const uint16_t best = maxU16Neon(ptr, count);
  for (int i = 0; i < count; ++i) {
    if (ptr[i] == best) return i;
  }
  return 0;
}

int countGreaterU16Neon(const uint16_t* ptr, int count, uint16_t candidate) {
  int i = 0;
  uint32_t greater = 0;
  const uint16x8_t candidate_v = vdupq_n_u16(candidate);
  for (; i + 8 <= count; i += 8) {
    const uint16x8_t v = vld1q_u16(ptr + i);
    const uint16x8_t gt = vshrq_n_u16(vcgtq_u16(v, candidate_v), 15);
    greater += static_cast<uint32_t>(vaddvq_u16(gt));
  }
  for (; i < count; ++i) {
    if (ptr[i] > candidate) ++greater;
  }
  return static_cast<int>(greater);
}
#endif

template <typename T>
T clampCast(int64_t value) {
  const auto lo = static_cast<int64_t>(std::numeric_limits<T>::min());
  const auto hi = static_cast<int64_t>(std::numeric_limits<T>::max());
  return static_cast<T>(std::clamp(value, lo, hi));
}

template <typename T>
void writeZeroes(TensorBuffer& dst) {
  auto* ptr = reinterpret_cast<T*>(dst.host.data());
  const size_t count = dst.host.size() / sizeof(T);
  std::fill(ptr, ptr + count, T{});
}

void fillQuantizedFloatBuffer(TensorBuffer& dst, const std::vector<float>& values) {
  const auto dtype = dst.desc.dtype;
  const auto quant = dst.desc.quant;
  const int64_t count = static_cast<int64_t>(dst.bytes() / qnnDtypeSize(dtype));
  const int64_t n = std::min<int64_t>(count, static_cast<int64_t>(values.size()));

  auto quantize = [&](float v) -> int64_t {
    double raw = static_cast<double>(v);
    if (hasScaleOffset(quant)) {
      raw = static_cast<double>(v) / static_cast<double>(quant.scaleOffsetEncoding.scale) -
            static_cast<double>(quant.scaleOffsetEncoding.offset);
    }
    return static_cast<int64_t>(std::llround(raw));
  };

  switch (dtype) {
    case QNN_DATATYPE_FLOAT_32: {
      auto* dst_ptr = reinterpret_cast<float*>(dst.host.data());
      for (int64_t i = 0; i < n; ++i) dst_ptr[i] = values[static_cast<size_t>(i)];
      for (int64_t i = n; i < count; ++i) dst_ptr[i] = 0.0f;
      return;
    }
    case QNN_DATATYPE_FLOAT_16: {
      auto* dst_ptr = reinterpret_cast<uint16_t*>(dst.host.data());
      for (int64_t i = 0; i < n; ++i) dst_ptr[i] = floatToFp16Bits(values[static_cast<size_t>(i)]);
      for (int64_t i = n; i < count; ++i) dst_ptr[i] = 0;
      return;
    }
    case QNN_DATATYPE_BFLOAT_16: {
      auto* dst_ptr = reinterpret_cast<uint16_t*>(dst.host.data());
      for (int64_t i = 0; i < n; ++i) dst_ptr[i] = floatToBf16Bits(values[static_cast<size_t>(i)]);
      for (int64_t i = n; i < count; ++i) dst_ptr[i] = 0;
      return;
    }
    case QNN_DATATYPE_INT_32: {
      auto* dst_ptr = reinterpret_cast<int32_t*>(dst.host.data());
      for (int64_t i = 0; i < n; ++i) dst_ptr[i] = clampCast<int32_t>(quantize(values[static_cast<size_t>(i)]));
      for (int64_t i = n; i < count; ++i) dst_ptr[i] = 0;
      return;
    }
    case QNN_DATATYPE_UINT_32: {
      auto* dst_ptr = reinterpret_cast<uint32_t*>(dst.host.data());
      for (int64_t i = 0; i < n; ++i) dst_ptr[i] = clampCast<uint32_t>(quantize(values[static_cast<size_t>(i)]));
      for (int64_t i = n; i < count; ++i) dst_ptr[i] = 0;
      return;
    }
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_SFIXED_POINT_16: {
      auto* dst_ptr = reinterpret_cast<int16_t*>(dst.host.data());
      for (int64_t i = 0; i < n; ++i) dst_ptr[i] = clampCast<int16_t>(quantize(values[static_cast<size_t>(i)]));
      for (int64_t i = n; i < count; ++i) dst_ptr[i] = 0;
      return;
    }
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16: {
      auto* dst_ptr = reinterpret_cast<uint16_t*>(dst.host.data());
      for (int64_t i = 0; i < n; ++i) dst_ptr[i] = clampCast<uint16_t>(quantize(values[static_cast<size_t>(i)]));
      for (int64_t i = n; i < count; ++i) dst_ptr[i] = 0;
      return;
    }
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_SFIXED_POINT_8: {
      auto* dst_ptr = reinterpret_cast<int8_t*>(dst.host.data());
      for (int64_t i = 0; i < n; ++i) dst_ptr[i] = clampCast<int8_t>(quantize(values[static_cast<size_t>(i)]));
      for (int64_t i = n; i < count; ++i) dst_ptr[i] = 0;
      return;
    }
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8: {
      auto* dst_ptr = reinterpret_cast<uint8_t*>(dst.host.data());
      for (int64_t i = 0; i < n; ++i) dst_ptr[i] = clampCast<uint8_t>(quantize(values[static_cast<size_t>(i)]));
      for (int64_t i = n; i < count; ++i) dst_ptr[i] = 0;
      return;
    }
    default:
      throw std::runtime_error("unsupported dtype for float fill: " + qnnDtypeName(dtype));
  }
}

bool sameShape(const std::vector<uint32_t>& dims, std::initializer_list<uint32_t> expect) {
  if (dims.size() != expect.size()) return false;
  size_t i = 0;
  for (auto d : expect) {
    if (dims[i++] != d) return false;
  }
  return true;
}

}  // namespace

TargetTensorBinding buildTargetTensorBinding(const Graph& graph, int num_layers) {
  TargetTensorBinding binding;
  binding.input_ids = findExact(graph.inputs, "input_ids");
  binding.position_ids = findExact(graph.inputs, "position_ids");
  binding.attention_mask = findExact(graph.inputs, "attention_mask");
  if (binding.input_ids == kNoTensor && !graph.inputs.empty()) binding.input_ids = 0;
  if (binding.position_ids == kNoTensor && graph.inputs.size() > 1) binding.position_ids = 1;
  if (binding.attention_mask == kNoTensor && graph.inputs.size() > 2) binding.attention_mask = 2;

  binding.past_key_inputs = collectByPrefix(graph.inputs, "past_key_");
  binding.past_value_inputs = collectByPrefix(graph.inputs, "past_value_");
  if (binding.past_key_inputs.empty() && graph.inputs.size() >= static_cast<size_t>(3 + num_layers * 2)) {
    for (int i = 0; i < num_layers; ++i) {
      binding.past_key_inputs.push_back(static_cast<size_t>(3 + i));
      binding.past_value_inputs.push_back(static_cast<size_t>(3 + num_layers + i));
    }
  }

  binding.logits_output = graph.outputs.empty() ? kNoTensor : 0;
  binding.present_key_outputs = collectByPrefix(graph.outputs, "present_key_");
  binding.present_value_outputs = collectByPrefix(graph.outputs, "present_value_");
  if (binding.present_key_outputs.empty() && graph.outputs.size() >= static_cast<size_t>(1 + num_layers * 2)) {
    for (int i = 0; i < num_layers; ++i) {
      binding.present_key_outputs.push_back(static_cast<size_t>(1 + i));
      binding.present_value_outputs.push_back(static_cast<size_t>(1 + num_layers + i));
    }
  }

  for (size_t i = 1 + binding.present_key_outputs.size() + binding.present_value_outputs.size();
       i < graph.outputs.size(); ++i) {
    binding.extra_outputs.push_back(i);
  }
  return binding;
}

DraftTensorBinding buildDraftTensorBinding(const Graph& graph, const DraftTensorGeometry& geom) {
  DraftTensorBinding binding;
  binding.cache_aware = graph.inputs.size() == static_cast<size_t>(8 + geom.num_layers * geom.num_heads * 2) &&
                        graph.outputs.size() == static_cast<size_t>(1 + geom.num_layers * geom.num_heads * 2);
  for (size_t i = 0; i < graph.inputs.size(); ++i) {
    const auto& desc = graph.inputs[i];
    const auto n = numel(desc.dims);
    if (desc.dtype == QNN_DATATYPE_INT_32 || n == geom.context_len + geom.block) {
      binding.position_input = i;
    } else if (n == static_cast<int64_t>(geom.block) * geom.draft_hidden) {
      binding.noise_input = i;
    } else if (n == static_cast<int64_t>(geom.context_len) * geom.target_hidden) {
      binding.target_hidden_input = i;
    } else if (sameShape(desc.dims, {static_cast<uint32_t>(geom.cache_ar_len), static_cast<uint32_t>(geom.head_dim)})) {
      binding.full_rope_inputs.push_back(i);
    } else if (sameShape(desc.dims, {static_cast<uint32_t>(geom.block), static_cast<uint32_t>(geom.head_dim)})) {
      binding.q_rope_inputs.push_back(i);
    } else if ((sameShape(desc.dims, {static_cast<uint32_t>(geom.block), static_cast<uint32_t>(geom.context_len + geom.block)}) ||
                desc.dims.size() == 4) &&
               binding.mask_scale_inputs.size() < 2) {
      binding.mask_scale_inputs.push_back(i);
    }
  }

  if (binding.cache_aware) {
    binding.noise_input = 0;
    binding.target_hidden_input = 1;
    binding.position_input = kNoTensor;
    binding.full_rope_inputs = {2, 3};
    binding.q_rope_inputs = {4, 5};
    binding.mask_scale_inputs = {6, 7};
  }

  binding.output0 = graph.outputs.empty() ? kNoTensor : 0;
  for (size_t i = 1; i < graph.outputs.size(); ++i) {
    const auto n = numel(graph.outputs[i].dims);
    if (n == static_cast<int64_t>(geom.block) * geom.draft_hidden && binding.hidden_output == kNoTensor) {
      binding.hidden_output = i;
    } else if (!binding.cache_aware) {
      binding.tap_outputs.push_back(i);
    }
  }
  return binding;
}

std::vector<int32_t> makePrefillAttentionMap(int ar_len) {
  std::vector<int32_t> map(ar_len);
  if (ar_len > 0) {
    map[0] = -1;
    for (int i = 1; i < ar_len; ++i) map[i] = i - 1;
  }
  return map;
}

std::vector<int32_t> makeVerifyAttentionMap(int ar_len) {
  std::vector<int32_t> map(ar_len);
  std::iota(map.begin(), map.end(), -1);
  return map;
}

void fillInt32Buffer(TensorBuffer& dst, const std::vector<int32_t>& values) {
  copyVectorToBuffer(dst, values);
}

void fillFloatBuffer(TensorBuffer& dst, const std::vector<float>& values) {
  fillQuantizedFloatBuffer(dst, values);
}

void fillUInt16Buffer(TensorBuffer& dst, const std::vector<uint16_t>& values) {
  copyVectorToBuffer(dst, values);
}

void fillBlockInputs(std::vector<TensorBuffer>& inputs,
                     size_t input_ids_index,
                     size_t position_ids_index,
                     const std::vector<int64_t>& block,
                     int64_t start_pos,
                     int ar_len) {
  if (input_ids_index == kNoTensor || position_ids_index == kNoTensor) return;
  std::vector<int32_t> ids(static_cast<size_t>(ar_len));
  std::vector<int32_t> pos(static_cast<size_t>(ar_len));
  for (int i = 0; i < ar_len; ++i) {
    ids[static_cast<size_t>(i)] = static_cast<int32_t>(i < static_cast<int>(block.size()) ? block[static_cast<size_t>(i)] : 0);
    pos[static_cast<size_t>(i)] = static_cast<int32_t>(start_pos + i);
  }
  fillInt32Buffer(inputs[input_ids_index], ids);
  fillInt32Buffer(inputs[position_ids_index], pos);
}

void fillPositionIdsInput(std::vector<TensorBuffer>& inputs,
                          size_t position_index,
                          int64_t position_base,
                          int positions) {
  if (position_index == kNoTensor || position_index >= inputs.size()) return;
  std::vector<int32_t> pos(static_cast<size_t>(std::max(0, positions)));
  for (int i = 0; i < positions; ++i) {
    pos[static_cast<size_t>(i)] = static_cast<int32_t>(position_base + i);
  }
  fillInt32Buffer(inputs[position_index], pos);
}

void fillAttentionMaskBuffer(TensorBuffer& dst,
                             const std::vector<int32_t>& attention_map,
                             int ar_len,
                             int n_past) {
  if (dst.host.empty()) return;
  const int context_len = static_cast<int>(dst.bytes() / sizeof(uint16_t) / static_cast<size_t>(ar_len));
  std::vector<uint16_t> mask(static_cast<size_t>(ar_len) * static_cast<size_t>(context_len), 0);
  constexpr uint16_t pos_val = 65535;
  for (int i = 0; i < ar_len; ++i) {
    uint16_t* past_ptr = mask.data() + static_cast<size_t>(i) * context_len;
    uint16_t* new_ptr = past_ptr + (context_len - ar_len);
    if (attention_map[i] < 0) {
      std::fill_n(past_ptr, n_past, pos_val);
    } else {
      const int pidx = attention_map[i];
      const uint16_t* parent_ptr = mask.data() + static_cast<size_t>(pidx) * context_len;
      std::memcpy(past_ptr, parent_ptr, static_cast<size_t>(context_len) * sizeof(uint16_t));
    }
    new_ptr[i] = pos_val;
  }
  fillUInt16Buffer(dst, mask);
}

void fillDraftNoiseInput(std::vector<TensorBuffer>& inputs, size_t noise_index, const std::vector<float>& noise) {
  if (noise_index == kNoTensor || noise_index >= inputs.size()) return;
  fillFloatBuffer(inputs[noise_index], noise);
}

void fillDraftTargetHiddenInput(std::vector<TensorBuffer>& inputs,
                                size_t target_hidden_index,
                                const std::vector<float>& history,
                                int64_t start_token,
                                int count,
                                int target_hidden,
                                int cache_delta) {
  if (target_hidden_index == kNoTensor || target_hidden_index >= inputs.size()) return;
  count = std::max(0, std::min(count, cache_delta));
  auto& dst = inputs[target_hidden_index];
  const int64_t history_tokens = static_cast<int64_t>(history.size()) / target_hidden;

  if (dst.desc.dtype == QNN_DATATYPE_FLOAT_32) {
    auto* out = reinterpret_cast<float*>(dst.host.data());
    const size_t out_count = dst.host.size() / sizeof(float);
    std::fill(out, out + out_count, 0.0f);
    const size_t row_elems = static_cast<size_t>(target_hidden);
    for (int t = 0; t < count && start_token + t < history_tokens; ++t) {
      const size_t dst_offset = static_cast<size_t>(t) * row_elems;
      if (dst_offset >= out_count) break;
      const size_t copy_elems = std::min(row_elems, out_count - dst_offset);
      const float* src = history.data() + (start_token + t) * static_cast<int64_t>(target_hidden);
      std::copy(src, src + copy_elems, out + dst_offset);
    }
    return;
  }

  std::vector<float> packed(static_cast<size_t>(cache_delta) * static_cast<size_t>(target_hidden), 0.0f);
  for (int t = 0; t < count && start_token + t < history_tokens; ++t) {
    const float* src = history.data() + (start_token + t) * static_cast<int64_t>(target_hidden);
    float* dst_row = packed.data() + static_cast<size_t>(t) * static_cast<size_t>(target_hidden);
    std::copy(src, src + target_hidden, dst_row);
  }
  fillFloatBuffer(dst, packed);
}

void fillDraftStandardRopeInputs(std::vector<TensorBuffer>& inputs,
                                 const DraftTensorBinding& binding,
                                 int64_t position_base,
                                 const DraftTensorGeometry& geom) {
  if (binding.cache_aware) return;
  if (binding.position_input != kNoTensor) {
    fillPositionIdsInput(inputs, binding.position_input, position_base, geom.context_len + geom.block);
  }
  fillDraftRopeInputs(inputs, binding, position_base, geom.block, geom);
}

void fillDraftStandardMaskScaleInputs(std::vector<TensorBuffer>& inputs,
                                     const DraftTensorBinding& binding,
                                     const DraftTensorGeometry& geom) {
  if (binding.cache_aware) return;
  fillDraftMaskScaleInputs(inputs, binding, geom.block, true, geom);
}

std::vector<float> buildRopeForPositions(int64_t position_base,
                                         int positions,
                                         int head_dim,
                                         bool want_sin,
                                         float rope_theta) {
  std::vector<float> rope(static_cast<size_t>(positions) * static_cast<size_t>(head_dim), 0.0f);
  for (int i = 0; i < positions; ++i) {
    for (int j = 0; j < head_dim / 2; ++j) {
      const float inv = std::pow(rope_theta, -2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
      const float angle = static_cast<float>(position_base + i) * inv;
      const float v = want_sin ? std::sin(angle) : std::cos(angle);
      rope[static_cast<size_t>(i) * head_dim + j] = v;
      rope[static_cast<size_t>(i) * head_dim + j + head_dim / 2] = v;
    }
  }
  return rope;
}

std::vector<float> buildRopeForCacheCurrent(int64_t position_base,
                                            int delta_count,
                                            int head_dim,
                                            int cache_ar_len,
                                            int cache_delta,
                                            bool want_sin,
                                            float rope_theta) {
  std::vector<float> rope(static_cast<size_t>(cache_ar_len) * static_cast<size_t>(head_dim), 0.0f);
  delta_count = std::max(0, std::min(delta_count, cache_delta));
  for (int i = 0; i < cache_ar_len; ++i) {
    const int64_t pos = (i < cache_delta)
                            ? position_base + i
                            : position_base + delta_count + (i - cache_delta);
    for (int j = 0; j < head_dim / 2; ++j) {
      const float inv = std::pow(rope_theta, -2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
      const float angle = static_cast<float>(pos) * inv;
      const float v = want_sin ? std::sin(angle) : std::cos(angle);
      rope[static_cast<size_t>(i) * head_dim + j] = v;
      rope[static_cast<size_t>(i) * head_dim + j + head_dim / 2] = v;
    }
  }
  return rope;
}

void fillDraftRopeInputs(std::vector<TensorBuffer>& inputs,
                         const DraftTensorBinding& binding,
                         int64_t position_base,
                         int delta_count,
                         const DraftTensorGeometry& geom) {
  if (!binding.cache_aware) return;
  for (size_t i = 0; i < binding.full_rope_inputs.size(); ++i) {
    const auto rope = buildRopeForCacheCurrent(position_base,
                                               delta_count,
                                               geom.head_dim,
                                               geom.cache_ar_len,
                                               geom.cache_delta,
                                               i == 0,
                                               geom.rope_theta);
    fillFloatBuffer(inputs[binding.full_rope_inputs[i]], rope);
  }
  for (size_t i = 0; i < binding.q_rope_inputs.size(); ++i) {
    const auto rope = buildRopeForPositions(position_base + delta_count,
                                            geom.block,
                                            geom.head_dim,
                                            i == 0,
                                            geom.rope_theta);
    fillFloatBuffer(inputs[binding.q_rope_inputs[i]], rope);
  }
}

void fillDraftMaskScaleInputs(std::vector<TensorBuffer>& inputs,
                              const DraftTensorBinding& binding,
                              int delta_count,
                              bool include_noise,
                              const DraftTensorGeometry& geom) {
  if (binding.mask_scale_inputs.empty()) return;
  const int total_len = geom.cache_context + geom.cache_ar_len;
  std::vector<float> mask(static_cast<size_t>(geom.block) * static_cast<size_t>(total_len), -20.0f);
  delta_count = std::max(0, std::min(delta_count, geom.cache_delta));
  for (int row = 0; row < geom.block; ++row) {
    float* dst = mask.data() + static_cast<size_t>(row) * static_cast<size_t>(total_len);
    for (int i = 0; i < geom.cache_delta + delta_count; ++i) {
      if (i < geom.cache_context) dst[i] = 0.0f;
    }
    if (include_noise) {
      for (int i = 0; i < geom.block; ++i) {
        dst[geom.cache_context + geom.cache_delta + i] = 0.0f;
      }
    }
  }
  std::vector<float> scale(mask.size(), 1.0f / std::sqrt(static_cast<float>(geom.head_dim)));
  fillFloatBuffer(inputs[binding.mask_scale_inputs[0]], mask);
  if (binding.mask_scale_inputs.size() > 1) {
    fillFloatBuffer(inputs[binding.mask_scale_inputs[1]], scale);
  }
}

int64_t tensorElementCount(const TensorBuffer& buffer) {
  int64_t n = 1;
  for (auto d : buffer.desc.dims) n *= static_cast<int64_t>(d);
  return n;
}

std::vector<float> tensorBufferToFloat(const TensorBuffer& buffer) {
  const auto dtype = buffer.desc.dtype;
  const auto quant = buffer.desc.quant;
  const int64_t n = tensorElementCount(buffer);
  std::vector<float> out(static_cast<size_t>(std::max<int64_t>(0, n)), 0.0f);
  switch (dtype) {
    case QNN_DATATYPE_FLOAT_32: {
      const auto* ptr = reinterpret_cast<const float*>(buffer.host.data());
      std::copy(ptr, ptr + n, out.begin());
      return out;
    }
    case QNN_DATATYPE_INT_32: {
      const auto* ptr = reinterpret_cast<const int32_t*>(buffer.host.data());
      for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = dequantRaw(ptr[i], dtype, quant);
      return out;
    }
    case QNN_DATATYPE_UINT_32: {
      const auto* ptr = reinterpret_cast<const uint32_t*>(buffer.host.data());
      for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = dequantRaw(ptr[i], dtype, quant);
      return out;
    }
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_BFLOAT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16: {
      const auto* ptr = reinterpret_cast<const uint16_t*>(buffer.host.data());
      for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = dequantRaw(ptr[i], dtype, quant);
      return out;
    }
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_SFIXED_POINT_16: {
      const auto* ptr = reinterpret_cast<const int16_t*>(buffer.host.data());
      for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = dequantRaw(ptr[i], dtype, quant);
      return out;
    }
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8: {
      const auto* ptr = reinterpret_cast<const uint8_t*>(buffer.host.data());
      for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = dequantRaw(ptr[i], dtype, quant);
      return out;
    }
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_SFIXED_POINT_8: {
      const auto* ptr = reinterpret_cast<const int8_t*>(buffer.host.data());
      for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = dequantRaw(ptr[i], dtype, quant);
      return out;
    }
    default:
      throw std::runtime_error("unsupported dtype for output read: " + qnnDtypeName(dtype));
  }
}

float readTensorValueAtRaw(const TensorBuffer& buffer, int64_t flat_index) {
  const int64_t n = tensorElementCount(buffer);
  if (flat_index < 0 || flat_index >= n) return 0.0f;
  switch (buffer.desc.dtype) {
    case QNN_DATATYPE_FLOAT_32:
      return reinterpret_cast<const float*>(buffer.host.data())[flat_index];
    case QNN_DATATYPE_INT_32:
      return dequantRaw(reinterpret_cast<const int32_t*>(buffer.host.data())[flat_index],
                        buffer.desc.dtype,
                        buffer.desc.quant);
    case QNN_DATATYPE_UINT_32:
      return dequantRaw(reinterpret_cast<const uint32_t*>(buffer.host.data())[flat_index],
                        buffer.desc.dtype,
                        buffer.desc.quant);
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_BFLOAT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      return dequantRaw(reinterpret_cast<const uint16_t*>(buffer.host.data())[flat_index],
                        buffer.desc.dtype,
                        buffer.desc.quant);
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
      return dequantRaw(reinterpret_cast<const int16_t*>(buffer.host.data())[flat_index],
                        buffer.desc.dtype,
                        buffer.desc.quant);
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
      return dequantRaw(reinterpret_cast<const uint8_t*>(buffer.host.data())[flat_index],
                        buffer.desc.dtype,
                        buffer.desc.quant);
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
      return dequantRaw(reinterpret_cast<const int8_t*>(buffer.host.data())[flat_index],
                        buffer.desc.dtype,
                        buffer.desc.quant);
    default:
      throw std::runtime_error("unsupported dtype for output read: " + qnnDtypeName(buffer.desc.dtype));
  }
}

void copyTensorRowToFloat(const TensorBuffer& buffer, int64_t row_base, float* dst, int count) {
  if (!dst || count <= 0) return;
  const int64_t n = tensorElementCount(buffer);
  if (row_base < 0 || row_base >= n) {
    std::fill(dst, dst + count, 0.0f);
    return;
  }
  const int64_t safe_count = std::min<int64_t>(count, n - row_base);
  const auto dtype = buffer.desc.dtype;
  const auto quant = buffer.desc.quant;
  switch (dtype) {
    case QNN_DATATYPE_FLOAT_32: {
      const auto* ptr = reinterpret_cast<const float*>(buffer.host.data()) + row_base;
      std::copy(ptr, ptr + safe_count, dst);
      break;
    }
    case QNN_DATATYPE_INT_32: {
      const auto* ptr = reinterpret_cast<const int32_t*>(buffer.host.data()) + row_base;
      for (int64_t i = 0; i < safe_count; ++i) dst[i] = dequantRaw(ptr[i], dtype, quant);
      break;
    }
    case QNN_DATATYPE_UINT_32: {
      const auto* ptr = reinterpret_cast<const uint32_t*>(buffer.host.data()) + row_base;
      for (int64_t i = 0; i < safe_count; ++i) dst[i] = dequantRaw(ptr[i], dtype, quant);
      break;
    }
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_BFLOAT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16: {
      const auto* ptr = reinterpret_cast<const uint16_t*>(buffer.host.data()) + row_base;
      if (dtype == QNN_DATATYPE_FLOAT_16) {
        for (int64_t i = 0; i < safe_count; ++i) dst[i] = fp16ToFloat(ptr[i]);
      } else if (dtype == QNN_DATATYPE_BFLOAT_16) {
        for (int64_t i = 0; i < safe_count; ++i) dst[i] = bf16ToFloat(ptr[i]);
      } else if (hasScaleOffset(quant)) {
        const float scale = quant.scaleOffsetEncoding.scale;
        const int32_t offset = quant.scaleOffsetEncoding.offset;
        for (int64_t i = 0; i < safe_count; ++i) {
          dst[i] = (static_cast<float>(ptr[i]) + static_cast<float>(offset)) * scale;
        }
      } else {
        for (int64_t i = 0; i < safe_count; ++i) dst[i] = static_cast<float>(ptr[i]);
      }
      break;
    }
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_SFIXED_POINT_16: {
      const auto* ptr = reinterpret_cast<const int16_t*>(buffer.host.data()) + row_base;
      for (int64_t i = 0; i < safe_count; ++i) dst[i] = dequantRaw(ptr[i], dtype, quant);
      break;
    }
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8: {
      const auto* ptr = reinterpret_cast<const uint8_t*>(buffer.host.data()) + row_base;
      for (int64_t i = 0; i < safe_count; ++i) dst[i] = dequantRaw(ptr[i], dtype, quant);
      break;
    }
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_SFIXED_POINT_8: {
      const auto* ptr = reinterpret_cast<const int8_t*>(buffer.host.data()) + row_base;
      for (int64_t i = 0; i < safe_count; ++i) dst[i] = dequantRaw(ptr[i], dtype, quant);
      break;
    }
    default:
      throw std::runtime_error("unsupported dtype for row copy: " + qnnDtypeName(dtype));
  }
  if (safe_count < count) {
    std::fill(dst + safe_count, dst + count, 0.0f);
  }
}

void argmaxRowsAtRaw(const TensorBuffer& buffer,
                     int first_seq_index,
                     int row_count,
                     int vocab_size,
                     int64_t* dst) {
  if (!dst || first_seq_index < 0 || row_count <= 0 || vocab_size <= 0) return;
  const int64_t n = tensorElementCount(buffer);
  const int64_t first_base = static_cast<int64_t>(first_seq_index) * vocab_size;
  if (first_base < 0 || first_base >= n) {
    std::fill(dst, dst + row_count, 0);
    return;
  }
  const int rows = static_cast<int>(std::min<int64_t>(
      row_count, std::max<int64_t>(0, (n - first_base) / vocab_size)));
  const auto dtype = buffer.desc.dtype;
  const auto quant = buffer.desc.quant;

  switch (dtype) {
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16: {
      const auto* ptr = reinterpret_cast<const uint16_t*>(buffer.host.data()) + first_base;
      for (int r = 0; r < rows; ++r) {
        const uint16_t* row = ptr + static_cast<int64_t>(r) * vocab_size;
#if defined(__aarch64__)
        dst[r] = argmaxU16Neon(row, vocab_size);
#else
        const auto* max_it = std::max_element(row, row + vocab_size);
        dst[r] = static_cast<int64_t>(std::distance(row, max_it));
#endif
      }
      break;
    }
    case QNN_DATATYPE_FLOAT_32: {
      const auto* ptr = reinterpret_cast<const float*>(buffer.host.data()) + first_base;
      for (int r = 0; r < rows; ++r) {
        const float* row = ptr + static_cast<int64_t>(r) * vocab_size;
        const auto* max_it = std::max_element(row, row + vocab_size);
        dst[r] = static_cast<int64_t>(std::distance(row, max_it));
      }
      break;
    }
    default:
      for (int r = 0; r < rows; ++r) {
        dst[r] = argmaxTensorAtRaw(buffer, first_seq_index + r, vocab_size);
      }
      break;
  }
  if (rows < row_count) {
    std::fill(dst + rows, dst + row_count, 0);
  }
}

int64_t readTokenIdAtRaw(const TensorBuffer& buffer, int seq_index) {
  const int64_t n = tensorElementCount(buffer);
  if (seq_index < 0 || seq_index >= n) return 0;
  switch (buffer.desc.dtype) {
    case QNN_DATATYPE_INT_32:
      return reinterpret_cast<const int32_t*>(buffer.host.data())[seq_index];
    case QNN_DATATYPE_UINT_32:
      return reinterpret_cast<const uint32_t*>(buffer.host.data())[seq_index];
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
      return reinterpret_cast<const int16_t*>(buffer.host.data())[seq_index];
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      return reinterpret_cast<const uint16_t*>(buffer.host.data())[seq_index];
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
      return reinterpret_cast<const int8_t*>(buffer.host.data())[seq_index];
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
      return reinterpret_cast<const uint8_t*>(buffer.host.data())[seq_index];
    default:
      return static_cast<int64_t>(std::llround(tensorBufferToFloat(buffer)[static_cast<size_t>(seq_index)]));
  }
}

int64_t argmaxTensorAtRaw(const TensorBuffer& buffer, int seq_index, int vocab_size) {
  if (seq_index < 0 || vocab_size <= 0) return 0;
  const int64_t base = static_cast<int64_t>(seq_index) * vocab_size;
  const int64_t n = tensorElementCount(buffer);
  if (base < 0 || base + vocab_size > n) return 0;
  const auto dtype = buffer.desc.dtype;
  const auto quant = buffer.desc.quant;
  int64_t best = 0;
  float best_v = -std::numeric_limits<float>::infinity();
  auto update = [&](int64_t token, float value) {
    if (value > best_v) {
      best_v = value;
      best = token;
    }
  };
  switch (dtype) {
    case QNN_DATATYPE_FLOAT_32: {
      const auto* ptr = reinterpret_cast<const float*>(buffer.host.data()) + base;
      for (int i = 0; i < vocab_size; ++i) update(i, ptr[i]);
      return best;
    }
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_BFLOAT_16: {
      const auto* ptr = reinterpret_cast<const uint16_t*>(buffer.host.data()) + base;
      for (int i = 0; i < vocab_size; ++i) update(i, dequantRaw(ptr[i], dtype, quant));
      return best;
    }
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16: {
      const auto* ptr = reinterpret_cast<const uint16_t*>(buffer.host.data()) + base;
#if defined(__aarch64__)
      return argmaxU16Neon(ptr, vocab_size);
#else
      const auto* max_it = std::max_element(ptr, ptr + vocab_size);
      return static_cast<int64_t>(std::distance(ptr, max_it));
#endif
    }
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_SFIXED_POINT_16: {
      const auto* ptr = reinterpret_cast<const int16_t*>(buffer.host.data()) + base;
      for (int i = 0; i < vocab_size; ++i) update(i, dequantRaw(ptr[i], dtype, quant));
      return best;
    }
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8: {
      const auto* ptr = reinterpret_cast<const uint8_t*>(buffer.host.data()) + base;
      for (int i = 0; i < vocab_size; ++i) update(i, dequantRaw(ptr[i], dtype, quant));
      return best;
    }
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_SFIXED_POINT_8: {
      const auto* ptr = reinterpret_cast<const int8_t*>(buffer.host.data()) + base;
      for (int i = 0; i < vocab_size; ++i) update(i, dequantRaw(ptr[i], dtype, quant));
      return best;
    }
    case QNN_DATATYPE_INT_32: {
      const auto* ptr = reinterpret_cast<const int32_t*>(buffer.host.data()) + base;
      for (int i = 0; i < vocab_size; ++i) update(i, dequantRaw(ptr[i], dtype, quant));
      return best;
    }
    case QNN_DATATYPE_UINT_32: {
      const auto* ptr = reinterpret_cast<const uint32_t*>(buffer.host.data()) + base;
      for (int i = 0; i < vocab_size; ++i) update(i, dequantRaw(ptr[i], dtype, quant));
      return best;
    }
    default:
      throw std::runtime_error("unsupported dtype for argmax: " + qnnDtypeName(dtype));
  }
}

RankAcceptResult rankAcceptTensorAtRaw(const TensorBuffer& buffer,
                                       int seq_index,
                                       int vocab_size,
                                       int token,
                                       int top_k) {
  RankAcceptResult result;
  if (seq_index < 0 || vocab_size <= 0 || token < 0 || token >= vocab_size || top_k <= 0) {
    return result;
  }
  const int64_t base = static_cast<int64_t>(seq_index) * vocab_size;
  const int64_t n = tensorElementCount(buffer);
  if (base < 0 || base + vocab_size > n) return result;
  const auto dtype = buffer.desc.dtype;
  const auto quant = buffer.desc.quant;

  if (dtype == QNN_DATATYPE_FLOAT_32) {
    const auto* ptr = reinterpret_cast<const float*>(buffer.host.data()) + base;
    const float candidate = ptr[token];
    int64_t best = 0;
    float best_v = ptr[0];
    int greater = 0;
    for (int i = 0; i < vocab_size; ++i) {
      const float v = ptr[i];
      if (v > best_v) {
        best_v = v;
        best = i;
      }
      if (v > candidate) ++greater;
    }
    result.argmax = best;
    result.greater_count = greater;
    result.accepted = top_k <= 1 ? best == token : greater < top_k;
    return result;
  }

  if (dtype == QNN_DATATYPE_UINT_16 || dtype == QNN_DATATYPE_UFIXED_POINT_16) {
    const auto* ptr = reinterpret_cast<const uint16_t*>(buffer.host.data()) + base;
    const uint16_t candidate = ptr[token];
#if defined(__aarch64__)
    result.argmax = argmaxU16Neon(ptr, vocab_size);
    result.greater_count = countGreaterU16Neon(ptr, vocab_size, candidate);
    result.accepted = top_k <= 1 ? result.argmax == token : result.greater_count < top_k;
    return result;
#else
    int64_t best = 0;
    uint16_t best_v = ptr[0];
    int greater = 0;
    for (int i = 0; i < vocab_size; ++i) {
      const uint16_t v = ptr[i];
      if (v > best_v) {
        best_v = v;
        best = i;
      }
      if (v > candidate) ++greater;
    }
    result.argmax = best;
    result.greater_count = greater;
    result.accepted = top_k <= 1 ? best == token : greater < top_k;
    return result;
#endif
  }

  auto read_value = [&](int idx) -> float {
    const int64_t offset = base + idx;
    switch (dtype) {
      case QNN_DATATYPE_FLOAT_16:
      case QNN_DATATYPE_BFLOAT_16:
        return dequantRaw(reinterpret_cast<const uint16_t*>(buffer.host.data())[offset], dtype, quant);
      case QNN_DATATYPE_INT_16:
      case QNN_DATATYPE_SFIXED_POINT_16:
        return dequantRaw(reinterpret_cast<const int16_t*>(buffer.host.data())[offset], dtype, quant);
      case QNN_DATATYPE_UINT_8:
      case QNN_DATATYPE_UFIXED_POINT_8:
        return dequantRaw(reinterpret_cast<const uint8_t*>(buffer.host.data())[offset], dtype, quant);
      case QNN_DATATYPE_INT_8:
      case QNN_DATATYPE_SFIXED_POINT_8:
        return dequantRaw(reinterpret_cast<const int8_t*>(buffer.host.data())[offset], dtype, quant);
      case QNN_DATATYPE_INT_32:
        return dequantRaw(reinterpret_cast<const int32_t*>(buffer.host.data())[offset], dtype, quant);
      case QNN_DATATYPE_UINT_32:
        return dequantRaw(reinterpret_cast<const uint32_t*>(buffer.host.data())[offset], dtype, quant);
      default:
        throw std::runtime_error("unsupported dtype for rank accept: " + qnnDtypeName(dtype));
    }
  };

  const float candidate = read_value(token);
  int64_t best = 0;
  float best_v = read_value(0);
  int greater = 0;
  for (int i = 0; i < vocab_size; ++i) {
    const float v = read_value(i);
    if (v > best_v) {
      best_v = v;
      best = i;
    }
    if (v > candidate) ++greater;
  }
  result.argmax = best;
  result.greater_count = greater;
  result.accepted = top_k <= 1 ? best == token : greater < top_k;
  return result;
}

bool tokenInTopKAtRaw(const TensorBuffer& buffer, int seq_index, int vocab_size, int token, int top_k) {
  if (seq_index < 0 || vocab_size <= 0 || token < 0 || token >= vocab_size) return false;
  if (top_k <= 1) return argmaxTensorAtRaw(buffer, seq_index, vocab_size) == token;
  const int64_t base = static_cast<int64_t>(seq_index) * vocab_size;
  const int64_t n = tensorElementCount(buffer);
  if (base < 0 || base + vocab_size > n) return false;
  const auto dtype = buffer.desc.dtype;
  const auto quant = buffer.desc.quant;

  auto read_value = [&](int idx) -> float {
    const int64_t offset = base + idx;
    switch (dtype) {
      case QNN_DATATYPE_FLOAT_32:
        return reinterpret_cast<const float*>(buffer.host.data())[offset];
      case QNN_DATATYPE_FLOAT_16:
      case QNN_DATATYPE_BFLOAT_16:
      case QNN_DATATYPE_UINT_16:
      case QNN_DATATYPE_UFIXED_POINT_16:
        return dequantRaw(reinterpret_cast<const uint16_t*>(buffer.host.data())[offset], dtype, quant);
      case QNN_DATATYPE_INT_16:
      case QNN_DATATYPE_SFIXED_POINT_16:
        return dequantRaw(reinterpret_cast<const int16_t*>(buffer.host.data())[offset], dtype, quant);
      case QNN_DATATYPE_UINT_8:
      case QNN_DATATYPE_UFIXED_POINT_8:
        return dequantRaw(reinterpret_cast<const uint8_t*>(buffer.host.data())[offset], dtype, quant);
      case QNN_DATATYPE_INT_8:
      case QNN_DATATYPE_SFIXED_POINT_8:
        return dequantRaw(reinterpret_cast<const int8_t*>(buffer.host.data())[offset], dtype, quant);
      case QNN_DATATYPE_INT_32:
        return dequantRaw(reinterpret_cast<const int32_t*>(buffer.host.data())[offset], dtype, quant);
      case QNN_DATATYPE_UINT_32:
        return dequantRaw(reinterpret_cast<const uint32_t*>(buffer.host.data())[offset], dtype, quant);
      default:
        throw std::runtime_error("unsupported dtype for top-k: " + qnnDtypeName(dtype));
    }
  };

  if (dtype == QNN_DATATYPE_UINT_16 || dtype == QNN_DATATYPE_UFIXED_POINT_16) {
    const auto* ptr = reinterpret_cast<const uint16_t*>(buffer.host.data()) + base;
    const uint16_t candidate = ptr[token];
    int greater = 0;
    for (int i = 0; i < vocab_size; ++i) {
      if (i == token) continue;
      if (ptr[i] > candidate) {
        ++greater;
        if (greater >= top_k) return false;
      }
    }
    return true;
  }

  const float candidate = read_value(token);
  int greater = 0;
  for (int i = 0; i < vocab_size; ++i) {
    if (i == token) continue;
    if (read_value(i) > candidate) {
      ++greater;
      if (greater >= top_k) return false;
    }
  }
  return true;
}

}  // namespace dflash
