#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dflash/qwen3_tokenizer.hpp"
#include "dflash/runtime_session.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string target;
  std::string target_prefill;
  std::string target_verify;
  std::string draft;
  std::string embedding;
  std::string tokens;
  std::string prompt_file;
  std::string tokenizer;
  std::string prefill_graph;
  std::string verify_graph;
  std::string draft_graph;
  std::string target_kv_dtype = "uint8";
  int target_layers = 36;
  int draft_layers = 5;
  int kv_heads = 8;
  int head_dim = 128;
  int draft_hidden = 2560;
  int target_hidden = 12800;
  int vocab = 151936;
  int context_len = 1024;
  int prefill_seq = 32;
  int verify_seq = 16;
  int block = 16;
  int max_new = 128;
  int accept_topk = 5;
  int mask_token = 151669;
};

struct HostTiming {
  double draft_noise_embed = 0.0;
  double draft_sync_host = 0.0;
  double draft_input_fill = 0.0;
  double draft_cache_append = 0.0;
  double draft_argmax = 0.0;
  double target_cache_rearrange = 0.0;
  double target_cache_fill_inputs = 0.0;
  double target_input_fill = 0.0;
  double target_cache_read_outputs = 0.0;
  double target_cache_update = 0.0;
  double append_hidden = 0.0;
  double output_commit = 0.0;
};

double elapsedMs(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

void usage(const char* argv0) {
  std::cerr
      << "usage: " << argv0 << " [options]\n"
      << "  --target <context.bin> | --target_prefill <bin> --target_verify <bin>\n"
      << "  --draft <context.bin>\n"
      << "  --embedding <embedding.fp16.bin>\n"
      << "  --tokens <ids.txt> | --prompt_file <prompt.txt> --tokenizer <tokenizer.json>\n"
      << "  --target_kv_dtype uint8|uint16\n"
      << "  --max_new <n> --block <n> --accept_topk <n>\n";
}

bool needValue(int argc, char** argv, int i) {
  if (i + 1 < argc) return true;
  std::cerr << "missing value for " << argv[i] << "\n";
  return false;
}

Options parseOptions(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto take_string = [&](std::string& dst) {
      if (!needValue(argc, argv, i)) throw std::runtime_error("bad arguments");
      dst = argv[++i];
    };
    auto take_int = [&](int& dst) {
      if (!needValue(argc, argv, i)) throw std::runtime_error("bad arguments");
      dst = std::atoi(argv[++i]);
    };
    if (arg == "--target") take_string(opt.target);
    else if (arg == "--target_prefill") take_string(opt.target_prefill);
    else if (arg == "--target_verify") take_string(opt.target_verify);
    else if (arg == "--draft") take_string(opt.draft);
    else if (arg == "--embedding") take_string(opt.embedding);
    else if (arg == "--tokens") take_string(opt.tokens);
    else if (arg == "--prompt_file") take_string(opt.prompt_file);
    else if (arg == "--tokenizer") take_string(opt.tokenizer);
    else if (arg == "--prefill_graph") take_string(opt.prefill_graph);
    else if (arg == "--verify_graph") take_string(opt.verify_graph);
    else if (arg == "--draft_graph") take_string(opt.draft_graph);
    else if (arg == "--target_kv_dtype") take_string(opt.target_kv_dtype);
    else if (arg == "--target_layers") take_int(opt.target_layers);
    else if (arg == "--draft_layers") take_int(opt.draft_layers);
    else if (arg == "--kv_heads") take_int(opt.kv_heads);
    else if (arg == "--head_dim") take_int(opt.head_dim);
    else if (arg == "--draft_hidden") take_int(opt.draft_hidden);
    else if (arg == "--target_hidden") take_int(opt.target_hidden);
    else if (arg == "--vocab") take_int(opt.vocab);
    else if (arg == "--context_len") take_int(opt.context_len);
    else if (arg == "--prefill_seq") take_int(opt.prefill_seq);
    else if (arg == "--verify_seq") take_int(opt.verify_seq);
    else if (arg == "--block") take_int(opt.block);
    else if (arg == "--max_new") take_int(opt.max_new);
    else if (arg == "--accept_topk") take_int(opt.accept_topk);
    else if (arg == "--mask_token") take_int(opt.mask_token);
    else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "unknown option: " << arg << "\n";
      throw std::runtime_error("bad arguments");
    }
  }
  if (opt.target.empty() && (opt.target_prefill.empty() || opt.target_verify.empty())) {
    throw std::runtime_error("target context missing");
  }
  if (opt.draft.empty()) throw std::runtime_error("draft context missing");
  if (opt.tokens.empty() && opt.prompt_file.empty()) throw std::runtime_error("token input missing");
  if (!opt.prompt_file.empty() && opt.tokenizer.empty()) throw std::runtime_error("tokenizer missing for prompt_file");
  if (opt.prefill_graph.empty()) opt.prefill_graph = "model.0.s" + std::to_string(opt.prefill_seq);
  if (opt.verify_graph.empty()) opt.verify_graph = "model.0.s" + std::to_string(opt.verify_seq);
  return opt;
}

std::string readTextFile(const std::string& path) {
  std::ifstream fin(path);
  if (!fin) throw std::runtime_error("failed to open text file: " + path);
  std::stringstream ss;
  ss << fin.rdbuf();
  return ss.str();
}

std::vector<int64_t> readTokenIds(const std::string& path) {
  std::ifstream fin(path);
  if (!fin) throw std::runtime_error("failed to open token id file: " + path);
  std::stringstream ss;
  ss << fin.rdbuf();
  std::string text = ss.str();
  for (char& c : text) {
    if (c == ',' || c == '[' || c == ']') c = ' ';
  }
  std::vector<int64_t> ids;
  std::istringstream in(text);
  int64_t id = 0;
  while (in >> id) ids.push_back(id);
  if (ids.empty()) throw std::runtime_error("empty token id file: " + path);
  return ids;
}

std::vector<int64_t> makeDraftNoiseTokens(const std::vector<int64_t>& seed_tokens, int block, int mask_token) {
  std::vector<int64_t> tokens(static_cast<size_t>(block), mask_token);
  const int count = std::min(block, static_cast<int>(seed_tokens.size()));
  for (int i = 0; i < count; ++i) tokens[static_cast<size_t>(i)] = seed_tokens[static_cast<size_t>(i)];
  return tokens;
}

uint16_t fp32ToFp16(float f) {
  uint32_t x = 0;
  std::memcpy(&x, &f, sizeof(x));
  uint32_t sign = (x >> 31) & 1u;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127;
  uint32_t mant = x & 0x7fffff;
  if (exp == 128) return static_cast<uint16_t>((sign << 15) | (0x1f << 10) | (mant ? 1 : 0));
  if (exp < -14) return static_cast<uint16_t>(sign << 15);
  if (exp > 15) return static_cast<uint16_t>((sign << 15) | (0x1f << 10));
  return static_cast<uint16_t>((sign << 15) | ((exp + 15) << 10) | (mant >> 13));
}

float fp16ToFp32(uint16_t h) {
  int sign = (h & 0x8000) ? -1 : 1;
  int exp = (h >> 10) & 0x1f;
  int mant = h & 0x03ff;
  if (exp == 0) return sign * std::ldexp(static_cast<float>(mant), -24);
  if (exp == 31) return sign * std::numeric_limits<float>::infinity();
  return sign * std::ldexp(static_cast<float>(1024 + mant), exp - 25);
}

class EmbeddingTable {
 public:
  bool load(const std::string& path, int vocab, int hidden) {
    vocab_ = vocab;
    hidden_ = hidden;
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return false;
    const auto size = in.tellg();
    in.seekg(0, std::ios::beg);
    data_.resize(static_cast<size_t>(size) / sizeof(uint16_t));
    if (!in.read(reinterpret_cast<char*>(data_.data()), size)) return false;
    if (static_cast<int64_t>(data_.size()) < static_cast<int64_t>(vocab_) * hidden_) {
      throw std::runtime_error("embedding table is smaller than vocab*hidden");
    }
    return true;
  }

  std::vector<float> lookupBlock(const std::vector<int64_t>& tokens, int block) const {
    std::vector<float> out(static_cast<size_t>(block) * static_cast<size_t>(hidden_), 0.0f);
    for (int i = 0; i < block && i < static_cast<int>(tokens.size()); ++i) {
      int64_t tok = tokens[static_cast<size_t>(i)];
      if (tok < 0 || tok >= vocab_) continue;
      const uint16_t* src = data_.data() + static_cast<size_t>(tok) * static_cast<size_t>(hidden_);
      float* dst = out.data() + static_cast<size_t>(i) * static_cast<size_t>(hidden_);
      for (int j = 0; j < hidden_; ++j) dst[j] = fp16ToFp32(src[j]);
    }
    return out;
  }

 private:
  int vocab_ = 0;
  int hidden_ = 0;
  std::vector<uint16_t> data_;
};

bool loadRuntime(dflash::QnnRuntime& runtime, const std::string& path, const char* label) {
  const auto t0 = Clock::now();
  if (!runtime.open(QNN_LOG_LEVEL_ERROR) || !runtime.loadContextBinary(path)) {
    std::cerr << "[runtime] failed to load " << label << ": " << path << "\n";
    return false;
  }
  const auto t1 = Clock::now();
  std::cout << "[runtime] loaded " << label << "=" << path << " in " << elapsedMs(t0, t1) << " ms\n";
  return true;
}

std::string findDraftGraph(dflash::QnnRuntime& runtime, const Options& opt) {
  if (!opt.draft_graph.empty() && runtime.hasGraph(opt.draft_graph)) return opt.draft_graph;
  const std::vector<std::string> candidates = {
      "model.0.s" + std::to_string(opt.draft_hidden),
      "draft.0.s" + std::to_string(opt.block),
      "model.0.s2560",
      "model.0.s2048",
      "model.0.s1024",
  };
  for (const auto& name : candidates) {
    if (runtime.hasGraph(name)) return name;
  }
  return "";
}

void appendTargetHidden(const dflash::TargetGraphSession& session,
                        int ar_len,
                        int start_index,
                        int count,
                        int target_hidden,
                        std::vector<float>& history) {
  const auto& binding = session.binding();
  const auto& outputs = session.outputs();
  if (binding.extra_outputs.empty()) {
    history.resize(history.size() + static_cast<size_t>(count) * static_cast<size_t>(target_hidden), 0.0f);
    return;
  }
  std::vector<int> tap_dims;
  std::vector<const dflash::TensorBuffer*> tap_buffers;
  int sum_dim = 0;
  for (size_t idx : binding.extra_outputs) {
    if (idx >= outputs.size()) continue;
    const auto& out = outputs[idx];
    const int64_t elems = dflash::tensorElementCount(out);
    int dim = static_cast<int>(elems / std::max(1, ar_len));
    if (dim <= 0) continue;
    sum_dim += dim;
    tap_dims.push_back(dim);
    tap_buffers.push_back(&out);
  }
  if (sum_dim <= 0) {
    history.resize(history.size() + static_cast<size_t>(count) * static_cast<size_t>(target_hidden), 0.0f);
    return;
  }
  const size_t old_size = history.size();
  history.resize(old_size + static_cast<size_t>(count) * static_cast<size_t>(target_hidden), 0.0f);
  for (int t = 0; t < count; ++t) {
    const size_t base = old_size + static_cast<size_t>(t) * static_cast<size_t>(target_hidden);
    size_t offset = base;
    for (size_t k = 0; k < tap_buffers.size(); ++k) {
      const int dim = tap_dims[k];
      const int src_index = start_index + t;
      if (src_index < 0 || src_index >= ar_len) continue;
      const int copy_dim = std::min<int>(dim, target_hidden - static_cast<int>(offset - base));
      if (copy_dim > 0) {
        const int64_t src_base = static_cast<int64_t>(src_index) * dim;
        dflash::copyTensorRowToFloat(*tap_buffers[k], src_base, history.data() + offset, copy_dim);
        offset += static_cast<size_t>(copy_dim);
      }
    }
  }
}

template <typename CacheT>
int bindTargetCacheToVerifyBuffers(dflash::TargetGraphSession& session, dflash::KvCacheManager<CacheT>& cache) {
  auto& layers = cache.layers();
  auto& inputs = session.inputs();
  auto& outputs = session.outputs();
  const auto& binding = session.binding();
  const size_t n = std::min(layers.size(), std::min(binding.past_key_inputs.size(), binding.past_value_inputs.size()));
  int bound = 0;
  for (size_t i = 0; i < n; ++i) {
    if (binding.past_key_inputs[i] != dflash::kNoTensor && binding.past_key_inputs[i] < inputs.size()) {
      auto& src = inputs[binding.past_key_inputs[i]];
      layers[i].key_in.bindExternal(reinterpret_cast<CacheT*>(src.host.data()), src.host.size() / sizeof(CacheT), true);
      ++bound;
    }
    if (binding.past_value_inputs[i] != dflash::kNoTensor && binding.past_value_inputs[i] < inputs.size()) {
      auto& src = inputs[binding.past_value_inputs[i]];
      layers[i].value_in.bindExternal(reinterpret_cast<CacheT*>(src.host.data()), src.host.size() / sizeof(CacheT), true);
      ++bound;
    }
    if (i < binding.present_key_outputs.size() &&
        binding.present_key_outputs[i] != dflash::kNoTensor &&
        binding.present_key_outputs[i] < outputs.size()) {
      auto& dst = outputs[binding.present_key_outputs[i]];
      layers[i].key_out.bindExternal(reinterpret_cast<CacheT*>(dst.host.data()), dst.host.size() / sizeof(CacheT), false);
      ++bound;
    }
    if (i < binding.present_value_outputs.size() &&
        binding.present_value_outputs[i] != dflash::kNoTensor &&
        binding.present_value_outputs[i] < outputs.size()) {
      auto& dst = outputs[binding.present_value_outputs[i]];
      layers[i].value_out.bindExternal(reinterpret_cast<CacheT*>(dst.host.data()), dst.host.size() / sizeof(CacheT), false);
      ++bound;
    }
  }
  return bound;
}

std::vector<int64_t> draftTokensFromOutput(const dflash::TensorBuffer& out, int draft_count, int vocab) {
  std::vector<int64_t> tokens(static_cast<size_t>(draft_count), 0);
  const int64_t elements = dflash::tensorElementCount(out);
  if (elements >= static_cast<int64_t>(draft_count + 1) * vocab) {
    dflash::argmaxRowsAtRaw(out, 1, draft_count, vocab, tokens.data());
  } else {
    for (int i = 0; i < draft_count; ++i) tokens[static_cast<size_t>(i)] = dflash::readTokenIdAtRaw(out, i + 1);
  }
  return tokens;
}

struct DraftCacheState {
  int layers = 0;
  int heads = 0;
  int head_dim = 0;
  int context = 0;
  int64_t origin = 0;
  int tokens = 0;
  std::vector<std::vector<uint16_t>> key;
  std::vector<std::vector<uint16_t>> value;

  void init(int layer_count, int head_count, int dim, int context_rows) {
    layers = layer_count;
    heads = head_count;
    head_dim = dim;
    context = context_rows;
    origin = 0;
    tokens = 0;
    key.assign(static_cast<size_t>(layers * heads),
               std::vector<uint16_t>(static_cast<size_t>(context) * head_dim, 0));
    value = key;
  }

  int slot(int layer, int head) const { return layer * heads + head; }
  int pastKeyInputIndex(int layer, int head) const { return 8 + layer * heads * 2 + head; }
  int pastValueInputIndex(int layer, int head) const { return 8 + layer * heads * 2 + heads + head; }
  int presentKeyOutputIndex(int layer, int head) const { return 1 + layer * heads * 2 + head; }
  int presentValueOutputIndex(int layer, int head) const { return 1 + layer * heads * 2 + heads + head; }

  void reset(int64_t new_origin = 0) {
    origin = new_origin;
    tokens = 0;
    for (auto& cache : key) std::fill(cache.begin(), cache.end(), 0);
    for (auto& cache : value) std::fill(cache.begin(), cache.end(), 0);
  }

  void fillInputs(dflash::DraftGraphSession& draft) const {
    auto& inputs = draft.inputs();
    for (int layer = 0; layer < layers; ++layer) {
      for (int h = 0; h < heads; ++h) {
        const int s = slot(layer, h);
        const int key_idx = pastKeyInputIndex(layer, h);
        const int value_idx = pastValueInputIndex(layer, h);
        if (key_idx >= 0 && key_idx < static_cast<int>(inputs.size())) {
          auto& dst = inputs[static_cast<size_t>(key_idx)];
          const size_t bytes = std::min(dst.host.size(), key[static_cast<size_t>(s)].size() * sizeof(uint16_t));
          std::memcpy(dst.host.data(), key[static_cast<size_t>(s)].data(), bytes);
          if (bytes < dst.host.size()) std::fill(dst.host.begin() + static_cast<std::ptrdiff_t>(bytes), dst.host.end(), 0);
        }
        if (value_idx >= 0 && value_idx < static_cast<int>(inputs.size())) {
          auto& dst = inputs[static_cast<size_t>(value_idx)];
          const size_t bytes = std::min(dst.host.size(), value[static_cast<size_t>(s)].size() * sizeof(uint16_t));
          std::memcpy(dst.host.data(), value[static_cast<size_t>(s)].data(), bytes);
          if (bytes < dst.host.size()) std::fill(dst.host.begin() + static_cast<std::ptrdiff_t>(bytes), dst.host.end(), 0);
        }
      }
    }
  }

  void dropOldRows(int drop) {
    if (drop <= 0) return;
    drop = std::min(drop, tokens);
    const int remain = tokens - drop;
    for (auto& cache : key) {
      for (int r = 0; r < remain; ++r) {
        std::copy(cache.data() + static_cast<int64_t>(r + drop) * head_dim,
                  cache.data() + static_cast<int64_t>(r + drop + 1) * head_dim,
                  cache.data() + static_cast<int64_t>(r) * head_dim);
      }
      std::fill(cache.data() + static_cast<int64_t>(remain) * head_dim, cache.data() + cache.size(), 0);
    }
    for (auto& cache : value) {
      for (int r = 0; r < remain; ++r) {
        std::copy(cache.data() + static_cast<int64_t>(r + drop) * head_dim,
                  cache.data() + static_cast<int64_t>(r + drop + 1) * head_dim,
                  cache.data() + static_cast<int64_t>(r) * head_dim);
      }
      std::fill(cache.data() + static_cast<int64_t>(remain) * head_dim, cache.data() + cache.size(), 0);
    }
    origin += drop;
    tokens = remain;
  }

  void appendOutputs(const dflash::DraftGraphSession& draft, int count) {
    count = std::max(0, std::min(count, context));
    if (count <= 0) return;
    if (tokens + count > context) dropOldRows(tokens + count - context);
    const auto& outputs = draft.outputs();
    const int dst_row = tokens;
    for (int layer = 0; layer < layers; ++layer) {
      for (int h = 0; h < heads; ++h) {
        const int s = slot(layer, h);
        const int key_idx = presentKeyOutputIndex(layer, h);
        const int value_idx = presentValueOutputIndex(layer, h);
        if (key_idx >= 0 && key_idx < static_cast<int>(outputs.size())) {
          const auto& src = outputs[static_cast<size_t>(key_idx)];
          const auto* ptr = reinterpret_cast<const uint16_t*>(src.host.data());
          const size_t elems = std::min<size_t>(static_cast<size_t>(count) * head_dim, src.host.size() / sizeof(uint16_t));
          std::copy(ptr, ptr + elems, key[static_cast<size_t>(s)].data() + static_cast<int64_t>(dst_row) * head_dim);
        }
        if (value_idx >= 0 && value_idx < static_cast<int>(outputs.size())) {
          const auto& src = outputs[static_cast<size_t>(value_idx)];
          const auto* ptr = reinterpret_cast<const uint16_t*>(src.host.data());
          const size_t elems = std::min<size_t>(static_cast<size_t>(count) * head_dim, src.host.size() / sizeof(uint16_t));
          std::copy(ptr, ptr + elems, value[static_cast<size_t>(s)].data() + static_cast<int64_t>(dst_row) * head_dim);
        }
      }
    }
    tokens += count;
  }
};

void fillDraftCacheMaskScale(dflash::DraftGraphSession& draft,
                             int cached_tokens,
                             int delta_count,
                             bool include_noise,
                             const dflash::DraftTensorGeometry& geom) {
  const auto& binding = draft.binding();
  if (binding.mask_scale_inputs.empty()) return;
  const int total_len = geom.cache_context + geom.cache_ar_len;
  std::vector<float> mask(static_cast<size_t>(geom.block) * static_cast<size_t>(total_len), -20.0f);
  delta_count = std::max(0, std::min(delta_count, geom.cache_delta));
  cached_tokens = std::max(0, std::min(cached_tokens, geom.cache_context));
  for (int row = 0; row < geom.block; ++row) {
    float* dst = mask.data() + static_cast<size_t>(row) * static_cast<size_t>(total_len);
    for (int i = 0; i < cached_tokens; ++i) dst[i] = 0.0f;
    for (int i = 0; i < delta_count; ++i) dst[geom.cache_context + i] = 0.0f;
    if (include_noise) {
      for (int i = 0; i < geom.block; ++i) dst[geom.cache_context + geom.cache_delta + i] = 0.0f;
    }
  }
  std::vector<float> scale(mask.size(), 1.0f / std::sqrt(static_cast<float>(geom.head_dim)));
  dflash::fillFloatBuffer(draft.inputs()[binding.mask_scale_inputs[0]], mask);
  if (binding.mask_scale_inputs.size() > 1) {
    dflash::fillFloatBuffer(draft.inputs()[binding.mask_scale_inputs[1]], scale);
  }
}

bool runDraftCacheFillChunk(dflash::DraftGraphSession& draft,
                            DraftCacheState& cache,
                            const std::vector<float>& target_hidden_history,
                            int64_t start_token,
                            int count,
                            const Options& opt,
                            const dflash::DraftTensorGeometry& geom,
                            double& draft_ms) {
  std::vector<float> zero_noise(static_cast<size_t>(geom.block) * static_cast<size_t>(opt.draft_hidden), 0.0f);
  dflash::fillDraftNoiseInput(draft.inputs(), draft.binding().noise_input, zero_noise);
  dflash::fillDraftTargetHiddenInput(draft.inputs(), draft.binding().target_hidden_input,
                                     target_hidden_history, start_token, count, opt.target_hidden, geom.cache_delta);
  cache.fillInputs(draft);
  dflash::fillDraftRopeInputs(draft.inputs(), draft.binding(), start_token, count, geom);
  fillDraftCacheMaskScale(draft, cache.tokens, count, false, geom);
  const auto t0 = Clock::now();
  if (!draft.execute()) return false;
  const auto t1 = Clock::now();
  draft_ms += elapsedMs(t0, t1);
  cache.appendOutputs(draft, count);
  return true;
}

std::pair<int64_t, int> syncDraftCacheToHistory(dflash::DraftGraphSession& draft,
                                                DraftCacheState& cache,
                                                const std::vector<float>& target_hidden_history,
                                                const Options& opt,
                                                const dflash::DraftTensorGeometry& geom,
                                                double& draft_ms) {
  const int64_t history_tokens = static_cast<int64_t>(target_hidden_history.size()) / opt.target_hidden;
  const int64_t pending_start_target = std::max<int64_t>(0, history_tokens - geom.cache_delta);
  if (history_tokens < cache.origin + cache.tokens) cache.reset();
  if (pending_start_target - cache.origin > geom.cache_context) {
    cache.reset(std::max<int64_t>(0, pending_start_target - geom.cache_context));
  }
  int64_t next_token = cache.origin + cache.tokens;
  while (next_token < pending_start_target) {
    const int count = static_cast<int>(std::min<int64_t>(geom.cache_delta, pending_start_target - next_token));
    if (!runDraftCacheFillChunk(draft, cache, target_hidden_history, next_token, count, opt, geom, draft_ms)) {
      throw std::runtime_error("draft cache fill failed");
    }
    next_token += count;
  }
  const int64_t pending_start = cache.origin + cache.tokens;
  int pending_count = static_cast<int>(std::max<int64_t>(0, history_tokens - pending_start));
  pending_count = std::min(pending_count, geom.cache_delta);
  return {pending_start, pending_count};
}

template <typename CacheT>
bool runDecodeTyped(dflash::QnnRuntime& target_prefill_runtime,
                    dflash::QnnRuntime& target_verify_runtime,
                    dflash::QnnRuntime& draft_runtime,
                    const Options& opt,
                    const std::vector<int64_t>& prompt_tokens,
                    const EmbeddingTable& embedding,
                    std::vector<int64_t>* generated_out,
                    const std::function<void(const std::vector<int64_t>&)>& text_printer) {
  dflash::TargetGraphSession prefill;
  dflash::TargetGraphSession verify;
  dflash::DraftGraphSession draft;
  if (!prefill.init(target_prefill_runtime, opt.prefill_graph, opt.target_layers)) return false;
  if (!verify.init(target_verify_runtime, opt.verify_graph, opt.target_layers)) return false;

  dflash::DraftTensorGeometry draft_geom;
  draft_geom.block = opt.block;
  draft_geom.cache_delta = opt.block;
  draft_geom.num_layers = opt.draft_layers;
  draft_geom.draft_hidden = opt.draft_hidden;
  draft_geom.target_hidden = opt.target_hidden;
  const std::string draft_graph = findDraftGraph(draft_runtime, opt);
  if (draft_graph.empty()) throw std::runtime_error("draft graph not found");
  if (!draft.init(draft_runtime, draft_graph, draft_geom)) return false;

  dflash::AotModelConfig target_cfg;
  target_cfg.context_len = opt.context_len;
  target_cfg.max_cache_len = opt.context_len - opt.verify_seq;
  target_cfg.max_ar_len = std::max(opt.prefill_seq, opt.verify_seq);
  target_cfg.num_layers = opt.target_layers;
  target_cfg.num_heads = opt.kv_heads;
  target_cfg.head_dim = opt.head_dim;
  dflash::KvCacheManager<CacheT> target_cache(target_cfg);
  target_cache.init(opt.prefill_seq);

  DraftCacheState draft_cache;
  draft_cache.init(opt.draft_layers, opt.kv_heads, opt.head_dim, draft_geom.cache_context);

  std::vector<float> target_hidden_history;
  target_hidden_history.reserve(static_cast<size_t>(prompt_tokens.size() + opt.max_new + opt.block) *
                                static_cast<size_t>(opt.target_hidden));
  std::vector<int64_t> generated;
  generated.reserve(static_cast<size_t>(opt.max_new));

  int64_t pos = 0;
  int prompt_offset = 0;
  int64_t current_token = prompt_tokens.back();

  const auto t_prefill0 = Clock::now();
  while (prompt_offset < static_cast<int>(prompt_tokens.size())) {
    const int chunk = std::min(opt.prefill_seq, static_cast<int>(prompt_tokens.size()) - prompt_offset);
    std::vector<int64_t> block(static_cast<size_t>(opt.prefill_seq), opt.mask_token);
    for (int i = 0; i < chunk; ++i) block[static_cast<size_t>(i)] = prompt_tokens[static_cast<size_t>(prompt_offset + i)];
    target_cache.rearrange(opt.prefill_seq);
    prefill.fillCacheInputs(target_cache);
    prefill.fillBlock(block, pos, opt.prefill_seq);
    prefill.fillPrefillAttention(opt.prefill_seq, static_cast<int>(pos));
    if (!prefill.execute()) return false;
    prefill.fillCacheOutputs(target_cache);
    target_cache.update(opt.prefill_seq, static_cast<int>(pos), chunk);
    appendTargetHidden(prefill, opt.prefill_seq, 0, chunk, opt.target_hidden, target_hidden_history);
    const auto& prefill_logits =
        prefill.outputs()[prefill.binding().logits_output == dflash::kNoTensor ? 0 : prefill.binding().logits_output];
    current_token = dflash::argmaxTensorAtRaw(prefill_logits, chunk - 1, opt.vocab);
    pos += chunk;
    prompt_offset += chunk;
  }
  const auto t_prefill1 = Clock::now();
  const auto t_bind0 = Clock::now();
  const int bound_verify_cache_buffers = bindTargetCacheToVerifyBuffers(verify, target_cache);
  const auto t_bind1 = Clock::now();
  if (bound_verify_cache_buffers > 0) {
    std::cout << "[runtime] target verify cache buffers bound=" << bound_verify_cache_buffers
              << " (external TensorBuffer storage), target_verify_cache_bind="
              << elapsedMs(t_bind0, t_bind1) << " ms\n";
  }

  int rounds = 0;
  int accepted_draft_tokens = 0;
  int draft_tokens_total = 0;
  double draft_ms = 0.0;
  double verify_ms = 0.0;
  double sample_ms = 0.0;
  HostTiming host_timing;

  const auto t_decode0 = Clock::now();
  while (static_cast<int>(generated.size()) < opt.max_new) {
    ++rounds;
    std::vector<int64_t> seed_and_draft;
    seed_and_draft.reserve(static_cast<size_t>(opt.block));
    seed_and_draft.push_back(current_token);
    std::vector<int64_t> prev_draft_tokens;
    prev_draft_tokens.push_back(current_token);
    int remaining_draft = opt.block - 1;

    while (remaining_draft > 0) {
      const int step = std::min(opt.block, remaining_draft);
      const auto draft_noise0 = Clock::now();
      const auto noise_tokens = makeDraftNoiseTokens(prev_draft_tokens, opt.block, opt.mask_token);
      auto noise = embedding.lookupBlock(noise_tokens, opt.block);
      const auto draft_noise1 = Clock::now();
      host_timing.draft_noise_embed += elapsedMs(draft_noise0, draft_noise1);
      if (draft.binding().cache_aware) {
        const auto draft_sync0 = Clock::now();
        const double draft_ms_before_sync = draft_ms;
        auto [pending_start, pending_count] =
            syncDraftCacheToHistory(draft, draft_cache, target_hidden_history, opt, draft_geom, draft_ms);
        const auto draft_sync1 = Clock::now();
        const double draft_graph_in_sync = draft_ms - draft_ms_before_sync;
        host_timing.draft_sync_host += std::max(0.0, elapsedMs(draft_sync0, draft_sync1) - draft_graph_in_sync);
        const auto draft_fill0 = Clock::now();
        dflash::fillDraftNoiseInput(draft.inputs(), draft.binding().noise_input, noise);
        dflash::fillDraftTargetHiddenInput(draft.inputs(),
                                           draft.binding().target_hidden_input,
                                           target_hidden_history,
                                           pending_start,
                                           pending_count,
                                           opt.target_hidden,
                                           draft_geom.cache_delta);
        draft_cache.fillInputs(draft);
        dflash::fillDraftRopeInputs(draft.inputs(), draft.binding(), pending_start, pending_count, draft_geom);
        fillDraftCacheMaskScale(draft, draft_cache.tokens, pending_count, true, draft_geom);
        const auto draft_fill1 = Clock::now();
        host_timing.draft_input_fill += elapsedMs(draft_fill0, draft_fill1);
      } else {
        const auto draft_fill0 = Clock::now();
        draft.fillStandardInputs(pos, noise, target_hidden_history);
        const auto draft_fill1 = Clock::now();
        host_timing.draft_input_fill += elapsedMs(draft_fill0, draft_fill1);
      }
      const auto td0 = Clock::now();
      if (!draft.execute()) return false;
      const auto td1 = Clock::now();
      if (draft.binding().cache_aware) {
        const auto draft_append0 = Clock::now();
        const int64_t history_tokens = static_cast<int64_t>(target_hidden_history.size()) / opt.target_hidden;
        const int pending_count = static_cast<int>(std::min<int64_t>(
            draft_geom.cache_delta,
            std::max<int64_t>(0, history_tokens - (draft_cache.origin + draft_cache.tokens))));
        draft_cache.appendOutputs(draft, pending_count);
        const auto draft_append1 = Clock::now();
        host_timing.draft_cache_append += elapsedMs(draft_append0, draft_append1);
      }
      draft_ms += elapsedMs(td0, td1);
      if (draft.binding().output0 == dflash::kNoTensor || draft.binding().output0 >= draft.outputs().size()) {
        throw std::runtime_error("draft output0 missing");
      }
      const auto draft_argmax0 = Clock::now();
      auto dtoks = draftTokensFromOutput(draft.outputs()[draft.binding().output0], step, opt.vocab);
      const auto draft_argmax1 = Clock::now();
      host_timing.draft_argmax += elapsedMs(draft_argmax0, draft_argmax1);
      prev_draft_tokens.assign(dtoks.begin(), dtoks.begin() + step);
      for (int i = 0; i < step; ++i) seed_and_draft.push_back(dtoks[static_cast<size_t>(i)]);
      remaining_draft -= step;
    }

    std::vector<int64_t> verify_block(static_cast<size_t>(opt.verify_seq), opt.mask_token);
    const int verify_count = std::min(opt.verify_seq, static_cast<int>(seed_and_draft.size()));
    for (int i = 0; i < verify_count; ++i) verify_block[static_cast<size_t>(i)] = seed_and_draft[static_cast<size_t>(i)];

    const auto tr0 = Clock::now();
    target_cache.rearrange(opt.verify_seq);
    const auto tr1 = Clock::now();
    host_timing.target_cache_rearrange += elapsedMs(tr0, tr1);

    const auto tci0 = Clock::now();
    verify.fillCacheInputs(target_cache);
    const auto tci1 = Clock::now();
    host_timing.target_cache_fill_inputs += elapsedMs(tci0, tci1);

    const auto tfi0 = Clock::now();
    verify.fillBlock(verify_block, pos, opt.verify_seq);
    verify.fillVerifyAttention(opt.verify_seq, static_cast<int>(std::max<int64_t>(0, pos)));
    const auto tfi1 = Clock::now();
    host_timing.target_input_fill += elapsedMs(tfi0, tfi1);

    const auto tv0 = Clock::now();
    if (!verify.execute()) return false;
    const auto tv1 = Clock::now();
    verify_ms += elapsedMs(tv0, tv1);

    const auto tco0 = Clock::now();
    verify.fillCacheOutputs(target_cache);
    const auto tco1 = Clock::now();
    host_timing.target_cache_read_outputs += elapsedMs(tco0, tco1);

    const auto ts0 = Clock::now();
    const auto& logits = verify.outputs()[verify.binding().logits_output == dflash::kNoTensor ? 0 : verify.binding().logits_output];
    int accepted_inputs = 1;
    std::vector<int64_t> target_argmax(static_cast<size_t>(verify_count), 0);
    std::vector<bool> target_argmax_valid(static_cast<size_t>(verify_count), false);
    for (int i = 1; i < verify_count && static_cast<int>(generated.size()) + accepted_inputs - 1 < opt.max_new; ++i) {
      const int64_t candidate = seed_and_draft[static_cast<size_t>(i)];
      const auto rank = dflash::rankAcceptTensorAtRaw(logits, i - 1, opt.vocab, static_cast<int>(candidate), opt.accept_topk);
      target_argmax[static_cast<size_t>(i - 1)] = rank.argmax;
      target_argmax_valid[static_cast<size_t>(i - 1)] = true;
      if (rank.accepted) {
        ++accepted_inputs;
      } else {
        break;
      }
    }
    const int correction_index = accepted_inputs - 1;
    if (correction_index >= 0 && correction_index < verify_count && !target_argmax_valid[static_cast<size_t>(correction_index)]) {
      target_argmax[static_cast<size_t>(correction_index)] =
          dflash::argmaxTensorAtRaw(logits, correction_index, opt.vocab);
      target_argmax_valid[static_cast<size_t>(correction_index)] = true;
    }
    const int64_t next_current =
        correction_index >= 0 && correction_index < verify_count ? target_argmax[static_cast<size_t>(correction_index)] : 0;
    const auto ts1 = Clock::now();
    sample_ms += elapsedMs(ts0, ts1);

    const auto commit0 = Clock::now();
    const int new_tokens = std::min(accepted_inputs, opt.max_new - static_cast<int>(generated.size()));
    for (int i = 0; i < new_tokens; ++i) generated.push_back(seed_and_draft[static_cast<size_t>(i)]);
    const auto commit1 = Clock::now();
    host_timing.output_commit += elapsedMs(commit0, commit1);

    accepted_draft_tokens += std::max(0, new_tokens - 1);
    draft_tokens_total += std::max(0, verify_count - 1);

    const auto tcu0 = Clock::now();
    target_cache.update(opt.verify_seq, static_cast<int>(pos), new_tokens);
    const auto tcu1 = Clock::now();
    host_timing.target_cache_update += elapsedMs(tcu0, tcu1);

    const auto th0 = Clock::now();
    appendTargetHidden(verify, opt.verify_seq, 0, new_tokens, opt.target_hidden, target_hidden_history);
    const auto th1 = Clock::now();
    host_timing.append_hidden += elapsedMs(th0, th1);
    pos += new_tokens;
    current_token = next_current;
  }
  const auto t_decode1 = Clock::now();

  const double prefill_ms = elapsedMs(t_prefill0, t_prefill1);
  const double decode_ms = elapsedMs(t_decode0, t_decode1);
  const double accept_rate = draft_tokens_total ? 100.0 * accepted_draft_tokens / draft_tokens_total : 0.0;
  const char* print_tokens_env = std::getenv("DFLASH_PRINT_TOKENS");
  const bool print_tokens = print_tokens_env && std::string(print_tokens_env) == "1";
  if (print_tokens) {
    std::cout << "[DFlash] output_tokens=";
    for (size_t i = 0; i < generated.size(); ++i) {
      if (i) std::cout << ",";
      std::cout << generated[i];
    }
    std::cout << "\n";
  }
  if (generated_out) *generated_out = generated;
  if (text_printer) text_printer(generated);
  std::cout << std::fixed << std::setprecision(3)
            << "[DFlash] summary: rounds=" << rounds
            << ", draft_tokens=" << draft_tokens_total
            << ", accepted_draft_tokens=" << accepted_draft_tokens
            << ", draft_accept_rate=" << accept_rate << "%"
            << ", accepted_inputs=" << generated.size()
            << ", avg_accept_inputs=" << (rounds ? static_cast<double>(generated.size()) / rounds : 0.0)
            << "\n";
  const double legacy_like_sampler_ms = sample_ms + host_timing.draft_argmax;
  std::cout << std::fixed << std::setprecision(3)
            << "[DFlash] timing: target_prefill=" << prefill_ms
            << " ms, target_verify=" << verify_ms
            << " ms, draft_model=" << draft_ms
            << " ms, sampler=" << legacy_like_sampler_ms
            << " ms, sampler_accept_only=" << sample_ms
            << " ms, decode_total=" << decode_ms
            << " ms\n";
  const double measured_decode_parts =
      draft_ms + verify_ms + sample_ms + host_timing.draft_noise_embed + host_timing.draft_sync_host +
      host_timing.draft_input_fill + host_timing.draft_cache_append + host_timing.draft_argmax +
      host_timing.target_cache_rearrange + host_timing.target_cache_fill_inputs + host_timing.target_input_fill +
      host_timing.target_cache_read_outputs + host_timing.target_cache_update + host_timing.append_hidden +
      host_timing.output_commit;
  std::cout << std::fixed << std::setprecision(3)
            << "[DFlash] host_breakdown:"
            << " draft_noise_embed=" << host_timing.draft_noise_embed
            << " draft_sync_host=" << host_timing.draft_sync_host
            << " draft_input_fill=" << host_timing.draft_input_fill
            << " draft_cache_append=" << host_timing.draft_cache_append
            << " draft_argmax=" << host_timing.draft_argmax
            << " target_cache_rearrange=" << host_timing.target_cache_rearrange
            << " target_cache_fill_inputs=" << host_timing.target_cache_fill_inputs
            << " target_input_fill=" << host_timing.target_input_fill
            << " target_cache_read_outputs=" << host_timing.target_cache_read_outputs
            << " target_cache_update=" << host_timing.target_cache_update
            << " append_hidden=" << host_timing.append_hidden
            << " output_commit=" << host_timing.output_commit
            << " measured_decode_parts=" << measured_decode_parts
            << " unaccounted=" << (decode_ms - measured_decode_parts)
            << " ms\n";
  std::cout << std::fixed << std::setprecision(3)
            << "#################################\n"
            << "prompt tokens num = " << prompt_tokens.size() << "\n"
            << "decode tokens num = " << generated.size() << "\n"
            << "prefill time = " << prefill_ms / 1000.0 << " s\n"
            << " decode time = " << decode_ms / 1000.0 << " s\n"
            << "prefill speed = " << (prompt_tokens.empty() ? 0.0 : prompt_tokens.size() * 1000.0 / prefill_ms) << " tok/s\n"
            << " decode speed = " << (generated.empty() ? 0.0 : generated.size() * 1000.0 / decode_ms) << " tok/s\n"
            << "#################################\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options opt = parseOptions(argc, argv);
    std::vector<int64_t> prompt_tokens;
    const bool prompt_uses_tokenizer = !opt.prompt_file.empty();
    const char* decode_text_env = std::getenv("DFLASH_DECODE_TEXT");
    const bool decode_text = prompt_uses_tokenizer && !(decode_text_env && std::string(decode_text_env) == "0");
    if (!opt.prompt_file.empty()) {
      dflash::Qwen3Tokenizer tokenizer;
      if (!tokenizer.load(opt.tokenizer)) {
        throw std::runtime_error("failed to load tokenizer: " + opt.tokenizer);
      }
      prompt_tokens = tokenizer.encodeChatPrompt(readTextFile(opt.prompt_file));
      std::cout << "[runtime] tokenized prompt_file=" << opt.prompt_file
                << " tokenizer=" << opt.tokenizer
                << " prompt_tokens=" << prompt_tokens.size() << "\n";
    } else {
      prompt_tokens = readTokenIds(opt.tokens);
    }

    dflash::QnnRuntime target_merged_runtime;
    dflash::QnnRuntime target_prefill_runtime;
    dflash::QnnRuntime target_verify_runtime;
    dflash::QnnRuntime draft_runtime;
    dflash::QnnRuntime* prefill_runtime = nullptr;
    dflash::QnnRuntime* verify_runtime = nullptr;
    if (!opt.target.empty()) {
      if (!loadRuntime(target_merged_runtime, opt.target, "target")) return 1;
      prefill_runtime = &target_merged_runtime;
      verify_runtime = &target_merged_runtime;
    } else {
      if (!loadRuntime(target_prefill_runtime, opt.target_prefill, "target_prefill")) return 1;
      if (!loadRuntime(target_verify_runtime, opt.target_verify, "target_verify")) return 1;
      prefill_runtime = &target_prefill_runtime;
      verify_runtime = &target_verify_runtime;
    }
    if (!loadRuntime(draft_runtime, opt.draft, "draft")) return 1;

    EmbeddingTable embedding;
    if (!embedding.load(opt.embedding, opt.vocab, opt.draft_hidden)) {
      throw std::runtime_error("failed to load embedding: " + opt.embedding);
    }
    std::cout << "[runtime] loaded embedding=" << opt.embedding
              << " vocab=" << opt.vocab
              << " hidden=" << opt.draft_hidden << "\n";

    std::vector<int64_t> generated_tokens;
    std::function<void(const std::vector<int64_t>&)> text_printer;
    if (decode_text) {
      text_printer = [&](const std::vector<int64_t>& tokens) {
        dflash::Qwen3Tokenizer tokenizer;
        if (!tokenizer.load(opt.tokenizer)) {
          throw std::runtime_error("failed to reload tokenizer for decode: " + opt.tokenizer);
        }
        std::cout << "========== Generated Text ==========\n"
                  << tokenizer.decode(tokens)
                  << "\n========== End Generated Text ==========\n";
      };
    }
    if (opt.target_kv_dtype == "uint16") {
      if (!runDecodeTyped<uint16_t>(*prefill_runtime, *verify_runtime, draft_runtime, opt, prompt_tokens, embedding,
                                    &generated_tokens, text_printer)) {
        return 1;
      }
    } else if (opt.target_kv_dtype == "uint8") {
      if (!runDecodeTyped<uint8_t>(*prefill_runtime, *verify_runtime, draft_runtime, opt, prompt_tokens, embedding,
                                   &generated_tokens, text_printer)) {
        return 1;
      }
    } else {
      throw std::runtime_error("unsupported target_kv_dtype: " + opt.target_kv_dtype);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[runtime] fatal: " << e.what() << "\n";
    usage(argv[0]);
    return 2;
  }
}
