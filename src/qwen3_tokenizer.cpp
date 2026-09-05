#include "dflash/qwen3_tokenizer.hpp"

#include <algorithm>
#include <cwctype>
#include <fstream>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace dflash {
namespace {

using json = nlohmann::json;

std::string wideToUtf8(const std::wstring& wstr) {
  std::string result;
  for (wchar_t wc : wstr) {
    if (wc <= 0x7fU) {
      result.push_back(static_cast<char>(wc));
    } else if (wc <= 0x7ffU) {
      result.push_back(static_cast<char>(0xc0U | ((wc >> 6U) & 0x1fU)));
      result.push_back(static_cast<char>(0x80U | (wc & 0x3fU)));
    } else if (wc <= 0xffffU) {
      result.push_back(static_cast<char>(0xe0U | ((wc >> 12U) & 0x0fU)));
      result.push_back(static_cast<char>(0x80U | ((wc >> 6U) & 0x3fU)));
      result.push_back(static_cast<char>(0x80U | (wc & 0x3fU)));
    } else if (wc <= 0x10ffffU) {
      result.push_back(static_cast<char>(0xf0U | ((wc >> 18U) & 0x07U)));
      result.push_back(static_cast<char>(0x80U | ((wc >> 12U) & 0x3fU)));
      result.push_back(static_cast<char>(0x80U | ((wc >> 6U) & 0x3fU)));
      result.push_back(static_cast<char>(0x80U | (wc & 0x3fU)));
    }
  }
  return result;
}

std::wstring utf8ToWide(const std::string& str) {
  std::wstring out;
  for (size_t i = 0; i < str.size();) {
    const auto byte = static_cast<unsigned char>(str[i]);
    if ((byte & 0x80U) == 0) {
      out.push_back(static_cast<wchar_t>(byte));
      ++i;
    } else if ((byte & 0xe0U) == 0xc0U && i + 1 < str.size()) {
      wchar_t wc = (static_cast<wchar_t>(byte & 0x1fU) << 6U) |
                   static_cast<wchar_t>(static_cast<unsigned char>(str[i + 1]) & 0x3fU);
      out.push_back(wc);
      i += 2;
    } else if ((byte & 0xf0U) == 0xe0U && i + 2 < str.size()) {
      wchar_t wc = (static_cast<wchar_t>(byte & 0x0fU) << 12U) |
                   (static_cast<wchar_t>(static_cast<unsigned char>(str[i + 1]) & 0x3fU) << 6U) |
                   static_cast<wchar_t>(static_cast<unsigned char>(str[i + 2]) & 0x3fU);
      out.push_back(wc);
      i += 3;
    } else if ((byte & 0xf8U) == 0xf0U && i + 3 < str.size()) {
      wchar_t wc = (static_cast<wchar_t>(byte & 0x07U) << 18U) |
                   (static_cast<wchar_t>(static_cast<unsigned char>(str[i + 1]) & 0x3fU) << 12U) |
                   (static_cast<wchar_t>(static_cast<unsigned char>(str[i + 2]) & 0x3fU) << 6U) |
                   static_cast<wchar_t>(static_cast<unsigned char>(str[i + 3]) & 0x3fU);
      out.push_back(wc);
      i += 4;
    } else {
      ++i;
    }
  }
  return out;
}

void initLocale() {
  try {
    std::locale::global(std::locale("en_US.UTF-8"));
  } catch (...) {
    try {
      std::locale::global(std::locale("C.UTF-8"));
    } catch (...) {
    }
  }
}

bool isLetter(wchar_t c) { return std::iswalpha(c) != 0; }
bool isDigit(wchar_t c) { return std::iswdigit(c) != 0; }

void makeBytesToUnicode(std::unordered_map<std::wint_t, wchar_t>& dict) {
  std::vector<std::wint_t> bs;
  for (std::wint_t i = L'!'; i <= L'~'; ++i) bs.push_back(i);
  for (std::wint_t i = L'¡'; i <= L'¬'; ++i) bs.push_back(i);
  for (std::wint_t i = L'®'; i <= L'ÿ'; ++i) bs.push_back(i);

  std::vector<std::wint_t> cs = bs;
  int n = 0;
  for (std::wint_t b = 0; b < 256; ++b) {
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
      bs.push_back(b);
      cs.push_back(256 + n);
      ++n;
    }
  }
  for (size_t i = 0; i < bs.size(); ++i) dict.emplace(bs[i], static_cast<wchar_t>(cs[i]));
}

class Trie {
  struct Node {
    std::unordered_map<wchar_t, std::unique_ptr<Node>> children;
    bool is_end = false;
  };

 public:
  void add(const std::wstring& word) {
    if (word.empty()) return;
    special_tokens_.insert(word);
    Node* current = root_.get();
    for (wchar_t c : word) {
      if (!current->children.count(c)) current->children[c] = std::make_unique<Node>();
      current = current->children[c].get();
    }
    current->is_end = true;
  }

  bool isSpecialToken(const std::wstring& token) const { return special_tokens_.count(token) != 0; }

  std::vector<std::wstring> split(const std::wstring& text) const {
    std::map<size_t, Node*> states;
    std::vector<size_t> offsets = {0};
    size_t skip = 0;

    for (size_t current = 0; current < text.size(); ++current) {
      if (skip > current) continue;

      std::unordered_set<size_t> to_remove;
      bool reset = false;
      wchar_t current_char = text[current];

      for (auto& kv : states) {
        const size_t start0 = kv.first;
        Node* node = kv.second;
        size_t start = start0;
        if (node->is_end) {
          size_t max_end = current;
          for (auto& look_kv : states) {
            const size_t look_start = look_kv.first;
            Node* look_node = look_kv.second;
            if (look_start > start) break;

            size_t lookahead = (look_start < start) ? current + 1 : current;
            size_t end = lookahead;
            Node* ptr = look_node;
            while (lookahead < text.size()) {
              wchar_t ch = text[lookahead];
              if (!ptr->children.count(ch)) break;
              ptr = ptr->children[ch].get();
              ++lookahead;
              if (ptr->is_end) {
                start = look_start;
                end = lookahead;
                skip = lookahead;
              }
            }
            if (ptr->is_end && end > max_end) max_end = end;
          }
          offsets.push_back(start);
          offsets.push_back(max_end);
          reset = true;
          break;
        }
        if (node->children.count(current_char)) {
          states[start] = node->children[current_char].get();
        } else {
          to_remove.insert(start);
        }
      }
      if (reset) {
        states.clear();
      } else {
        for (size_t start : to_remove) states.erase(start);
      }
      if (current >= skip && root_->children.count(current_char)) {
        states[current] = root_->children.at(current_char).get();
      }
    }
    for (auto& kv : states) {
      if (kv.second->is_end) {
        offsets.push_back(kv.first);
        offsets.push_back(text.size());
        break;
      }
    }

    std::sort(offsets.begin(), offsets.end());
    std::vector<std::wstring> result;
    for (size_t i = 1; i < offsets.size(); ++i) {
      if (offsets[i - 1] != offsets[i]) {
        result.push_back(text.substr(offsets[i - 1], offsets[i] - offsets[i - 1]));
      }
    }
    if (offsets.back() != text.size()) result.push_back(text.substr(offsets.back()));
    return result;
  }

 private:
  std::unique_ptr<Node> root_ = std::make_unique<Node>();
  std::unordered_set<std::wstring> special_tokens_;
};

struct BPEPairHash {
  std::size_t operator()(const std::pair<std::wstring, std::wstring>& key) const {
    return std::hash<std::wstring>{}(key.first + key.second);
  }
};

class BPE {
 public:
  bool load(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    json data = json::parse(f);
    if (!data.contains("model") || !data["model"].contains("vocab") || !data["model"].contains("merges")) {
      return false;
    }

    for (const auto& item : data["model"]["vocab"].items()) {
      const auto token = utf8ToWide(item.key());
      const int64_t id = item.value().get<int64_t>();
      vocab_[token] = id;
      vocab_inverse_[id] = token;
    }
    if (data.contains("added_tokens")) {
      for (const auto& add_token : data["added_tokens"]) {
        const int64_t id = add_token["id"].get<int64_t>();
        const auto token = utf8ToWide(add_token["content"].get<std::string>());
        vocab_[token] = id;
        vocab_inverse_[id] = token;
      }
    }

    int64_t rank = 0;
    for (const auto& merge_item : data["model"]["merges"]) {
      std::wstring first;
      std::wstring second;
      if (merge_item.is_string()) {
        const auto wide = utf8ToWide(merge_item.get<std::string>());
        const auto blank = wide.find(L' ');
        if (blank == std::wstring::npos) continue;
        first = wide.substr(0, blank);
        second = wide.substr(blank + 1);
      } else if (merge_item.is_array() && merge_item.size() >= 2) {
        first = utf8ToWide(merge_item[0].get<std::string>());
        second = utf8ToWide(merge_item[1].get<std::string>());
      } else {
        continue;
      }
      bpe_ranks_[{first, second}] = rank++;
    }
    return true;
  }

  std::vector<std::wstring> bpe(const std::wstring& token) const {
    std::vector<std::wstring> word;
    for (wchar_t c : token) word.push_back(std::wstring{c});
    auto pairs = getPairs(word);
    if (pairs.empty()) return {token};

    while (true) {
      bool has_bigram = false;
      int64_t best_rank = std::numeric_limits<int64_t>::max();
      std::pair<std::wstring, std::wstring> best;
      for (const auto& p : pairs) {
        auto it = bpe_ranks_.find(p);
        if (it != bpe_ranks_.end() && it->second < best_rank) {
          best_rank = it->second;
          best = p;
          has_bigram = true;
        }
      }
      if (!has_bigram) break;

      const auto& first = best.first;
      const auto& second = best.second;
      std::vector<std::wstring> new_word;
      size_t i = 0;
      while (i < word.size()) {
        size_t j = i;
        while (j < word.size() && word[j] != first) ++j;
        if (j > i) new_word.insert(new_word.end(), word.begin() + static_cast<std::ptrdiff_t>(i),
                                   word.begin() + static_cast<std::ptrdiff_t>(j));
        if (j < word.size() - 1 && word[j] == first && word[j + 1] == second) {
          new_word.push_back(first + second);
          i = j + 2;
        } else if (j < word.size()) {
          new_word.push_back(word[j]);
          i = j + 1;
        } else {
          i = j;
        }
      }
      word = std::move(new_word);
      if (word.size() == 1) break;
      pairs = getPairs(word);
    }
    return word;
  }

  int64_t idForToken(const std::wstring& token) const {
    auto it = vocab_.find(token);
    return it == vocab_.end() ? 0 : it->second;
  }

  std::wstring tokenForId(int64_t id) const {
    auto it = vocab_inverse_.find(id);
    return it == vocab_inverse_.end() ? L"" : it->second;
  }

 private:
  std::unordered_set<std::pair<std::wstring, std::wstring>, BPEPairHash> getPairs(
      const std::vector<std::wstring>& word) const {
    std::unordered_set<std::pair<std::wstring, std::wstring>, BPEPairHash> pairs;
    if (word.size() < 2) return pairs;
    auto prev = word[0];
    for (size_t i = 1; i < word.size(); ++i) {
      pairs.insert({prev, word[i]});
      prev = word[i];
    }
    return pairs;
  }

  std::unordered_map<std::wstring, int64_t> vocab_;
  std::unordered_map<int64_t, std::wstring> vocab_inverse_;
  std::unordered_map<std::pair<std::wstring, std::wstring>, int64_t, BPEPairHash> bpe_ranks_;
};

bool qwen3TokenizerMatchPattern(const std::wstring& str, size_t& pos, std::wstring& matched) {
  if (pos >= str.size()) return false;

  static const std::wstring contractions[] = {L"'s", L"'t", L"'re", L"'ve", L"'m", L"'ll", L"'d"};
  for (const auto& contraction : contractions) {
    if (pos + contraction.size() <= str.size() && str.compare(pos, contraction.size(), contraction) == 0) {
      matched = contraction;
      pos += contraction.size();
      return true;
    }
  }

  {
    const size_t original_pos = pos;
    bool has_prefix = false;
    matched.clear();
    if (!isLetter(str[pos]) && !isDigit(str[pos]) && str[pos] != L'\r' && str[pos] != L'\n') {
      matched += str[pos];
      ++pos;
      has_prefix = true;
    }
    if (pos < str.size() && isLetter(str[pos])) {
      do {
        matched += str[pos];
        ++pos;
      } while (pos < str.size() && isLetter(str[pos]));
      return true;
    }
    if (has_prefix) {
      pos = original_pos;
      matched.clear();
    }
  }

  if (isDigit(str[pos])) {
    matched = str.substr(pos, 1);
    ++pos;
    return true;
  }

  {
    const size_t original_pos = pos;
    size_t start = pos;
    matched.clear();
    if (str[pos] == L' ') ++pos;
    if (pos < str.size() && !std::iswspace(str[pos]) && !isLetter(str[pos]) && !isDigit(str[pos])) {
      do {
        ++pos;
      } while (pos < str.size() && !std::iswspace(str[pos]) && !isLetter(str[pos]) && !isDigit(str[pos]));
      matched = str.substr(start, pos - start);
      while (pos < str.size() && (str[pos] == L'\r' || str[pos] == L'\n')) {
        matched += str[pos];
        ++pos;
      }
      return true;
    }
    pos = original_pos;
  }

  {
    const size_t start = pos;
    while (pos < str.size() && std::iswspace(str[pos])) ++pos;
    if (pos < str.size() && (str[pos] == L'\r' || str[pos] == L'\n')) {
      while (pos < str.size() && (str[pos] == L'\r' || str[pos] == L'\n')) ++pos;
      matched = str.substr(start, pos - start);
      return true;
    }
    pos = start;
  }

  if (std::iswspace(str[pos])) {
    const size_t start = pos;
    while (pos < str.size() && std::iswspace(str[pos])) ++pos;
    if (pos >= str.size() || std::iswspace(str[pos])) {
      matched = str.substr(start, pos - start);
      return true;
    }
    pos = start;
  }

  if (std::iswspace(str[pos])) {
    const size_t start = pos;
    while (pos < str.size() && std::iswspace(str[pos])) ++pos;
    matched = str.substr(start, pos - start);
    return true;
  }
  return false;
}

std::vector<std::wstring> qwen3Regex(const std::string& text) {
  std::vector<std::wstring> out;
  const auto wide = utf8ToWide(text);
  size_t pos = 0;
  while (pos < wide.size()) {
    std::wstring matched;
    if (qwen3TokenizerMatchPattern(wide, pos, matched)) {
      out.push_back(matched);
    } else {
      ++pos;
    }
  }
  return out;
}

}  // namespace

class Qwen3Tokenizer::Impl {
 public:
  bool load(const std::string& tokenizer_json) {
    initLocale();
    makeBytesToUnicode(bytes_to_unicode_);
    unicode_to_bytes_.clear();
    for (const auto& kv : bytes_to_unicode_) unicode_to_bytes_[kv.second] = kv.first;
    const bool ok = bpe_.load(tokenizer_json);
    if (!ok) return false;
    for (const char* token : {
             "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
             "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>",
             "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", "<think>", "</think>"}) {
      special_tokens_.add(utf8ToWide(token));
    }
    return true;
  }

  std::vector<int64_t> encodeText(const std::string& text) const {
    std::vector<int64_t> ids;
    for (const auto& token : tokenize(text)) ids.push_back(bpe_.idForToken(token));
    return ids;
  }

  std::vector<int64_t> encodeChatPrompt(const std::string& prompt) const {
    std::string templ = "<|im_start|>user\n{{{prompt}}}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    const auto pos = templ.find("{{{prompt}}}");
    templ.replace(pos, 12, prompt);
    return encodeText(templ);
  }

  std::string decodeToken(int64_t id) const {
    const auto token = bpe_.tokenForId(id);
    std::string bytes;
    bytes.reserve(token.size());
    for (wchar_t c : token) {
      auto it = unicode_to_bytes_.find(c);
      if (it != unicode_to_bytes_.end()) bytes.push_back(static_cast<char>(it->second));
    }
    return wideToUtf8(utf8ToWide(bytes));
  }

  std::string decode(const std::vector<int64_t>& ids) const {
    std::string out;
    for (int64_t id : ids) out += decodeToken(id);
    return out;
  }

 private:
  std::vector<std::wstring> tokenize(const std::string& text) const {
    const auto pieces = special_tokens_.split(utf8ToWide(text));
    std::vector<std::wstring> all;
    for (const auto& piece : pieces) {
      if (special_tokens_.isSpecialToken(piece)) {
        all.push_back(piece);
        continue;
      }
      for (const auto& regex_piece : qwen3Regex(wideToUtf8(piece))) {
        const auto utf8_piece = wideToUtf8(regex_piece);
        std::wstring mapped;
        for (unsigned char c : utf8_piece) mapped.push_back(bytes_to_unicode_.at(c));
        auto bpe_tokens = bpe_.bpe(mapped);
        all.insert(all.end(), bpe_tokens.begin(), bpe_tokens.end());
      }
    }
    return all;
  }

  BPE bpe_;
  Trie special_tokens_;
  std::unordered_map<std::wint_t, wchar_t> bytes_to_unicode_;
  std::unordered_map<wchar_t, std::wint_t> unicode_to_bytes_;
};

Qwen3Tokenizer::Qwen3Tokenizer() : impl_(std::make_unique<Impl>()) {}
Qwen3Tokenizer::~Qwen3Tokenizer() = default;

bool Qwen3Tokenizer::load(const std::string& tokenizer_json) { return impl_->load(tokenizer_json); }
std::vector<int64_t> Qwen3Tokenizer::encodeText(const std::string& text) const { return impl_->encodeText(text); }
std::vector<int64_t> Qwen3Tokenizer::encodeChatPrompt(const std::string& prompt) const {
  return impl_->encodeChatPrompt(prompt);
}
std::string Qwen3Tokenizer::decodeToken(int64_t id) const { return impl_->decodeToken(id); }
std::string Qwen3Tokenizer::decode(const std::vector<int64_t>& ids) const { return impl_->decode(ids); }

}  // namespace dflash
