#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dflash {

class Qwen3Tokenizer {
 public:
  Qwen3Tokenizer();
  ~Qwen3Tokenizer();

  bool load(const std::string& tokenizer_json);

  std::vector<int64_t> encodeText(const std::string& text) const;
  std::vector<int64_t> encodeChatPrompt(const std::string& prompt) const;

  std::string decodeToken(int64_t id) const;
  std::string decode(const std::vector<int64_t>& ids) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace dflash
