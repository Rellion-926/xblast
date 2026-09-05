#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "QnnInterface.h"
#include "QnnMem.h"
#include "QnnTypes.h"
#include "HTP/QnnHtpDevice.h"
#include "System/QnnSystemInterface.h"

namespace dflash {

struct TensorDesc {
  std::string name;
  Qnn_TensorType_t type = QNN_TENSOR_TYPE_UNDEFINED;
  Qnn_DataType_t dtype = QNN_DATATYPE_UNDEFINED;
  Qnn_QuantizeParams_t quant = QNN_QUANTIZE_PARAMS_INIT;
  std::vector<uint32_t> dims;
  uint32_t id = 0;
};

struct HostBuffer {
  std::vector<uint8_t> owned;
  uint8_t* external = nullptr;
  size_t external_size = 0;

  uint8_t* data() { return external ? external : owned.data(); }
  const uint8_t* data() const { return external ? external : owned.data(); }
  size_t size() const { return external ? external_size : owned.size(); }
  bool empty() const { return size() == 0; }
  uint8_t* begin() { return data(); }
  uint8_t* end() { return data() + size(); }
  const uint8_t* begin() const { return data(); }
  const uint8_t* end() const { return data() + size(); }

  void assign(size_t n, uint8_t value) {
    if (external) {
      if (n > external_size) std::abort();
      std::fill(begin(), begin() + static_cast<std::ptrdiff_t>(n), value);
      if (n < external_size) std::fill(begin() + static_cast<std::ptrdiff_t>(n), end(), 0);
      return;
    }
    owned.assign(n, value);
  }

  void bindExternal(void* ptr, size_t n) {
    owned.clear();
    owned.shrink_to_fit();
    external = static_cast<uint8_t*>(ptr);
    external_size = n;
  }
};

struct TensorBuffer {
  TensorDesc desc;
  HostBuffer host;
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  Qnn_MemHandle_t mem_handle = nullptr;
  bool shared_buffer = false;

  void resizeForDesc();
  void bindClientBuffer(Qnn_TensorType_t io_type);
  void bindSharedBuffer(Qnn_TensorType_t io_type, Qnn_MemHandle_t handle);
  size_t bytes() const;
};

struct Graph {
  std::string name;
  Qnn_GraphHandle_t handle = nullptr;
  std::vector<TensorDesc> inputs;
  std::vector<TensorDesc> outputs;
};

class QnnRuntime {
 public:
  QnnRuntime() = default;
  ~QnnRuntime();

  QnnRuntime(const QnnRuntime&) = delete;
  QnnRuntime& operator=(const QnnRuntime&) = delete;

  bool open(QnnLog_Level_t log_level = QNN_LOG_LEVEL_ERROR);
  bool loadContextBinary(const std::string& context_path);
  bool hasGraph(const std::string& graph_name) const;
  const Graph& graph(const std::string& graph_name) const;
  std::vector<std::string> graphNames() const;

  std::vector<TensorBuffer> makeInputBuffers(const std::string& graph_name);
  std::vector<TensorBuffer> makeOutputBuffers(const std::string& graph_name);
  bool execute(const std::string& graph_name,
               std::vector<TensorBuffer>& inputs,
               std::vector<TensorBuffer>& outputs);

  const QNN_INTERFACE_VER_TYPE& qnn() const { return qnn_; }
  Qnn_ContextHandle_t context() const { return context_; }

 private:
  using RpcMemAllocFn = void* (*)(int, uint32_t, int);
  using RpcMemFreeFn = void (*)(void*);
  using RpcMemToFdFn = int (*)(void*);

  struct SharedAllocation {
    void* ptr = nullptr;
    size_t size = 0;
    int fd = -1;
    Qnn_MemHandle_t handle = nullptr;
  };

  bool loadSymbols();
  bool loadRpcMemSymbols();
  bool createBackendAndDevice(QnnLog_Level_t log_level);
  bool configureHtpPerformance();
  bool loadGraphMetadata(const std::vector<uint8_t>& context_binary);
  bool prepareBuffer(TensorBuffer& buffer, Qnn_TensorType_t io_type);
  bool registerSharedBuffer(TensorBuffer& buffer, Qnn_TensorType_t io_type);
  void releaseGraphs();

  QNN_INTERFACE_VER_TYPE qnn_{};
  QNN_SYSTEM_INTERFACE_VER_TYPE qnn_system_{};
  Qnn_LogHandle_t log_ = nullptr;
  Qnn_BackendHandle_t backend_ = nullptr;
  Qnn_DeviceHandle_t device_ = nullptr;
  Qnn_ContextHandle_t context_ = nullptr;
  Qnn_ProfileHandle_t profile_ = nullptr;
  QnnHtpDevice_PerfInfrastructure_t perf_infra_{};
  uint32_t power_config_id_ = 0;
  bool perf_configured_ = false;
  bool shared_buffers_enabled_ = true;

  void* qnn_lib_ = nullptr;
  void* qnn_system_lib_ = nullptr;
  void* cdsprpc_lib_ = nullptr;
  RpcMemAllocFn rpcmem_alloc_ = nullptr;
  RpcMemFreeFn rpcmem_free_ = nullptr;
  RpcMemToFdFn rpcmem_to_fd_ = nullptr;
  std::vector<SharedAllocation> shared_allocations_;
  std::vector<Graph> graphs_;
  std::unordered_map<std::string, size_t> graph_index_;
};

size_t qnnDtypeSize(Qnn_DataType_t dtype);
std::string qnnDtypeName(Qnn_DataType_t dtype);

}  // namespace dflash
