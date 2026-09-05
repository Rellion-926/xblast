#include "dflash/qnn_runtime.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "HTP/QnnHtpDevice.h"

namespace dflash {
namespace {

using QnnInterfaceGetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
using QnnSystemInterfaceGetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*);

constexpr int kRpcmemHeapIdSystem = 25;
constexpr uint32_t kRpcmemDefaultFlags = 1;

QnnInterfaceGetProvidersFn qnnGetProviders = nullptr;
QnnSystemInterfaceGetProvidersFn qnnSystemGetProviders = nullptr;

void qnnLogCallback(const char* fmt, QnnLog_Level_t level, uint64_t timestamp, va_list args) {
  (void)timestamp;
  if (level > QNN_LOG_LEVEL_WARN) return;
  std::fputs("[QNN] ", stderr);
  std::vfprintf(stderr, fmt, args);
}

std::vector<uint8_t> readFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("failed to open QNN context: " + path);
  const auto size = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> data(static_cast<size_t>(size));
  if (!in.read(reinterpret_cast<char*>(data.data()), size)) {
    throw std::runtime_error("failed to read QNN context: " + path);
  }
  return data;
}

TensorDesc copyTensorDesc(const Qnn_Tensor_t& tensor, Qnn_TensorType_t io_type) {
  TensorDesc out;
  out.name = tensor.v2.name ? tensor.v2.name : "";
  out.id = tensor.v2.id;
  out.type = io_type;
  out.dtype = tensor.v2.dataType;
  out.quant = tensor.v2.quantizeParams;
  out.dims.assign(tensor.v2.dimensions, tensor.v2.dimensions + tensor.v2.rank);
  return out;
}

template <typename GraphInfoT>
Graph makeGraphFromInfo(const GraphInfoT& info) {
  Graph graph;
  graph.name = info.graphName ? info.graphName : "";
  for (uint32_t i = 0; i < info.numGraphInputs; ++i) {
    graph.inputs.push_back(copyTensorDesc(info.graphInputs[i], QNN_TENSOR_TYPE_APP_WRITE));
  }
  for (uint32_t i = 0; i < info.numGraphOutputs; ++i) {
    graph.outputs.push_back(copyTensorDesc(info.graphOutputs[i], QNN_TENSOR_TYPE_APP_READ));
  }
  return graph;
}

std::vector<Graph> copyGraphsFromBinaryInfo(const QnnSystemContext_BinaryInfo_t* binary_info) {
  if (!binary_info) throw std::runtime_error("QNN binary info is null");

  const QnnSystemContext_GraphInfo_t* graphs = nullptr;
  uint32_t num_graphs = 0;
  if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    graphs = binary_info->contextBinaryInfoV1.graphs;
    num_graphs = binary_info->contextBinaryInfoV1.numGraphs;
  } else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    graphs = binary_info->contextBinaryInfoV2.graphs;
    num_graphs = binary_info->contextBinaryInfoV2.numGraphs;
  } else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
    graphs = binary_info->contextBinaryInfoV3.graphs;
    num_graphs = binary_info->contextBinaryInfoV3.numGraphs;
  } else {
    throw std::runtime_error("unsupported QNN binary info version");
  }

  std::vector<Graph> out;
  out.reserve(num_graphs);
  for (uint32_t i = 0; i < num_graphs; ++i) {
    if (graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
      out.push_back(makeGraphFromInfo(graphs[i].graphInfoV1));
    } else if (graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
      out.push_back(makeGraphFromInfo(graphs[i].graphInfoV3));
    } else {
      throw std::runtime_error("unsupported QNN graph info version");
    }
  }
  return out;
}

}  // namespace

size_t qnnDtypeSize(Qnn_DataType_t dtype) {
  switch (dtype) {
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
      return 1;
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_BFLOAT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      return 2;
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_FLOAT_32:
      return 4;
    default:
      throw std::runtime_error("unsupported QNN dtype size");
  }
}

std::string qnnDtypeName(Qnn_DataType_t dtype) {
  switch (dtype) {
    case QNN_DATATYPE_INT_32: return "int32";
    case QNN_DATATYPE_UINT_32: return "uint32";
    case QNN_DATATYPE_FLOAT_32: return "float32";
    case QNN_DATATYPE_FLOAT_16: return "float16";
    case QNN_DATATYPE_UINT_16: return "uint16";
    case QNN_DATATYPE_UFIXED_POINT_16: return "ufix16";
    case QNN_DATATYPE_SFIXED_POINT_16: return "sfix16";
    case QNN_DATATYPE_BFLOAT_16: return "bfloat16";
    case QNN_DATATYPE_UINT_8: return "uint8";
    case QNN_DATATYPE_UFIXED_POINT_8: return "ufix8";
    case QNN_DATATYPE_INT_8: return "int8";
    case QNN_DATATYPE_SFIXED_POINT_8: return "sfix8";
    default: return "unknown";
  }
}

size_t TensorBuffer::bytes() const {
  size_t elems = 1;
  for (uint32_t d : desc.dims) elems *= d;
  return elems * qnnDtypeSize(desc.dtype);
}

void TensorBuffer::resizeForDesc() {
  host.assign(bytes(), 0);
}

void TensorBuffer::bindClientBuffer(Qnn_TensorType_t io_type) {
  tensor = QNN_TENSOR_INIT;
  tensor.version = QNN_TENSOR_VERSION_2;
  tensor.v2.id = desc.id;
  tensor.v2.name = desc.name.c_str();
  tensor.v2.type = io_type;
  tensor.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tensor.v2.dataType = desc.dtype;
  tensor.v2.quantizeParams = desc.quant;
  tensor.v2.rank = static_cast<uint32_t>(desc.dims.size());
  tensor.v2.dimensions = const_cast<uint32_t*>(desc.dims.data());
  tensor.v2.memType = QNN_TENSORMEMTYPE_RAW;
  tensor.v2.clientBuf = Qnn_ClientBuffer_t{host.data(), static_cast<uint32_t>(host.size())};
}

void TensorBuffer::bindSharedBuffer(Qnn_TensorType_t io_type, Qnn_MemHandle_t handle) {
  tensor = QNN_TENSOR_INIT;
  tensor.version = QNN_TENSOR_VERSION_2;
  tensor.v2.id = desc.id;
  tensor.v2.name = desc.name.c_str();
  tensor.v2.type = io_type;
  tensor.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tensor.v2.dataType = desc.dtype;
  tensor.v2.quantizeParams = desc.quant;
  tensor.v2.rank = static_cast<uint32_t>(desc.dims.size());
  tensor.v2.dimensions = const_cast<uint32_t*>(desc.dims.data());
  tensor.v2.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
  tensor.v2.memHandle = handle;
}

QnnRuntime::~QnnRuntime() {
  const char* explicit_release = std::getenv("DFLASH_QNN_EXPLICIT_RELEASE");
  if (!explicit_release || std::string(explicit_release) != "1") {
    // Some HTP/QNN driver stacks are not robust when several contexts/backends
    // are destroyed at process teardown. The device runner is process-scoped,
    // so the OS can reclaim these handles safely on exit.
    releaseGraphs();
    context_ = nullptr;
    profile_ = nullptr;
    device_ = nullptr;
    backend_ = nullptr;
    log_ = nullptr;
    cdsprpc_lib_ = nullptr;
    qnn_system_lib_ = nullptr;
    qnn_lib_ = nullptr;
    return;
  }
  releaseGraphs();
  for (auto& alloc : shared_allocations_) {
    if (alloc.handle && qnn_.memDeRegister) qnn_.memDeRegister(&alloc.handle, 1);
  }
  for (auto& alloc : shared_allocations_) {
    if (alloc.ptr && rpcmem_free_) rpcmem_free_(alloc.ptr);
  }
  shared_allocations_.clear();
  if (context_ && qnn_.contextFree) qnn_.contextFree(context_, nullptr);
  if (profile_ && qnn_.profileFree) qnn_.profileFree(profile_);
  if (perf_configured_ && perf_infra_.destroyPowerConfigId) {
    perf_infra_.destroyPowerConfigId(power_config_id_);
  }
  if (device_ && qnn_.deviceFree) qnn_.deviceFree(device_);
  if (backend_ && qnn_.backendFree) qnn_.backendFree(backend_);
  if (log_ && qnn_.logFree) qnn_.logFree(log_);
  if (cdsprpc_lib_) dlclose(cdsprpc_lib_);
  if (qnn_system_lib_) dlclose(qnn_system_lib_);
  if (qnn_lib_) dlclose(qnn_lib_);
}

bool QnnRuntime::open(QnnLog_Level_t log_level) {
  return loadSymbols() && createBackendAndDevice(log_level);
}

bool QnnRuntime::loadSymbols() {
  qnn_lib_ = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
  if (!qnn_lib_) {
    std::cerr << "failed to open libQnnHtp.so: " << dlerror() << "\n";
    return false;
  }
  qnnGetProviders = reinterpret_cast<QnnInterfaceGetProvidersFn>(dlsym(qnn_lib_, "QnnInterface_getProviders"));
  if (!qnnGetProviders) {
    std::cerr << "missing QnnInterface_getProviders\n";
    return false;
  }

  qnn_system_lib_ = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
  if (!qnn_system_lib_) {
    std::cerr << "failed to open libQnnSystem.so: " << dlerror() << "\n";
    return false;
  }
  qnnSystemGetProviders =
      reinterpret_cast<QnnSystemInterfaceGetProvidersFn>(dlsym(qnn_system_lib_, "QnnSystemInterface_getProviders"));
  if (!qnnSystemGetProviders) {
    std::cerr << "missing QnnSystemInterface_getProviders\n";
    return false;
  }
  return true;
}

bool QnnRuntime::loadRpcMemSymbols() {
  if (cdsprpc_lib_) return true;
  const char* candidates[] = {
      "/vendor/lib64/libcdsprpc.so",
      "/system/lib64/libcdsprpc.so",
      "libcdsprpc.so",
      nullptr,
  };
  const char* loaded_path = nullptr;
  for (const char** path = candidates; *path; ++path) {
    cdsprpc_lib_ = dlopen(*path, RTLD_NOW | RTLD_LOCAL);
    if (cdsprpc_lib_) {
      loaded_path = *path;
      break;
    }
  }
  if (!cdsprpc_lib_) {
    std::cerr << "[QNN] warning: failed to open libcdsprpc.so: " << dlerror() << "\n";
    return false;
  }
  rpcmem_alloc_ = reinterpret_cast<RpcMemAllocFn>(dlsym(cdsprpc_lib_, "rpcmem_alloc"));
  rpcmem_free_ = reinterpret_cast<RpcMemFreeFn>(dlsym(cdsprpc_lib_, "rpcmem_free"));
  rpcmem_to_fd_ = reinterpret_cast<RpcMemToFdFn>(dlsym(cdsprpc_lib_, "rpcmem_to_fd"));
  if (!rpcmem_alloc_ || !rpcmem_free_ || !rpcmem_to_fd_) {
    std::cerr << "[QNN] warning: missing rpcmem symbols\n";
    return false;
  }
  std::cout << "[QNN] rpcmem library loaded: " << loaded_path << "\n";
  return true;
}

bool QnnRuntime::createBackendAndDevice(QnnLog_Level_t log_level) {
  const QnnInterface_t** providers = nullptr;
  uint32_t num_providers = 0;
  if (qnnGetProviders(&providers, &num_providers) != QNN_SUCCESS || num_providers == 0) return false;

  bool found = false;
  for (uint32_t i = 0; i < num_providers; ++i) {
    if (providers[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
        providers[i]->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR) {
      qnn_ = providers[i]->QNN_INTERFACE_VER_NAME;
      found = true;
      break;
    }
  }
  if (!found) return false;

  if (qnn_.logCreate && qnn_.logCreate(qnnLogCallback, log_level, &log_) != QNN_SUCCESS) return false;
  if (qnn_.backendCreate(log_, nullptr, &backend_) != QNN_SUCCESS) return false;

  if (qnn_.deviceCreate) {
    QnnHtpDevice_CustomConfig_t pd_custom{};
#if defined(QNN_HTP_DEVICE_CONFIG_OPTION_UNSIGNEDPD)
    pd_custom.option = QNN_HTP_DEVICE_CONFIG_OPTION_UNSIGNEDPD;
#elif defined(QNN_HTP_DEVICE_CONFIG_OPTION_USE_UNSIGNED_PD)
    pd_custom.option = QNN_HTP_DEVICE_CONFIG_OPTION_USE_UNSIGNED_PD;
#endif
#if defined(QNN_HTP_DEVICE_CONFIG_OPTION_UNSIGNEDPD) || defined(QNN_HTP_DEVICE_CONFIG_OPTION_USE_UNSIGNED_PD)
    pd_custom.useUnsignedProcessDomain.useUnsignedProcessDomain = true;
    pd_custom.useUnsignedProcessDomain.deviceId = 0;
    QnnDevice_Config_t dev_cfg = QNN_DEVICE_CONFIG_INIT;
    dev_cfg.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
    dev_cfg.customConfig = &pd_custom;
    const QnnDevice_Config_t* dev_cfgs[] = {&dev_cfg, nullptr};
    auto rc = qnn_.deviceCreate(log_, dev_cfgs, &device_);
    if (rc != QNN_SUCCESS) rc = qnn_.deviceCreate(log_, nullptr, &device_);
    if (rc != QNN_SUCCESS) return false;
#else
    if (qnn_.deviceCreate(log_, nullptr, &device_) != QNN_SUCCESS) return false;
#endif
  }

  const char* perf_env = std::getenv("DFLASH_QNN_PERF");
  if (!perf_env || std::string(perf_env) != "0") {
    if (!configureHtpPerformance()) {
      std::cerr << "[QNN] warning: failed to configure HTP performance; continuing with defaults\n";
    }
  }

  const QnnSystemInterface_t** sys_providers = nullptr;
  uint32_t num_sys_providers = 0;
  if (qnnSystemGetProviders(&sys_providers, &num_sys_providers) != QNN_SUCCESS || num_sys_providers == 0) return false;

  found = false;
  for (uint32_t i = 0; i < num_sys_providers; ++i) {
    if (sys_providers[i]->systemApiVersion.major == QNN_SYSTEM_API_VERSION_MAJOR &&
        sys_providers[i]->systemApiVersion.minor >= QNN_SYSTEM_API_VERSION_MINOR) {
      qnn_system_ = sys_providers[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
      found = true;
      break;
    }
  }
  return found;
}

bool QnnRuntime::configureHtpPerformance() {
  if (!qnn_.deviceGetInfrastructure) return false;

  QnnDevice_Infrastructure_t device_infra = nullptr;
  auto rc = qnn_.deviceGetInfrastructure(&device_infra);
  if ((rc & 0xFFFF) != QNN_SUCCESS || !device_infra) return false;

  auto* htp_infra = static_cast<QnnHtpDevice_Infrastructure_t*>(device_infra);
  perf_infra_ = htp_infra->perfInfra;
  if (!perf_infra_.createPowerConfigId || !perf_infra_.setPowerConfig) return false;

  rc = perf_infra_.createPowerConfigId(0, 0, &power_config_id_);
  if ((rc & 0xFFFF) != QNN_SUCCESS) return false;
  perf_configured_ = true;

  QnnHtpPerfInfrastructure_PowerConfig_t dcvs = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
  dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  dcvs.dcvsV3Config.contextId = power_config_id_;
  dcvs.dcvsV3Config.setDcvsEnable = 1;
  dcvs.dcvsV3Config.dcvsEnable = 0;
  dcvs.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  dcvs.dcvsV3Config.setSleepLatency = 1;
  dcvs.dcvsV3Config.sleepLatency = 40;
  dcvs.dcvsV3Config.setSleepDisable = 0;
  dcvs.dcvsV3Config.sleepDisable = 0;
  dcvs.dcvsV3Config.setBusParams = 1;
  dcvs.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.setCoreParams = 1;
  dcvs.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  const QnnHtpPerfInfrastructure_PowerConfig_t* dcvs_cfg[] = {&dcvs, nullptr};
  rc = perf_infra_.setPowerConfig(power_config_id_, dcvs_cfg);
  if ((rc & 0xFFFF) != QNN_SUCCESS) return false;

  QnnHtpPerfInfrastructure_PowerConfig_t rpc_latency = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
  rpc_latency.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
  rpc_latency.rpcControlLatencyConfig = 100;
  const QnnHtpPerfInfrastructure_PowerConfig_t* rpc_latency_cfg[] = {&rpc_latency, nullptr};
  rc = perf_infra_.setPowerConfig(power_config_id_, rpc_latency_cfg);
  if ((rc & 0xFFFF) != QNN_SUCCESS) return false;

  QnnHtpPerfInfrastructure_PowerConfig_t rpc_polling = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
  rpc_polling.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
  rpc_polling.rpcPollingTimeConfig = 9999;
  const QnnHtpPerfInfrastructure_PowerConfig_t* rpc_polling_cfg[] = {&rpc_polling, nullptr};
  rc = perf_infra_.setPowerConfig(power_config_id_, rpc_polling_cfg);
  if ((rc & 0xFFFF) != QNN_SUCCESS) return false;

  std::cout << "[QNN] HTP performance configured: burst dcvs + rpc latency/polling\n";
  return true;
}

bool QnnRuntime::loadContextBinary(const std::string& context_path) {
  auto binary = readFile(context_path);
  if (!loadGraphMetadata(binary)) return false;

  Qnn_ContextBinarySize_t written_size = 0;
  if (qnn_.contextCreateFromBinary(backend_,
                                   device_,
                                   nullptr,
                                   binary.data(),
                                   binary.size(),
                                   &context_,
                                   profile_) != QNN_SUCCESS) {
    std::cerr << "contextCreateFromBinary failed: " << context_path << "\n";
    return false;
  }

  for (auto& graph : graphs_) {
    if (qnn_.graphRetrieve(context_, graph.name.c_str(), &graph.handle) != QNN_SUCCESS) {
      std::cerr << "graphRetrieve failed: " << graph.name << "\n";
      return false;
    }
  }

  const char* shared_env = std::getenv("DFLASH_QNN_SHARED_BUFFER");
  shared_buffers_enabled_ = !shared_env || std::string(shared_env) != "0";
  if (shared_buffers_enabled_ && !loadRpcMemSymbols()) {
    shared_buffers_enabled_ = false;
  }
  if (shared_buffers_enabled_) {
    std::cout << "[QNN] shared buffers enabled: rpcmem + QNN MEMHANDLE\n";
  }
  return true;
}

bool QnnRuntime::loadGraphMetadata(const std::vector<uint8_t>& context_binary) {
  QnnSystemContext_Handle_t sys_ctx = nullptr;
  if (qnn_system_.systemContextCreate(&sys_ctx) != QNN_SUCCESS) return false;
  const QnnSystemContext_BinaryInfo_t* binary_info = nullptr;
  Qnn_ContextBinarySize_t binary_info_size = 0;
  auto rc = qnn_system_.systemContextGetBinaryInfo(sys_ctx,
                                                   const_cast<uint8_t*>(context_binary.data()),
                                                   context_binary.size(),
                                                   &binary_info,
                                                   &binary_info_size);
  if (rc != QNN_SUCCESS) {
    qnn_system_.systemContextFree(sys_ctx);
    return false;
  }

  graphs_ = copyGraphsFromBinaryInfo(binary_info);
  graph_index_.clear();
  for (size_t i = 0; i < graphs_.size(); ++i) graph_index_[graphs_[i].name] = i;
  qnn_system_.systemContextFree(sys_ctx);
  return true;
}

void QnnRuntime::releaseGraphs() {
  graphs_.clear();
  graph_index_.clear();
}

bool QnnRuntime::hasGraph(const std::string& graph_name) const {
  return graph_index_.find(graph_name) != graph_index_.end();
}

const Graph& QnnRuntime::graph(const std::string& graph_name) const {
  auto it = graph_index_.find(graph_name);
  if (it == graph_index_.end()) throw std::runtime_error("missing graph: " + graph_name);
  return graphs_[it->second];
}

std::vector<std::string> QnnRuntime::graphNames() const {
  std::vector<std::string> names;
  names.reserve(graphs_.size());
  for (const auto& graph : graphs_) names.push_back(graph.name);
  return names;
}

bool QnnRuntime::prepareBuffer(TensorBuffer& buffer, Qnn_TensorType_t io_type) {
  if (shared_buffers_enabled_ && registerSharedBuffer(buffer, io_type)) return true;
  buffer.resizeForDesc();
  buffer.bindClientBuffer(io_type);
  return true;
}

bool QnnRuntime::registerSharedBuffer(TensorBuffer& buffer, Qnn_TensorType_t io_type) {
  if (!context_ || !qnn_.memRegister || !rpcmem_alloc_ || !rpcmem_to_fd_) return false;

  const size_t required = buffer.bytes();
  const size_t aligned = (required + 15u) & ~static_cast<size_t>(15u);
  if (aligned > static_cast<size_t>(std::numeric_limits<int>::max())) return false;

  void* ptr = rpcmem_alloc_(kRpcmemHeapIdSystem, kRpcmemDefaultFlags, static_cast<int>(aligned));
  if (!ptr) {
    std::cerr << "[QNN] warning: rpcmem_alloc failed bytes=" << aligned << "\n";
    return false;
  }
  std::memset(ptr, 0, aligned);

  int fd = rpcmem_to_fd_(ptr);
  if (fd == -1) {
    std::cerr << "[QNN] warning: rpcmem_to_fd failed; falling back to raw buffer\n";
    if (rpcmem_free_) rpcmem_free_(ptr);
    return false;
  }

  buffer.host.bindExternal(ptr, aligned);
  buffer.bindClientBuffer(io_type);

  Qnn_MemDescriptor_t mem_desc = QNN_MEM_DESCRIPTOR_INIT;
  mem_desc.memShape.numDim = buffer.tensor.v2.rank;
  mem_desc.memShape.dimSize = buffer.tensor.v2.dimensions;
  mem_desc.memShape.shapeConfig = nullptr;
  mem_desc.dataType = buffer.tensor.v2.dataType;
  mem_desc.memType = QNN_MEM_TYPE_ION;
  mem_desc.ionInfo.fd = fd;

  Qnn_MemHandle_t handle = nullptr;
  auto rc = qnn_.memRegister(context_, &mem_desc, 1u, &handle);
  if ((rc & 0xFFFF) != QNN_SUCCESS || !handle) {
    std::cerr << "[QNN] warning: memRegister failed rc=" << (rc & 0xFFFF)
              << "; falling back to raw buffer\n";
    if (rpcmem_free_) rpcmem_free_(ptr);
    buffer.host.external = nullptr;
    buffer.host.external_size = 0;
    buffer.mem_handle = nullptr;
    buffer.shared_buffer = false;
    return false;
  }

  buffer.mem_handle = handle;
  buffer.shared_buffer = true;
  buffer.bindSharedBuffer(io_type, handle);
  shared_allocations_.push_back(SharedAllocation{ptr, aligned, fd, handle});
  return true;
}

std::vector<TensorBuffer> QnnRuntime::makeInputBuffers(const std::string& graph_name) {
  std::vector<TensorBuffer> buffers;
  for (const auto& desc : graph(graph_name).inputs) {
    TensorBuffer buf;
    buf.desc = desc;
    prepareBuffer(buf, QNN_TENSOR_TYPE_APP_WRITE);
    buffers.push_back(std::move(buf));
  }
  return buffers;
}

std::vector<TensorBuffer> QnnRuntime::makeOutputBuffers(const std::string& graph_name) {
  std::vector<TensorBuffer> buffers;
  for (const auto& desc : graph(graph_name).outputs) {
    TensorBuffer buf;
    buf.desc = desc;
    prepareBuffer(buf, QNN_TENSOR_TYPE_APP_READ);
    buffers.push_back(std::move(buf));
  }
  return buffers;
}

bool QnnRuntime::execute(const std::string& graph_name,
                         std::vector<TensorBuffer>& inputs,
                         std::vector<TensorBuffer>& outputs) {
  auto& g = graph(graph_name);
  if (inputs.size() != g.inputs.size() || outputs.size() != g.outputs.size()) return false;

  std::vector<Qnn_Tensor_t> qnn_inputs;
  std::vector<Qnn_Tensor_t> qnn_outputs;
  qnn_inputs.reserve(inputs.size());
  qnn_outputs.reserve(outputs.size());
  for (auto& input : inputs) {
    if (input.shared_buffer) {
      input.bindSharedBuffer(QNN_TENSOR_TYPE_APP_WRITE, input.mem_handle);
    } else {
      input.bindClientBuffer(QNN_TENSOR_TYPE_APP_WRITE);
    }
    qnn_inputs.push_back(input.tensor);
  }
  for (auto& output : outputs) {
    if (output.shared_buffer) {
      output.bindSharedBuffer(QNN_TENSOR_TYPE_APP_READ, output.mem_handle);
    } else {
      output.bindClientBuffer(QNN_TENSOR_TYPE_APP_READ);
    }
    qnn_outputs.push_back(output.tensor);
  }

  auto rc = qnn_.graphExecute(g.handle,
                              qnn_inputs.data(),
                              qnn_inputs.size(),
                              qnn_outputs.data(),
                              qnn_outputs.size(),
                              profile_,
                              nullptr);
  if ((rc & 0xFFFF) != QNN_SUCCESS) {
    std::cerr << "graphExecute failed graph=" << graph_name << " rc=" << (rc & 0xFFFF) << "\n";
    return false;
  }
  return true;
}

}  // namespace dflash
