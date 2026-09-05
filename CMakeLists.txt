cmake_minimum_required(VERSION 3.22)
project(dflash_npu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED QAIRT_SDK_ROOT)
  set(QAIRT_SDK_ROOT "$ENV{QAIRT_SDK_ROOT}")
endif()
if(NOT QAIRT_SDK_ROOT)
  message(FATAL_ERROR "Set QAIRT_SDK_ROOT to the Qualcomm AI Runtime SDK path")
endif()

add_library(dflash_qnn_runtime
  src/qnn_runtime.cpp
  src/kv_cache.cpp
  src/qwen3_tokenizer.cpp
  src/runtime_io.cpp
  src/runtime_session.cpp
)
target_include_directories(dflash_qnn_runtime
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${QAIRT_SDK_ROOT}/include/QNN
)
target_link_libraries(dflash_qnn_runtime PUBLIC dl)

add_executable(dflash-runtime runtime/dflash_runtime.cpp)
target_link_libraries(dflash-runtime PRIVATE dflash_qnn_runtime)
