cmake_minimum_required(VERSION 3.18)
project(recorder_lidar_decoder LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Avoid accidentally mixing a local Protobuf 6 static library with Ubuntu's
# packaged PCL/VTK/Protobuf 3 libraries. Protobuf 6 additionally requires an
# ABI-matching Abseil installation (lts_20250127 in the reported build).
set(RECORDER_PROTOBUF_PROVIDER "system" CACHE STRING
    "Protobuf provider: system (Ubuntu packages) or config (matching custom installation)")
set_property(CACHE RECORDER_PROTOBUF_PROVIDER PROPERTY STRINGS system config)

if(RECORDER_PROTOBUF_PROVIDER STREQUAL "system")
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR
      "RECORDER_PROTOBUF_PROVIDER=system targets Ubuntu/Linux; select 'config' "
      "for a custom Protobuf installation on other systems.")
  endif()

  # Deliberately do not search /usr/local. Ubuntu 22.04's PCL/VTK packages use
  # the distribution's shared Protobuf runtime and its matching public headers.
  find_path(RECORDER_DISTRO_PROTOBUF_INCLUDE NAMES google/protobuf/message.h
    PATHS /usr/include NO_DEFAULT_PATH)
  find_library(RECORDER_DISTRO_PROTOBUF_LIBRARY NAMES protobuf
    PATHS "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}"
          "/lib/${CMAKE_LIBRARY_ARCHITECTURE}" /usr/lib /lib
    NO_DEFAULT_PATH)
  find_program(RECORDER_DISTRO_PROTOC NAMES protoc
    PATHS /usr/bin NO_DEFAULT_PATH)

  if(NOT RECORDER_DISTRO_PROTOBUF_INCLUDE OR
     NOT RECORDER_DISTRO_PROTOBUF_LIBRARY OR
     NOT RECORDER_DISTRO_PROTOC)
    message(FATAL_ERROR
      "Ubuntu Protobuf development files are incomplete. Install matching packages:\n"
      "  sudo apt install libprotobuf-dev protobuf-compiler\n"
      "Expected: /usr/bin/protoc, /usr/include/google/protobuf, "
      "/usr/lib/<multiarch>/libprotobuf.so.\n"
      "Do not link Ubuntu PCL to /usr/local/lib/libprotobuf.a.")
  endif()

  # Forcefully override any cached /usr/local discovery from a previous build.
  set(Protobuf_INCLUDE_DIR "${RECORDER_DISTRO_PROTOBUF_INCLUDE}" CACHE PATH
      "Selected Protobuf headers" FORCE)
  set(Protobuf_LIBRARY "${RECORDER_DISTRO_PROTOBUF_LIBRARY}" CACHE FILEPATH
      "Selected Protobuf library" FORCE)
  set(Protobuf_PROTOC_EXECUTABLE "${RECORDER_DISTRO_PROTOC}" CACHE FILEPATH
      "Selected protoc" FORCE)
  set(Protobuf_USE_STATIC_LIBS OFF)
  find_package(Protobuf MODULE REQUIRED)
  set(RECORDER_PROTOBUF_IMPORT_ROOT "${RECORDER_DISTRO_PROTOBUF_INCLUDE}")
  set(RECORDER_PROTOC_COMMAND "${Protobuf_PROTOC_EXECUTABLE}")

  execute_process(COMMAND "${RECORDER_PROTOC_COMMAND}" --version
    RESULT_VARIABLE _protoc_status
    OUTPUT_VARIABLE _protoc_text OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _protoc_status EQUAL 0)
    message(FATAL_ERROR "Failed to run ${RECORDER_PROTOC_COMMAND} --version")
  endif()
  string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" _protoc_version "${_protoc_text}")
  if(NOT _protoc_version)
    message(FATAL_ERROR "Unrecognized protoc version: ${_protoc_text}")
  endif()
  if(Protobuf_VERSION VERSION_LESS "3.12")
    message(FATAL_ERROR
      "Your protoc/libprotobuf are too old for proto3 optional fields: ${Protobuf_VERSION}")
  endif()
  if(NOT _protoc_version VERSION_EQUAL Protobuf_VERSION)
    message(FATAL_ERROR
      "Mixed Protobuf toolchain: protoc=${_protoc_version} "
      "library=${Protobuf_VERSION}. Install matching Ubuntu packages.")
  endif()
  # Protobuf 3.12-3.14 requires an explicit flag for proto3 optional fields
  # (used by lidar.proto); newer compilers accept them without the flag.
  if(_protoc_version VERSION_LESS "3.15")
    set(RECORDER_PROTOC_EXTRA_ARGS --experimental_allow_proto3_optional)
  endif()

  # GCC searches /usr/local/include ahead of /usr/include by default, even
  # when the library has been correctly selected from /usr/lib. Shadow just
  # the google/ include subtree with an ordinary -I path, so the compiler
  # cannot silently compile distro-generated code against Protobuf 6 headers.
  set(_protobuf_overlay "${CMAKE_CURRENT_BINARY_DIR}/distro-protobuf-include")
  file(MAKE_DIRECTORY "${_protobuf_overlay}")
  if(NOT EXISTS "${_protobuf_overlay}/google")
    file(CREATE_LINK "${RECORDER_PROTOBUF_IMPORT_ROOT}/google"
      "${_protobuf_overlay}/google" SYMBOLIC RESULT _overlay_result)
    if(NOT _overlay_result STREQUAL "0")
      message(FATAL_ERROR "Cannot create Protobuf header overlay: ${_overlay_result}")
    endif()
  endif()
elseif(RECORDER_PROTOBUF_PROVIDER STREQUAL "config")
  # An upstream static Protobuf 6 build is valid only when its exported CMake
  # target also brings in matching Abseil dependencies. Prefer that config
  # target over FindProtobuf's legacy static archive discovery.
  find_package(Protobuf CONFIG REQUIRED)
  if(NOT TARGET protobuf::libprotobuf)
    message(FATAL_ERROR "Custom Protobuf package lacks protobuf::libprotobuf")
  endif()
  if(TARGET protobuf::protoc)
    set(RECORDER_PROTOC_COMMAND "$<TARGET_FILE:protobuf::protoc>")
  elseif(Protobuf_PROTOC_EXECUTABLE)
    set(RECORDER_PROTOC_COMMAND "${Protobuf_PROTOC_EXECUTABLE}")
  else()
    message(FATAL_ERROR "Custom Protobuf config lacks protobuf::protoc")
  endif()
  set(RECORDER_PROTOBUF_IMPORT_ROOT "" CACHE PATH
      "Directory containing google/protobuf/timestamp.proto for custom Protobuf")
  if(NOT RECORDER_PROTOBUF_IMPORT_ROOT)
    get_target_property(_protobuf_include_dirs protobuf::libprotobuf INTERFACE_INCLUDE_DIRECTORIES)
    foreach(_candidate IN LISTS _protobuf_include_dirs)
      if(IS_DIRECTORY "${_candidate}" AND EXISTS "${_candidate}/google/protobuf/timestamp.proto")
        set(RECORDER_PROTOBUF_IMPORT_ROOT "${_candidate}")
        break()
      endif()
    endforeach()
  endif()
  if(NOT RECORDER_PROTOBUF_IMPORT_ROOT OR
     NOT EXISTS "${RECORDER_PROTOBUF_IMPORT_ROOT}/google/protobuf/timestamp.proto")
    message(FATAL_ERROR
      "Set -DRECORDER_PROTOBUF_IMPORT_ROOT=<prefix>/include to the custom "
      "Protobuf include directory containing google/protobuf/timestamp.proto")
  endif()
else()
  message(FATAL_ERROR "Invalid RECORDER_PROTOBUF_PROVIDER=${RECORDER_PROTOBUF_PROVIDER}")
endif()

find_package(PCL REQUIRED COMPONENTS common io)

message(STATUS "Recorder Protobuf provider: ${RECORDER_PROTOBUF_PROVIDER}")
message(STATUS "Recorder Protobuf import root: ${RECORDER_PROTOBUF_IMPORT_ROOT}")
if(RECORDER_PROTOBUF_PROVIDER STREQUAL "system")
  message(STATUS "Recorder Protobuf library: ${Protobuf_LIBRARY}")
  message(STATUS "Recorder protoc: ${RECORDER_PROTOC_COMMAND} (${_protoc_version})")
endif()

set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GENERATED_DIR}")
foreach(PROTO bag lidar)
  add_custom_command(
    OUTPUT "${GENERATED_DIR}/${PROTO}.pb.cc" "${GENERATED_DIR}/${PROTO}.pb.h"
    COMMAND "${RECORDER_PROTOC_COMMAND}"
            ${RECORDER_PROTOC_EXTRA_ARGS}
            "--proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto"
            "--proto_path=${RECORDER_PROTOBUF_IMPORT_ROOT}"
            "--cpp_out=${GENERATED_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/proto/${PROTO}.proto"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/proto/${PROTO}.proto"
    COMMENT "Generating C++ protobuf for ${PROTO}.proto"
    VERBATIM)
endforeach()

add_executable(decode_lidar cpp/decode_lidar.cpp
  "${GENERATED_DIR}/bag.pb.cc" "${GENERATED_DIR}/lidar.pb.cc")
target_include_directories(decode_lidar PRIVATE "${GENERATED_DIR}")
if(RECORDER_PROTOBUF_PROVIDER STREQUAL "system")
  target_include_directories(decode_lidar BEFORE PRIVATE "${_protobuf_overlay}")
endif()
target_include_directories(decode_lidar SYSTEM PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(decode_lidar PRIVATE protobuf::libprotobuf ${PCL_LIBRARIES})
target_compile_options(decode_lidar PRIVATE -Wall -Wextra -Wpedantic)

include(CTest)
if(BUILD_TESTING)
  add_executable(test_xyzi_writer tests/test_xyzi_writer.cpp)
  target_include_directories(test_xyzi_writer PRIVATE cpp)
  target_compile_options(test_xyzi_writer PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME xyzi_writer_byte_equivalence COMMAND test_xyzi_writer)
endif()
