cmake_minimum_required(VERSION 3.20)
project(fpsi_refactored LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)
endif()

option(FPSI_ENABLE_NATIVE "Enable -march=native for local optimized builds" ON)
option(FPSI_ENABLE_AVX512 "Enable AVX-512/VAES/VPCLMUL code paths" ON)
option(FPSI_REQUIRE_AVX512_VAES_VPCLMUL "Fail configure if the default AVX512+VAES+VPCLMUL toolchain flags are unavailable" ON)
option(FPSI_ENABLE_LTO "Enable interprocedural/LTO optimization when supported" ON)

# The cleaned FPSI benchmark tree targets a modern x86-64 server profile by
# default: AVX-512 + VAES + VPCLMULQDQ.  Portable fallback builds are kept only
# for explicit debugging and are not the reference benchmark configuration.
if(FPSI_ENABLE_AVX512)
  include(CheckCCompilerFlag)
  include(CheckCXXCompilerFlag)
  set(FPSI_REQUIRED_SIMD_FLAGS
      -mavx512f -mavx512bw -mavx512vl -mavx512dq
      -mvaes -mvpclmulqdq -mpclmul -maes -msse4.1)
  string(REPLACE ";" " " FPSI_REQUIRED_SIMD_FLAGS_STR "${FPSI_REQUIRED_SIMD_FLAGS}")

  set(CMAKE_REQUIRED_FLAGS "${FPSI_REQUIRED_SIMD_FLAGS_STR}")
  check_c_compiler_flag("" FPSI_C_HAS_REQUIRED_SIMD_FLAGS)
  check_cxx_compiler_flag("" FPSI_CXX_HAS_REQUIRED_SIMD_FLAGS)
  unset(CMAKE_REQUIRED_FLAGS)

  if(FPSI_REQUIRE_AVX512_VAES_VPCLMUL AND
     (NOT FPSI_C_HAS_REQUIRED_SIMD_FLAGS OR NOT FPSI_CXX_HAS_REQUIRED_SIMD_FLAGS))
    message(FATAL_ERROR
      "Default build requires AVX512+VAES+VPCLMUL compiler flags: ${FPSI_REQUIRED_SIMD_FLAGS_STR}. "
      "Configure with -DFPSI_ENABLE_AVX512=OFF -DFPSI_REQUIRE_AVX512_VAES_VPCLMUL=OFF only for explicit portable debugging.")
  endif()

  if(FPSI_C_HAS_REQUIRED_SIMD_FLAGS AND FPSI_CXX_HAS_REQUIRED_SIMD_FLAGS)
    add_compile_options(${FPSI_REQUIRED_SIMD_FLAGS})
    add_compile_definitions(FPSI_DEFAULT_AVX512_VAES_VPCLMUL=1)
    message(STATUS "FPSI default SIMD profile: AVX512 + VAES + VPCLMULQDQ")
  endif()
endif()

if(FPSI_ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT FPSI_IPO_SUPPORTED OUTPUT FPSI_IPO_ERROR)
  if(FPSI_IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  endif()
endif()

enable_testing()

# Dependency direction: gf64 -> pcg, okvs -> opprf.
add_subdirectory(gf64)
add_subdirectory(pcg)
add_subdirectory(okvs)
add_subdirectory(opprf)
add_subdirectory(fpsi)
