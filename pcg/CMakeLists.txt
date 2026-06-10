option(DMPF64_NATIVE "Enable native CPU optimizations" ON)
if(NOT DEFINED FPSI_ENABLE_AVX512)
    option(FPSI_ENABLE_AVX512 "Enable AVX-512 code paths" ON)
endif()

add_library(dmpf64
    src/node.c src/gf64.c src/rng.c src/prg.c src/dpf.c src/polynomial.c
    src/ring.c src/fft_additive.c src/ole_pcg.c src/dmpf_wrappers.c
    src/big_state.c src/dmpf_any.c src/polynomial_any.c src/ole_pcg_backend.c
)
add_library(pcg::dmpf64 ALIAS dmpf64)
target_include_directories(dmpf64 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(dmpf64 PUBLIC gf64::gf64)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(dmpf64 PRIVATE -O3 -Wall -Wextra -Wno-unused-parameter)
    if(DMPF64_NATIVE AND FPSI_ENABLE_NATIVE)
        target_compile_options(dmpf64 PRIVATE -march=native)
    endif()
    if(NOT FPSI_ENABLE_AVX512)
        target_compile_options(dmpf64 PRIVATE -mno-avx512f -mno-avx512bw -mno-avx512vl -mno-vpclmulqdq -mno-vaes)
    endif()

    if(FPSI_ENABLE_AVX512)
        target_compile_options(dmpf64 PRIVATE
            -mavx512f -mavx512bw -mavx512vl -mavx512dq
            -mvaes -mvpclmulqdq -mpclmul -maes -msse4.1)
        target_compile_definitions(dmpf64 PRIVATE DMPF64_FORCE_AVX512_VAES_VPCLMUL=1)
    endif()
endif()

add_executable(test_dpf test/test_dpf.c)
target_link_libraries(test_dpf PRIVATE dmpf64)
add_test(NAME dpf COMMAND test_dpf)

add_executable(test_ole test/test_ole.c)
target_link_libraries(test_ole PRIVATE dmpf64)
add_test(NAME ole COMMAND test_ole)

add_executable(test_big_state test/test_big_state.c)
target_link_libraries(test_big_state PRIVATE dmpf64)
add_test(NAME big_state COMMAND test_big_state)

add_executable(test_ole_backend test/test_ole_backend.c)
target_link_libraries(test_ole_backend PRIVATE dmpf64)
add_test(NAME ole_backend COMMAND test_ole_backend)


add_executable(test_afft test/test_afft.c)
target_link_libraries(test_afft PRIVATE dmpf64)
add_test(NAME afft COMMAND test_afft)

add_executable(bench_pcg bench/bench_pcg.c)
target_link_libraries(bench_pcg PRIVATE dmpf64)
