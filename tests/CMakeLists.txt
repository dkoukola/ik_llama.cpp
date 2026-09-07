#llama_add_compile_flags()

function(llama_build source)
    set(TEST_SOURCES ${source} ${ARGN})

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE common)
    install(TARGETS ${TEST_TARGET} RUNTIME)
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
function(llama_target_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS} get-model.cpp)

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    target_link_libraries(${TEST_TARGET} PRIVATE common)
    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS} get-model.cpp)

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE common)

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
add_executable(test-tokenizer-0 test-tokenizer-0.cpp)
target_link_libraries(test-tokenizer-0 PRIVATE common)
install(TARGETS test-tokenizer-0 RUNTIME)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-starcoder.gguf)

if (LLAMA_LLGUIDANCE)
    llama_target_and_test(test-grammar-llguidance.cpp ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-bpe.gguf)
endif ()

if (NOT WIN32)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API
    #llama_target_and_test(test-sampling.cpp)
    #llama_target_and_test(test-grammar-parser.cpp)
    #llama_target_and_test(test-grammar-integration.cpp)
    #llama_target_and_test(test-llama-grammar.cpp)
    #llama_target_and_test(test-chat.cpp)
    # TODO: disabled on loongarch64 because the ggml-ci node lacks Python 3.8
    #if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
    #    llama_target_and_test(test-json-schema-to-grammar.cpp   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
    #    target_include_directories(test-json-schema-to-grammar PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../examples/server)
    #endif()


    # build test-tokenizer-1-bpe target once and add many tests
    # add_executable(test-tokenizer-1-bpe test-tokenizer-1-bpe.cpp)
    # target_link_libraries(test-tokenizer-1-bpe PRIVATE common)
    # install(TARGETS test-tokenizer-1-bpe RUNTIME)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    # add_executable(test-tokenizer-1-spm test-tokenizer-1-spm.cpp)
    # target_link_libraries(test-tokenizer-1-spm PRIVATE common)
    # install(TARGETS test-tokenizer-1-spm RUNTIME)

    # llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-baichuan.gguf)

    # llama_target_and_test(test-double-float.cpp) # SLOW
endif()

#llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp )
llama_build_and_test(test-json-partial.cpp )
#llama_build_and_test(test-log.cpp)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-json-serialization.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
)
llama_build_and_test(test-regex-partial.cpp)

llama_build_and_test(test-iqk-mul-multi-add.cpp)
target_include_directories(test-iqk-mul-multi-add PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)

add_executable(test-numa-barrier-concurrency test-numa-barrier-concurrency.cpp)
target_link_libraries(test-numa-barrier-concurrency PRIVATE ggml)
llama_test(test-numa-barrier-concurrency)
set_tests_properties(test-numa-barrier-concurrency PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)

if (GGML_IQK_MUL_MAT AND GGML_IQK_FLASH_ATTENTION)
    llama_build_and_test(test-iqk-fa-work-buffer.cpp)
    target_include_directories(test-iqk-fa-work-buffer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)

    add_executable(test-iqk-indexed-fa test-iqk-indexed-fa.cpp)
    target_include_directories(test-iqk-indexed-fa PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
    target_link_libraries(test-iqk-indexed-fa PRIVATE ggml)
    llama_test(test-iqk-indexed-fa)
endif()
if (GGML_IQK_MUL_MAT)
    add_executable(test-iqk-dsv4-attn test-iqk-dsv4-attn.cpp)
    target_include_directories(test-iqk-dsv4-attn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
    target_link_libraries(test-iqk-dsv4-attn PRIVATE ggml)
    llama_test(test-iqk-dsv4-attn)

    add_executable(test-iqk-numa-mirror-small test-iqk-numa-mirror-small.cpp)
    target_include_directories(test-iqk-numa-mirror-small PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
    target_link_libraries(test-iqk-numa-mirror-small PRIVATE ggml)
    llama_test(test-iqk-numa-mirror-small)

    add_executable(test-iqk-hc-f32 test-iqk-hc-f32.cpp)
    target_include_directories(test-iqk-hc-f32 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
    target_link_libraries(test-iqk-hc-f32 PRIVATE ggml)
    llama_test(test-iqk-hc-f32)
endif()
llama_build_and_test(test-rms-norm-parallel.cpp)
add_executable(test-rope-parallel test-rope-parallel.cpp)
target_link_libraries(test-rope-parallel PRIVATE ggml)
llama_test(test-rope-parallel)
llama_build_and_test(test-speculative-config.cpp)
target_include_directories(test-speculative-config PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_executable(test-dflash-schema test-dflash-schema.cpp)
target_include_directories(test-dflash-schema PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test-dflash-schema PRIVATE llama)
add_test(NAME test-dflash-schema COMMAND test-dflash-schema)
set_property(TEST test-dflash-schema PROPERTY LABELS model)
llama_build_and_test(test-iq4-ks-kt-decode.cpp)
target_include_directories(test-iq4-ks-kt-decode PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
llama_build_and_test(test-quantize-shape-fallback.cpp)

# llama_target_and_test(test-opt.cpp) # SLOW

llama_target_and_test(test-model-load-cancel.cpp  LABEL "model")
llama_target_and_test(test-autorelease.cpp        LABEL "model")


# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

add_executable(test-context-memory test-context-memory.cpp)
target_include_directories(test-context-memory PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/ggml/src)
target_link_libraries(test-context-memory PRIVATE llama)
add_test(NAME test-context-memory COMMAND test-context-memory)

add_executable(test-swa-state test-swa-state.cpp)
target_include_directories(test-swa-state PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test-swa-state PRIVATE llama)
add_test(NAME test-swa-state COMMAND test-swa-state)
set_property(TEST test-swa-state PROPERTY LABELS model)

add_executable(test-qwen4exp-state test-qwen4exp-state.cpp)
target_include_directories(test-qwen4exp-state PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test-qwen4exp-state PRIVATE llama)
add_test(NAME test-qwen4exp-state COMMAND test-qwen4exp-state)
set_property(TEST test-qwen4exp-state PROPERTY LABELS model)

add_executable(test-qwen4exp-mtp test-qwen4exp-mtp.cpp)
target_include_directories(test-qwen4exp-mtp PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test-qwen4exp-mtp PRIVATE llama)
if (TARGET llama::speculative)
    target_link_libraries(test-qwen4exp-mtp PRIVATE llama::speculative)
    target_compile_definitions(test-qwen4exp-mtp PRIVATE LLAMA_TEST_SPECULATIVE_BRIDGE)
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)
set(QWEN4EXP_MTP_NUMPY_RESULT 1)
if (Python3_Interpreter_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import numpy"
        RESULT_VARIABLE QWEN4EXP_MTP_NUMPY_RESULT
        OUTPUT_QUIET
        ERROR_QUIET)
endif()

if (Python3_Interpreter_FOUND AND QWEN4EXP_MTP_NUMPY_RESULT EQUAL 0)
    set(QWEN4EXP_MTP_FIXTURE_DIR ${CMAKE_CURRENT_BINARY_DIR}/qwen4exp-mtp-fixtures)
    set(QWEN4EXP_MTP_EMBEDDED ${QWEN4EXP_MTP_FIXTURE_DIR}/embedded.gguf)
    set(QWEN4EXP_MTP_SPLIT ${QWEN4EXP_MTP_FIXTURE_DIR}/split.gguf)
    set(QWEN4EXP_MTP_FUSED ${QWEN4EXP_MTP_FIXTURE_DIR}/fused.gguf)
    set(QWEN4EXP_MTP_SHARED ${QWEN4EXP_MTP_FIXTURE_DIR}/shared.gguf)
    set(QWEN4EXP_MTP_INCOMPLETE ${QWEN4EXP_MTP_FIXTURE_DIR}/incomplete.gguf)
    set(QWEN4EXP_MTP_MIXED ${QWEN4EXP_MTP_FIXTURE_DIR}/mixed.gguf)
    set(QWEN4EXP_MTP_SHARED_EMBEDDED ${QWEN4EXP_MTP_FIXTURE_DIR}/shared-embedded.gguf)
    set(QWEN4EXP_MTP_SHARED_IO ${QWEN4EXP_MTP_FIXTURE_DIR}/shared-io.gguf)
    file(MAKE_DIRECTORY ${QWEN4EXP_MTP_FIXTURE_DIR})

    add_test(
        NAME generate-qwen4exp-mtp-fixtures
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/generate-qwen4exp-mtp-gguf.py
            ${QWEN4EXP_MTP_EMBEDDED}
            ${QWEN4EXP_MTP_SPLIT}
            ${QWEN4EXP_MTP_FUSED}
            ${QWEN4EXP_MTP_SHARED}
            ${QWEN4EXP_MTP_INCOMPLETE}
            ${QWEN4EXP_MTP_MIXED}
            ${QWEN4EXP_MTP_SHARED_EMBEDDED}
            ${QWEN4EXP_MTP_SHARED_IO})
    set_property(
        TEST generate-qwen4exp-mtp-fixtures
        PROPERTY FIXTURES_SETUP qwen4exp-mtp-fixtures)

    add_test(
        NAME test-qwen4exp-mtp
        COMMAND test-qwen4exp-mtp
            ${QWEN4EXP_MTP_EMBEDDED}
            ${QWEN4EXP_MTP_SPLIT}
            ${QWEN4EXP_MTP_FUSED}
            ${QWEN4EXP_MTP_SHARED}
            --reject-load ${QWEN4EXP_MTP_INCOMPLETE}
            --reject-load ${QWEN4EXP_MTP_MIXED}
            --reject-load ${QWEN4EXP_MTP_SHARED_EMBEDDED}
            --reject-load ${QWEN4EXP_MTP_SHARED_IO})
    set_tests_properties(
        test-qwen4exp-mtp
        PROPERTIES
            FIXTURES_REQUIRED qwen4exp-mtp-fixtures
            LABELS model)
else()
    message(STATUS "test-qwen4exp-mtp is not registered: Python 3 with NumPy is required")
endif()

if (TARGET llama::speculative)
    add_executable(test-speculative-api test-speculative-api.c)
    target_link_libraries(test-speculative-api PRIVATE llama::speculative llama)
    add_test(NAME test-speculative-api COMMAND test-speculative-api)

    add_executable(test-speculative-sampling
        test-speculative-sampling.cpp
        ../common/speculative-sampling.cpp)
    target_include_directories(test-speculative-sampling PRIVATE ${PROJECT_SOURCE_DIR}/common)
    target_link_libraries(test-speculative-sampling PRIVATE llama)
    add_test(NAME test-speculative-sampling COMMAND test-speculative-sampling)

    add_executable(test-speculative-carry
        test-speculative-carry.cpp
        ../common/speculative-sampling.cpp)
    target_include_directories(test-speculative-carry PRIVATE ${PROJECT_SOURCE_DIR}/common)
    target_link_libraries(test-speculative-carry PRIVATE llama)
    add_test(NAME test-speculative-carry COMMAND test-speculative-carry)
endif()
