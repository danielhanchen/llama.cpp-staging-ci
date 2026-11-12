# Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
llama_add_compile_flags()

# Optional args:
# - LINK: libraries to link against (defaults to llama-common, which also brings llama)
function(llama_build source)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs LINK)
    cmake_parse_arguments(LLAMA_BUILD "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_BUILD_UNPARSED_ARGUMENTS})

    if (NOT LLAMA_BUILD_LINK)
        set(LLAMA_BUILD_LINK llama-common)
    endif()

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    target_link_libraries(${TEST_TARGET} PRIVATE ${LLAMA_BUILD_LINK})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
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

function(llama_test_cmd target)
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

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
# - LINK: libraries to link against (defaults to llama-common, which also brings llama)
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS LINK)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS})

    if (NOT LLAMA_TEST_LINK)
        set(LLAMA_TEST_LINK llama-common)
    endif()

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
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE ${LLAMA_TEST_LINK})

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gemma-4           ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen35            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen35.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
endif ()

# llama_context state tests: KV and recurrent state save, load, copy, rollback
set(LLAMA_CONTEXT_TEST_SRC
    test-llama-context.cpp
    llama-context/cases-kv.cpp
    llama-context/cases-rs.cpp
    llama-context/test-case.cpp
    llama-context/test-log.cpp
    llama-context/test-runner.cpp
    llama-context/test-util.cpp
)
llama_build(${LLAMA_CONTEXT_TEST_SRC})
llama_build(test-state-async.cpp)

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-unicode.cpp LINK llama)
    llama_build_and_test(test-sampling.cpp LINK llama)
    llama_build_and_test(test-reasoning-budget.cpp)
    llama_build_and_test(test-grammar-parser.cpp LINK llama)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp LINK llama)
    llama_build_and_test(test-batch-alloc.cpp LINK llama)
    llama_build_and_test(test-chat.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    llama_build_and_test(test-server-component.cpp)
    target_include_directories(test-server-component PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-component PRIVATE server-context)
    llama_build_and_test(test-server-prompt-ssd.cpp)
    target_include_directories(test-server-prompt-ssd PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-prompt-ssd PRIVATE server-context)
    llama_build_and_test(test-server-tpp.cpp)
    target_include_directories(test-server-tpp PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    llama_build_and_test(test-json-schema-to-grammar.cpp)

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp LINK llama)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW

    # TODO: after `test-llama-archs` is refactored to not link internal symbols, we can move all related tests out of
    #       this branch and run them on all platforms
    llama_build(test-llama-archs.cpp)

    set(MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models/")
    file(MAKE_DIRECTORY "${MODEL_DIR}")

    llama_test(
        test-llama-archs
        NAME test-generate-models
        LABEL main
        ARGS -o "${MODEL_DIR}"
    )
    set_tests_properties(test-generate-models PROPERTIES
        FIXTURES_SETUP generate-models
    )

    # Test context state handling across all architectures, using the generated dummy models
    llama_test(
        test-llama-context
        LABEL main
        ARGS --models "${MODEL_DIR}"
    )
    # abort on graph reallocations that the initial reserve should have covered
    set_tests_properties(test-llama-context PROPERTIES
        FIXTURES_REQUIRED generate-models
        ENVIRONMENT "GGML_SCHED_DEBUG_REALLOC=1"
    )

    # recurrent single-token decodes (states read in place) equal one prompt of the same tokens on the CPU
    llama_test(
        test-llama-archs
        NAME test-rs-decode-parity
        LABEL main
        ARGS --rs-decode-parity -s 1
    )

    # an exact verify row keeps every cell of its sequence when the cells do not follow the positions (two sequences in a unified KV)
    llama_build(test-exact-verify-cells.cpp)
    llama_test(
        test-exact-verify-cells
        LABEL main
        ARGS "${MODEL_DIR}/llama-dense.gguf"
    )
    set_tests_properties(test-exact-verify-cells PROPERTIES FIXTURES_REQUIRED generate-models)

    # --moe-cache-mib with the layers on a GPU outside ggml-cuda (a mock device): the context runs without the MoE cache
    llama_build(test-moe-cache-device.cpp)
    llama_test(
        test-moe-cache-device
        LABEL main
        ARGS "${MODEL_DIR}/llama-moe.gguf"
    )
    set_tests_properties(test-moe-cache-device PROPERTIES FIXTURES_REQUIRED generate-models)

    # k19-graph: flash attention with its KV length on the device vs the unpadded node, byte for byte (GPU; SKIP without one)
    llama_build(test-fa-kvlen-exact.cpp)
    llama_build(test-fa-kvlen-rows.cpp)   # k19-nanfix: small-batch flash attention with the device KV length, finite and against a reference

    # the SSD prompt-cache namespace covers the --override-kv metadata overrides (a restart with another override restored stale state)
    llama_build(test-server-prompt-ssd-ns.cpp)
    target_include_directories(test-server-prompt-ssd-ns PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-prompt-ssd-ns PRIVATE server-context)
    llama_test(
        test-server-prompt-ssd-ns
        LABEL main
        ARGS "${MODEL_DIR}/llama-dense.gguf"
    )
    set_tests_properties(test-server-prompt-ssd-ns PROPERTIES FIXTURES_REQUIRED generate-models)

    # k19-rollback: a drafted window rolled back by the host copy of the recurrent state (no rollback rows) = the accepted tokens decoded alone
    llama_build(test-rs-snap.cpp)
    llama_test(
        test-rs-snap
        LABEL main
        ARGS "${MODEL_DIR}/qwen4exp-moe.gguf"
    )
    set_tests_properties(test-rs-snap PROPERTIES FIXTURES_REQUIRED generate-models)
    llama_test(
        test-rs-snap
        NAME test-rs-snap-qwen35
        LABEL main
        ARGS "${MODEL_DIR}/qwen35-dense.gguf"
    )
    set_tests_properties(test-rs-snap-qwen35 PROPERTIES FIXTURES_REQUIRED generate-models)

    # -sm tensor (Meta(CPU, CPU)) with LLAMA_RS_REPLAY=1: the delta-net rollback keeps the snapshot rows and matches the CPU decode
    llama_build(test-rs-replay-meta.cpp)
    llama_test(
        test-rs-replay-meta
        LABEL main
        ARGS "${MODEL_DIR}/qwen35-dense.gguf"
    )
    set_tests_properties(test-rs-replay-meta PROPERTIES FIXTURES_REQUIRED generate-models)

    # -sm tensor with MTP drafting: the restricted draft head (request-token rows read from the vocabulary-split LM head) and the
    # full head run on Meta(CPU, CPU) and give the tokens of the CPU run
    llama_build(test-mtp-meta.cpp)
    llama_test(
        test-mtp-meta
        LABEL main
        ARGS "${MODEL_DIR}/mtp/qwen35moe-mtp.gguf"
    )
    set_tests_properties(test-mtp-meta PROPERTIES FIXTURES_REQUIRED generate-models)

    # k19-dtopk: the MTP draft top-k over the restricted head's rows in token order gives the full-vocabulary top-k's candidates,
    # logits and order (ops on every backend, with ties), and the same tokens and drafts as LLAMA_MTP_DRAFT_TOPK_FULL=1
    llama_build(test-mtp-draft-topk.cpp)
    llama_test(
        test-mtp-draft-topk
        LABEL main
        ARGS "${MODEL_DIR}/mtp/qwen35moe-mtp.gguf"
    )
    set_tests_properties(test-mtp-draft-topk PROPERTIES FIXTURES_REQUIRED generate-models)

    # the greedy fast path of common_sampler_sample samples what the sampler chain samples, in every mode of its env knobs
    llama_build(test-sampling-greedy-fast.cpp)
    foreach(GF_ENV "" "LLAMA_SAMPLE_GREEDY_FAST=0" "LLAMA_SAMPLE_GREEDY_FAST=1" "LLAMA_SAMPLE_GREEDY_FAST=3"
                   "LLAMA_SAMPLE_BIAS_SPARSE=0" "LLAMA_SAMPLE_GREEDY_PENALTY=0")
        if (GF_ENV STREQUAL "")
            set(GF_NAME test-sampling-greedy-fast)
        else()
            string(REGEX REPLACE "^LLAMA_SAMPLE_([A-Z_]+)=([0-9])$" "\\1_\\2" GF_SUFFIX "${GF_ENV}")
            string(TOLOWER "${GF_SUFFIX}" GF_SUFFIX)
            set(GF_NAME test-sampling-greedy-fast-${GF_SUFFIX})
        endif()
        llama_test(
            test-sampling-greedy-fast
            NAME ${GF_NAME}
            LABEL main
            ARGS "${MODEL_DIR}/llama-dense.gguf"
        )
        set_tests_properties(${GF_NAME} PROPERTIES FIXTURES_REQUIRED generate-models)
        if (NOT GF_ENV STREQUAL "")
            set_tests_properties(${GF_NAME} PROPERTIES ENVIRONMENT "${GF_ENV}")
        endif()
    endforeach()

    # k19-sharednoise: the shared-noise draw samples the chain's distribution (with and without LLAMA_SHARED_NOISE=1), and a drafter with
    # the target's logits proposes the target's token
    llama_build(test-shared-noise.cpp)
    foreach(SN_ENV "LLAMA_SHARED_NOISE=0" "LLAMA_SHARED_NOISE=1")
        string(REGEX REPLACE "^LLAMA_SHARED_NOISE=([0-9])$" "\\1" SN_SUFFIX "${SN_ENV}")
        llama_test(
            test-shared-noise
            NAME test-shared-noise-${SN_SUFFIX}
            LABEL main
            ARGS "${MODEL_DIR}/qwen3tts-dense.gguf" 8000   # the generated model with a vocab of at least 1000 tokens (3072)
        )
        set_tests_properties(test-shared-noise-${SN_SUFFIX} PROPERTIES FIXTURES_REQUIRED generate-models ENVIRONMENT "${SN_ENV}")
    endforeach()

    # the advisory PLE prefetch predicts the rows set_input gathers, on the generated qwen4exp model
    llama_build(test-ple-prefetch.cpp)
    llama_test(
        test-ple-prefetch
        LABEL main
        ARGS "${MODEL_DIR}/qwen4exp-moe.gguf"
    )
    set_tests_properties(test-ple-prefetch PROPERTIES FIXTURES_REQUIRED generate-models)

    # k19-plerow: the PLE row cache leaves every gather and the logits bit-identical
    llama_build(test-ple-rowcache.cpp)
    llama_test(
        test-ple-rowcache
        LABEL main
        ARGS "${MODEL_DIR}/qwen4exp-moe.gguf"
    )
    set_tests_properties(test-ple-rowcache PROPERTIES FIXTURES_REQUIRED generate-models)

    llama_build(test-fusion.cpp)
endif()

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_build_and_test(test-sysmem.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
llama_build_and_test(test-json-schema.cpp)
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp)
# debug tool for chat template differential analysis (not registered as a test, run it manually)
llama_build(test-chat-analysis.cpp)
llama_build_and_test(test-log.cpp)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
)

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    set(MODEL_NAME "tinyllamas/stories15M-q4_0.gguf")
    set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
    set(MODEL_NAME "tinyllamas/stories15M-be.Q4_0.gguf")
    set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")

add_test(NAME test-download-model COMMAND ${CMAKE_COMMAND}
    -DDEST=${MODEL_DEST}
    -DNAME=${MODEL_NAME}
    -DHASH=${MODEL_HASH}
    -P ${PROJECT_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

# the context state cases against a real model, the sweep above uses the generated dummy ones
llama_test(test-llama-context NAME test-llama-context-model LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-llama-context-model PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-thread-safety.cpp ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-arg-parser.cpp)
# --fit: the run-time reserve of a small-device placement (common/fit.h)
llama_build_and_test(test-fit-small-reserve.cpp)
# --fit: the margin step-down of a plain dense launch on a device with no display (common/fit.h)
llama_build_and_test(test-fit-plain-spill.cpp)
# token-tiled gated delta-net vs the AR kernel, byte for byte on the GPU backend (SKIP without one)
llama_build_and_test(test-gdn-tiled-exact.cpp LINK ggml)
# the ops added here (KQ_MASK, MOE_CACHE_LOOKUP, the GDN replay window) are refused by the meta backend (-sm tensor) and RPC and run on the CPU
llama_build_and_test(test-meta-k15-ops.cpp LINK ggml)
# k19-fuse prompt fusions of qwen4exp (PLE conv taps, conv-input concat, QSA mask) vs the plain graphs, byte for byte (F4 also on the CPU)
llama_build_and_test(test-k19-fuse-exact.cpp LINK ggml)
llama_build_and_test(test-topk384-exact.cpp LINK ggml)
llama_build_and_test(test-topk-oneblock-exact.cpp LINK ggml)
# k19-moebw routed-expert mat-vec vs the kernels it replaces, byte for byte (SKIP without a CUDA device)
llama_build_and_test(test-moe-bw-exact.cpp LINK ggml)
# k19-handoff: ggml_dflash2_conv vs the add / mul / pad chain of the DFlash2 drafter's conv, byte for byte on every device (CPU always)
llama_build_and_test(test-dflash2-conv-exact.cpp LINK ggml)
# k19-mmvqcols: the rows-per-block lever of the dense 2..8 column mat-vec vs the launch it replaces, byte for byte (SKIP without a CUDA device)
llama_build_and_test(test-mmvqcols-exact.cpp LINK ggml)
# k19-iqsmem: IQ3 MUL_MAT_ID grid rows + IQ2_XS / IQ2_S codebook in shared memory vs the launches they replace, byte for byte (SKIP without a CUDA device)
llama_build_and_test(test-iqsmem-exact.cpp LINK ggml)
# k19-exactdet: single column MUL_MAT_ID with a partial last row block stays inside its expert's rows (sentinel, CPU reference, repeat bytes; SKIP without a CUDA device)
llama_build_and_test(test-mmid-tail.cpp LINK ggml)
# k19-ikfix: the CUDA mat-vec does not read past the last weight row of a partial last row block (weights before an unmapped page,
# CPU reference, same bytes as an ordinary buffer; SKIP without a CUDA device)
llama_build_and_test(test-mmvq-lastrow.cpp LINK ggml)
# k19-q4xt: single token MUL_MAT_ID warp per row launch (short K-quant / IQ4 expert rows) vs the launch it replaces, byte for byte (SKIP without a CUDA device)
llama_build_and_test(test-q4xt-exact.cpp LINK ggml)
llama_build_and_test(test-spec-type-gguf.cpp)
# the build identity of the MoE CPU-miss calibration key: unknown-commit builds differ by content, a relaunch keeps its identity
llama_build_and_test(test-cal-build-id.cpp LINK ggml)
llama_build_and_test(test-moe-trial-persist.cpp)   # k19-cmtpersist: CPU-miss trial verdicts in the calibration file
if (NOT WIN32)
    target_link_libraries(test-cal-build-id PRIVATE ${CMAKE_DL_LIBS})
endif()

# the weight store is internal to libllama (exported symbols on non-Windows builds), the test uses the CPU backend directly
if (NOT WIN32 AND NOT GGML_BACKEND_DL)
    llama_build_and_test(test-weight-store.cpp)
    llama_build_and_test(test-moe-host-cache.cpp)
    llama_build_and_test(test-moe-ram-profile.cpp) # k19-ramprof: routing profile file + RAM-tier fill order
    llama_build_and_test(test-moe-cache-tail-skip.cpp)
endif()

# CPU work buffer: every node stays inside the size ggml_graph_plan() reserved (guard page after the buffer)
if (NOT WIN32)
    llama_build_and_test(test-cpu-wsize-guard.cpp LINK ggml)
endif()
# k19-q4xcache: the assign standing in for the plain lookup keeps the lookup's cache state (GPU backend, skipped without one)
if (NOT WIN32)
    llama_build_and_test(test-moe-cache-plain-assign.cpp)
endif()
llama_build_and_test(test-model-resolution.cpp)
# the test serves its repos from an httplib server, and the library links it privately
target_link_libraries(test-model-resolution PRIVATE cpp-httplib)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp LINK ggml)
endif()
if (GGML_METAL AND NOT GGML_BACKEND_DL)
    # calls the Metal backend's graph optimizer and fusion table directly
    llama_build_and_test(test-metal-graph-optimize.cpp LINK ggml ggml-metal)
    target_include_directories(test-metal-graph-optimize PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-metal)
endif()

llama_build(test-backend-ops.cpp LINK ggml)
# k19-mmq-pipe: the pipelined MMQ MoE GEMM, md5 + time per case (needs a GPU backend; not a ctest)
llama_build(test-mmq-pipe.cpp LINK ggml)
llama_build(test-graph-src-type.cpp LINK ggml) # k15-mmid: a CUDA graph is not replayed for a source of another type
llama_build(test-gdn-replay.cpp LINK ggml) # k16-draft: ggml_gated_delta_net_replay vs the unfused replay path, bitwise

llama_build_and_test(test-tiled-mulmat.cpp LINK ggml)
llama_build_and_test(test-model-load-cancel.cpp LABEL "model")
llama_build_and_test(test-autorelease.cpp       LABEL "model")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model")

if (APPLE)
    llama_build(test-rset-release.cpp)
endif()

if (NOT GGML_BACKEND_DL AND GGML_CPU)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp LINK ggml)
    llama_build_and_test(test-quantize-fns.cpp LINK ggml)
    llama_build_and_test(test-quantize-perf.cpp LINK ggml)
    llama_build_and_test(test-mmid-multi.cpp LINK ggml)
    llama_build_and_test(test-moe-host-ffn.cpp LINK ggml)
    llama_build_and_test(test-rope.cpp LINK ggml)
    llama_build_and_test(test-col2im-1d.cpp LINK ggml)
    llama_build_and_test(test-conv-1d.cpp LINK ggml)
endif()

# libmtmd
llama_build_and_test(test-mtmd-c-api.c LINK mtmd)

llama_build_and_test(test-mtmd-impl.cpp LINK mtmd)

# GGUF model data fetcher library for tests that need real model metadata
# Only compile when cpp-httplib has SSL support (CPPHTTPLIB_OPENSSL_SUPPORT)
if (TARGET cpp-httplib)
    get_target_property(_cpp_httplib_defs cpp-httplib INTERFACE_COMPILE_DEFINITIONS)
    if (_cpp_httplib_defs MATCHES "CPPHTTPLIB_OPENSSL_SUPPORT")
        add_library(gguf-model-data STATIC gguf-model-data.cpp)
        target_link_libraries(gguf-model-data PRIVATE llama-common cpp-httplib)
        target_include_directories(gguf-model-data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

        add_executable(test-gguf-model-data test-gguf-model-data.cpp)
        target_link_libraries(test-gguf-model-data PRIVATE gguf-model-data llama-common)
        llama_test(test-gguf-model-data LABEL "model")

        # test-quant-type-selection requires gguf-model-data for remote model metadata
        llama_build_and_test(test-quant-type-selection.cpp LABEL "model" LINK llama)
        target_link_libraries(test-quant-type-selection PRIVATE gguf-model-data)
        target_compile_definitions(test-quant-type-selection PRIVATE
            SNAPSHOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/snapshots")
    endif()
endif()

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

if (NOT LLAMA_USE_SYSTEM_GGML)
    # Needs non-public ggml-impl.h
    llama_build_and_test(test-gguf.cpp LINK ggml)

    # Needs non-public ggml{,-backend}-impl.h
    llama_build_and_test(test-alloc.cpp LINK ggml)

    # Needs non-public ggml-backend-impl.h: host weight prefetch with a lender on a mock device with a capacity
    llama_build_and_test(test-sched-lend.cpp LINK ggml)
endif()

llama_build(test-export-graph-ops.cpp)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()
