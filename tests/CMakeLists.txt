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
    llama_build_and_test(test-sampling.cpp)
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
    # the task-queue yield with and without its fast end (LLAMA_SERVER_YIELD_FAST)
    llama_build(test-server-queue-yield.cpp)
    target_include_directories(test-server-queue-yield PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-queue-yield PRIVATE server-context)
    foreach(YF 0 1)
        llama_test(test-server-queue-yield NAME test-server-queue-yield-fast${YF} LABEL main)
        set_tests_properties(test-server-queue-yield-fast${YF} PROPERTIES ENVIRONMENT "LLAMA_SERVER_YIELD_FAST=${YF}")
    endforeach()
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
        # fixed seed: the fork's tests that read these models (test-dtopk-grammar, test-mtp-draft-topk, ...) need the same weights
        # on every run; with the default random seed they were flaky
        ARGS -s 1234 -o "${MODEL_DIR}"
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
    # every architecture and case: several minutes on a 3-4 vCPU runner, so it holds 4 slots and its own timeout
    set_tests_properties(test-llama-context PROPERTIES
        FIXTURES_REQUIRED generate-models
        ENVIRONMENT "GGML_SCHED_DEBUG_REALLOC=1"
        PROCESSORS 4
        TIMEOUT 2400
    )

    # the same with an accelerator device that takes host buffers but no op (as BLAS does): every graph must still run on a
    # backend that supports its ops (without the realloc abort: a second backend on the CPU buffer type makes the scheduler,
    # not the graph allocator, re-plan a graph whose recurrent state gather changed size)
    # (the recurrent cases, which run the replay commit graphs; its own directory for the session files of the cases)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/llama-context-accel")
    llama_test(
        test-llama-context
        NAME test-llama-context-accel
        LABEL main
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/llama-context-accel"
        ARGS --models "${MODEL_DIR}" --accel --test "rollback|replay.*|shared|.*_move|.*_fuzz" -t 2
    )
    set_tests_properties(test-llama-context-accel PROPERTIES
        FIXTURES_REQUIRED generate-models
        PROCESSORS 2
        TIMEOUT 1800
    )

    # recurrent single-token decodes (states read in place) equal one prompt of the same tokens on the CPU
    llama_test(
        test-llama-archs
        NAME test-rs-decode-parity
        LABEL main
        ARGS --rs-decode-parity -s 1
    )

    # delta-net replay commits of a state split over two devices (two in-process RPC servers over the CPU; skipped without
    # the RPC backend) give the logits of one device, with the cached commit graphs of alternating windows reused through the scheduler
    llama_test(
        test-llama-archs
        NAME test-rs-replay-sched
        LABEL main
        ARGS --rs-replay-sched -s 1
    )
    # MTP draft contexts (qwen35, qwen35moe, deepseek4): a batch with no output row writes the cache bytes of the whole block
    foreach(MKV_MODE "" "kill")
        if (MKV_MODE STREQUAL "")
            set(MKV_NAME test-mtp-kv-only)
        else()
            set(MKV_NAME test-mtp-kv-only-${MKV_MODE})
        endif()
        llama_test(test-llama-archs NAME ${MKV_NAME} LABEL main ARGS --mtp-kv-only -s 1)
        if (MKV_MODE STREQUAL "kill")
            set_tests_properties(${MKV_NAME} PROPERTIES ENVIRONMENT "LLAMA_MTP_KV_ONLY_QWEN35=0;LLAMA_MTP_KV_ONLY_DSV4=0")
        endif()
    endforeach()
    llama_test(
        test-llama-archs
        NAME test-small-backends
        LABEL main
        ARGS --small-backends -s 1
    )
    # a layer-major prompt group after a large two-sequence ubatch that released the MoE cache banks seeds nothing and gives the
    # logits of a run without the cache (skipped without a CUDA / ROCm / MUSA device)
    llama_test(
        test-llama-archs
        NAME test-moe-released-seed
        LABEL main
        ARGS --moe-released-seed -s 1
    )
    # an exact verify batch over a q8_0 KV cache with flash attention (one attention per row, windows of 32 and 128 rows) fits the
    # graph and matches single decodes
    llama_test(
        test-llama-archs
        NAME test-verify-quant-kv
        LABEL main
        ARGS --verify-quant-kv -s 1
    )
    # a layer-major group member with outputs beside released banks, or beside banks the group must release, reads the host experts
    # (skipped without a CUDA / ROCm / MUSA device)
    llama_test(
        test-llama-archs
        NAME test-moe-released-out
        LABEL main
        ARGS --moe-released-out -s 1
    )
    # a long sequence of a model without memory (diffusion) runs with the MoE cache banks released, a short one gets them back
    # (skipped without a CUDA / ROCm / MUSA device)
    llama_test(
        test-llama-archs
        NAME test-moe-encode-release
        LABEL main
        ARGS --moe-encode-release -s 1
    )

    # an exact verify row keeps every cell of its sequence when the cells do not follow the positions (two sequences in a unified KV)
    llama_build(test-exact-verify-cells.cpp)
    llama_test(
        test-exact-verify-cells
        LABEL main
        ARGS "${MODEL_DIR}/llama-dense.gguf"
    )
    set_tests_properties(test-exact-verify-cells PROPERTIES FIXTURES_REQUIRED generate-models)

    # the recent-window ring beside a quantized KV cache (probe = plain q8_0 and control = plain q4_0 bit for bit, routing,
    # draft rollback, state save / restore, context shift, multi-sequence, the concat stopgap), on generated models of several families
    llama_build(test-kv-win.cpp)
    llama_test(
        test-kv-win
        LABEL main
        ARGS --models "${MODEL_DIR}"
    )
    set_tests_properties(test-kv-win PROPERTIES FIXTURES_REQUIRED generate-models RUN_SERIAL TRUE PROCESSORS 4)

    # the CPU two-source flash attention (ggml_flash_attn_ext_set_src2) against the concatenation and a double reference
    llama_build(test-fa-src2.cpp)
    llama_test(test-fa-src2 LABEL main)

    # --moe-cache-mib with the layers on a GPU outside ggml-cuda (a mock device): the context runs without the MoE cache
    llama_build(test-moe-cache-device.cpp)
    llama_test(
        test-moe-cache-device
        LABEL main
        ARGS "${MODEL_DIR}/llama-moe.gguf"
    )
    set_tests_properties(test-moe-cache-device PROPERTIES FIXTURES_REQUIRED generate-models)

    # flash attention with its KV length on the device vs the unpadded node, byte for byte (GPU; SKIP without one)
    llama_build(test-fa-kvlen-exact.cpp)
    llama_build(test-fa-kvlen-rows.cpp)   # small-batch flash attention with the device KV length, finite and against a reference

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

    # a drafted window rolled back by the host copy of the recurrent state (no rollback rows) = the accepted tokens decoded alone
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

    # the recent-window ring: sinks-only, its off switch beside LLAMA_KV_WIN_SINKS, a two-stream sparse attention cache
    llama_build(test-kv-win-guards.cpp)
    llama_test(
        test-kv-win-guards
        LABEL main
        ARGS "${MODEL_DIR}/llama-dense.gguf" "${MODEL_DIR}/qwen4exp-moe.gguf"
    )
    set_tests_properties(test-kv-win-guards PROPERTIES FIXTURES_REQUIRED generate-models)

    # a partial seq_rm into a prompt ubatch restores the delta-net snapshot row of that position (fresh prefill = rollback);
    # the chunked CUDA kernel needs 16 key heads x 128 (a model with that layout, not this one, exercises it on a GPU)
    llama_build(test-gdn-snap-rollback.cpp)
    llama_test(
        test-gdn-snap-rollback
        LABEL main
        ARGS "${MODEL_DIR}/qwen35-dense.gguf" 200 2 2
    )
    set_tests_properties(test-gdn-snap-rollback PROPERTIES FIXTURES_REQUIRED generate-models)

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

    # the MTP draft top-k over the restricted head's rows in token order gives the full-vocabulary top-k's candidates,
    # logits and order (ops on every backend, with ties), and the same tokens and drafts as LLAMA_MTP_DRAFT_TOPK_FULL=1
    llama_build(test-mtp-draft-topk.cpp)
    llama_test(
        test-mtp-draft-topk
        LABEL main
        ARGS "${MODEL_DIR}/mtp/qwen35moe-mtp.gguf"
    )
    set_tests_properties(test-mtp-draft-topk PROPERTIES FIXTURES_REQUIRED generate-models RUN_SERIAL TRUE PROCESSORS 4)

    # the QSA selection without indexer scoring while every pool fits the top_k (LLAMA_QSA_ALLFIT) == the scored one, byte for byte
    llama_build(test-qsa-allfit-exact.cpp)
    llama_test(
        test-qsa-allfit-exact
        LABEL main
        ARGS "${MODEL_DIR}/qwen4exp-moe.gguf"
    )
    set_tests_properties(test-qsa-allfit-exact PROPERTIES FIXTURES_REQUIRED generate-models)

    # KV seq_rm over the used cells (LLAMA_KV_SEQ_RM_USED) leaves tokens, drafts and the memory state unchanged
    llama_build(test-hostgap-exact.cpp)
    llama_test(
        test-hostgap-exact
        LABEL main
        ARGS "${MODEL_DIR}/mtp/qwen35moe-mtp.gguf"
    )
    set_tests_properties(test-hostgap-exact PROPERTIES FIXTURES_REQUIRED generate-models)

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

    # the shared-noise draw samples the chain's distribution (with and without LLAMA_SHARED_NOISE=1), and a drafter with
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

    # grammar / reasoning budget requests sample the same tokens from device top-k candidates (CPU backend sampler) as
    # from the full rows; the kill switch keeps them on the full rows; the deliberately inexact no-fallback mode must make it fail
    llama_build(test-dtopk-grammar.cpp)
    foreach(DG_MODE "" "kill" "broken")
        if (DG_MODE STREQUAL "")
            set(DG_NAME test-dtopk-grammar)
        else()
            set(DG_NAME test-dtopk-grammar-${DG_MODE})
        endif()
        if (DG_MODE STREQUAL "broken")
            llama_test(test-dtopk-grammar NAME ${DG_NAME} LABEL main ARGS "${MODEL_DIR}/qwen3tts-dense.gguf" --expect-diff)
        else()
            llama_test(test-dtopk-grammar NAME ${DG_NAME} LABEL main ARGS "${MODEL_DIR}/qwen3tts-dense.gguf")
        endif()
        # run alone: a thread pool of up to 4 threads per test; beside other tests on a 3-4 vCPU runner they starved for 15-30 min
        set_tests_properties(${DG_NAME} PROPERTIES FIXTURES_REQUIRED generate-models RUN_SERIAL TRUE PROCESSORS 4)
        if (DG_MODE STREQUAL "kill")
            set_tests_properties(${DG_NAME} PROPERTIES ENVIRONMENT "LLAMA_DEVICE_TOPK_GRAMMAR=0")
        endif()
    endforeach()

    # the verify window's backend top-k in one launch (LLAMA_DEV_TOPK_BATCH) gives every row the candidates, logits and
    # tokens of one top-k per row
    llama_build(test-topk-batch-sampling.cpp)
    llama_test(test-topk-batch-sampling LABEL main ARGS "${MODEL_DIR}/qwen3tts-dense.gguf")
    set_tests_properties(test-topk-batch-sampling PROPERTIES FIXTURES_REQUIRED generate-models)

    # the advisory PLE prefetch predicts the rows set_input gathers, on the generated qwen4exp model
    llama_build(test-ple-prefetch.cpp)
    llama_test(
        test-ple-prefetch
        LABEL main
        ARGS "${MODEL_DIR}/qwen4exp-moe.gguf"
    )
    set_tests_properties(test-ple-prefetch PROPERTIES FIXTURES_REQUIRED generate-models)

    # the PLE row cache leaves every gather and the logits bit-identical
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
# --fit: the full margin beside a MoE cache without flash attention, and the flash attention the measure reads back (common/fit.h)
llama_build_and_test(test-fit-faoff.cpp)
# --fit: the MoE cache size of a decode-sized placement, at most the size requested (common/fit.h)
llama_build_and_test(test-fit-place8-grow.cpp)
# --fit: the mid-size placement prices a lent drafter decode-sized (common/fit.h)
llama_build_and_test(test-fit-placemid-lent.cpp)
# --fit: the K/V cells of an all-sliding DFlash drafter, with and without --swa-full (common_fit_dflash_draft_bytes)
llama_build_and_test(test-fit-dflash-swa.cpp)
# token-tiled gated delta-net vs the AR kernel, byte for byte on the GPU backend (SKIP without one)
llama_build_and_test(test-gdn-tiled-exact.cpp LINK ggml)
# CPU GET_ROWS results gathered ahead (the next prompt batch's embedding rows) vs the plain gather, byte for byte
llama_build_and_test(test-get-rows-ahead.cpp LINK ggml)
# the exact int-to-float product of the CUDA q8_0 MMQ vec_dot (GGML_CUDA_MMQ_EXACT_CVT) == the conversion and product, on the CPU
llama_build_and_test(test-mmq-cvt-exact.cpp LINK ggml)
# the ops added here (KQ_MASK, MOE_CACHE_LOOKUP, the GDN replay window) are refused by the meta backend (-sm tensor) and RPC and run on the CPU
llama_build_and_test(test-meta-k15-ops.cpp LINK ggml)
# prompt fusions of qwen4exp (PLE conv taps, conv-input concat, QSA mask) vs the plain graphs, byte for byte (F4 also on the CPU)
llama_build_and_test(test-k19-fuse-exact.cpp LINK ggml)
# GDN fusion predicate (replay windows), silu gated norm, one-barrier mmvf == the code before them and the unfused graph, byte for byte
llama_build_and_test(test-k19-p1fuse-exact.cpp LINK ggml)
# MoE cache-bank MMQ products one block per exact tile == the compact tiling without the bank, else the split as before;
# the fused-SwiGLU down reusing its gate / up routing maps and the match-any scatter == the code before them, byte for byte (SKIP without CUDA)
llama_build_and_test(test-k19-p5mmq-exact.cpp LINK ggml)
llama_build_and_test(test-topk384-exact.cpp LINK ggml)
llama_build_and_test(test-topk-oneblock-exact.cpp LINK ggml)
llama_build_and_test(test-topk-twostage-exact.cpp LINK ggml)   # two-stage one-launch top-k == radix select, byte for byte
llama_build_and_test(test-topk-batch-exact.cpp LINK ggml)      # one top-k over a verify window == one per row, byte for byte
llama_build_and_test(test-fa-sparse-smem-exact.cpp LINK ggml)   # sparse MMA flash attention at its one-stage shared memory == the two-stage launch, byte for byte
llama_build_and_test(test-fattn-q8-exact.cpp LINK ggml)   # MMA flash attention reading q8_0 K / V in place == the staged f16 copy, byte for byte
llama_build_and_test(test-fattn-q4-exact.cpp LINK ggml)   # MMA flash attention reading q4_0 / mixed q4_0 + q8_0 K / V in place == the staged f16 copy, byte for byte
llama_build_and_test(test-fattn-src2-exact.cpp LINK ggml) # two-source flash attention (body + window ring) in place == staged == one node over the concatenation, byte for byte
# routed-expert mat-vec vs the kernels it replaces, byte for byte (SKIP without a CUDA device)
llama_build_and_test(test-moe-bw-exact.cpp LINK ggml)
llama_build_and_test(test-moe-tc-exact.cpp LINK ggml)
# ggml_dflash2_conv vs the add / mul / pad chain of the DFlash2 drafter's conv, byte for byte on every device (CPU always)
llama_build_and_test(test-dflash2-conv-exact.cpp LINK ggml)
# the MoE expert reduction (MUL, VIEWs, ADD chain) on Metal, fused or not, vs the CPU backend, byte for byte (SKIP without a Metal device)
llama_build_and_test(test-moe-reduce-exact.cpp LINK ggml)
# Metal asynchronous host <-> device copies at page and non-page offsets, round trip byte for byte (SKIP without a Metal device)
llama_build_and_test(test-metal-async-copy.cpp LINK ggml)
# host models of the Metal MUL_MAT_ID expert map / simdgroup blocks and of the selection mask scratch (no device needed)
llama_build_and_test(test-metal-mmid-map.cpp LINK ggml)
# the list of paired routes of the Metal mat-vec MUL_MAT_ID (reference of its map kernel) vs a sequential pairing (CPU only)
llama_build_and_test(test-metal-mmid-pair-model.cpp LINK ggml)
if (GGML_METAL)
    # the paired-route mat-vec MUL_MAT_ID on Metal (GGML_METAL_MMID_PAIR=1) vs the single-route kernel, byte for byte (SKIP when unsupported)
    llama_build_and_test(test-metal-mmid-pair-exact.cpp LINK ggml)
endif()
# Metal host bank binding (the MoE copy callback on unified memory reads host expert banks in place): bound reads return the host
# bytes, a later graph reads the tensor's own bytes, and GGML_METAL_HOST_BANK_BIND=0 refuses every bind (SKIP without a Metal device)
llama_build_and_test(test-metal-host-bind.cpp LINK ggml)
llama_test(test-metal-host-bind NAME test-metal-host-bind-off ARGS --off)
set_tests_properties(test-metal-host-bind PROPERTIES ENVIRONMENT "GGML_METAL_HOST_BANK_BIND=1")
set_tests_properties(test-metal-host-bind-off PROPERTIES ENVIRONMENT "GGML_METAL_HOST_BANK_BIND=0")
# the rows-per-block lever of the dense 2..8 column mat-vec vs the launch it replaces, byte for byte (SKIP without a CUDA device)
llama_build_and_test(test-mmvqcols-exact.cpp LINK ggml)
# the ported few-row int8 MMQ (GGML_CUDA_MMQ_FEW) on dense K-quant / IQ4_XS / Q8_0 products at 2..8 columns vs mmvq + fp64 reference (NMSE gates,
# column invariance and exact-verify bytes; SKIP without a CUDA device)
llama_build_and_test(test-kqtc-exact.cpp LINK ggml)
# IQ3 MUL_MAT_ID grid rows + IQ2_XS / IQ2_S codebook in shared memory vs the launches they replace, byte for byte (SKIP without a CUDA device)
llama_build_and_test(test-iqsmem-exact.cpp LINK ggml)
# single column MUL_MAT_ID with a partial last row block stays inside its expert's rows (sentinel, CPU reference, repeat bytes; SKIP without a CUDA device)
llama_build_and_test(test-mmid-tail.cpp LINK ggml)
# the CUDA mat-vec does not read past the last weight row of a partial last row block (weights before an unmapped page,
# CPU reference, same bytes as an ordinary buffer; SKIP without a CUDA device)
llama_build_and_test(test-mmvq-lastrow.cpp LINK ggml)
target_link_libraries(test-mmvq-lastrow PRIVATE ${CMAKE_DL_LIBS}) # dlopen / dlsym (libdl before glibc 2.34)
# single token MUL_MAT_ID warp per row launch (short K-quant / IQ4 expert rows) vs the launch it replaces, byte for byte (SKIP without a CUDA device)
llama_build_and_test(test-q4xt-exact.cpp LINK ggml)
# DeepSeek-V4.1 candidate scoring (lightning indexer over the candidate blocks only) == indexer + block mask, byte for byte
# (scores and top-k ids, ties and -inf), with a negative control; the CPU, and the first CUDA / ROCm device (SKIP without one)
llama_build_and_test(test-dsv41-cand-score-exact.cpp LINK ggml)
llama_test(test-dsv41-cand-score-exact NAME test-dsv41-cand-score-gpu-exact ARGS --gpu)
set_tests_properties(test-dsv41-cand-score-exact test-dsv41-cand-score-gpu-exact PROPERTIES SKIP_RETURN_CODE 77)
llama_build_and_test(test-spec-type-gguf.cpp)
llama_build_and_test(test-dflash-ev-auto.cpp)
# the build identity of the MoE CPU-miss calibration key: unknown-commit builds differ by content, a relaunch keeps its identity
llama_build_and_test(test-cal-build-id.cpp LINK ggml)
llama_build_and_test(test-moe-trial-persist.cpp)   # CPU-miss trial verdicts in the calibration file
llama_build_and_test(test-moe-overlap-reset.cpp)   # fetch overlap rule restarts with a fixed-share request
if (NOT WIN32)
    target_link_libraries(test-cal-build-id PRIVATE ${CMAKE_DL_LIBS})
endif()

llama_build_and_test(test-env-alias.cpp) # renamed environment switches: earlier names still read

# the weight store is internal to libllama (exported symbols on non-Windows builds), the test uses the CPU backend directly
if (NOT WIN32 AND NOT GGML_BACKEND_DL)
    llama_build_and_test(test-weight-store.cpp)
    llama_build_and_test(test-moe-host-cache.cpp)
    llama_build_and_test(test-moe-ram-profile.cpp) # routing profile file + RAM-tier fill order
    llama_build_and_test(test-autoprof.cpp)        # self-learned profile files, atomic write, PLE row recorder
    llama_build_and_test(test-moe-cache-tail-skip.cpp)
endif()

# CPU work buffer: every node stays inside the size ggml_graph_plan() reserved (guard page after the buffer); it calls the CPU
# backend's graph API directly, which a loadable CPU backend (GGML_BACKEND_DL) does not link
if (NOT WIN32 AND NOT GGML_BACKEND_DL)
    llama_build_and_test(test-cpu-wsize-guard.cpp LINK ggml)
endif()
# the assign standing in for the plain lookup keeps the lookup's cache state (GPU backend, skipped without one)
if (NOT WIN32)
    llama_build_and_test(test-moe-cache-plain-assign.cpp)
    # with the hybrid admission margin set: the share 1 assign still fetches every miss, as the lookup does
    llama_test(test-moe-cache-plain-assign NAME test-moe-cache-plain-assign-margin LABEL main)
    set_tests_properties(test-moe-cache-plain-assign-margin PROPERTIES ENVIRONMENT "LLAMA_MOE_CACHE_ADMIT_MARGIN=1.25")
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
# the pipelined MMQ MoE GEMM, md5 + time per case (needs a GPU backend; not a ctest)
llama_build(test-mmq-pipe.cpp LINK ggml)
llama_build(test-graph-src-type.cpp LINK ggml) # a CUDA graph is not replayed for a source of another type
llama_build(test-gdn-replay.cpp LINK ggml) # ggml_gated_delta_net_replay vs the unfused replay path, bitwise

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
    llama_build_and_test(test-q8k-exact.cpp LINK ggml)
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
