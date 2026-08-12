# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

if (EMSCRIPTEN)
else()
    add_subdirectory(batched-bench)
    add_subdirectory(gguf-split)
    add_subdirectory(imatrix)
    add_subdirectory(llama-bench)
    add_subdirectory(completion)
    add_subdirectory(perplexity)
    add_subdirectory(quantize)
    # llama-cli needs server-context (from tools/server) but not the web UI bundle, so
    # server always builds (server-context itself is gated off LLAMA_BUILD_SERVER inside
    # tools/server/CMakeLists.txt, only llama-server/llama-server-impl are) while ui stays
    # gated here: it needs a working npm/prebuilt UI asset pipeline that this fork branch
    # doesn't reliably produce, and only llama-server links against it.
    add_subdirectory(server)
    add_subdirectory(cli)
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(ui)
    endif()
    add_subdirectory(tokenize)
    add_subdirectory(parser)
    add_subdirectory(tts)
    add_subdirectory(mtmd)
    if (GGML_RPC)
        add_subdirectory(rpc)
    endif()
    if (NOT GGML_BACKEND_DL AND GGML_CPU)
        # these tools use backends directly (no dynamic loading) and depend on CPU backend symbols
        add_subdirectory(cvector-generator)
        add_subdirectory(export-lora)
    endif()
    add_subdirectory(fit-params)
    add_subdirectory(results)
endif()
