#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Ref: https://huggingface.co/docs/hub/local-cache.md

namespace hf_cache {

struct hf_file {
    std::string path;
    std::string url;
    std::string local_path;
    std::string final_path;
    std::string oid;
    std::string repo_id;
    uint64_t    size = 0; // bytes; 0 if unknown (e.g. offline cache listing couldn't stat it)
};

using hf_files = std::vector<hf_file>;

// Get files from HF API
hf_files get_repo_files(
    const std::string & repo_id,
    const std::string & token
);

hf_files get_cached_files(const std::string & repo_id = {});

// Create snapshot path (link or move/copy) and return it
std::string finalize_file(const hf_file & file);

// Remove the entire cached directory for a repo, returns true if removed
bool remove_cached_repo(const std::string & repo_id);

// The base HF hub cache directory downloads land in (respects LLAMA_CACHE / HF_HUB_CACHE /
// HUGGINGFACE_HUB_CACHE / HF_HOME / XDG_CACHE_HOME, same as the rest of this file) - exposed so
// UIs can tell the user where their disk space is actually going.
std::string get_cache_dir();

} // namespace hf_cache
