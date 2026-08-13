#pragma once

#include "hf-cache.h"

#include <string>
#include <vector>
#include <functional>

struct common_params_model;

using common_header      = std::pair<std::string, std::string>;
using common_header_list = std::vector<common_header>;

struct common_download_progress {
    std::string url;
    size_t downloaded = 0;
    size_t total      = 0;
    bool cached       = false;
};

class common_download_callback {
public:
    virtual ~common_download_callback() = default;
    virtual void on_start(const common_download_progress & p) = 0;
    virtual void on_update(const common_download_progress & p) = 0;
    virtual void on_done(const common_download_progress & p, bool ok) = 0;
    virtual bool is_cancelled() const { return false; }
};

struct common_remote_params {
    common_header_list headers;
    long timeout  = 0;           // in seconds, 0 means no timeout
    long max_size = 0;           // unlimited if 0
};

// get remote file content, returns <http_code, raw_response_body>
std::pair<long, std::vector<char>> common_remote_get_content(const std::string & url, const common_remote_params & params);

// split HF repo with tag into <repo, tag>, for example:
// - "ggml-org/models:F16" -> <"ggml-org/models", "F16">
// tag is optional and can be empty
std::pair<std::string, std::string> common_download_split_repo_tag(const std::string & hf_repo_with_tag);

// Result of common_list_cached_models
struct common_cached_model_info {
    std::string repo;
    std::string tag;
    std::string to_string() const {
        return repo + ":" + tag;
    }
};

// Options for common_download_file_single
struct common_download_opts {
    std::string bearer_token;
    common_header_list headers;
    bool offline = false;
    bool download_mmproj = false;
    bool download_mtp = false;
    common_download_callback * callback = nullptr;
};

struct common_download_task {
    common_download_opts opts;
    std::string url;
    std::string local_path;
    std::function<void()> on_done;
    bool is_hf = false;

    common_download_task() = default;
    common_download_task(hf_cache::hf_file f,
            const common_download_opts & opts,
            std::function<void()> on_done = nullptr)
        : opts(opts), url(f.url), local_path(f.local_path), on_done(on_done), is_hf(true) {}
};

void common_download_run_tasks(const std::vector<common_download_task> & tasks);

// if url is a multi-part GGUF file, returns all parts, otherwise returns the single file
std::vector<std::string> common_download_get_all_parts(const std::string & url);

// returns list of cached models
std::vector<common_cached_model_info> common_list_cached_models();

// download single file from url to local path
// returns status code or -1 on error
// skip_etag: if true, don't read/write .etag files (for HF cache where filename is the hash)
int common_download_file_single(const std::string & url,
                                const std::string & path,
                                const common_download_opts & opts = {},
                                bool skip_etag = false);

// resolve and download model from Docker registry
// return local path to downloaded model file
std::string common_docker_resolve_model(const std::string & docker);

// Remove a cached model from disk
// input format: "user/model" or "user/model:tag"
// - if tag is omitted, removes the entire repo cache directory
// - if tag is present, removes only files matching that tag (and orphaned blobs)
// returns true if anything was removed
bool common_download_remove(const std::string & hf_repo_with_tag);

struct common_download_hf_plan {
    hf_cache::hf_file primary;
    hf_cache::hf_files model_files;
    hf_cache::hf_file mmproj;
    hf_cache::hf_file mtp;
    hf_cache::hf_file preset; // if set, only this file is downloaded
    // if true, model_files holds the ordered legacy `.partNofM` parts (not modern split shards),
    // and primary is a synthetic placeholder for the reconstructed file - see
    // common_download_reconstruct_legacy_split()
    bool primary_is_legacy_split = false;
};
common_download_hf_plan common_download_get_hf_plan(const common_params_model & model, const common_download_opts & opts);

// Resolve which real GGUF file(s) in `files` match `tag` (modern split, single file, or legacy
// `.partNofM` split) - the same matching common_download_get_hf_plan() uses internally, so a
// caller checking "does this quant really exist, and how big is it" (e.g. the model picker's
// verification step) can never disagree with what an actual download resolves to. Empty tag means
// the default priority order (Q4_K_M, Q8_0). Returns a plan with an empty primary.path if nothing
// matches; only primary/model_files/primary_is_legacy_split are populated (no mmproj/mtp/preset).
common_download_hf_plan common_download_resolve_model_files(const hf_cache::hf_files & files, const std::string & tag);

// Sequentially download and concatenate a legacy `.partNofM` GGUF split (still used by some
// uploaders, e.g. mradermacher, whose pipeline predates llama.cpp's native split format) into one
// file at primary.final_path. Each part is deleted immediately after being appended, so peak extra
// disk usage is bounded by the largest single part instead of doubling the whole model. Downloads
// are sequential, not parallel across parts, trading download speed for that bound. Throws on
// failure, including when there isn't enough free disk for the reconstruction.
std::string common_download_reconstruct_legacy_split(const hf_cache::hf_files    & parts,
                                                      const hf_cache::hf_file     & primary,
                                                      const common_download_opts & opts);
