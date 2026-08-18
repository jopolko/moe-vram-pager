#include "server-model-picker.h"

#include "common.h"
#include "download.h"
#include "hf-cache.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "preset.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <condition_variable>
#include <fstream>
#include <future>
#include <map>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// Native port of scripts/model_picker.py. Deliberately self-contained (no
// server_context/task-queue dependency) since this isn't part of the
// inference path. Like the Python reference, derestricted detection checks
// name + known_repo_ids + live HF tags (fetched concurrently, disk-cached
// for HF_TAGS_TTL_SECONDS since tags rarely change) to catch abliteration
// tool brand names in a model's tags even when its name doesn't mention
// them (e.g. "Heretic-..." output repos that self-tag but aren't named after
// the tool).

using json = nlohmann::json;

namespace {

const char * UGI_CSV_URL              = "https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard/resolve/main/ugi-leaderboard-data.csv";
const char * DERESTRICTED_FILTER_URL  = "https://raw.githubusercontent.com/jopolko/moe-vram-pager/main/derestricted-filter.json";
const char * ARCH_MAP_URL             = "https://raw.githubusercontent.com/jopolko/moe-vram-pager/main/arch-map.json";

constexpr long DEFAULT_TTL_SECONDS = 24 * 3600;
// HF tags rarely change once a repo is published, so this is cached far longer than
// the other TTLs - keeps the per-page-load cost near-zero after the first cold run.
constexpr long HF_TAGS_TTL_SECONDS = 30L * 24 * 3600;

// Progress counters for the (one-time, cold-cache-only) batch HF tags fetch, polled by
// GET /model-picker/tags-progress so the frontend can show "42/162" instead of a blind
// spinner. Global rather than per-request since this endpoint is a local single-user
// dev tool; two concurrent page loads would just interleave, which is an acceptable
// simplification here.
std::atomic<int> g_tags_progress_done{0};
std::atomic<int> g_tags_progress_total{0};

// Bounds total concurrent outbound requests to huggingface.co from this process. HF applies a
// per-IP rate limit ("429: We had to rate limit your IP") that an unbounded burst of parallel
// std::async tasks blows straight through - the tags batch, the composite-scored search, and the
// per-candidate real-file-size verification (verify_gguf_row) can together fire hundreds of HTTP
// calls in the same instant on a cold cache. A blocking counting semaphore serializes that burst
// down to a sustainable rate without changing any call site's own retry/fallback logic - a
// rate-limited call still just fails and falls through to "best effort", same as any other network
// hiccup, it just becomes far less likely to happen in the first place.
class hf_request_limiter {
    std::mutex mutex;
    std::condition_variable cv;
    int available;
public:
    explicit hf_request_limiter(int n) : available(n) {}
    void acquire() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] { return available > 0; });
        available--;
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        available++;
        cv.notify_one();
    }
};
hf_request_limiter g_hf_request_limiter(4);

struct hf_request_guard {
    hf_request_guard()  { g_hf_request_limiter.acquire(); }
    ~hf_request_guard() { g_hf_request_limiter.release(); }
};

// Ordered best (highest quality/bpw) to worst, so each model can be checked
// against progressively more aggressive quants until one actually fits.
const std::vector<std::pair<std::string, double>> QUANT_CANDIDATES = {
    {"Q8_0", 8.50}, {"Q6_K", 6.56}, {"Q5_K_M", 5.67}, {"Q5_K_S", 5.54},
    {"Q4_K_M", 4.83}, {"Q4_K_S", 4.58}, {"IQ4_XS", 4.25}, {"IQ3_M", 3.66},
};

// How many composite-scored candidate repos the verification step (verify_gguf_row) will actually
// fetch a real file listing for, per model. Kept small since each candidate costs a real HF tree
// API round trip (unlike the cheap search query that ranks them) - the top-scored repo is right
// often enough that a handful of runners-up is enough to recover from the cases (e.g. a repo with
// no model card, or one that turns out to only host a different quant than expected) where it isn't.
constexpr int VERIFY_CANDIDATE_REPOS = 3;

// How long a candidate repo's real HF search results / file listing stay cached (gguf_search_candidates,
// get_repo_files_cached below). Real GGUF uploads and search rankings rarely change minute to minute,
// and the point of this cache is purely to stop the picker from re-fetching the same ~150-repo
// candidate pool from scratch on every page load / filter toggle - hf_request_limiter only bounds
// concurrent connections, not total request volume over time, and that redundant repeat traffic is
// exactly what was tripping HF's per-IP rate limit.
constexpr long VERIFY_CACHE_TTL_SECONDS = 6 * 3600;

// Full-response cache for /model-picker/models, keyed by the request params that actually change
// the result set. Every sub-fetch this handler does (UGI CSV, derestricted terms, per-repo GGUF
// verification) is already TTL-cached on disk, but the handler still re-parses/re-scores/re-ranks
// the whole ~150-repo candidate pool from those caches on every single call - a few real seconds
// of CPU/disk work that a plain page refresh has no reason to redo, since the underlying catalog
// doesn't change minute to minute. This caches the assembled model list (everything except the
// live hardware/disk-space readings, which are cheap and always recomputed fresh) in memory for
// the same TTL as the data it's built from.
struct models_response_cache_entry {
    long  cached_at = 0;
    json  models_arr;
    size_t count = 0;
};
std::mutex g_models_response_cache_mutex;
std::map<std::string, models_response_cache_entry> g_models_response_cache;

// Offline-only fallback, used solely if both the GitHub fetch and the local
// disk cache are unavailable (first-ever run with no network). The live
// files in the repo are the real source; these are intentionally minimal.
// Kept in sync with derestricted-filter.json and scripts/model_picker.py's
// DEFAULT_DERESTRICTED_TERMS. The second row was added after cross-checking
// real HF tags pulled for ~160 candidate repos (fetch_hf_tags_batch): each
// term below was confirmed on an actual model page, not guessed. "jailbreak"
// is the one borderline case - seen once, alongside "red-teaming"/
// "evaluation" tags, so it may also catch a red-team eval repo rather than
// a ready-to-use uncensored finetune.
const std::vector<std::string> FALLBACK_DERESTRICTED_TERMS = {
    "abliterat", "derestrict", "uncensor", "decensor", "unalign",
    "unshackl", "unfilter", "unbound", "unrestrict", "heretic", "heresy",
    "nsfw", "not-for-all-audiences", "de-alignment",
    "obliterat", "ablated", "unlimited", "refusal-remov", "amoral", "jailbreak",
};
const std::map<std::string, std::string> FALLBACK_ARCH_MAP = {
    {"Qwen3MoeForCausalLM",  "qwen3moe"},  {"Qwen2MoeForCausalLM", "qwen2moe"},
    {"MixtralForCausalLM",   "mixtral"},   {"Llama4ForConditionalGeneration", "llama4"},
    {"Glm4MoeForCausalLM",   "glm4moe"},   {"GptOssForCausalLM",   "gpt-oss"},
    {"DeepseekV2ForCausalLM","deepseek2"}, {"DeepseekV3ForCausalLM","deepseek2"},
    {"MiniMaxM2ForCausalLM", "minimax-m2"},{"Lfm2MoeForCausalLM",  "lfm2moe"},
};

// ---------------------------------------------------------------------
// disk cache: same shape as scripts/model_picker.py's TTL-cached fetches
// ---------------------------------------------------------------------

std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void write_file(const std::string & path, const std::string & content) {
    std::ofstream out(path, std::ios::binary);
    out << content;
}

bool cache_is_fresh(const std::string & path, long ttl_seconds) {
    std::error_code ec;
    auto ftime = std::filesystem::last_write_time(path, ec);
    if (ec) return false;
    auto age = std::chrono::duration_cast<std::chrono::seconds>(
        std::filesystem::file_time_type::clock::now() - ftime).count();
    return age < ttl_seconds;
}

// Fetch url, cached on disk under cache_name with the given TTL. Falls back
// to a stale cache, then to fallback_content, in that order, so a network
// hiccup never breaks the endpoint outright.
std::string fetch_cached(const std::string & url, const std::string & cache_name,
                          long ttl_seconds, bool refresh, const std::string & fallback_content) {
    std::string cache_path = fs_get_cache_file("model-picker-" + cache_name);

    if (!refresh && cache_is_fresh(cache_path, ttl_seconds)) {
        std::string cached = read_file(cache_path);
        if (!cached.empty()) return cached;
    }

    common_remote_params params;
    params.timeout = 15;
    try {
        auto [http_code, body] = common_remote_get_content(url, params);
        if (http_code == 200 && !body.empty()) {
            std::string content(body.begin(), body.end());
            write_file(cache_path, content);
            return content;
        }
    } catch (const std::exception &) {
        // fall through to stale cache / fallback below
    }

    std::string stale = read_file(cache_path);
    if (!stale.empty()) return stale;
    return fallback_content;
}

// ---------------------------------------------------------------------
// minimal CSV parser (handles quoted fields with embedded commas/quotes)
// ---------------------------------------------------------------------

std::vector<std::string> parse_csv_line(const std::string & line) {
    std::vector<std::string> fields;
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); i++) {
        char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    cur += '"';
                    i++;
                } else {
                    in_quotes = false;
                }
            } else {
                cur += c;
            }
        } else {
            if (c == '"') {
                in_quotes = true;
            } else if (c == ',') {
                fields.push_back(cur);
                cur.clear();
            } else {
                cur += c;
            }
        }
    }
    fields.push_back(cur);
    return fields;
}

std::vector<std::map<std::string, std::string>> parse_csv(std::string text) {
    // strip UTF-8 BOM if present
    if (text.size() >= 3 && (unsigned char) text[0] == 0xEF &&
        (unsigned char) text[1] == 0xBB && (unsigned char) text[2] == 0xBF) {
        text = text.substr(3);
    }

    std::vector<std::map<std::string, std::string>> rows;
    std::istringstream stream(text);
    std::string line;

    if (!std::getline(stream, line)) return rows;
    // strip trailing \r (CRLF line endings)
    if (!line.empty() && line.back() == '\r') line.pop_back();
    auto header = parse_csv_line(line);

    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        auto fields = parse_csv_line(line);
        std::map<std::string, std::string> row;
        for (size_t i = 0; i < header.size() && i < fields.size(); i++) {
            row[header[i]] = fields[i];
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

double to_double(const std::string & s) {
    try {
        return std::stod(s);
    } catch (const std::exception &) {
        return 0.0;
    }
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

// Strips a trailing UGI-Leaderboard display annotation like " (thinking=True)",
// not part of the real HF repo id (repo ids can't contain spaces/parens).
std::string clean_repo_id(const std::string & name) {
    static const std::regex re(R"(\s*\([^)]*\)\s*$)");
    return std::regex_replace(name, re, "");
}

// Pulls out that same trailing annotation (e.g. "reasoning=disabled", "thinking=True") instead
// of discarding it - the same base model often has multiple separate UGI leaderboard rows that
// differ ONLY by this (reasoning on/off, effort level), each with genuinely different scores.
// Stripping it from the display name entirely (as clean_repo_id does for the repo id) would make
// those rows look like exact duplicates with no explanation for the different numbers.
std::string extract_variant(const std::string & name) {
    static const std::regex re(R"(\(([^)]*)\)\s*$)");
    std::smatch match;
    if (std::regex_search(name, match, re)) {
        return match[1].str();
    }
    return "";
}

// ---------------------------------------------------------------------
// hardware detection: reuses the same ggml_backend_dev_memory() calls
// common/fit.cpp already relies on for --fit, portable across Linux/Windows
// for free since that's ggml's own abstraction, not something to reinvent.
// ---------------------------------------------------------------------

struct hardware_info {
    double vram_gb      = 0.0; // total - what fit-tier budgets are computed against
    double ram_gb       = 0.0; // total - ditto
    double vram_free_gb = 0.0; // currently free - display only, not used in fit math
    double ram_free_gb  = 0.0; // ditto
};

// ggml_backend_cpu_device_get_memory() hardcodes free=total on Linux ("free system
// memory is ill-defined, for practical purposes assume that all of it is free" -
// ggml/src/ggml-cpu/ggml-cpu.cpp), so RAM free never differs from RAM total in the
// hardware cards otherwise. The kernel already computes a real, meaningful number
// for this - MemAvailable in /proc/meminfo, which (unlike MemFree) accounts for
// reclaimable page cache/buffers correctly. Returns -1 if unavailable (non-Linux,
// or the line's missing), so the caller can fall back to ggml's value.
double linux_ram_available_gb() {
    std::ifstream f("/proc/meminfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("MemAvailable:", 0) != 0) continue;
        std::istringstream iss(line.substr(strlen("MemAvailable:")));
        double kb;
        if (iss >> kb) return kb * 1e3 / 1e9; // kB -> GB
        break;
    }
    return -1.0;
}

// cudaMemGetInfo() (what ggml_backend_dev_memory() calls for CUDA devices)
// is unreliable under WSL2's paravirtualized GPU when more than one process
// holds a CUDA context on the device - e.g. this router process plus a
// router-mode model child it spawned. Each process's view of "free" doesn't
// reliably reflect the other's allocations there, so the router can report
// several GB free while nvidia-smi (and Task Manager, on the Windows host)
// correctly shows the device nearly full. Query nvidia-smi/NVML directly
// instead, same fix shape as linux_ram_available_gb() above. Sums across
// all reported GPUs; returns -1 if nvidia-smi isn't available so the caller
// falls back to ggml's value (e.g. non-NVIDIA hardware).
double nvidia_smi_vram_free_gb() {
#ifdef _WIN32
    FILE * pipe = _popen("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>NUL", "r");
#else
    FILE * pipe = popen("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null", "r");
#endif
    if (!pipe) return -1.0;

    double total_mib = 0.0;
    bool   got_any   = false;
    char   buf[128];
    while (fgets(buf, sizeof(buf), pipe)) {
        double mib;
        if (sscanf(buf, "%lf", &mib) == 1) {
            total_mib += mib;
            got_any    = true;
        }
    }

#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif

    if (!got_any) return -1.0;
    return total_mib * 1048576.0 / 1e9; // MiB -> GB
}

hardware_info detect_hardware() {
    hardware_info hw;

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev) {
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(cpu_dev, &free, &total);
        hw.ram_gb      = (double) total / 1e9;
        hw.ram_free_gb = (double) free  / 1e9;
    }
    double linux_free_gb = linux_ram_available_gb();
    if (linux_free_gb >= 0.0) hw.ram_free_gb = linux_free_gb;

    size_t vram_total = 0;
    size_t vram_free  = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            vram_total += total;
            vram_free  += free;
        }
    }
    hw.vram_gb      = (double) vram_total / 1e9;
    hw.vram_free_gb = (double) vram_free  / 1e9;

    double smi_vram_free_gb = nvidia_smi_vram_free_gb();
    if (smi_vram_free_gb >= 0.0) hw.vram_free_gb = smi_vram_free_gb;

    return hw;
}

// ---------------------------------------------------------------------
// model row + fit classification, mirrors scripts/model_picker.py
// ---------------------------------------------------------------------

struct model_row {
    std::string name;
    std::string variant; // e.g. "reasoning=disabled" - see extract_variant()
    std::string repo_id;
    std::string link;
    double active_b = 0.0;
    double total_b  = 0.0;
    std::string hf_arch;
    bool arch_known = false;     // UGI CSV populated an Architecture value at all
    bool arch_supported = false; // and it's one this fork can convert/run
    double ugi_score = 0.0;
    double willingness = 0.0;
    double active_gb = 0.0;
    double total_gb  = 0.0;
    std::string quant;
    std::string fit_tier;
    bool is_derestricted = false;
    std::string gguf_repo; // empty if not looked up / not found
    int  n_ctx_train = 0;  // 0 if unverified
    bool kv_verified = false; // true once fetch_kv_hparams succeeded for this row
    std::vector<double> ctx_options_gb; // parallel to CTX_SIZE_OPTIONS, empty if unverified
};

// The model - every expert, dense or routed - always lives on SSD; --moe-stream reads
// routed-expert weights off disk on demand into a bounded VRAM cache. Dense weights and
// the working expert set (active_gb) are the only thing that must actually be VRAM-
// resident, so that's the only thing gating whether a model is usable at all here - RAM
// plays no part in this. VRAM_FIT_FRACTION is deliberately a mid-point (not "everything
// that technically streams"): it holds real headroom back for KV-cache/context and
// compute buffers so the card is never maxed out just by loading the model, and it's a
// fraction of free VRAM rather than a fixed GiB figure so the same rule scales correctly
// from a 1080 Ti up to a 5090 without separate tuning per card size.
constexpr double VRAM_FIT_FRACTION = 0.55;

// The floor the fit check itself now enforces, instead of leaving "how much context actually
// fits" as an afterthought decided by whatever's left over once a model is already marked
// "fits" (that afterthought used to bottom out at a flat, model-agnostic 8192 - see
// DEFAULT_CTX_SIZE below). When a row's real KV-cache hyperparameters are known
// (kv_cache_gb > 0, from estimate_kv_cache_gb(h, MIN_USEFUL_CTX_TOKENS)), classify_fit
// requires active_gb + kv_cache_gb to clear VRAM_FIT_FRACTION together, so a model that
// would only leave room for a useless few thousand tokens of context no longer reports "fits."
constexpr int MIN_USEFUL_CTX_TOKENS = 32768;

// Mirrors CTX_SIZE_OPTIONS in tools/ui/src/routes/models/+page.svelte's ctxPicker - kept here
// rather than reimplementing estimate_kv_cache_gb's SWA-aware math in TypeScript, so the
// frontend only ever compares precomputed numbers against headroom instead of duplicating (and
// risking drift from) the formula itself.
constexpr int CTX_SIZE_OPTIONS[] = {4096, 8192, 16384, 32768, 65536};

std::string classify_fit(double active_gb, double total_gb, double vram_gb, double disk_free_gb, double kv_cache_gb = 0.0) {
    if (total_gb > disk_free_gb) return "no-disk-space"; // can't even be stored on SSD
    if (active_gb + kv_cache_gb <= vram_gb * VRAM_FIT_FRACTION) return "fits";
    return "too-large"; // active footprint (+ a useful context, when known) would starve compute - not shown
}

// Only two outcomes ever reach the final result list ("fits", scored by UGI; "no-disk-
// space", shown so the user can see what more storage would unlock) - "too-large" rows
// are dropped well before this, so fit_order just needs "fits" to sort first.
int fit_order(const std::string & tier) {
    return tier == "fits" ? 0 : 1;
}

bool is_usable(const std::string & tier) {
    return tier == "fits";
}

// The router has no auto-fit pass for models it launches (unlike a manually-run llama-server),
// so leaving --ctx-size unset means "0" reaches llama.cpp as-is, which resolves to the model's
// full native training context - unbounded as far as any of this file's VRAM/RAM budgeting is
// concerned, and enough on its own to blow way past any headroom fraction classify_fit reserves
// (e.g. 262144 tokens on a 26B model). A fixed, generous-for-normal-chat default here instead;
// editable per-model afterward directly in the preset INI.
constexpr int DEFAULT_CTX_SIZE = 8192;

// Once active_gb (dense weights + working expert set) is accounted for, whatever's still
// free in VRAM gets split in half: one half grows --moe-stream-cache (fewer disk-streaming
// misses on the routed experts), the other half stays reserved for KV-cache/context and
// compute buffers. A flat fraction of *leftover* VRAM rather than a fixed GiB figure, so
// it's a "medium" tune that scales the same way on a 1080 Ti as a 5090 - never handing the
// whole card over to the cache alone.
constexpr double CACHE_HEADROOM_FRACTION = 0.5;

// Same idea, for the second-tier host-RAM cache (--moe-stream-ram-cache): a flat fraction of
// whatever RAM detect_hardware() reports as free (hw.ram_free_gb, already the OS's own
// "available for new allocations" figure - MemAvailable on Linux, which on WSL2 is capped by
// .wslconfig's [wsl2] memory= setting, so this scales down automatically on a constrained
// Windows host same as it scales up on a bare-metal Linux box with lots of RAM). The other half
// stays free for the OS, the model's own mmap'd dense weights, and everything else already
// competing for host memory - the RAM cache is a bonus tier, not the primary residency source.
constexpr double RAM_CACHE_HEADROOM_FRACTION = 0.5;

bool text_matches_terms(const std::vector<std::string> & terms, const std::string & text) {
    std::string lower = to_lower(text);
    for (const auto & t : terms) {
        if (lower.find(t) != std::string::npos) return true;
    }
    return false;
}

// On WSL2, the filesystem under the user's home dir lives inside a
// dynamically-growing virtual disk (a VHDX on the Windows host). Its
// self-reported free space is unallocated blocks *within that virtual disk*,
// which can be far larger than what's actually left on the physical host
// drive it still needs to grow into. /mnt/c (when present) is the real host
// C: drive via DrvFs, so cap by whichever is smaller. Assumes the common
// default single-drive (C:) WSL setup; won't catch a WSL install relocated
// to another drive letter.
bool running_on_wsl() {
    std::ifstream f("/proc/version");
    std::string line;
    std::getline(f, line);
    return to_lower(line).find("microsoft") != std::string::npos;
}

double disk_free_gb(const std::string & path) {
    std::error_code ec;
    auto space = std::filesystem::space(path, ec);
    if (ec) return 0.0;
    double free_gb = (double) space.available / 1e9;

    if (running_on_wsl()) {
        std::error_code host_ec;
        auto host_space = std::filesystem::space("/mnt/c", host_ec);
        if (!host_ec) {
            free_gb = std::min(free_gb, (double) host_space.available / 1e9);
        }
    }
    return free_gb;
}

// Total capacity, not free space - used as a hard ceiling: a model bigger than this can
// never fit no matter what the user deletes, vs. one that's merely bigger than *current*
// free space (deletable-fixable, still worth showing). Same WSL host-drive-capping logic
// as disk_free_gb: the VHDX's reported capacity can be set larger than the physical host
// drive it still needs to grow into.
double disk_total_gb(const std::string & path) {
    std::error_code ec;
    auto space = std::filesystem::space(path, ec);
    if (ec) return 0.0;
    double total_gb = (double) space.capacity / 1e9;

    if (running_on_wsl()) {
        std::error_code host_ec;
        auto host_space = std::filesystem::space("/mnt/c", host_ec);
        if (!host_ec) {
            total_gb = std::min(total_gb, (double) host_space.capacity / 1e9);
        }
    }
    return total_gb;
}

// run one HF model-search query, returning every result that actually hosts GGUF files, ranked
// best-first. checked via the "gguf" tag HF attaches to any repo containing .gguf files, not the
// repo name - plenty of repos (huihui-ai's abliterated models among them) host GGUF quants
// alongside safetensors in the same repo without "GGUF" anywhere in the name itself.
//
// Ranked by a downloads+likes composite (log-scaled so neither magnitude dominates) rather than
// whichever HF's search ranks first - plain search relevance skews toward big-name uploaders
// (bartowski/unsloth) regardless of whether a less prominent but well-used quantizer (e.g.
// mradermacher) actually has a better, or the only, quant of this particular model.
std::vector<std::string> gguf_search_query_ranked(const std::string & url) {
    common_remote_params params;
    params.timeout = 8;
    std::vector<std::pair<double, std::string>> scored;
    try {
        hf_request_guard guard;
        auto [http_code, body] = common_remote_get_content(url, params);
        if (http_code == 200 && !body.empty()) {
            json results = json::parse(std::string(body.begin(), body.end()));

            for (const auto & r : results) {
                if (!r.contains("tags") || !r["tags"].is_array()) continue;
                bool has_gguf_tag = false;
                for (const auto & t : r["tags"]) {
                    if (t.is_string() && to_lower(t.get<std::string>()) == "gguf") {
                        has_gguf_tag = true;
                        break;
                    }
                }
                if (!has_gguf_tag) continue;

                double downloads = r.value("downloads", 0.0);
                double likes      = r.value("likes", 0.0);
                double score = std::log10(downloads + 1.0) + std::log10(likes + 1.0);
                scored.emplace_back(score, r.value("id", ""));
            }
        }
    } catch (const std::exception &) {
        // ignore, best-effort only
    }

    std::sort(scored.begin(), scored.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
    std::vector<std::string> ids;
    ids.reserve(scored.size());
    for (auto & [score, id] : scored) {
        if (!id.empty()) ids.push_back(id);
    }
    return ids;
}

// best-effort search for existing community GGUF quants, ranked best-scored-first, top
// `max_candidates` only. Falls back to huihui-ai's namespace specifically when the name search
// comes up short: they're a prolific publisher of abliterated (uncensored) derivatives that host
// their own GGUF quants, covering a lot of ground a plain name search misses since the derivative
// has a different repo name entirely.
std::vector<std::string> gguf_search_candidates_uncached(const std::string & repo_id, int max_candidates) {
    std::string base = repo_id;
    auto slash = base.find_last_of('/');
    if (slash != std::string::npos) base = base.substr(slash + 1);

    std::vector<std::string> ids = gguf_search_query_ranked(
        "https://huggingface.co/api/models?search=" + base + "%20GGUF&limit=15");

    if ((int) ids.size() < max_candidates) {
        auto fallback = gguf_search_query_ranked(
            "https://huggingface.co/api/models?search=" + base + "&author=huihui-ai&limit=5");
        for (auto & id : fallback) {
            if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
                ids.push_back(id);
            }
        }
    }
    if ((int) ids.size() > max_candidates) {
        ids.resize(max_candidates);
    }
    return ids;
}

std::mutex g_gguf_search_cache_mutex;

// TTL-disk-cached wrapper, same rationale and TTL as get_repo_files_cached above - the search step
// costs 1-2 HF API calls per row and gets re-run for largely the same candidate pool on every page
// load / filter toggle otherwise.
std::vector<std::string> gguf_search_candidates(const std::string & repo_id, int max_candidates) {
    std::string cache_key = repo_id + "::" + std::to_string(max_candidates);
    std::string cache_path = fs_get_cache_file("model-picker-gguf-search.json");
    long now = (long) std::time(nullptr);

    {
        std::lock_guard<std::mutex> lock(g_gguf_search_cache_mutex);
        json cache = json::parse(read_file(cache_path), nullptr, false);
        if (!cache.is_discarded() && cache.is_object()) {
            auto it = cache.find(cache_key);
            if (it != cache.end() && it->contains("t") && it->contains("ids") &&
                now - (*it)["t"].get<long>() < VERIFY_CACHE_TTL_SECONDS) {
                return it->at("ids").get<std::vector<std::string>>();
            }
        }
    }

    std::vector<std::string> ids = gguf_search_candidates_uncached(repo_id, max_candidates);

    {
        std::lock_guard<std::mutex> lock(g_gguf_search_cache_mutex);
        json cache = json::parse(read_file(cache_path), nullptr, false);
        if (cache.is_discarded() || !cache.is_object()) cache = json::object();
        cache[cache_key] = { {"t", now}, {"ids", ids} };
        write_file(cache_path, cache.dump());
    }

    return ids;
}

json hf_file_to_json(const hf_cache::hf_file & f) {
    // local_path/final_path deliberately omitted - verification only ever reads path/size, never
    // downloads, so there's nothing to invalidate if the real cache-dir layout changes later.
    return { {"path", f.path}, {"url", f.url}, {"oid", f.oid}, {"repo_id", f.repo_id}, {"size", f.size} };
}

hf_cache::hf_file hf_file_from_json(const json & j) {
    hf_cache::hf_file f;
    f.path    = j.value("path", std::string());
    f.url     = j.value("url", std::string());
    f.oid     = j.value("oid", std::string());
    f.repo_id = j.value("repo_id", std::string());
    f.size    = j.value("size", (uint64_t) 0);
    return f;
}

std::mutex g_repo_files_cache_mutex;

// TTL-disk-cached wrapper around hf_cache::get_repo_files, used only by the verification step -
// the real download path always calls hf_cache::get_repo_files directly for a live, uncached
// lookup. One combined JSON file keyed by repo_id, same shape as fetch_hf_tags_batch's cache.
hf_cache::hf_files get_repo_files_cached(const std::string & repo_id) {
    std::string cache_path = fs_get_cache_file("model-picker-repo-files.json");
    long now = (long) std::time(nullptr);

    {
        std::lock_guard<std::mutex> lock(g_repo_files_cache_mutex);
        json cache = json::parse(read_file(cache_path), nullptr, false);
        if (!cache.is_discarded() && cache.is_object()) {
            auto it = cache.find(repo_id);
            if (it != cache.end() && it->contains("t") && it->contains("files") &&
                now - (*it)["t"].get<long>() < VERIFY_CACHE_TTL_SECONDS) {
                hf_cache::hf_files files;
                for (auto & fj : (*it)["files"]) files.push_back(hf_file_from_json(fj));
                return files;
            }
        }
    }

    hf_cache::hf_files files;
    {
        hf_request_guard guard;
        files = hf_cache::get_repo_files(repo_id, "");
    }

    {
        std::lock_guard<std::mutex> lock(g_repo_files_cache_mutex);
        json cache = json::parse(read_file(cache_path), nullptr, false);
        if (cache.is_discarded() || !cache.is_object()) cache = json::object();
        json files_json = json::array();
        for (auto & f : files) files_json.push_back(hf_file_to_json(f));
        cache[repo_id] = { {"t", now}, {"files", files_json} };
        write_file(cache_path, cache.dump());
    }

    return files;
}

// Per-model attention-shape hyperparameters needed to price a KV-cache in real bytes -
// none of this is in HF's ?expand[]=gguf metadata (verified live: that only ever returns
// total/architecture/context_length/chat_template/tokenizer tokens/totalFileSize), but it
// is reliably in the base model's own config.json, which model_row.repo_id already points
// at (separate from gguf_repo, the resolved quant repo verify_gguf_row finds).
struct kv_hparams {
    bool ok = false;
    int n_layer = 0;
    int n_head_kv = 0;
    int head_dim = 0;
    int n_ctx_train = 0;
    int sliding_window = 0;         // 0 = no SWA
    int sliding_window_pattern = 0; // every Nth layer is full attention; 0/1 = treat all layers as full
};

// Reads the handful of fields classify_fit's KV-cache estimate needs out of a parsed
// config.json object. Returns ok=false if num_hidden_layers isn't present - the one field
// with no safe default, so its absence means "can't verify this model" rather than "assume
// zero layers." Caller retries this against a nested "text_config" object first for VLM-
// wrapper architectures (confirmed live against gemma-3-12b-it/gemma-3-4b-it) where the text
// model's real hparams aren't at the top level.
kv_hparams parse_kv_hparams(const json & cfg) {
    kv_hparams h;
    if (!cfg.contains("num_hidden_layers") || !cfg["num_hidden_layers"].is_number()) return h;

    h.n_layer = cfg["num_hidden_layers"].get<int>();

    int n_head = cfg.value("num_attention_heads", 0);
    h.n_head_kv = cfg.value("num_key_value_heads", n_head); // no GQA -> same as query heads

    int hidden_size = cfg.value("hidden_size", 0);
    h.head_dim = cfg.value("head_dim", 0);
    if (h.head_dim <= 0 && n_head > 0) h.head_dim = hidden_size / n_head;

    h.n_ctx_train = cfg.value("max_position_embeddings", 0);
    h.sliding_window = cfg.value("sliding_window", 0);
    h.sliding_window_pattern = cfg.value("sliding_window_pattern", 0);

    h.ok = h.n_layer > 0 && h.n_head_kv > 0 && h.head_dim > 0;
    return h;
}

// GET the base model's config.json - not the GGUF repo's, the original HF repo model_row.repo_id
// already carries from the UGI CSV. A small, standard, well-documented JSON file (unlike a GGUF's
// own header, which would require a binary range-request parser and risks pulling in several MB of
// embedded tokenizer vocab before reaching the fields we actually want).
kv_hparams fetch_kv_hparams(const std::string & repo_id) {
    common_remote_params params;
    params.timeout = 6;
    kv_hparams h;
    try {
        hf_request_guard guard;
        auto [http_code, body] = common_remote_get_content(
            "https://huggingface.co/" + repo_id + "/raw/main/config.json", params);
        if (http_code != 200 || body.empty()) return h;
        json cfg = json::parse(std::string(body.begin(), body.end()), nullptr, false);
        if (cfg.is_discarded() || !cfg.is_object()) return h;

        h = parse_kv_hparams(cfg);
        if (!h.ok && cfg.contains("text_config") && cfg["text_config"].is_object()) {
            h = parse_kv_hparams(cfg["text_config"]);
        }
    } catch (const std::exception &) {
        // best-effort only
    }
    return h;
}

std::mutex g_kv_hparams_cache_mutex;

// TTL-disk-cached wrapper around fetch_kv_hparams, same shape as get_repo_files_cached but
// keyed by the base repo_id and cached for HF_TAGS_TTL_SECONDS - a model's architecture
// hyperparameters never change post-release, so this is effectively a permanent cache that
// just re-warms itself monthly.
kv_hparams get_kv_hparams_cached(const std::string & repo_id) {
    std::string cache_path = fs_get_cache_file("model-picker-kv-hparams.json");
    long now = (long) std::time(nullptr);

    {
        std::lock_guard<std::mutex> lock(g_kv_hparams_cache_mutex);
        json cache = json::parse(read_file(cache_path), nullptr, false);
        if (!cache.is_discarded() && cache.is_object()) {
            auto it = cache.find(repo_id);
            if (it != cache.end() && it->contains("t") && it->contains("h") &&
                now - (*it)["t"].get<long>() < HF_TAGS_TTL_SECONDS) {
                kv_hparams h;
                auto & hj = (*it)["h"];
                h.ok                     = hj.value("ok", false);
                h.n_layer                = hj.value("n_layer", 0);
                h.n_head_kv              = hj.value("n_head_kv", 0);
                h.head_dim               = hj.value("head_dim", 0);
                h.n_ctx_train            = hj.value("n_ctx_train", 0);
                h.sliding_window         = hj.value("sliding_window", 0);
                h.sliding_window_pattern = hj.value("sliding_window_pattern", 0);
                return h;
            }
        }
    }

    kv_hparams h = fetch_kv_hparams(repo_id);

    {
        std::lock_guard<std::mutex> lock(g_kv_hparams_cache_mutex);
        json cache = json::parse(read_file(cache_path), nullptr, false);
        if (cache.is_discarded() || !cache.is_object()) cache = json::object();
        cache[repo_id] = {
            {"t", now},
            {"h", {
                {"ok", h.ok}, {"n_layer", h.n_layer}, {"n_head_kv", h.n_head_kv},
                {"head_dim", h.head_dim}, {"n_ctx_train", h.n_ctx_train},
                {"sliding_window", h.sliding_window}, {"sliding_window_pattern", h.sliding_window_pattern},
            }},
        };
        write_file(cache_path, cache.dump());
    }

    return h;
}

// Estimate KV-cache VRAM for ctx_tokens of context, F16 (llama.cpp's default; this UI doesn't
// expose --cache-type-k/v) - so this only ever over-estimates real usage, never under. SWA layers
// (when sliding_window_pattern is known) are priced at min(ctx_tokens, sliding_window) instead of
// the full context, since only every Nth layer pays the full price; with no pattern info every
// layer is conservatively priced as full-attention.
double estimate_kv_cache_gb(const kv_hparams & h, int ctx_tokens) {
    constexpr int kv_bytes_per_elem = 2; // F16
    int pattern     = h.sliding_window_pattern > 1 ? h.sliding_window_pattern : 0;
    int full_layers = pattern ? (h.n_layer + pattern - 1) / pattern : h.n_layer;
    int swa_layers  = h.n_layer - full_layers;
    int swa_ctx     = h.sliding_window > 0 ? std::min(ctx_tokens, h.sliding_window) : ctx_tokens;

    double bytes = 0.0;
    bytes += (double) full_layers * h.n_head_kv * h.head_dim * 2 /*K+V*/ * kv_bytes_per_elem * ctx_tokens;
    bytes += (double) swa_layers  * h.n_head_kv * h.head_dim * 2         * kv_bytes_per_elem * swa_ctx;
    return bytes / 1e9;
}

// Assessment of an arbitrary user-supplied GGUF (local file or direct URL), not from the
// curated UGI list - read straight out of the GGUF's own metadata header instead of a base
// HF repo's config.json, so it works for any file, including ones with no matching HF repo.
struct gguf_probe {
    bool ok = false;
    std::string arch;
    int n_layer = 0;
    int n_head_kv = 0;
    int head_dim = 0;
    int n_ctx_train = 0;
    int sliding_window = 0;
    int sliding_window_pattern = 0;
    int n_expert = 0;
    int n_expert_used = 0;
    double total_gb = 0.0;  // sum of every tensor's byte size
    double active_gb = 0.0; // dense + (moe experts scaled by n_expert_used/n_expert)
    std::string quant_label;
    bool is_split = false; // "split.count" KV present - size may be understated, see caller
};

kv_hparams gguf_probe_to_kv_hparams(const gguf_probe & p) {
    kv_hparams h;
    h.ok                     = p.ok;
    h.n_layer                = p.n_layer;
    h.n_head_kv               = p.n_head_kv;
    h.head_dim               = p.head_dim;
    h.n_ctx_train            = p.n_ctx_train;
    h.sliding_window         = p.sliding_window;
    h.sliding_window_pattern = p.sliding_window_pattern;
    return h;
}

// Shared metadata extraction, once a gguf_context has been opened (no_alloc, header-only) by
// either the local-file or remote-callback path below. Expert-merged tensors in this fork's
// GGUF convention hold every routed expert's weights in one tensor per layer/projection (name
// containing "_exps."), so their total byte size scales down by n_expert_used/n_expert for the
// active-VRAM estimate; everything else (dense/shared weights, embeddings, norms) always
// counts in full.
gguf_probe gguf_probe_from_ctx(struct gguf_context * ctx) {
    gguf_probe p;

    int64_t arch_id = gguf_find_key(ctx, "general.architecture");
    p.arch = arch_id >= 0 ? gguf_get_val_str(ctx, arch_id) : "";
    p.is_split = gguf_find_key(ctx, "split.count") >= 0;

    auto get_int = [&](const std::string & suffix, int def = 0) -> int {
        std::string key = p.arch + "." + suffix;
        int64_t id = gguf_find_key(ctx, key.c_str());
        if (id < 0) return def;
        switch (gguf_get_kv_type(ctx, id)) {
            case GGUF_TYPE_UINT8:  return gguf_get_val_u8(ctx, id);
            case GGUF_TYPE_INT8:   return gguf_get_val_i8(ctx, id);
            case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, id);
            case GGUF_TYPE_INT16:  return gguf_get_val_i16(ctx, id);
            case GGUF_TYPE_UINT32: return (int) gguf_get_val_u32(ctx, id);
            case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx, id);
            case GGUF_TYPE_UINT64: return (int) gguf_get_val_u64(ctx, id);
            case GGUF_TYPE_INT64:  return (int) gguf_get_val_i64(ctx, id);
            default: return def;
        }
    };

    p.n_layer   = get_int("block_count");
    int n_head  = get_int("attention.head_count");
    p.n_head_kv = get_int("attention.head_count_kv", n_head);
    int embd    = get_int("embedding_length");
    p.head_dim  = get_int("attention.key_length");
    if (p.head_dim <= 0 && n_head > 0) p.head_dim = embd / n_head;
    p.n_ctx_train            = get_int("context_length");
    p.sliding_window         = get_int("attention.sliding_window");
    p.sliding_window_pattern = get_int("attention.sliding_window_pattern");
    p.n_expert               = get_int("expert_count");
    p.n_expert_used          = get_int("expert_used_count");

    int64_t n_tensors = gguf_get_n_tensors(ctx);
    double dense_bytes = 0.0, moe_bytes = 0.0;
    std::map<std::string, int64_t> type_votes; // by cumulative bytes, to find the dominant quant
    for (int64_t i = 0; i < n_tensors; i++) {
        std::string name = gguf_get_tensor_name(ctx, i);
        size_t sz = gguf_get_tensor_size(ctx, i);
        if (name.find("_exps.") != std::string::npos) {
            moe_bytes += (double) sz;
        } else {
            dense_bytes += (double) sz;
        }
        if (sz > 1000000) { // skip tiny norm/bias tensors when picking a representative quant
            type_votes[ggml_type_name(gguf_get_tensor_type(ctx, i))] += (int64_t) sz;
        }
    }
    p.total_gb = (dense_bytes + moe_bytes) / 1e9;
    p.active_gb = (p.n_expert > 0 && p.n_expert_used > 0)
        ? (dense_bytes + moe_bytes * ((double) p.n_expert_used / p.n_expert)) / 1e9
        : p.total_gb;

    int64_t best_bytes = 0;
    for (auto & [type_name, bytes] : type_votes) {
        if (bytes > best_bytes) { best_bytes = bytes; p.quant_label = type_name; }
    }

    p.ok = p.n_layer > 0 && p.n_head_kv > 0 && p.head_dim > 0 && n_tensors > 0;
    return p;
}

gguf_probe probe_gguf_local(const std::string & path) {
    gguf_init_params params;
    params.no_alloc = true;
    params.ctx = nullptr;
    struct gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    if (!ctx) return gguf_probe{};

    gguf_probe p = gguf_probe_from_ctx(ctx);
    gguf_free(ctx);

    // Ground truth for a local file - trust the filesystem over the tensor-size sum (which
    // omits header/KV/alignment overhead, a rounding error in practice but free to correct).
    std::error_code ec;
    auto file_size = std::filesystem::file_size(path, ec);
    if (!ec && file_size > 0) {
        double real_total_gb = (double) file_size / 1e9;
        if (p.total_gb > 0) {
            double scale = real_total_gb / p.total_gb;
            p.active_gb *= scale;
        }
        p.total_gb = real_total_gb;
    }
    return p;
}

// Hard ceiling on how much of a remote GGUF's header we'll ever fetch/buffer - real headers
// (KV pairs + tensor-info table) are a tiny fraction of this even for huge vocabs. Also handed
// to gguf_init_from_callback as max_expected_size so the parser treats reads past it as a
// legitimate end-of-data rather than assuming unlimited bytes remain (a true unbounded value
// there let a truncated/corrupt remote header run the parser past valid memory).
constexpr uint64_t GGUF_REMOTE_MAX_HEADER_BYTES = 256ull * 1024 * 1024;

// Backs gguf_init_from_callback with ranged HTTP GETs against a remote URL, so only the header
// (magic/version/KV pairs/tensor-info table) is ever fetched - never the multi-GB tensor data
// blob. A single generous prefetch covers virtually every model in one round trip; only issues
// a second, larger ranged GET if the parser asks for bytes beyond it (e.g. an unusually large
// vocab/KV array).
struct gguf_remote_reader {
    std::string url;
    std::vector<char> buf;
    uint64_t buf_len = 0;
    bool failed = false;

    bool ensure(uint64_t offset, size_t len) {
        if (offset + len <= buf_len) return true;
        if (failed) return false;

        // A GGUF header is read as thousands of small fields (one HTTP round trip per field
        // would be requested here otherwise, since the parser tracks its own read position and
        // calls the callback per-field). Doubling the fetch size each time turns a header whose
        // KV/vocab data runs past the initial prefetch into O(log n) requests instead of O(n),
        // capped well above any real header (tensor data is never touched - no_alloc skips it).
        uint64_t want = std::max<uint64_t>(offset + (uint64_t) len, buf_len == 0 ? 0 : buf_len * 2);
        want = std::min<uint64_t>(want, GGUF_REMOTE_MAX_HEADER_BYTES);
        if (want < offset + (uint64_t) len) {
            failed = true; // header apparently larger than any real GGUF's - bail out
            return false;
        }
        common_remote_params params;
        params.timeout = 15;
        params.headers.push_back({"Range", "bytes=0-" + std::to_string(want - 1)});
        auto [http_code, body] = common_remote_get_content(url, params);
        if ((http_code != 200 && http_code != 206) || body.empty()) {
            failed = true;
            return false;
        }
        buf.assign(body.begin(), body.end());
        buf_len = buf.size();
        return buf_len > offset; // may still be < want if that's genuinely the whole file
    }
};

constexpr size_t GGUF_REMOTE_PREFETCH_BYTES = 2 * 1024 * 1024;

size_t gguf_remote_read_cb(void * userdata, void * output, uint64_t offset, size_t len) {
    auto * r = (gguf_remote_reader *) userdata;
    if (!r->ensure(offset, len)) return 0;
    if (offset >= r->buf_len) return 0;
    size_t avail = (size_t) std::min<uint64_t>((uint64_t) len, r->buf_len - offset);
    std::memcpy(output, r->buf.data() + offset, avail);
    return avail;
}

gguf_probe probe_gguf_remote(const std::string & url) {
    gguf_remote_reader reader;
    reader.url = url;
    if (!reader.ensure(0, GGUF_REMOTE_PREFETCH_BYTES)) return gguf_probe{};

    gguf_init_params params;
    params.no_alloc = true;
    params.ctx = nullptr;
    // max_expected_size isn't a hint - passing 0 makes the reader believe 0 bytes remain and
    // every read fails immediately (gguf_init_from_callback only special-cases max_chunk_read==0,
    // not this parameter). We don't know the real remote file size up front, so bound it by our
    // own header-fetch ceiling instead of passing something like UINT64_MAX - that let a
    // truncated/corrupt remote header be treated as having unlimited bytes remaining, which
    // crashed the parser instead of failing the read cleanly.
    struct gguf_context * ctx =
        gguf_init_from_callback(gguf_remote_read_cb, &reader, 0, GGUF_REMOTE_MAX_HEADER_BYTES, params);
    if (!ctx) return gguf_probe{};

    gguf_probe p = gguf_probe_from_ctx(ctx);
    gguf_free(ctx);
    return p;
}

// Verify a candidate row against real HF file listings instead of trusting the CSV-estimate size:
// try each of the top VERIFY_CANDIDATE_REPOS composite-scored repos, and within each, each quant
// best-to-worst, until a REAL resolvable file (or split group) is found that actually fits - reusing
// the exact same matching common_download_get_hf_plan() uses (common_download_resolve_model_files),
// so a row this returns as "IQ3_M, 7.8 GB" can never turn into an actual bf16 download at click
// time. On success overwrites gguf_repo/quant/active_gb/total_gb/fit_tier with the real numbers; on
// failure leaves gguf_repo empty so the row gets dropped downstream, same as an outright search miss.
void verify_gguf_row(model_row & m, double vram_gb, double free_gb) {
    // Real per-model KV-cache hyperparameters, fetched once per row regardless of which
    // candidate repo/quant ends up winning below - it's a property of the base model
    // (m.repo_id), not of any particular GGUF quant. h.ok == false (fetch failed, or an
    // architecture parse_kv_hparams can't read cleanly, e.g. recurrent/hybrid-memory
    // archs) falls back to kv_cache_gb = 0.0, i.e. classify_fit's pre-existing
    // active_gb-only check - unverifiable, not zero-cost.
    kv_hparams h = get_kv_hparams_cached(m.repo_id);
    double kv_cache_gb = h.ok ? estimate_kv_cache_gb(h, MIN_USEFUL_CTX_TOKENS) : 0.0;

    for (auto & repo : gguf_search_candidates(m.repo_id, VERIFY_CANDIDATE_REPOS)) {
        hf_cache::hf_files files = get_repo_files_cached(repo);
        if (files.empty()) continue;

        int         best_order = -1;
        std::string best_quant;
        std::string best_tier;
        double      best_active_gb = 0.0;
        double      best_total_gb  = 0.0;

        for (auto & [qname, qbpw] : QUANT_CANDIDATES) {
            (void) qbpw;
            auto plan = common_download_resolve_model_files(files, qname);
            if (plan.primary.path.empty()) continue;

            uint64_t total_bytes = plan.primary_is_legacy_split ? plan.primary.size : 0;
            if (!plan.primary_is_legacy_split) {
                for (auto & f : plan.model_files) total_bytes += f.size;
            }
            if (total_bytes == 0) continue; // size unknown - can't trust it

            double total_gb  = (double) total_bytes / 1e9;
            // same active/total ratio the CSV-estimate math used, now anchored to a real total
            // size instead of an assumed one - bpw is uniform across tensors for a given quant, so
            // the active-parameter share of file size scales with active_b/total_b either way.
            double active_gb = total_gb * (m.active_b / m.total_b);
            std::string tier = classify_fit(active_gb, total_gb, vram_gb, free_gb, kv_cache_gb);
            if (!is_usable(tier)) continue;

            int order = fit_order(tier);
            if (best_order < 0 || order < best_order) {
                best_order     = order;
                best_quant     = qname;
                best_tier      = tier;
                best_active_gb = active_gb;
                best_total_gb  = total_gb;
            }
        }

        if (best_order >= 0) {
            m.gguf_repo    = repo;
            m.quant        = best_quant;
            m.fit_tier     = best_tier;
            m.active_gb    = best_active_gb;
            m.total_gb     = best_total_gb;
            m.n_ctx_train  = h.n_ctx_train;
            m.kv_verified  = h.ok;
            if (h.ok) {
                for (int ctx : CTX_SIZE_OPTIONS) m.ctx_options_gb.push_back(estimate_kv_cache_gb(h, ctx));
            }
            return; // first real usable match wins - candidates are already priority ordered
        }
    }
    // nothing real found in any candidate repo - gguf_repo stays empty, dropped downstream
}

// fetch one model repo's HF tags - the signal that catches abliteration-tool-branded
// repos (e.g. "Heretic-...") a name-keyword match alone would miss, since the tool
// tags its own output even when the repo isn't named after it.
std::vector<std::string> fetch_hf_tags(const std::string & repo_id) {
    common_remote_params params;
    params.timeout = 6;
    std::vector<std::string> tags;
    try {
        hf_request_guard guard;
        auto [http_code, body] = common_remote_get_content(
            "https://huggingface.co/api/models/" + repo_id + "?expand[]=tags", params);
        if (http_code != 200 || body.empty()) return tags;
        json j = json::parse(std::string(body.begin(), body.end()), nullptr, false);
        if (j.is_discarded() || !j.contains("tags") || !j["tags"].is_array()) return tags;
        for (auto & t : j["tags"]) {
            if (t.is_string()) tags.push_back(t.get<std::string>());
        }
    } catch (const std::exception &) {
        // best-effort only
    }
    return tags;
}

// disk-cached, parallel HF tags lookup for a batch of repo ids. Mirrors
// scripts/model_picker.py's hf-tags.json cache: fetch once, keep for
// HF_TAGS_TTL_SECONDS, only hit the network for repos that are missing or stale.
std::unordered_map<std::string, std::vector<std::string>> fetch_hf_tags_batch(
        const std::vector<std::string> & repo_ids, bool refresh) {
    std::string cache_path = fs_get_cache_file("model-picker-hf-tags.json");
    json cache = json::object();
    if (!refresh) {
        json parsed = json::parse(read_file(cache_path), nullptr, false);
        if (!parsed.is_discarded() && parsed.is_object()) cache = parsed;
    }

    long now = (long) std::time(nullptr);
    std::unordered_map<std::string, std::vector<std::string>> result;
    std::vector<std::string> to_fetch;
    for (auto & rid : repo_ids) {
        if (result.count(rid)) continue; // duplicate repo_id in the batch
        auto it = cache.find(rid);
        if (it != cache.end() && it->contains("t") && it->contains("tags") &&
            now - (*it)["t"].get<long>() < HF_TAGS_TTL_SECONDS) {
            result[rid] = (*it)["tags"].get<std::vector<std::string>>();
        } else {
            to_fetch.push_back(rid);
        }
    }

    g_tags_progress_done  = 0;
    g_tags_progress_total = (int) to_fetch.size();

    std::vector<std::future<std::vector<std::string>>> futures;
    futures.reserve(to_fetch.size());
    for (auto & rid : to_fetch) {
        futures.push_back(std::async(std::launch::async, [rid]() {
            std::vector<std::string> tags = fetch_hf_tags(rid);
            // incremented here (not after .get() below) so progress reflects real
            // completion order across the parallel fetches, not join order
            g_tags_progress_done.fetch_add(1, std::memory_order_relaxed);
            return tags;
        }));
    }
    for (size_t i = 0; i < to_fetch.size(); i++) {
        std::vector<std::string> tags = futures[i].get();
        result[to_fetch[i]] = tags;
        cache[to_fetch[i]] = { {"t", now}, {"tags", tags} };
    }

    if (!to_fetch.empty()) {
        write_file(cache_path, cache.dump());
    }
    return result;
}

} // namespace

void server_model_picker_register_routes(const server_http_context & ctx_http, const std::string & models_preset_path) {
    ctx_http.get("/model-picker/models", [models_preset_path](const server_http_req & req) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        res->content_type = "application/json; charset=utf-8";

        try {
            bool refresh          = req.get_param("refresh") == "true";
            bool derestricted_only = req.get_param("derestricted_only") == "true";
            bool unsupported_only  = req.get_param("unsupported_only") == "true";
            // Same candidate pool for all three - just a different priority
            // within each fit-tier group. "ugi" (default) = best-known overall
            // quality (UGI's own blended score). "size" = biggest active-param
            // model that still fits, ignoring quality/censorship entirely.
            // "willingness" = least likely to refuse, ignoring general capability.
            std::string rank_by    = req.get_param("rank_by").empty() ? "ugi" : req.get_param("rank_by");
            double min_ugi         = req.get_param("min_ugi").empty() ? 0.0 : std::stod(req.get_param("min_ugi"));
            int top                = req.get_param("top").empty() ? 20 : std::stoi(req.get_param("top"));
            // how many ranked candidates (before top-N truncation) to search for a community GGUF
            // quant; larger than `top` since some won't have one and get filtered out afterward.
            // searched in parallel, so this doesn't cost much page-load latency to keep generous.
            int gguf_lookup        = req.get_param("gguf_lookup").empty() ? 60 : std::stoi(req.get_param("gguf_lookup"));

            hardware_info hw = detect_hardware();
            double vram_gb = req.get_param("vram_gb").empty() ? hw.vram_gb : std::stod(req.get_param("vram_gb"));
            double ram_gb  = req.get_param("ram_gb").empty()  ? hw.ram_gb  : std::stod(req.get_param("ram_gb"));
            // What fit-tier budgets and moe-stream-cache sizing actually compute against - vram_gb/
            // ram_gb above (total) are reported for display only now. Assuming the whole card/whole
            // machine is available ignores whatever else is already using it: the OS, a browser,
            // another loaded model, or someone actually gaming on the same GPU. Same free-not-total
            // reasoning disk_free_gb below already applies; overridable for the same testability
            // reason too.
            double vram_free_gb = req.get_param("vram_free_gb").empty() ? hw.vram_free_gb : std::stod(req.get_param("vram_free_gb"));
            double ram_free_gb  = req.get_param("ram_free_gb").empty()  ? hw.ram_free_gb  : std::stod(req.get_param("ram_free_gb"));
            // overridable like vram_gb/ram_gb above, mainly so the no-disk-space / too-big-for-
            // this-drive tiers are actually testable without needing a physically full disk
            double free_gb = req.get_param("disk_free_gb").empty()
                                  ? disk_free_gb(".") : std::stod(req.get_param("disk_free_gb"));
            double total_capacity_gb = req.get_param("disk_total_gb").empty()
                                  ? disk_total_gb(".") : std::stod(req.get_param("disk_total_gb"));

            std::string cache_dir_for_key;
            try {
                cache_dir_for_key = hf_cache::get_cache_dir();
            } catch (const std::exception &) {
                // fall through with an empty key component - just means this fallback path
                // never shares a cache entry with a successful lookup, which is fine
            }
            // free/total disk space rounded to the nearest GB so trivial fluctuations (a few MB
            // written/freed between requests) don't force a full recompute, while a real change
            // (a big download finishing, a drive filling up) still busts the cache and re-buckets
            // fit tiers correctly.
            std::string cache_key = derestricted_only ? "d1" : "d0";
            cache_key += unsupported_only ? "u1" : "u0";
            cache_key += "|" + rank_by + "|" + std::to_string(top) + "|" + std::to_string(gguf_lookup)
                       + "|" + std::to_string((long) std::round(vram_gb))
                       + "|" + std::to_string((long) std::round(ram_gb))
                       + "|" + std::to_string((long) std::round(vram_free_gb))
                       + "|" + std::to_string((long) std::round(ram_free_gb))
                       + "|" + std::to_string((long) std::round(free_gb))
                       + "|" + std::to_string((long) std::round(total_capacity_gb))
                       + "|" + cache_dir_for_key;

            json cached_arr;
            size_t cached_count = 0;
            bool have_cached_list = false;
            if (!refresh) {
                std::lock_guard<std::mutex> lk(g_models_response_cache_mutex);
                auto it = g_models_response_cache.find(cache_key);
                if (it != g_models_response_cache.end() &&
                    (long) std::time(nullptr) - it->second.cached_at < VERIFY_CACHE_TTL_SECONDS) {
                    cached_arr        = it->second.models_arr;
                    cached_count      = it->second.count;
                    have_cached_list  = true;
                }
            }

            if (have_cached_list) {
                json out;
                std::string cache_dir = cache_dir_for_key;
                out["hardware"] = {
                    {"vram_gb", vram_gb}, {"ram_gb", ram_gb},
                    {"vram_free_gb", hw.vram_free_gb}, {"ram_free_gb", hw.ram_free_gb},
                    {"disk_free_gb", free_gb}, {"disk_total_gb", total_capacity_gb}, {"cache_dir", cache_dir},
                };
                out["router_available"] = !models_preset_path.empty();
                out["count"]  = cached_count;
                out["models"] = cached_arr;
                res->status = 200;
                res->data = out.dump();
                return res;
            }

            // ---- fetch the three data sources (GitHub primary, cache/fallback if unreachable) ----

            json fallback_filter = { {"terms", FALLBACK_DERESTRICTED_TERMS}, {"known_repo_ids", json::array()} };
            std::string filter_raw = fetch_cached(DERESTRICTED_FILTER_URL, "derestricted-filter.json",
                                                   DEFAULT_TTL_SECONDS, refresh, fallback_filter.dump());
            json filter_json = json::parse(filter_raw, nullptr, false);
            std::vector<std::string> terms = FALLBACK_DERESTRICTED_TERMS;
            std::vector<std::string> known_repo_ids;
            if (!filter_json.is_discarded()) {
                if (filter_json.contains("terms")) {
                    terms = filter_json["terms"].get<std::vector<std::string>>();
                }
                if (filter_json.contains("known_repo_ids")) {
                    known_repo_ids = filter_json["known_repo_ids"].get<std::vector<std::string>>();
                }
            }

            json fallback_arch = FALLBACK_ARCH_MAP;
            std::string arch_raw = fetch_cached(ARCH_MAP_URL, "arch-map.json",
                                                 DEFAULT_TTL_SECONDS, refresh, fallback_arch.dump());
            json arch_json = json::parse(arch_raw, nullptr, false);
            std::map<std::string, std::string> arch_map = FALLBACK_ARCH_MAP;
            if (!arch_json.is_discarded() && arch_json.is_object()) {
                arch_map.clear();
                for (auto & [k, v] : arch_json.items()) {
                    arch_map[k] = v.get<std::string>();
                }
            }

            std::string csv_raw = fetch_cached(UGI_CSV_URL, "ugi-leaderboard-data.csv",
                                                DEFAULT_TTL_SECONDS, refresh, "");
            auto rows = parse_csv(csv_raw);

            // ---- build + filter + score ----

            std::vector<model_row> models;
            for (auto & r : rows) {
                double active_b = to_double(r["Active Parameters"]);
                double total_b  = to_double(r["Total Parameters"]);
                if (active_b <= 0 || total_b <= 0 || active_b >= total_b) continue; // not MoE / missing data

                model_row m;
                // The raw UGI display name carries a trailing annotation like
                // " (thinking=True)" - stripped from the display name (clutter, users click
                // through to the model link for detail) but kept separately as `variant`: the
                // same base model often has multiple leaderboard rows differing only by this
                // (reasoning on/off, effort level) with genuinely different scores, so silently
                // dropping it would make those rows look like unexplained exact duplicates.
                m.variant = extract_variant(r["author/model_name"]);
                m.name    = clean_repo_id(r["author/model_name"]);
                m.link    = r["Model Link"];
                m.active_b = active_b;
                m.total_b  = total_b;
                m.hf_arch  = r["Architecture"];
                m.repo_id  = m.name;
                m.ugi_score   = to_double(r["UGI \xF0\x9F\x8F\x86"]); // "UGI 🏆"
                m.willingness = to_double(r["W/10 \xF0\x9F\x91\x8D"]); // "W/10 👍"

                m.arch_known = !m.hf_arch.empty();
                auto ait = arch_map.find(m.hf_arch);
                m.arch_supported = ait != arch_map.end();

                // Picking the first best-to-worst quant whose active footprint merely fits under
                // VRAM_FIT_FRACTION would pick Q8_0 whenever it happens to clear the bar, even
                // when a smaller quant would leave meaningfully more headroom. Instead: find the
                // best (lowest fit_order) tier any candidate quant can reach, then take the
                // highest-bpw quant that still reaches it - best quality that's actually "fits".
                bool fits_at_any_quant = false;
                int  best_order        = -1;
                for (auto & [qname, qbpw] : QUANT_CANDIDATES) {
                    double a_gb = active_b * 1e9 * qbpw / 8 / 1e9;
                    double t_gb = total_b  * 1e9 * qbpw / 8 / 1e9;
                    std::string tier = classify_fit(a_gb, t_gb, vram_free_gb, free_gb);
                    if (!is_usable(tier)) continue;
                    const int order = fit_order(tier);
                    if (best_order < 0 || order < best_order) {
                        best_order  = order;
                        m.quant     = qname;
                        m.active_gb = a_gb;
                        m.total_gb  = t_gb;
                        m.fit_tier  = tier;
                        fits_at_any_quant = true;
                    }
                }
                if (!fits_at_any_quant) {
                    // Doesn't fit at any quant, for one of two different reasons - tell them
                    // apart using the smallest (most compressed) candidate's footprint, since
                    // that's the realistic "what would it actually take" figure either way.
                    const auto & [qname, qbpw] = QUANT_CANDIDATES.back();
                    double a_gb = active_b * 1e9 * qbpw / 8 / 1e9;
                    double t_gb = total_b  * 1e9 * qbpw / 8 / 1e9;
                    // Bigger than the drive's total capacity - no amount of deleting ever makes
                    // this one fit, so it's excluded entirely rather than shown as "needs space".
                    if (t_gb > total_capacity_gb) continue;
                    m.quant     = qname;
                    m.active_gb = a_gb;
                    m.total_gb  = t_gb;
                    // Still show "doesn't currently fit on disk, free some space" (actionable);
                    // "active footprint too big for VRAM at any quant" isn't fixable by the user
                    // at all, so it's filtered out below instead of cluttering the list.
                    m.fit_tier  = t_gb > free_gb ? "no-disk-space" : "too-large";
                }

                models.push_back(std::move(m));
            }

            // Derestricted detection: name keywords + a curated repo-id allowlist +
            // live HF tags (batched, parallel, disk-cached) - tags catch abliteration-tool
            // brand names in a model's tags even when its name doesn't mention them.
            {
                std::vector<std::string> repo_ids;
                repo_ids.reserve(models.size());
                for (auto & m : models) repo_ids.push_back(m.repo_id);
                auto tags_by_id = fetch_hf_tags_batch(repo_ids, refresh);

                for (auto & m : models) {
                    bool known = std::find(known_repo_ids.begin(), known_repo_ids.end(), m.repo_id) != known_repo_ids.end();
                    std::string tag_text;
                    auto tit = tags_by_id.find(m.repo_id);
                    if (tit != tags_by_id.end()) {
                        for (auto & t : tit->second) { tag_text += t; tag_text += ' '; }
                    }
                    m.is_derestricted = known || text_matches_terms(terms, m.name + " " + tag_text);
                }
            }

            std::vector<model_row> filtered;
            for (auto & m : models) {
                // The UGI CSV leaves Architecture blank for some (often very
                // new) models - that means "unknown," not "confirmed this
                // fork can't load it." Default view: include confirmed-
                // supported and unknown, exclude only confirmed-unsupported.
                // unsupported_only (gap-finding): show only confirmed gaps,
                // not unknowns.
                if (unsupported_only) {
                    if (!m.arch_known || m.arch_supported) continue;
                } else {
                    if (m.arch_known && !m.arch_supported) continue;
                }
                if (derestricted_only && !m.is_derestricted) continue;
                if (m.ugi_score < min_ugi) continue;
                // Active footprint doesn't fit VRAM at any quant - no RAM/disk fallback tier
                // anymore, so there's nothing to show here, actionable or otherwise.
                if (m.fit_tier == "too-large") continue;
                filtered.push_back(m);
            }

            std::sort(filtered.begin(), filtered.end(), [&rank_by](const model_row & a, const model_row & b) {
                int fa = fit_order(a.fit_tier), fb = fit_order(b.fit_tier);
                if (fa != fb) return fa < fb;
                if (rank_by == "size") return a.active_gb > b.active_gb;
                if (rank_by == "willingness") return a.willingness > b.willingness;
                return a.ugi_score > b.ugi_score; // "ugi" and any unrecognized value
            });

            // Bucket by the CSV-estimate tier purely to decide verification priority below - this
            // is not the final bucketing shown to the user, since verification can move a row to a
            // different real tier (or drop it outright). Splitting first still matters: it's what
            // stops the no-disk-space tail from being crowded out of its own gguf_lookup
            // verification budget by whatever's estimate-ranked ahead of it.
            std::vector<model_row> est_ideal, est_no_space;
            for (auto & m : filtered) {
                if (m.fit_tier == "no-disk-space") {
                    est_no_space.push_back(std::move(m));
                } else {
                    est_ideal.push_back(std::move(m));
                }
            }

            // Verify a pool bigger than `top` (gguf_lookup), in parallel, before truncating - a
            // ranked candidate with no real, resolvable community GGUF quant isn't something the
            // user can click Download on, so it gets filtered out below rather than shown as a dead
            // (or worse, mis-sized) row; verifying only the eventual top-N would make the "nothing
            // found" rows a lookup-limit artifact instead of an honest "we checked, there isn't
            // one". Run per-group (not one flat pool) so each estimate-tier tail gets its own
            // gguf_lookup budget.
            auto verify_group = [gguf_lookup, vram_free_gb, free_gb](std::vector<model_row> & group) {
                const int n = std::min((int) group.size(), gguf_lookup);
                std::vector<std::future<void>> futures;
                futures.reserve(n);
                for (int i = 0; i < n; i++) {
                    futures.push_back(std::async(std::launch::async, verify_gguf_row,
                                                  std::ref(group[i]), vram_free_gb, free_gb));
                }
                for (int i = 0; i < n; i++) {
                    futures[i].get();
                }
                std::vector<model_row> verified;
                for (int i = 0; i < n; i++) {
                    // beyond the verified pool, gguf_repo is unknown, not confirmed-absent - drop
                    // those too, same reasoning as an actual miss: nothing to click Download on
                    if (!group[i].gguf_repo.empty()) {
                        verified.push_back(std::move(group[i]));
                    }
                }
                group = std::move(verified);
            };
            verify_group(est_ideal);
            verify_group(est_no_space);

            // Re-bucket by the now-real fit_tier verify_gguf_row resolved - a row that only
            // survives verification is always "fits" (is_usable() already excluded anything
            // else), so this just separates that from the no-disk-space informational tail.
            std::vector<model_row> ideal, no_space;
            for (auto * group : { &est_ideal, &est_no_space }) {
                for (auto & m : *group) {
                    if (m.fit_tier == "no-disk-space") {
                        no_space.push_back(std::move(m));
                    } else {
                        ideal.push_back(std::move(m));
                    }
                }
            }

            auto sort_by_rank = [&rank_by](std::vector<model_row> & group) {
                std::sort(group.begin(), group.end(), [&rank_by](const model_row & a, const model_row & b) {
                    if (rank_by == "size") return a.active_gb > b.active_gb;
                    if (rank_by == "willingness") return a.willingness > b.willingness;
                    return a.ugi_score > b.ugi_score; // "ugi" and any unrecognized value
                });
            };
            sort_by_rank(ideal);
            sort_by_rank(no_space);

            // `top` only bounds the ideal ("fits") group; every actionable no-disk-space
            // candidate is kept regardless, so a flat top-N truncation doesn't cut it off first.
            if ((int) ideal.size() > top) ideal.resize(top);
            ideal.insert(ideal.end(), std::make_move_iterator(no_space.begin()), std::make_move_iterator(no_space.end()));
            filtered = std::move(ideal);

            json out;
            std::string cache_dir;
            try {
                cache_dir = hf_cache::get_cache_dir();
            } catch (const std::exception &) {
                // best-effort only; the UI just skips showing a path if this is empty
            }
            out["hardware"] = {
                {"vram_gb", vram_gb}, {"ram_gb", ram_gb},
                {"vram_free_gb", hw.vram_free_gb}, {"ram_free_gb", hw.ram_free_gb},
                {"disk_free_gb", free_gb}, {"disk_total_gb", total_capacity_gb}, {"cache_dir", cache_dir},
            };
            out["router_available"] = !models_preset_path.empty();
            out["count"] = filtered.size();
            json arr = json::array();
            for (auto & m : filtered) {
                arr.push_back({
                    {"name", m.name}, {"variant", m.variant}, {"repo_id", m.repo_id}, {"link", m.link},
                    {"active_b", m.active_b}, {"total_b", m.total_b},
                    {"active_gb", m.active_gb}, {"total_gb", m.total_gb},
                    {"quant", m.quant},
                    {"hf_arch", m.hf_arch}, {"fit_tier", m.fit_tier},
                    {"ugi_score", m.ugi_score}, {"willingness", m.willingness},
                    {"is_derestricted", m.is_derestricted},
                    {"gguf_repo", m.gguf_repo},
                    {"n_ctx_train", m.n_ctx_train},
                    {"kv_verified", m.kv_verified},
                    {"ctx_options_gb", m.ctx_options_gb},
                });
            }
            out["models"] = arr;

            {
                std::lock_guard<std::mutex> lk(g_models_response_cache_mutex);
                g_models_response_cache[cache_key] = {
                    (long) std::time(nullptr), arr, filtered.size()
                };
            }

            res->status = 200;
            res->data = out.dump();
        } catch (const std::exception & e) {
            res->status = 500;
            res->data = json{{"error", e.what()}}.dump();
        }

        return res;
    });

    // Cheap, no CSV/tags/gguf-search - called right before firing a download so the
    // frontend can re-check free space against the model's size at click time, not just
    // whatever was true when the page/table last loaded (which can go stale: another
    // download finished, disk filled up elsewhere, user left the tab open a while).
    ctx_http.get("/model-picker/disk-free", [](const server_http_req & req) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        res->content_type = "application/json; charset=utf-8";
        // overridable like /model-picker/models's disk_free_gb param, for testing
        double free_gb = req.get_param("disk_free_gb").empty()
                              ? disk_free_gb(".") : std::stod(req.get_param("disk_free_gb"));
        res->data = json{{"disk_free_gb", free_gb}}.dump();
        return res;
    });

    // Polled by the frontend while GET /model-picker/models is in flight, to show
    // progress during the (cold-cache-only, ~one-time) batch HF tags fetch instead of
    // a blind spinner. total==0 means nothing needed fetching (warm cache / not started).
    ctx_http.get("/model-picker/tags-progress", [](const server_http_req &) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        res->content_type = "application/json; charset=utf-8";
        res->data = json{
            {"done",  g_tags_progress_done.load(std::memory_order_relaxed)},
            {"total", g_tags_progress_total.load(std::memory_order_relaxed)},
        }.dump();
        return res;
    });

    // stash a per-model --moe-stream-cache override in the router's --models-preset INI file
    // before the caller triggers the actual download/load via POST /models. auto-fit (common/
    // fit.cpp) has no idea --moe-stream-cache exists, so without this every downloaded model
    // would load with the engine's generic 2*n_expert_used default instead of a size tuned to
    // this machine's actual VRAM.
    ctx_http.post("/model-picker/prepare-download", [models_preset_path](const server_http_req & req) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        res->content_type = "application/json; charset=utf-8";

        if (models_preset_path.empty()) {
            res->status = 404;
            res->data = json{{"error", "model-picker downloads require router mode (launch with --models-preset)"}}.dump();
            return res;
        }

        try {
            json body = json::parse(req.body);
            std::string gguf_repo = body.value("gguf_repo", std::string());
            std::string repo_id   = body.value("repo_id", std::string()); // base HF repo, for KV-cache sizing
            std::string quant     = body.value("quant", std::string());
            std::string path      = body.value("path", std::string()); // ad-hoc local-file source
            std::string url       = body.value("url", std::string());  // ad-hoc direct-URL source
            double      active_gb = body.value("active_gb", 0.0);
            int         ctx_size  = body.value("ctx_size", DEFAULT_CTX_SIZE);

            // Ad-hoc GGUF assessment (server-model-picker.cpp's /model-picker/assess-gguf) has
            // no repo_id to re-derive KV-cache hparams from - the client passes the same
            // gguf_probe fields it already got back from that endpoint instead.
            bool has_inline_hparams = body.contains("n_layer");
            kv_hparams inline_h;
            if (has_inline_hparams) {
                inline_h.n_layer                = body.value("n_layer", 0);
                inline_h.n_head_kv              = body.value("n_head_kv", 0);
                inline_h.head_dim               = body.value("head_dim", 0);
                inline_h.n_ctx_train            = body.value("n_ctx_train", 0);
                inline_h.sliding_window         = body.value("sliding_window", 0);
                inline_h.sliding_window_pattern = body.value("sliding_window_pattern", 0);
                inline_h.ok = inline_h.n_layer > 0 && inline_h.n_head_kv > 0 && inline_h.head_dim > 0;
            }

            std::string model_id;
            if (!gguf_repo.empty() && !quant.empty()) {
                model_id = gguf_repo + ":" + quant;
            } else if (!path.empty()) {
                model_id = path;
            } else if (!url.empty()) {
                model_id = url;
            } else {
                throw std::invalid_argument("gguf_repo+quant, path, or url is required");
            }
            if (ctx_size <= 0) {
                ctx_size = DEFAULT_CTX_SIZE;
            }

            // Freshly detected here rather than trusting client-supplied VRAM figures - the same
            // "recheck right before firing" reasoning downloadModel() already applies to disk
            // space: whatever's free can have changed since the page loaded (another app claimed
            // VRAM, another model got loaded).
            hardware_info hw = detect_hardware();

            // active_gb (dense weights + working expert set) is going to be VRAM-resident no
            // matter what - see VRAM_FIT_FRACTION, which already confirmed this row fits under
            // that basis. Real KV-cache bytes for the actually-requested ctx_size are reserved
            // next (when repo_id's hparams are known - see fetch_kv_hparams), so moe-stream-cache
            // can no longer eat the space this specific context size needs. Only unverifiable rows
            // (repo_id missing, or hparams fetch failed) fall back to the old blind 50/50 split of
            // whatever's left after active_gb alone.
            double headroom_gb = std::max(0.0, hw.vram_free_gb - active_gb);

            double kv_cache_gb = 0.0;
            if (!repo_id.empty()) {
                kv_hparams h = get_kv_hparams_cached(repo_id);
                if (h.ok) kv_cache_gb = estimate_kv_cache_gb(h, ctx_size);
            } else if (inline_h.ok) {
                kv_cache_gb = estimate_kv_cache_gb(inline_h, ctx_size);
            }

            // --moe-stream-cache only gets a cut of whatever's left over after active_gb AND this
            // context's real KV-cache reservation, and only half of THAT (CACHE_HEADROOM_FRACTION),
            // so loading the model never eats the headroom left for compute buffers. --moe-stream-
            // cache only parses integer GiB (or integer slot counts), so round down to a whole GiB
            // with a 1 GiB floor.
            double cache_headroom_gb = std::max(0.0, headroom_gb - kv_cache_gb);
            uint64_t cache_gb = std::max<uint64_t>(1, (uint64_t) (cache_headroom_gb * CACHE_HEADROOM_FRACTION));

            std::string cache_val = std::to_string(cache_gb) + "G";

            // second-tier host-RAM cache: sized off hw.ram_free_gb alone (independent of VRAM/
            // active_gb/kv_cache_gb above - it's a different physical pool), floored at 0 rather
            // than 1 since unlike moe-stream-cache this tier is optional and fine to skip
            // entirely on a RAM-starved box (e.g. WSL2 with a small .wslconfig memory= cap).
            uint64_t ram_cache_gb = (uint64_t) (std::max(0.0, hw.ram_free_gb) * RAM_CACHE_HEADROOM_FRACTION);

            common_preset_write_ini_section(models_preset_path, model_id, {
                {"moe-stream-cache", cache_val},
                {"moe-stream-ram-cache", std::to_string(ram_cache_gb)},
                {"ctx-size", std::to_string(ctx_size)},
            });

            res->status = 200;
            res->data = json{
                {"success", true},
                {"model_id", model_id},
                {"moe_stream_cache_gb", cache_gb},
                {"moe_stream_ram_cache_gb", ram_cache_gb},
                {"ctx_size", ctx_size},
            }.dump();
        } catch (const std::exception & e) {
            res->status = 400;
            res->data = json{{"error", e.what()}}.dump();
        }

        return res;
    });

    // Instant hardware-fit assessment for an arbitrary GGUF not in the curated UGI list -
    // "local" reads a file already on disk, "url" range-reads just the header (never the
    // multi-GB tensor data) over HTTP. Both go through the same probe_gguf_* -> gguf_probe
    // pipeline and the same classify_fit()/estimate_kv_cache_gb() the curated rows use, so
    // the verdict means the same thing here as it does anywhere else on this page.
    ctx_http.post("/model-picker/assess-gguf", [](const server_http_req & req) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        res->content_type = "application/json; charset=utf-8";

        try {
            json body = json::parse(req.body);
            std::string source = body.value("source", std::string());
            std::string path   = body.value("path", std::string());
            std::string url    = body.value("url", std::string());

            gguf_probe p;
            if (source == "local") {
                if (path.empty()) throw std::invalid_argument("path is required for source=local");
                p = probe_gguf_local(path);
            } else if (source == "url") {
                if (url.empty()) throw std::invalid_argument("url is required for source=url");
                p = probe_gguf_remote(url);
            } else {
                throw std::invalid_argument("source must be \"local\" or \"url\"");
            }

            if (!p.ok) {
                res->status = 422;
                res->data = json{{"error", "couldn't read GGUF metadata from that " +
                    std::string(source == "local" ? "path" : "URL") +
                    " - check it exists and is a valid GGUF file"}}.dump();
                return res;
            }

            hardware_info hw = detect_hardware();
            double free_gb = disk_free_gb(".");
            kv_hparams h = gguf_probe_to_kv_hparams(p);
            double kv_cache_gb = h.ok ? estimate_kv_cache_gb(h, MIN_USEFUL_CTX_TOKENS) : 0.0;
            std::string fit = classify_fit(p.active_gb, p.total_gb, hw.vram_free_gb, free_gb, kv_cache_gb);

            res->status = 200;
            res->data = json{
                {"fit", fit},
                {"arch", p.arch},
                {"is_moe", p.n_expert > 0},
                {"quant_label", p.quant_label},
                {"active_gb", p.active_gb},
                {"total_gb", p.total_gb},
                {"kv_cache_gb", kv_cache_gb},
                {"n_ctx_train", p.n_ctx_train},
                {"is_split", p.is_split},
                {"vram_free_gb", hw.vram_free_gb},
                {"disk_free_gb", free_gb},
                {"n_layer", p.n_layer},
                {"n_head_kv", p.n_head_kv},
                {"head_dim", p.head_dim},
                {"sliding_window", p.sliding_window},
                {"sliding_window_pattern", p.sliding_window_pattern},
                {"n_expert", p.n_expert},
                {"n_expert_used", p.n_expert_used},
            }.dump();
        } catch (const std::exception & e) {
            res->status = 400;
            res->data = json{{"error", e.what()}}.dump();
        }

        return res;
    });
}
