#include "server-model-picker.h"

#include "common.h"
#include "download.h"
#include "hf-cache.h"
#include "ggml-backend.h"
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
    double ram_spillover_gb = 0.0; // only meaningful when fit_tier == "ram-cache"
    // total_gb alone (mmap'd file + OS page cache), independent of fit_tier's active-set-only
    // budget - a model whose active set is comfortably VRAM-resident can still be most of the
    // machine's RAM in total size, which thrashes the OS page cache under sustained streaming.
    // Set once total_gb is real (post-verification); excluded from the default view, shown only
    // with allow_ram_heavy=true.
    bool ram_risk = false;
    bool is_derestricted = false;
    std::string gguf_repo; // empty if not looked up / not found
};

// --moe-stream serves the full model from disk with a bounded RAM-LRU cache
// in front of it, not "everything must be simultaneously resident." So the
// only hard failure is not being able to store the model at all (disk); a
// working set bigger than VRAM+RAM just means more cache misses streamed
// from disk per token (slower), not "won't run." Tiers below are purely
// informational (expected speed), never exclude a model.
std::string classify_fit(double active_gb, double total_gb, double vram_gb, double ram_gb, double disk_free_gb) {
    double easy_budget = vram_gb * 0.35;
    double vram_budget = vram_gb * 0.75;
    double ram_budget  = ram_gb * 0.7;

    if (total_gb > disk_free_gb) return "no-disk-space"; // can't even be stored
    if (active_gb <= easy_budget) return "easy";                    // VRAM-resident, plenty of headroom
    if (active_gb <= vram_budget) return "comfortable";              // VRAM-resident
    if (active_gb <= vram_gb + ram_budget) return "ram-cache";       // spills into RAM cache, no disk streaming needed for the hot set
    return "disk-streaming";                                        // active set exceeds VRAM+RAM, expect real per-token disk streaming
}

// Used only to group results (fastest-expected first, then by UGI within a
// group) so the top-N selection isn't dominated by a faster-but-lower-UGI
// pick over a slower-but-much-better one; the tier distinction is
// informational (shown via fit_tier), not a ranking exclusion - nothing
// below is ever dropped from results except "no-disk-space".
int fit_order(const std::string & tier) {
    if (tier == "easy" || tier == "comfortable") return 0;
    if (tier == "ram-cache")      return 1;
    if (tier == "disk-streaming") return 2;
    return 3; // no-disk-space
}

bool is_usable(const std::string & tier) {
    return tier != "no-disk-space";
}

// Above this fraction of total RAM, just holding the file's mmap'd pages leaves too little
// headroom for the OS/other apps and the active-expert cache itself, regardless of how well the
// active set alone fits VRAM+RAM - the machine ends up thrashing under sustained streaming even
// though fit_tier (judged on the active set only) reports "easy". Scales with the machine: a
// generous-RAM box naturally clears far more models than a tight one, rather than a fixed cutoff.
constexpr double RAM_THRASH_FRACTION = 0.75;

// The router has no auto-fit pass for models it launches (unlike a manually-run llama-server),
// so leaving --ctx-size unset means "0" reaches llama.cpp as-is, which resolves to the model's
// full native training context - unbounded as far as any of this file's VRAM/RAM budgeting is
// concerned, and enough on its own to blow way past any headroom fraction classify_fit reserves
// (e.g. 262144 tokens on a 26B model). A fixed, generous-for-normal-chat default here instead;
// editable per-model afterward directly in the preset INI.
constexpr int DEFAULT_CTX_SIZE = 8192;

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

// Verify a candidate row against real HF file listings instead of trusting the CSV-estimate size:
// try each of the top VERIFY_CANDIDATE_REPOS composite-scored repos, and within each, each quant
// best-to-worst, until a REAL resolvable file (or split group) is found that actually fits - reusing
// the exact same matching common_download_get_hf_plan() uses (common_download_resolve_model_files),
// so a row this returns as "IQ3_M, 7.8 GB" can never turn into an actual bf16 download at click
// time. On success overwrites gguf_repo/quant/active_gb/total_gb/fit_tier with the real numbers; on
// failure leaves gguf_repo empty so the row gets dropped downstream, same as an outright search miss.
void verify_gguf_row(model_row & m, double vram_gb, double ram_gb, double free_gb) {
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
            std::string tier = classify_fit(active_gb, total_gb, vram_gb, ram_gb, free_gb);
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
            m.gguf_repo = repo;
            m.quant     = best_quant;
            m.fit_tier  = best_tier;
            m.active_gb = best_active_gb;
            m.total_gb  = best_total_gb;
            m.ram_spillover_gb = m.fit_tier == "ram-cache"
                                      ? std::max(0.0, m.active_gb - vram_gb * 0.75) : 0.0;
            m.ram_risk  = m.total_gb > ram_gb * RAM_THRASH_FRACTION;
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
            // default view hides models whose total size alone risks thrashing this machine's
            // RAM (see RAM_THRASH_FRACTION); this opts back in for a specific model worth the
            // slower/heavier streaming, without changing what shows up by default for everyone else
            bool allow_ram_heavy   = req.get_param("allow_ram_heavy") == "true";
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
            cache_key += allow_ram_heavy ? "r1" : "r0";
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

                // Every quant that fits on disk at all is technically "usable" (that's the whole
                // point of --moe-stream: disk-streaming still runs, just slower), so picking the
                // first best-to-worst quant that merely fits on disk picked Q8_0 for nearly every
                // model regardless of how slow it'd actually be. Instead: find the best fit_tier
                // any candidate quant can reach, then take the highest-bpw quant that reaches it -
                // e.g. prefer a smaller quant that's VRAM-resident over Q8_0 that's disk-streaming.
                bool fits_at_any_quant = false;
                int  best_order        = -1;
                for (auto & [qname, qbpw] : QUANT_CANDIDATES) {
                    double a_gb = active_b * 1e9 * qbpw / 8 / 1e9;
                    double t_gb = total_b  * 1e9 * qbpw / 8 / 1e9;
                    std::string tier = classify_fit(a_gb, t_gb, vram_free_gb, ram_free_gb, free_gb);
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
                    // Doesn't fit in *current free* space at any quant - still show the row
                    // instead of silently dropping it: the user needs to see it to judge what to
                    // delete to make room for it, not have it vanish with no explanation. Use the
                    // smallest (most compressed) candidate's footprint, since that's the realistic
                    // "how much space would I actually need" figure, not the biggest quant's.
                    const auto & [qname, qbpw] = QUANT_CANDIDATES.back();
                    double a_gb = active_b * 1e9 * qbpw / 8 / 1e9;
                    double t_gb = total_b  * 1e9 * qbpw / 8 / 1e9;
                    // ...unless it's bigger than the drive's total capacity, not just what's
                    // currently free - no amount of deleting ever makes that one fit, so this is
                    // the one case still worth excluding entirely rather than showing as "needs
                    // more space".
                    if (t_gb > total_capacity_gb) continue;
                    m.quant     = qname;
                    m.active_gb = a_gb;
                    m.total_gb  = t_gb;
                    m.fit_tier  = "no-disk-space";
                }

                // ram_spillover_gb is left unset here (0.0) even for an estimate-time "ram-cache"
                // tier - verify_gguf_row recomputes it from the real resolved size once the row is
                // verified below, since the estimate tier can change once real sizes are known.

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
            // stops the degraded/no-disk-space tails from being crowded out of their own
            // gguf_lookup verification budget by whatever's estimate-ranked ahead of them.
            std::vector<model_row> est_ideal, est_degraded, est_no_space;
            for (auto & m : filtered) {
                if (m.fit_tier == "no-disk-space") {
                    est_no_space.push_back(std::move(m));
                } else if (m.fit_tier == "ram-cache" || m.fit_tier == "disk-streaming") {
                    est_degraded.push_back(std::move(m));
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
            auto verify_group = [gguf_lookup, vram_free_gb, ram_free_gb, free_gb](std::vector<model_row> & group) {
                const int n = std::min((int) group.size(), gguf_lookup);
                std::vector<std::future<void>> futures;
                futures.reserve(n);
                for (int i = 0; i < n; i++) {
                    futures.push_back(std::async(std::launch::async, verify_gguf_row,
                                                  std::ref(group[i]), vram_free_gb, ram_free_gb, free_gb));
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
            verify_group(est_degraded);
            verify_group(est_no_space);

            // Re-bucket by the now-real fit_tier verify_gguf_row resolved - a row estimated as
            // "easy" can turn out to only have a real disk-streaming-tier quant available, or vice
            // versa, and the user needs the tier they see to reflect what they'd actually get.
            std::vector<model_row> ideal, degraded, no_space;
            for (auto * group : { &est_ideal, &est_degraded, &est_no_space }) {
                for (auto & m : *group) {
                    if (!allow_ram_heavy && m.ram_risk) continue;
                    if (m.fit_tier == "no-disk-space") {
                        no_space.push_back(std::move(m));
                    } else if (m.fit_tier == "ram-cache" || m.fit_tier == "disk-streaming") {
                        degraded.push_back(std::move(m));
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
            sort_by_rank(degraded);
            sort_by_rank(no_space);

            // `top` only bounds the ideal (easy/comfortable) group; every actionable degraded and
            // no-disk-space candidate is kept regardless - see the comment this logic used to carry
            // above, before re-bucketing moved here: a flat top-N truncation would otherwise always
            // cut off before reaching a single degraded or no-disk-space row.
            if ((int) ideal.size() > top) ideal.resize(top);
            ideal.insert(ideal.end(), std::make_move_iterator(degraded.begin()), std::make_move_iterator(degraded.end()));
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
                    {"hf_arch", m.hf_arch}, {"fit_tier", m.fit_tier}, {"ram_spillover_gb", m.ram_spillover_gb},
                    {"ram_risk", m.ram_risk},
                    {"ugi_score", m.ugi_score}, {"willingness", m.willingness},
                    {"is_derestricted", m.is_derestricted},
                    {"gguf_repo", m.gguf_repo},
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
            std::string quant     = body.value("quant", std::string());
            double      total_gb  = body.value("total_gb", 0.0);
            if (gguf_repo.empty() || quant.empty()) {
                throw std::invalid_argument("gguf_repo and quant are required");
            }

            // Freshly detected here rather than trusting a client-supplied vram_gb (dropped as a
            // param entirely) - the same "recheck right before firing" reasoning downloadModel()
            // already applies to disk space: whatever's free can have changed since the page
            // loaded (another app claimed VRAM, another model got loaded), and total was never
            // the right basis for this to begin with - see vram_free_gb in the /models handler.
            hardware_info hw = detect_hardware();

            // Half of *free* VRAM, not classify_fit()'s 0.75-of-total "comfortable" ceiling - that
            // top end leaves too little headroom for the KV cache/context/compute buffers (and
            // anything else sharing the card, including another already-loaded model) even before
            // considering it was sized against the whole card instead of what's actually free.
            // --moe-stream-cache only parses integer GiB (or integer slot counts), so round down
            // to a whole GiB with a 1 GiB floor.
            uint64_t cache_gb = std::max<uint64_t>(1, (uint64_t) (hw.vram_free_gb * 0.5));

            // Full coverage: cache_gb (GiB) can hold every expert in the file, not just the
            // active subset, so generation never hits the disk-streaming path at all - it's
            // compute-bound and KV cache belongs in VRAM (default, fastest). Below that line
            // we're going to take disk-streaming misses regardless, so trade the KV cache's
            // VRAM for more --moe-stream-cache headroom instead (-nkvo): fewer misses is worth
            // more than fast attention when attention was never the bottleneck to begin with.
            // total_gb is the whole GGUF (dense + every expert, not just the active ones) - a
            // slight overestimate of "expert bytes alone" for the coverage check, but this file
            // is dominated by expert weight anyway (heavily sparse MoE), so it's close enough.
            bool full_coverage = total_gb > 0.0 && (double) cache_gb >= total_gb;

            std::string model_id = gguf_repo + ":" + quant;
            std::string cache_val = std::to_string(cache_gb) + "G";

            common_preset_write_ini_section(models_preset_path, model_id, {
                {"moe-stream-cache", cache_val},
                {"ctx-size", std::to_string(DEFAULT_CTX_SIZE)},
                {"no-kv-offload", full_coverage ? "false" : "true"},
            });

            res->status = 200;
            res->data = json{
                {"success", true},
                {"model_id", model_id},
                {"moe_stream_cache_gb", cache_gb},
                {"ctx_size", DEFAULT_CTX_SIZE},
                {"no_kv_offload", !full_coverage},
            }.dump();
        } catch (const std::exception & e) {
            res->status = 400;
            res->data = json{{"error", e.what()}}.dump();
        }

        return res;
    });
}
