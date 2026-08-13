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
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
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

// Ordered best (highest quality/bpw) to worst, so each model can be checked
// against progressively more aggressive quants until one actually fits.
const std::vector<std::pair<std::string, double>> QUANT_CANDIDATES = {
    {"Q8_0", 8.50}, {"Q6_K", 6.56}, {"Q5_K_M", 5.67}, {"Q5_K_S", 5.54},
    {"Q4_K_M", 4.83}, {"Q4_K_S", 4.58}, {"IQ4_XS", 4.25}, {"IQ3_M", 3.66},
};

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
    "unshackl", "unfilter", "unbound", "unrestrict", "heretic",
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

hardware_info detect_hardware() {
    hardware_info hw;

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev) {
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(cpu_dev, &free, &total);
        hw.ram_gb      = (double) total / 1e9;
        hw.ram_free_gb = (double) free  / 1e9;
    }

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

// run one HF model-search query, returning the first result that actually hosts GGUF files.
// checked via the "gguf" tag HF attaches to any repo containing .gguf files, not the repo name -
// plenty of repos (huihui-ai's abliterated models among them) host GGUF quants alongside
// safetensors in the same repo without "GGUF" anywhere in the name itself.
std::string gguf_search_query(const std::string & url) {
    common_remote_params params;
    params.timeout = 8;
    try {
        auto [http_code, body] = common_remote_get_content(url, params);
        if (http_code != 200 || body.empty()) return "";
        json results = json::parse(std::string(body.begin(), body.end()));
        for (const auto & r : results) {
            if (!r.contains("tags") || !r["tags"].is_array()) continue;
            for (const auto & t : r["tags"]) {
                if (t.is_string() && to_lower(t.get<std::string>()) == "gguf") {
                    return r.value("id", "");
                }
            }
        }
    } catch (const std::exception &) {
        // ignore, best-effort only
    }
    return "";
}

// best-effort search for an existing community GGUF quant, top-N only. falls back to huihui-ai's
// namespace specifically when nothing turns up under the model's own name: they're a prolific
// publisher of abliterated (uncensored) derivatives that host their own GGUF quants, covering a
// lot of ground a plain name search misses since the derivative has a different repo name entirely.
std::string gguf_search(const std::string & repo_id) {
    std::string base = repo_id;
    auto slash = base.find_last_of('/');
    if (slash != std::string::npos) base = base.substr(slash + 1);

    std::string found = gguf_search_query("https://huggingface.co/api/models?search=" + base + "%20GGUF&limit=15");
    if (!found.empty()) {
        return found;
    }

    return gguf_search_query("https://huggingface.co/api/models?search=" + base + "&author=huihui-ai&limit=5");
}

// fetch one model repo's HF tags - the signal that catches abliteration-tool-branded
// repos (e.g. "Heretic-...") a name-keyword match alone would miss, since the tool
// tags its own output even when the repo isn't named after it.
std::vector<std::string> fetch_hf_tags(const std::string & repo_id) {
    common_remote_params params;
    params.timeout = 6;
    std::vector<std::string> tags;
    try {
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
            // overridable like vram_gb/ram_gb above, mainly so the no-disk-space / too-big-for-
            // this-drive tiers are actually testable without needing a physically full disk
            double free_gb = req.get_param("disk_free_gb").empty()
                                  ? disk_free_gb(".") : std::stod(req.get_param("disk_free_gb"));
            double total_capacity_gb = req.get_param("disk_total_gb").empty()
                                  ? disk_total_gb(".") : std::stod(req.get_param("disk_total_gb"));

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
                    std::string tier = classify_fit(a_gb, t_gb, vram_gb, ram_gb, free_gb);
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

            // Split before the top-N cut. fit_order groups sort easy/comfortable (0) ahead of
            // ram-cache/disk-streaming (1-2) ahead of no-disk-space (3), and in real data the
            // easy/comfortable group alone routinely exceeds `top` on its own (e.g. 99 vs the
            // default top=30) - a single flat top-N truncation would then *always* cut off
            // before reaching a single ram-cache or no-disk-space row, no matter how large `top`
            // reasonably gets, even though the user needs to see both (spillover models to judge
            // performance, no-disk-space ones to judge what to delete). So: `top` only bounds the
            // ideal (easy/comfortable) group; every actionable degraded and no-disk-space
            // candidate is kept regardless.
            std::vector<model_row> ideal, degraded, no_space;
            for (auto & m : filtered) {
                if (m.fit_tier == "no-disk-space") {
                    no_space.push_back(std::move(m));
                } else if (m.fit_tier == "ram-cache" || m.fit_tier == "disk-streaming") {
                    degraded.push_back(std::move(m));
                } else {
                    ideal.push_back(std::move(m));
                }
            }

            // search a pool bigger than `top` (gguf_lookup), in parallel, before truncating - a
            // ranked candidate with no findable community GGUF quant isn't something the user can
            // click Download on, so it gets filtered out below rather than shown as a dead row;
            // searching only the eventual top-N would make the "no GGUF found" rows a lookup-limit
            // artifact instead of an honest "we checked, there isn't one". Run per-group (not one
            // flat pool) so the degraded/no-disk-space tails each get their own gguf_lookup budget
            // instead of being crowded out by whatever's ranked ahead of them in the ideal group.
            auto gguf_search_group = [gguf_lookup](std::vector<model_row> & group) {
                const int n = std::min((int) group.size(), gguf_lookup);
                std::vector<std::future<std::string>> futures;
                futures.reserve(n);
                for (int i = 0; i < n; i++) {
                    futures.push_back(std::async(std::launch::async, gguf_search, group[i].repo_id));
                }
                for (int i = 0; i < n; i++) {
                    group[i].gguf_repo = futures[i].get();
                }
                std::vector<model_row> actionable;
                for (int i = 0; i < n; i++) {
                    // beyond the searched pool, gguf_repo is unknown, not confirmed-absent - drop
                    // those too, same reasoning as an actual miss: nothing to click Download on
                    if (!group[i].gguf_repo.empty()) {
                        actionable.push_back(std::move(group[i]));
                    }
                }
                group = std::move(actionable);
            };
            gguf_search_group(ideal);
            gguf_search_group(degraded);
            gguf_search_group(no_space);

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
                    {"hf_arch", m.hf_arch}, {"fit_tier", m.fit_tier},
                    {"ugi_score", m.ugi_score}, {"willingness", m.willingness},
                    {"is_derestricted", m.is_derestricted},
                    {"gguf_repo", m.gguf_repo},
                });
            }
            out["models"] = arr;

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
            if (gguf_repo.empty() || quant.empty()) {
                throw std::invalid_argument("gguf_repo and quant are required");
            }

            hardware_info hw = detect_hardware();
            double vram_gb = body.contains("vram_gb") ? body.at("vram_gb").get<double>() : hw.vram_gb;

            // same VRAM headroom fraction as classify_fit()'s "comfortable" tier budget: leave
            // room for the KV cache/context/compute buffers alongside the resident expert cache.
            // --moe-stream-cache only parses integer GiB (or integer slot counts), so round down
            // to a whole GiB with a 1 GiB floor.
            uint64_t cache_gb = std::max<uint64_t>(1, (uint64_t) (vram_gb * 0.75));

            std::string model_id = gguf_repo + ":" + quant;
            std::string cache_val = std::to_string(cache_gb) + "G";

            common_preset_write_ini_section(models_preset_path, model_id, {
                {"moe-stream-cache", cache_val},
            });

            res->status = 200;
            res->data = json{
                {"success", true},
                {"model_id", model_id},
                {"moe_stream_cache_gb", cache_gb},
            }.dump();
        } catch (const std::exception & e) {
            res->status = 400;
            res->data = json{{"error", e.what()}}.dump();
        }

        return res;
    });
}
