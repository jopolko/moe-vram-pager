#pragma once

#include "server-http.h"

#include <string>

// Registers the model-picker routes (GET /model-picker/models, POST /model-picker/prepare-download)
// onto the given HTTP context.
//
// Deliberately self-contained: no dependency on server_context/the task queue/server_models, this
// isn't part of the inference path, it's a hardware-fit ranker over the UGI-Leaderboard. See
// scripts/model_picker.py for the reference implementation this is a native port of, and
// derestricted-filter.json / arch-map.json in the repo root for the two GitHub-hosted data files
// it fetches at runtime (with a small baked-in fallback for offline use).
//
// models_preset_path: path to the router's --models-preset INI file (see common/preset.h). Used
// only by POST /model-picker/prepare-download to stash a per-model --moe-stream-cache override
// before the router downloads/loads it; pass "" (single-model / non-router mode) to disable that
// route (it responds 404).
void server_model_picker_register_routes(const server_http_context & ctx_http, const std::string & models_preset_path);
