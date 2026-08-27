<script lang="ts">
	import {
		ExternalLink,
		ArrowUp,
		ArrowDown,
		ArrowUpDown,
		ChevronDown,
		Check,
		Download,
		Trash2,
		MessageSquare,
		Loader2,
		Play,
		Info,
		Square,
		RefreshCw,
		Power
	} from '@lucide/svelte';
	import { Switch } from '$lib/components/ui/switch';
	import { Label } from '$lib/components/ui/label';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import * as Table from '$lib/components/ui/table';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import DialogLoadGguf from '$lib/components/app/dialogs/DialogLoadGguf.svelte';
	import { routerModels } from '$lib/stores/models.svelte.ts';

	interface ModelRow {
		name: string;
		variant: string;
		repo_id: string;
		link: string;
		active_b: number;
		total_b: number;
		active_gb: number;
		total_gb: number;
		quant: string;
		hf_arch: string;
		fit_tier: string;
		ugi_score: number;
		willingness: number;
		is_derestricted: boolean;
		gguf_repo: string;
		n_ctx_train: number;
		kv_verified: boolean;
		ctx_options_gb: number[]; // parallel to CTX_SIZE_OPTIONS, real KV-cache GB per option; empty if unverified
	}

	interface HardwareInfo {
		vram_gb: number;
		ram_gb: number;
		vram_free_gb: number;
		ram_free_gb: number;
		disk_free_gb: number;
		disk_total_gb: number;
		cache_dir: string;
	}

	let models = $state<ModelRow[]>([]);
	let hardware = $state<HardwareInfo | null>(null);
	let loading = $state(true);
	let error = $state('');
	let derestrictedOnly = $state(false);
	let tagsProgress = $state<{ done: number; total: number } | null>(null);
	// Not $state - purely an internal handle for loadModels() to cancel its own previous in-flight
	// request, never read by the template.
	let loadModelsAbort: AbortController | null = null;

	// Router-mode download/load management. Deliberately independent of
	// $lib/stores/models.svelte.ts (the chat UI's own model switcher) - this
	// page only needs a thin read of /models + /models/sse, and touching the
	// shared store risks rippling into the chat flow. If /model-picker/models
	// reports routerAvailable=false (single-model mode, no --models-preset),
	// downloads are unsupported and the Actions column just shows the HF link.
	type DlPhase = 'downloading' | 'downloaded' | 'loading' | 'loaded' | 'unloaded' | 'failed';
	interface DlState {
		phase: DlPhase;
		doneBytes?: number;
		totalBytes?: number;
		etaSeconds?: number;
		reason?: string;
		loadPct?: number;
		loadStage?: string;
	}
	let routerAvailable = $state(false);
	let dlState = $state<Record<string, DlState>>({});
	let loadGgufOpen = $state(false);

	// Context size is a launch-time (not download-time) choice - it sizes the KV-cache buffer
	// for the *process* being spawned. bestCtxForRow() picks the largest option verified to fit
	// in current headroom and is what download/load use by default. ctxOverride lets a user pick
	// a different size per row after download - never written back to the preset (matches the
	// backend's own one-off /models/load ctx_size, see server-models.cpp's post_router_models_load
	// comment), so it only affects the *next* load and reverts to bestCtxForRow() on a router
	// restart. It never applies to an already-running instance - that's a launch-time param, so
	// changing it always means unload + reload, not a live update.
	let ctxOverride = $state<Record<string, number>>({});

	// DEFAULT_CTX is only the fallback for unverified rows and must match DEFAULT_CTX_SIZE in
	// server-model-picker.cpp.
	const DEFAULT_CTX = 8192;
	const CTX_SIZE_OPTIONS = [4096, 8192, 16384, 32768, 65536];

	// Matches RAM_OS_RESERVE_GB in server-model-picker.cpp - see that constant's comment. Applied
	// here too because bestCtxForRow's VRAM check alone isn't sufficient: a ctx size can clear
	// the VRAM headroom check yet still be the thing that pushes total host RAM usage over the
	// edge once it doesn't fit, since --moe-stream disables mmap (nothing is OS-evictable) and a
	// KV cache that doesn't fit VRAM falls back to host RAM same as any other llama.cpp load. On
	// a small-RAM box (e.g. an 18GB WSL2 .wslconfig cap) that's the difference between a normal
	// load and one that swaps a VHDX to the Windows host and drops to well under 1 token/sec.
	const RAM_SAFETY_RESERVE_GB = 2;

	// Largest context size verified to fit this model's real KV-cache size within current free
	// VRAM headroom AND leave a safe amount of system RAM (m.ctx_options_gb, computed
	// server-side by estimate_kv_cache_gb, doubles as the worst-case host-RAM fallback size -
	// see RAM_SAFETY_RESERVE_GB above). Falls back to DEFAULT_CTX when the row's KV
	// hyperparameters couldn't be verified (config.json fetch failed, or an architecture
	// parse_kv_hparams can't read cleanly) - unverifiable, not zero-cost.
	function bestCtxForRow(m: ModelRow): number {
		if (!hardware || !m.kv_verified || m.ctx_options_gb.length !== CTX_SIZE_OPTIONS.length) {
			return DEFAULT_CTX;
		}
		const vramHeadroomGb = Math.max(0, hardware.vram_free_gb - m.active_gb);
		const ramHeadroomGb = Math.max(0, hardware.ram_free_gb - RAM_SAFETY_RESERVE_GB);
		let best = DEFAULT_CTX;
		for (let i = 0; i < CTX_SIZE_OPTIONS.length; i++) {
			if (m.ctx_options_gb[i] <= vramHeadroomGb && m.ctx_options_gb[i] <= ramHeadroomGb) {
				best = CTX_SIZE_OPTIONS[i];
			}
		}
		return best;
	}

	// ctxOverride, if the user picked one for this row, else bestCtxForRow()'s pick. This is what
	// downloadModel()/loadRouterModel() actually send as ctx_size, and what the estimate column
	// (kvCacheGbForRow/estTotalGbForRow) reflects, so the displayed number always matches what a
	// Load click will actually request.
	function effectiveCtxForRow(m: ModelRow): number {
		const id = modelIdFor(m);
		return (id && ctxOverride[id]) || bestCtxForRow(m);
	}

	// Real KV-cache GB for whatever ctx size effectiveCtxForRow() actually picks for this row -
	// kept in sync so the estimate shown always matches the ctx size a Load click will actually
	// request. 0 (not "unknown") when unverified, same fallback semantics as ctx_options_gb
	// itself - see its declaration above.
	function kvCacheGbForRow(m: ModelRow): number {
		if (!m.kv_verified || m.ctx_options_gb.length !== CTX_SIZE_OPTIONS.length) {
			return 0;
		}
		const idx = CTX_SIZE_OPTIONS.indexOf(effectiveCtxForRow(m));
		return idx >= 0 ? m.ctx_options_gb[idx] : 0;
	}

	// Rough allowance for what neither active_gb nor the KV-cache estimate accounts for: ubatch/
	// compute scratch buffers and CUDA graph memory. Not modeled precisely (it depends on batch
	// size, architecture, and backend) - a flat pad keeps the headline total from undershooting
	// real usage, which is the whole point of showing a total instead of active_gb alone. Still
	// label the number as an estimate in the UI; this is a floor, not a promise.
	const COMPUTE_BUFFER_FUDGE_GB = 0.5;

	// Best-effort total VRAM a row will actually use once loaded: active weights + KV cache at
	// this row's default ctx size + compute buffer pad. This is what gets shown as the headline
	// number - active_gb alone (still available as a tooltip breakdown) was the smaller, more
	// precise-looking figure that didn't include KV cache or compute buffers, so it read as a
	// promise it wasn't meant to be.
	function estTotalGbForRow(m: ModelRow): number {
		return m.active_gb + kvCacheGbForRow(m) + COMPUTE_BUFFER_FUDGE_GB;
	}

	// Download speed, EMA-smoothed from successive download_progress events - plain object, not
	// $state, since it's write-only scratch state for computing etaSeconds, not itself rendered.
	const rateTracker: Record<string, { bytes: number; time: number; emaBps: number | null }> = {};

	// last time each id's dlState was set to 'downloading', from either an SSE progress event or
	// firing the download - lets refreshRouterModels() tell a real live download (recent SSE
	// traffic) apart from a stale local 'downloading' left behind by a missed terminal event
	// (dropped SSE connection, router restart mid-download) that a backend poll should override.
	const lastDownloadingAt: Record<string, number> = {};

	function formatEta(seconds: number): string {
		if (!Number.isFinite(seconds) || seconds < 0) return '';
		if (seconds < 60) return `${Math.ceil(seconds)}s left`;
		const mins = Math.round(seconds / 60);
		if (mins < 60) return `${mins}m left`;
		const hrs = Math.floor(mins / 60);
		return `${hrs}h ${mins % 60}m left`;
	}
	let loadedDropdownOpen = $state(false);
	let busyIds = $state<Set<string>>(new Set());

	function modelIdFor(m: ModelRow): string | null {
		return m.gguf_repo ? `${m.gguf_repo}:${m.quant}` : null;
	}

	let currentModelId = $derived(
		Object.entries(dlState).find(([, s]) => s.phase === 'loaded')?.[0] ?? null
	);

	let dlEntries = $derived(Object.entries(dlState).sort(([a], [b]) => a.localeCompare(b)));

	// id -> friendly alias, from the router's live /models response, so the "Loaded:"
	// dropdown can show names instead of raw local/url/ollama paths (dlState only has id/phase)
	let dlAliases = $derived(
		Object.fromEntries(
			routerModels()
				.filter((m) => m.aliases?.[0])
				.map((m) => [m.id, m.aliases[0]])
		) as Record<string, string>
	);

	async function refreshRouterModels() {
		if (!routerAvailable) return;
		try {
			const resp = await fetch('./models');
			if (!resp.ok) return;
			const data = await resp.json();
			const next: Record<string, DlState> = { ...dlState };
			for (const row of data.data ?? []) {
				const status = row?.status?.value as string | undefined;
				if (!status) continue;
				// don't clobber a live "downloading" with a stale unloaded row from
				// before the SSE feed folded the finished download in - but only while SSE
				// traffic for it is actually recent; if the last progress event was more than
				// 10s ago (missed/never-arrived terminal event, dropped SSE connection, router
				// restart mid-download) trust the backend's authoritative status instead
				const downloadingRecently =
					next[row.id]?.phase === 'downloading' &&
					Date.now() - (lastDownloadingAt[row.id] ?? 0) < 10_000;
				if (downloadingRecently && status === 'unloaded') continue;
				next[row.id] = { phase: status as DlPhase };
			}
			dlState = next;
		} catch {
			// best-effort; the SSE feed is the real source of truth once connected
		}
	}

	function sumProgress(progress: Record<string, { done: number; total: number }>) {
		let done = 0;
		let total = 0;
		for (const p of Object.values(progress)) {
			done += p.done ?? 0;
			total += p.total ?? 0;
		}
		return { done, total };
	}

	$effect(() => {
		if (!routerAvailable) return;
		let es: EventSource | null = null;
		try {
			es = new EventSource('./models/sse');
			// (re)connect - including the browser's automatic reconnect after a dropped
			// connection (router restart, network blip, tab wake from background) - means we
			// may have missed events while disconnected, so resync from the backend
			es.onopen = () => void refreshRouterModels();
			es.onmessage = (ev) => {
				let envelope: { model?: string; event?: string; data?: Record<string, unknown> };
				try {
					envelope = JSON.parse(ev.data);
				} catch {
					return;
				}
				const id = envelope.model;
				if (!id) return;

				if (envelope.event === 'status_change') {
					const status = envelope.data?.status as string | undefined;
					// real tensor-loading progress from the child's GGML load_progress_callback,
					// piped through as-is - not just a spinner, this is actual bytes-read progress
					const progress = envelope.data?.progress as
						| { value?: number; current?: string }
						| undefined;
					if (status) {
						dlState = {
							...dlState,
							[id]: {
								phase: status as DlPhase,
								loadPct:
									status === 'loading' && typeof progress?.value === 'number'
										? Math.round(progress.value * 100)
										: undefined,
								loadStage: status === 'loading' ? progress?.current : undefined
							}
						};
						// the Loaded dropdown otherwise stays open through the whole load - only a
						// user gesture (click-away, select) closes a bits-ui dropdown by default, a
						// background state change like "finished loading" doesn't trigger that
						if (status === 'loaded') {
							loadedDropdownOpen = false;
						}
					}
				} else if (envelope.event === 'download_progress') {
					const progress = envelope.data?.progress as
						| Record<string, { done: number; total: number }>
						| undefined;
					const { done, total } = sumProgress(progress ?? {});

					const now = Date.now();
					const prev = rateTracker[id];
					let emaBps = prev?.emaBps ?? null;
					if (prev && now > prev.time && done >= prev.bytes) {
						const instantBps = ((done - prev.bytes) / (now - prev.time)) * 1000;
						emaBps = emaBps === null ? instantBps : emaBps * 0.7 + instantBps * 0.3;
					}
					rateTracker[id] = { bytes: done, time: now, emaBps };
					lastDownloadingAt[id] = now;

					const remaining = total > done ? total - done : 0;
					const etaSeconds = emaBps && emaBps > 0 ? remaining / emaBps : undefined;

					dlState = {
						...dlState,
						[id]: { phase: 'downloading', doneBytes: done, totalBytes: total, etaSeconds }
					};
				} else if (envelope.event === 'download_finished') {
					delete rateTracker[id];
					dlState = { ...dlState, [id]: { phase: 'downloaded' } };
					void refreshRouterModels();
				} else if (envelope.event === 'download_failed') {
					delete rateTracker[id];
					const reason = envelope.data?.reason as string | undefined;
					dlState = { ...dlState, [id]: { phase: 'failed', reason } };
				} else if (envelope.event === 'model_remove') {
					delete rateTracker[id];
					const next = { ...dlState };
					delete next[id];
					dlState = next;
				}
			};
		} catch {
			// router mode not available; the Actions column falls back silently
		}
		void refreshRouterModels();
		// Fallback reconciliation poll: an EventSource can go silently dead (WSL2/proxy
		// idle timeout, tab backgrounded, etc.) without the browser ever firing onerror/
		// onopen to trigger a resync - so a dropped terminal event (download_finished,
		// status_change) can otherwise leave a row stuck showing stale progress forever.
		// This poll is the self-healing backstop; the SSE feed stays the fast path.
		const reconcileTimer = setInterval(() => void refreshRouterModels(), 5000);
		return () => {
			es?.close();
			clearInterval(reconcileTimer);
		};
	});

	async function downloadModel(m: ModelRow) {
		const id = modelIdFor(m);
		if (!id || busyIds.has(id)) return;
		busyIds = new Set(busyIds).add(id);
		try {
			// Re-check free space right before firing the download - the table's fit_tier was
			// computed at page-load time and can go stale (another download finished since,
			// disk filled up elsewhere, tab left open a while) by the time the user clicks.
			const freeResp = await fetch('./model-picker/disk-free');
			if (freeResp.ok) {
				const { disk_free_gb } = await freeResp.json();
				if (typeof disk_free_gb === 'number' && m.total_gb > disk_free_gb) {
					dlState = {
						...dlState,
						[id]: {
							phase: 'failed',
							reason: `Not enough disk space: needs ${m.total_gb.toFixed(1)} GB, only ${disk_free_gb.toFixed(1)} GB free`
						}
					};
					return;
				}
			}
			await fetch('./model-picker/prepare-download', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					gguf_repo: m.gguf_repo,
					repo_id: m.repo_id,
					quant: m.quant,
					active_gb: m.active_gb,
					ctx_size: bestCtxForRow(m)
				})
			});
			const resp = await fetch('./models', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ model: id })
			});
			if (resp.ok) {
				lastDownloadingAt[id] = Date.now();
				dlState = { ...dlState, [id]: { phase: 'downloading', doneBytes: 0, totalBytes: 0 } };
			} else {
				const body = await resp.json().catch(() => ({}));
				// the router's error body is {error: {message, type, code}} (server-models.cpp's
				// res_err), not a plain string like the model-picker endpoints below use - so
				// body.error itself isn't a valid Error message, only body.error.message is.
				const message = body.error?.message || body.error || `Request failed (${resp.status})`;
				throw new Error(message);
			}
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			const next = new Set(busyIds);
			next.delete(id);
			busyIds = next;
		}
	}

	async function loadRouterModel(id: string) {
		if (busyIds.has(id)) return;
		busyIds = new Set(busyIds).add(id);
		try {
			// One-off for this load only - not persisted, so the next load (or a router
			// restart) reverts to whatever's actually saved in the preset from download time,
			// unless the user picked a ctxOverride for this row (see effectiveCtxForRow).
			// Looked up by id (not passed in) since this is called both from the main table
			// (where the row is right there) and the "Loaded:" dropdown (which only has id/state).
			const row = models.find((mm) => modelIdFor(mm) === id);
			const ctxSize = row ? effectiveCtxForRow(row) : ctxOverride[id] || DEFAULT_CTX;
			await fetch('./models/load', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ model: id, ctx_size: ctxSize })
			});
		} finally {
			const next = new Set(busyIds);
			next.delete(id);
			busyIds = next;
		}
	}

	async function unloadRouterModel(id: string) {
		if (busyIds.has(id)) return;
		busyIds = new Set(busyIds).add(id);
		try {
			const resp = await fetch('./models/unload', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ model: id })
			});
			if (!resp.ok) {
				const body = await resp.json().catch(() => ({}));
				throw new Error(body.error?.message || body.error || `Request failed (${resp.status})`);
			}
			// optimistic - the SSE status_change event normally does this, but don't leave the
			// button showing "loaded" for the round trip if that event is slow/dropped
			dlState = { ...dlState, [id]: { phase: 'unloaded' } };
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			const next = new Set(busyIds);
			next.delete(id);
			busyIds = next;
		}
	}

	async function deleteRouterModel(id: string) {
		if (busyIds.has(id)) return;
		const isLoaded = dlState[id]?.phase === 'loaded';
		const isDownloading = dlState[id]?.phase === 'downloading';
		const message = isDownloading
			? `Stop the download for ${id}? Any partial data will be removed; you can restart it from scratch later.`
			: isLoaded
				? `Delete downloaded files for ${id}? It's currently loaded, so it will be stopped first. This frees disk space; re-downloading later works the same as the first time.`
				: `Delete downloaded files for ${id}? This frees disk space; re-downloading later works the same as the first time.`;
		if (!confirm(message)) {
			return;
		}
		busyIds = new Set(busyIds).add(id);
		try {
			const resp = await fetch(`./models?model=${encodeURIComponent(id)}`, { method: 'DELETE' });
			if (!resp.ok) {
				const body = await resp.json().catch(() => ({}));
				throw new Error(body.error?.message || body.error || `Request failed (${resp.status})`);
			}
			const next = { ...dlState };
			delete next[id];
			dlState = next;
			await loadModels();
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			const nextBusy = new Set(busyIds);
			nextBusy.delete(id);
			busyIds = nextBusy;
		}
	}

	// Server-side ranking is fixed to UGI (highest-scored first within each hardware-fit
	// group) - it decides which models are worth spending GGUF-search budget on before
	// the top-N cut. Re-ordering the resulting page by size or willingness instead is a
	// client-side concern now, already covered by the sortable column headers below.
	const RANK_BY = 'ugi';

	async function pollTagsProgress() {
		try {
			const resp = await fetch('./model-picker/tags-progress');
			if (!resp.ok) return;
			const data = await resp.json();
			tagsProgress = data.total > 0 ? { done: data.done, total: data.total } : null;
		} catch {
			// best-effort only; the main request is the source of truth
		}
	}

	async function loadModels(forceRefresh = false) {
		// Cancel any still-in-flight request from a previous call (e.g. the user toggled the
		// derestricted filter again before the last load finished) - without this, whichever
		// request happens to resolve last wins, which can silently overwrite fresh results with a
		// stale (or by-then-irrelevant) response instead of the current toggle state's data.
		loadModelsAbort?.abort();
		const controller = new AbortController();
		loadModelsAbort = controller;

		loading = true;
		error = '';
		tagsProgress = null;
		// Only relevant on a cold cache (first-ever load, or once every ~30 days when HF
		// tags are re-checked) - most loads finish before this ever ticks.
		const progressTimer = setInterval(pollTagsProgress, 300);
		try {
			const params = new URLSearchParams({
				top: '30',
				// Each lookup now does real per-file HF verification (up to a few tree-API round
				// trips), not just one cheap search call, and not every lookup resolves to an
				// actionable real match - budget needs headroom above the ~25 final results wanted
				// per tier. Safe to keep generous since server-model-picker.cpp's hf_request_limiter
				// caps actual concurrent HF requests process-wide regardless of this number; a
				// bigger budget just means a longer queue through that cap, not a bigger burst.
				gguf_lookup: '50',
				derestricted_only: String(derestrictedOnly),
				rank_by: RANK_BY,
				...(forceRefresh ? { refresh: 'true' } : {})
			});

			const resp = await fetch(`./model-picker/models?${params.toString()}`, {
				signal: controller.signal
			});
			if (!resp.ok) {
				const body = await resp.json().catch(() => ({}));
				throw new Error(body.error || `Request failed (${resp.status})`);
			}
			const data = await resp.json();
			hardware = data.hardware;
			models = data.models;
			routerAvailable = Boolean(data.router_available);
		} catch (e) {
			// superseded by a newer loadModels() call, not a real failure - the newer call already
			// owns loading/error state, so leave it alone here.
			if (e instanceof DOMException && e.name === 'AbortError') return;
			error = e instanceof Error ? e.message : String(e);
		} finally {
			// progressTimer is per-call and must always be cleared regardless of which request
			// "wins" - only the shared loading/tagsProgress state is guarded, so a superseded call
			// doesn't stomp on the newer call's still-in-progress state.
			clearInterval(progressTimer);
			if (loadModelsAbort === controller) {
				tagsProgress = null;
				loading = false;
			}
		}
	}

	// $effect fires once on mount and again whenever derestrictedOnly changes, no separate
	// onMount needed (that caused a duplicate concurrent fetch).
	$effect(() => {
		// eslint-disable-next-line @typescript-eslint/no-unused-expressions
		derestrictedOnly;
		loadModels();
	});

	type SortKey = 'name' | 'quant' | 'active' | 'total' | 'ugi' | 'willingness' | 'gguf';

	// Best (highest quality) to worst, mirrors QUANT_CANDIDATES in
	// tools/server/server-model-picker.cpp so "sort by quant" reads best-first.
	const QUANT_ORDER: Record<string, number> = {
		Q8_0: 0,
		Q6_K: 1,
		Q5_K_M: 2,
		Q5_K_S: 3,
		Q4_K_M: 4,
		Q4_K_S: 5,
		IQ4_XS: 6,
		IQ3_M: 7
	};

	// numeric/quality columns default to descending (best or biggest first),
	// name/quant/gguf default to ascending (A-Z / best quant first)
	const DEFAULT_SORT_DIR: Record<SortKey, 1 | -1> = {
		name: 1,
		quant: 1,
		active: 1,
		total: 1,
		ugi: -1,
		willingness: -1,
		gguf: 1
	};

	let sortKey = $state<SortKey | null>(null);
	let sortDir = $state<1 | -1>(1);

	function sortBy(key: SortKey) {
		if (sortKey === key) {
			sortDir = sortDir === 1 ? -1 : 1;
		} else {
			sortKey = key;
			sortDir = DEFAULT_SORT_DIR[key];
		}
	}

	let sortedModels = $derived.by(() => {
		if (!sortKey) return models;
		const key = sortKey;
		const dir = sortDir;
		return [...models].sort((a, b) => {
			let cmp = 0;
			switch (key) {
				case 'name':
					cmp = a.name.localeCompare(b.name);
					break;
				case 'quant':
					cmp = (QUANT_ORDER[a.quant] ?? 9) - (QUANT_ORDER[b.quant] ?? 9);
					break;
				case 'active':
					cmp = estTotalGbForRow(a) - estTotalGbForRow(b);
					break;
				case 'total':
					cmp = a.total_gb - b.total_gb;
					break;
				case 'ugi':
					cmp = a.ugi_score - b.ugi_score;
					break;
				case 'willingness':
					cmp = a.willingness - b.willingness;
					break;
				case 'gguf':
					cmp = a.gguf_repo.localeCompare(b.gguf_repo);
					break;
			}
			return cmp * dir;
		});
	});

	// Only names that actually collide in the *currently shown* rows need the
	// variant hint - a model whose reasoning-mode sibling got filtered/ranked
	// out of view isn't a visible conflict, so showing the dot there would
	// just be unexplained clutter with nothing on screen to point at.
	let duplicateNames = $derived.by(() => {
		const counts = new Map<string, number>();
		for (const m of sortedModels) counts.set(m.name, (counts.get(m.name) ?? 0) + 1);
		const dupes = new Set<string>();
		for (const [name, count] of counts) {
			if (count > 1) dupes.add(name);
		}
		return dupes;
	});
</script>

<div class="mx-auto flex h-dvh w-full max-w-5xl flex-col p-4 md:p-8">
	<div class="mb-6 shrink-0 text-center">
		<h1 class="text-2xl font-bold tracking-tight">MoE VRAM Pager</h1>
		<p class="text-base font-bold italic text-foreground">Run MoE models that don't fit in your VRAM.</p>
	</div>

	{#snippet hwCard(label: string, total: number, free: number, decimals: number, totalTitle?: string, freeTitle?: string)}
		<div class="flex items-center gap-3 rounded-lg border px-4 py-3 text-sm whitespace-nowrap">
			<span class="text-base font-bold">{label}</span>
			<span class="text-muted-foreground" title={totalTitle}>Total</span>
			<span class="font-bold">{total.toFixed(decimals)} GB</span>
			<span class="text-muted-foreground" title={freeTitle}>Free</span>
			<span class="font-bold">{free.toFixed(decimals)} GB</span>
		</div>
	{/snippet}

	{#if hardware}
		<div class="mb-4 grid shrink-0 grid-cols-1 gap-3 sm:grid-cols-3">
			{@render hwCard(
				'VRAM',
				hardware.vram_gb,
				hardware.vram_free_gb,
				1,
				'Shown for reference only - fit ranking is budgeted against Free, not this',
				"What fit ranking is actually budgeted against. Close other GPU apps and this goes up - unlike disk space, VRAM is never overcommitted so this is a real live number"
			)}
			{@render hwCard(
				'RAM',
				hardware.ram_gb,
				hardware.ram_free_gb,
				1,
				'Shown for reference only - not part of fit ranking',
				"Informational only. The full model always lives on SSD and only the active experts page into VRAM, so RAM isn't part of the fit budget. On Linux/WSL Free always equals Total here - ggml treats free RAM as ill-defined and just assumes it's all available, rather than fighting the reclaimable-page-cache accounting mess. Real on Windows builds."
			)}
			<div class="flex items-center gap-3 rounded-lg border px-4 py-3 text-sm whitespace-nowrap">
				<span class="flex items-center gap-1 text-base font-bold">
					Storage
					{#if hardware.cache_dir}
						<span
							class="text-muted-foreground hover:text-foreground"
							title="Downloaded models are stored in the Hugging Face cache, not a Downloads folder: {hardware.cache_dir}"
						>
							<Info class="h-3.5 w-3.5" />
						</span>
					{/if}
				</span>
				<span class="text-muted-foreground">Total</span>
				<span class="font-bold">{hardware.disk_total_gb.toFixed(0)} GB</span>
				<span class="text-muted-foreground">Free</span>
				<span class="font-bold">{hardware.disk_free_gb.toFixed(0)} GB</span>
			</div>
		</div>
	{/if}

	<div class="mb-4 flex shrink-0 flex-wrap items-center gap-6">
		<div class="flex items-center gap-2">
			<Switch id="derestricted-only" bind:checked={derestrictedOnly} />
			<Label
				class="text-sm"
				title="Only show models that are specifically known derestricted/abliterated finetunes - a curated filter, not a score"
			>
				Derestricted finetunes only
			</Label>
		</div>
		<Button
			size="sm"
			variant="outline"
			disabled={loading}
			onclick={() => loadModels(true)}
			title="The list below is cached for up to 6 hours so a page refresh stays instant - use this to pull the latest UGI-leaderboard data and re-check for new GGUF quants right now"
		>
			<RefreshCw class="h-3.5 w-3.5 {loading ? 'animate-spin' : ''}" />
			Check for new models
		</Button>
		{#if routerAvailable}
			<Button size="sm" variant="outline" onclick={() => (loadGgufOpen = true)}>
				<Download class="h-3.5 w-3.5" />
				Load GGUF
			</Button>
			<DropdownMenu.Root bind:open={loadedDropdownOpen}>
				<DropdownMenu.Trigger>
					{#snippet child({ props })}
						<button
							{...props}
							type="button"
							class="-mx-1 inline-flex max-w-[20rem] items-center gap-1 rounded-sm px-1 hover:bg-muted-foreground/10"
						>
							<span class="text-muted-foreground">Loaded:</span>
							<span class="min-w-0 truncate"
								>{currentModelId ? (dlAliases[currentModelId] ?? currentModelId) : 'None'}</span
							>
							<ChevronDown class="h-3 w-3 shrink-0 text-muted-foreground" />
						</button>
					{/snippet}
				</DropdownMenu.Trigger>
				<DropdownMenu.Content align="start" class="w-80">
					{#if dlEntries.length === 0}
						<div class="px-2 py-3 text-center text-xs text-muted-foreground">
							No downloaded models yet
						</div>
					{:else}
						{#each dlEntries as [id, state] (id)}
							{@const busy = busyIds.has(id)}
							<div class="flex items-center gap-1">
								<DropdownMenu.Item
									class="min-w-0 flex-1"
									disabled={busy ||
										state.phase === 'loaded' ||
										state.phase === 'loading' ||
										state.phase === 'downloading' ||
										state.phase === 'downloaded'}
									onclick={() => loadRouterModel(id)}
								>
									{#if state.phase === 'loaded'}
										<Check class="h-3.5 w-3.5 text-primary" />
									{:else if state.phase === 'loading' || state.phase === 'downloading' || state.phase === 'downloaded'}
										<Loader2 class="h-3.5 w-3.5 animate-spin" />
									{/if}
									<span class="min-w-0 truncate" title={id}>{dlAliases[id] ?? id}</span>
								</DropdownMenu.Item>
								{#if state.phase === 'loaded'}
									<button
										type="button"
										class="shrink-0 rounded-sm p-1.5 text-muted-foreground hover:bg-muted-foreground/10 hover:text-foreground disabled:pointer-events-none disabled:opacity-50"
										title="Unload (frees VRAM/RAM, keeps the downloaded files)"
										disabled={busy}
										onclick={(e) => {
											e.stopPropagation();
											unloadRouterModel(id);
										}}
									>
										<Power class="h-3.5 w-3.5" />
									</button>
								{/if}
								{#if state.phase === 'unloaded' || state.phase === 'failed' || state.phase === 'loaded'}
									<button
										type="button"
										class="shrink-0 rounded-sm p-1.5 text-destructive hover:bg-destructive/10 hover:text-destructive disabled:pointer-events-none disabled:opacity-50"
										title="Delete downloaded files"
										disabled={busy}
										onclick={(e) => {
											e.stopPropagation();
											deleteRouterModel(id);
										}}
									>
										<Trash2 class="h-3.5 w-3.5" />
									</button>
								{:else if state.phase === 'downloading'}
									<button
										type="button"
										class="shrink-0 rounded-sm p-1.5 text-destructive hover:bg-destructive/10 hover:text-destructive disabled:pointer-events-none disabled:opacity-50"
										title="Stop download"
										disabled={busy}
										onclick={(e) => {
											e.stopPropagation();
											deleteRouterModel(id);
										}}
									>
										<Square class="h-3.5 w-3.5" />
									</button>
								{/if}
							</div>
						{/each}
					{/if}
				</DropdownMenu.Content>
			</DropdownMenu.Root>
		{/if}
	</div>

	<DialogLoadGguf bind:open={loadGgufOpen} onLoaded={() => void refreshRouterModels()} />

	{#if error}
		<div class="shrink-0 rounded-lg border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">
			{error}
		</div>
	{:else if loading && models.length === 0}
		<div class="shrink-0 p-8 text-center text-sm text-muted-foreground">
			{#if tagsProgress}
				<div class="mx-auto w-64">
					<div class="mb-1">
						Checking uncensored-finetune tags... {tagsProgress.done}/{tagsProgress.total}
					</div>
					<div class="h-1.5 w-full overflow-hidden rounded-full bg-muted">
						<div
							class="h-full rounded-full bg-primary transition-all"
							style="width: {Math.round((100 * tagsProgress.done) / tagsProgress.total)}%"
						></div>
					</div>
					<div class="mt-1 text-xs opacity-70">Initial load only - cached for next time</div>
				</div>
			{:else}
				Loading...
			{/if}
		</div>
	{:else}
		{#snippet sortIcon(key: SortKey)}
			{#if sortKey === key}
				{#if sortDir === 1}
					<ArrowUp class="h-3 w-3" />
				{:else}
					<ArrowDown class="h-3 w-3" />
				{/if}
			{:else}
				<ArrowUpDown class="h-3 w-3 opacity-40" />
			{/if}
		{/snippet}

		<div
			class="min-h-0 flex-1 overflow-hidden rounded-lg border [&>div]:h-full [&>div]:overflow-y-auto"
		>
			<Table.Root>
				<Table.Header>
					<Table.Row class="hover:[&>th]:bg-background!">
						<Table.Head class="sticky top-0 z-10 bg-background will-change-transform [transform:translateZ(0)] [backface-visibility:hidden]">
							<button
								type="button"
								onclick={() => sortBy('name')}
								class="inline-flex items-center gap-1 hover:text-foreground"
							>
								Model {@render sortIcon('name')}
							</button>
						</Table.Head>
						<Table.Head class="sticky top-0 z-10 bg-background will-change-transform [transform:translateZ(0)] [backface-visibility:hidden]">
							<button
								type="button"
								onclick={() => sortBy('quant')}
								class="inline-flex items-center gap-1 hover:text-foreground"
							>
								Quant {@render sortIcon('quant')}
							</button>
						</Table.Head>
						<Table.Head class="sticky top-0 z-10 bg-background will-change-transform [transform:translateZ(0)] [backface-visibility:hidden] text-right">
							<button
								type="button"
								title="Estimated total VRAM once loaded: active weights + KV cache at this model's default context size + a compute-buffer allowance. An estimate, not a guarantee - actual usage depends on backend and batch size."
								onclick={() => sortBy('active')}
								class="inline-flex w-full items-center justify-end gap-1 hover:text-foreground"
							>
								VRAM {@render sortIcon('active')}
							</button>
						</Table.Head>
						<Table.Head class="sticky top-0 z-10 bg-background will-change-transform [transform:translateZ(0)] [backface-visibility:hidden] text-right">
							<button
								type="button"
								title="Total params (B) x bits-per-weight at this quant / 8 = GB. The full model on disk, since --moe-stream streams the inactive experts in on demand."
								onclick={() => sortBy('total')}
								class="inline-flex w-full items-center justify-end gap-1 hover:text-foreground"
							>
								SSD {@render sortIcon('total')}
							</button>
						</Table.Head>
						<Table.Head class="sticky top-0 z-10 bg-background will-change-transform [transform:translateZ(0)] [backface-visibility:hidden] text-right">
							<button
								type="button"
								title="UGI benchmark score: how much knowledge/reasoning the model demonstrates on sensitive topics without refusing - not a general-purpose benchmark like MMLU"
								onclick={() => sortBy('ugi')}
								class="inline-flex w-full items-center justify-end gap-1 hover:text-foreground"
							>
								UGI {@render sortIcon('ugi')}
							</button>
						</Table.Head>
						<Table.Head class="sticky top-0 z-10 bg-background will-change-transform [transform:translateZ(0)] [backface-visibility:hidden] text-center">
							<button
								type="button"
								title="Willingness"
								onclick={() => sortBy('willingness')}
								class="inline-flex w-full items-center justify-center gap-1 hover:text-foreground"
							>
								Will. {@render sortIcon('willingness')}
							</button>
						</Table.Head>
						<Table.Head class="sticky top-0 z-10 bg-background will-change-transform [transform:translateZ(0)] [backface-visibility:hidden] text-center">
							<button
								type="button"
								title="Community GGUF repo this model would download from"
								onclick={() => sortBy('gguf')}
								class="inline-flex w-full items-center justify-center gap-1 hover:text-foreground"
							>
								GGUF {@render sortIcon('gguf')}
							</button>
						</Table.Head>
						{#if routerAvailable}
							<Table.Head class="sticky top-0 z-10 bg-background will-change-transform [transform:translateZ(0)] [backface-visibility:hidden]">Actions</Table.Head>
						{/if}
					</Table.Row>
				</Table.Header>
				<Table.Body>
					{#each sortedModels as m, i (m.name + i)}
						<Table.Row class={i % 2 === 1 ? 'bg-muted/30' : ''}>
							<Table.Cell class="max-w-xs whitespace-normal break-words">
								<a
									href={m.link}
									target="_blank"
									rel="noopener noreferrer"
									class="font-medium hover:underline"
									title={m.variant && duplicateNames.has(m.name)
										? `Separate UGI leaderboard entry (${m.variant}) - same model, different score for this configuration`
										: undefined}
								>
									{m.name}
								</a>
								{#if m.variant && duplicateNames.has(m.name)}
									<sup
										class="cursor-help text-sm leading-none text-muted-foreground"
										title={`Separate UGI leaderboard entry (${m.variant}) - same model, different score for this configuration`}
									>
										●
									</sup>
								{/if}
								{#if m.is_derestricted && !derestrictedOnly}
									<Badge
										variant="outline"
										class="ml-2 align-middle border-violet-300 bg-violet-50 text-violet-700 dark:border-violet-800 dark:bg-violet-950/40 dark:text-violet-300"
									>
										Derestricted
									</Badge>
								{/if}
								{#if m.fit_tier === 'no-disk-space' && hardware}
									<Badge
										variant="destructive"
										class="ml-2 align-middle"
										title="Fits on this drive, but not in the space free right now - free up {Math.max(0, m.total_gb - hardware.disk_free_gb).toFixed(1)} GB to download it"
									>
										Storage +{Math.max(0, m.total_gb - hardware.disk_free_gb).toFixed(0)}GB
									</Badge>
								{/if}
							</Table.Cell>
							<Table.Cell class="text-xs text-muted-foreground">{m.quant}</Table.Cell>
							<Table.Cell
								class="text-right tabular-nums"
								title="Active weights {m.active_gb.toFixed(1)} GB + KV cache {kvCacheGbForRow(m).toFixed(1)} GB (at {effectiveCtxForRow(m).toLocaleString()} ctx) + ~{COMPUTE_BUFFER_FUDGE_GB} GB compute buffer"
							>
								{estTotalGbForRow(m).toFixed(1)} GB
							</Table.Cell>
							<Table.Cell class="text-right tabular-nums">{m.total_gb.toFixed(1)} GB</Table.Cell>
							<Table.Cell class="text-right tabular-nums">{m.ugi_score.toFixed(1)}</Table.Cell>
							<Table.Cell class="text-center tabular-nums">{m.willingness.toFixed(1)}</Table.Cell>
							<Table.Cell class="text-center">
								{#if m.gguf_repo}
									<a
										href="https://huggingface.co/{m.gguf_repo}"
										target="_blank"
										rel="noopener noreferrer"
										title={m.gguf_repo}
										class="inline-flex text-teal-600 hover:text-teal-700 dark:text-teal-400 dark:hover:text-teal-300"
									>
										<ExternalLink class="h-3.5 w-3.5" />
									</a>
								{:else}
									<span class="text-xs text-muted-foreground">-</span>
								{/if}
							</Table.Cell>
							{#if routerAvailable}
								{@const id = modelIdFor(m)}
								{@const state = id ? dlState[id] : undefined}
								{@const busy = id ? busyIds.has(id) : false}
								<Table.Cell class="min-w-[7rem]">
									{#if !id}
										<span class="text-xs text-muted-foreground">no GGUF found</span>
									{:else}
										<div class="flex flex-col gap-1">
											<div class="flex items-center gap-1">
												{#if !state || state.phase === 'failed'}
													<Button
														size="icon-sm"
														variant="outline"
														disabled={busy}
														onclick={() => downloadModel(m)}
														title={state?.phase === 'failed' ? 'Retry download' : 'Download'}
													>
														<Download class="h-3.5 w-3.5 text-blue-600 dark:text-blue-400" />
													</Button>
												{:else if state.phase === 'downloading'}
													<Button size="icon-sm" variant="outline" disabled title="Downloading">
														<Loader2 class="h-3.5 w-3.5 animate-spin" />
													</Button>
													<Button
														size="icon-sm"
														variant="ghost"
														disabled={busy}
														onclick={() => deleteRouterModel(id)}
														title="Stop download"
													>
														<Square class="h-3.5 w-3.5 text-destructive" />
													</Button>
												{:else if state.phase === 'downloaded'}
													<Button size="icon-sm" variant="outline" disabled title="Downloading">
														<Loader2 class="h-3.5 w-3.5 animate-spin" />
													</Button>
												{:else if state.phase === 'loading'}
													<Button
							size="icon-sm"
							variant="outline"
							disabled
							title={state.loadPct !== undefined ? `Loading (${state.loadPct}%)` : 'Loading'}
						>
														<Loader2 class="h-3.5 w-3.5 animate-spin" />
													</Button>
												{:else if state.phase === 'unloaded'}
													<Button
														size="icon-sm"
														variant="outline"
														disabled={busy}
														onclick={() => downloadModel(m)}
														title="Re-download"
													>
														<Download class="h-3.5 w-3.5 text-blue-600 dark:text-blue-400" />
													</Button>
													<Button
														size="icon-sm"
														variant="default"
														disabled={busy}
														onclick={() => loadRouterModel(id)}
														title="Load"
													>
														<Play class="h-3.5 w-3.5" />
													</Button>
													<Button
														size="icon-sm"
														variant="ghost"
														disabled={busy}
														onclick={() => deleteRouterModel(id)}
														title="Delete downloaded files"
													>
														<Trash2 class="h-3.5 w-3.5" />
													</Button>
												{:else if state.phase === 'loaded'}
													<Button size="icon-sm" variant="default" href="../?model={encodeURIComponent(id)}" title="Chat">
														<MessageSquare class="h-3.5 w-3.5" />
													</Button>
													<Button
														size="icon-sm"
														variant="outline"
														disabled={busy}
														onclick={() => unloadRouterModel(id)}
														title="Unload (frees VRAM/RAM, keeps the downloaded files)"
													>
														<Power class="h-3.5 w-3.5" />
													</Button>
												{/if}
											</div>
											{#if state?.phase === 'downloading'}
												{@const pct = state.totalBytes && state.totalBytes > 0 ? Math.min(100, Math.round((100 * (state.doneBytes ?? 0)) / state.totalBytes)) : null}
												<div class="w-24">
													<div class="h-1 w-full overflow-hidden rounded-full bg-muted">
														<div class="h-full rounded-full bg-primary transition-all" style="width: {pct ?? 8}%"></div>
													</div>
													<span class="text-[10px] text-muted-foreground">
														{pct !== null ? `${pct}%` : '...'}{state.etaSeconds !== undefined
															? ` · ${formatEta(state.etaSeconds)}`
															: ''}
													</span>
												</div>
											{:else if state?.phase === 'loading'}
												<div class="w-24">
													<div class="h-1 w-full overflow-hidden rounded-full bg-muted">
														<div
															class="h-full rounded-full bg-primary transition-all"
															style="width: {state.loadPct ?? 8}%"
														></div>
													</div>
													<span class="text-[10px] text-muted-foreground"
														>{state.loadPct !== undefined ? `${state.loadPct}%` : '...'}</span
													>
												</div>
											{:else if state?.phase === 'failed' && state.reason}
												<span class="block w-28 text-[10px] whitespace-normal break-words text-destructive" title={state.reason}>
													{state.reason}
												</span>
											{:else if state?.phase === 'unloaded' || state?.phase === 'loaded'}
												<div class="flex flex-col gap-0.5">
													<select
														class="w-24 rounded border bg-background text-[10px] leading-tight"
														value={effectiveCtxForRow(m)}
														onchange={(e) => {
															ctxOverride = {
																...ctxOverride,
																[id]: Number((e.target as HTMLSelectElement).value)
															};
														}}
														title="Context size for the next Load - does not affect an already-loaded instance"
													>
														{#each CTX_SIZE_OPTIONS as opt (opt)}
															<option value={opt}>{opt.toLocaleString()} ctx</option>
														{/each}
													</select>
													{#if state.phase === 'loaded'}
														<span class="text-[10px] text-muted-foreground">Adjusting context requires reload</span>
													{/if}
												</div>
											{/if}
										</div>
									{/if}
								</Table.Cell>
							{/if}
						</Table.Row>
					{/each}
				</Table.Body>
			</Table.Root>
		</div>
	{/if}
</div>
