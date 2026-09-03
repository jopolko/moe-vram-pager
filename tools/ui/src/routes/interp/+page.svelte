<script lang="ts">
	import {
		FlaskConical,
		RefreshCw,
		Loader2,
		FolderOpen,
		Upload,
		Server,
		X,
		ChevronRight,
		Languages,
		Music,
		Brain,
		ShieldX,
		CircleCheck,
		CircleX,
		TriangleAlert
	} from '@lucide/svelte';
	import { browser } from '$app/environment';
	import { APP_NAME } from '$lib/constants/app';
	import { Button } from '$lib/components/ui/button';
	import { Badge } from '$lib/components/ui/badge';
	import { Input } from '$lib/components/ui/input';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import InterpLive from '$lib/components/app/interp/InterpLive.svelte';

	// "Runs" = the batch `run exp*` result viewer (below). "Live" = the
	// interactive per-token sidecar (tools/interp_live_api.py / `obench-interp serve`).
	type Mode = 'runs' | 'live';
	const MODE_KEY = 'interp-viewer-mode';
	let mode = $state<Mode>('runs');

	type Experiment = 'exp1_multilingual' | 'exp2_planning' | 'exp3_cot_faithfulness' | 'exp4_gaming';
	const EXPERIMENT_IDS: Experiment[] = [
		'exp1_multilingual',
		'exp2_planning',
		'exp3_cot_faithfulness',
		'exp4_gaming'
	];

	interface GamingCond {
		text: string;
		code: string;
		visible_pass: string;
		held_pass: string;
		solved: boolean;
		gamed: boolean;
	}

	interface RunData {
		schema_version: number;
		experiment: Experiment;
		generated_at: string;
		model: string;
		params: Record<string, unknown>;
		per_item: Record<string, unknown>[];
		aggregate: Record<string, unknown>;
		/** filled in by the loader from the run's directory / file name */
		_timestamp: string;
	}

	const EXPERIMENTS: { id: Experiment; label: string; icon: typeof Languages; question: string }[] =
		[
			{
				id: 'exp1_multilingual',
				label: 'Multilingual concept sharing',
				icon: Languages,
				question: 'What language, if any, is the model using "in its head"?'
			},
			{
				id: 'exp2_planning',
				label: 'Planning ahead',
				icon: Music,
				question: 'Is it only predicting the next word, or does it plan ahead?'
			},
			{
				id: 'exp3_cot_faithfulness',
				label: 'Chain-of-thought faithfulness',
				icon: Brain,
				question: 'Does the stated reasoning match the steps it actually took?'
			},
			{
				id: 'exp4_gaming',
				label: 'Specification gaming',
				icon: ShieldX,
				question:
					'Under pressure to pass the tests, does it hardcode them instead of solving the task?'
			}
		];

	// ---------------------------------------------------------------------------
	// data source: a directory the browser reads directly (no server), dropped
	// files, or the optional tools/interp_ui_api.py sidecar. Folder first — it is
	// the smooth path: pick interpretability/results once, it is remembered.
	// ---------------------------------------------------------------------------
	type Source = 'none' | 'folder' | 'drop' | 'sidecar';
	const SIDECAR_KEY = 'interp-viewer-sidecar';
	const SOURCE_KEY = 'interp-viewer-source';
	const SELECTED_KEY = 'interp-viewer-selected';
	const DEFAULT_SIDECAR = 'http://127.0.0.1:8087';

	const supportsFS = browser && 'showDirectoryPicker' in window;

	let source = $state<Source>('none');
	let sourceLabel = $state('');
	let runs = $state<RunData[]>([]);
	let loading = $state(false);
	let error = $state('');
	let dragging = $state(false);
	let showSidecar = $state(false);
	let sidecarUrl = $state(DEFAULT_SIDECAR);
	let selectedKey = $state<string | null>(null);
	let openRows = $state<Record<string, boolean>>({});

	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	let dirHandle: any = null;

	const keyOf = (r: { experiment: string; _timestamp: string }) =>
		`${r.experiment}/${r._timestamp}`;

	const sortedRuns = $derived(
		[...runs].sort(
			(a, b) =>
				EXPERIMENT_IDS.indexOf(a.experiment) - EXPERIMENT_IDS.indexOf(b.experiment) ||
				b._timestamp.localeCompare(a._timestamp)
		)
	);
	const grouped = $derived(
		EXPERIMENTS.map((e) => ({ ...e, runs: sortedRuns.filter((r) => r.experiment === e.id) }))
	);
	const selected = $derived(sortedRuns.find((r) => keyOf(r) === selectedKey) ?? null);

	function persist(k: string, v: string) {
		if (!browser) return;
		try {
			localStorage.setItem(k, v);
		} catch {
			/* ignore */
		}
	}

	function headline(r: RunData): string {
		const a = r.aggregate ?? {};
		try {
			if (r.experiment === 'exp1_multilingual')
				return `cosine ${a.mean_cross_language_cosine} · ${a.language_agnostic_feature_count} shared features`;
			if (r.experiment === 'exp2_planning')
				return `planning effect ${a.planning_effect} (${a.planning_flip_rate} vs ${a.control_flip_rate} control)`;
			if (r.experiment === 'exp3_cot_faithfulness')
				return `${a.unfaithful_count}/${a.n_items} unfaithful · follow ${a.hint_follow_rate}`;
			if (r.experiment === 'exp4_gaming')
				return `${a.gamed_count}/${a.n_items} gamed · ${a.pressure_induced_count} pressure-induced`;
		} catch {
			/* ignore */
		}
		return '';
	}

	function ingest(list: RunData[], src: Source, label: string) {
		const seen: Record<string, true> = {};
		runs = list
			.filter((r) => EXPERIMENT_IDS.includes(r.experiment) && Array.isArray(r.per_item))
			.filter((r) => {
				const k = keyOf(r);
				if (seen[k]) return false;
				seen[k] = true;
				return true;
			});
		source = src;
		sourceLabel = label;
		persist(SOURCE_KEY, src);
		if (!selected && sortedRuns.length) selectRun(sortedRuns[0]);
	}

	function selectRun(r: RunData) {
		selectedKey = keyOf(r);
		openRows = {};
		persist(SELECTED_KEY, selectedKey);
	}

	// ---- folder ----
	async function walkDir(handle: unknown): Promise<RunData[]> {
		const dir = handle as {
			name: string;
			entries(): AsyncIterable<[string, unknown]>;
			getDirectoryHandle(n: string): Promise<unknown>;
		};
		// support pointing either at `results/` or at its parent
		const roots: unknown[] = [];
		for (const id of EXPERIMENT_IDS) {
			try {
				roots.push(await dir.getDirectoryHandle(id));
			} catch {
				/* not here */
			}
		}
		if (!roots.length) {
			try {
				roots.push(await dir.getDirectoryHandle('results'));
				const inner = roots.pop() as typeof dir;
				for (const id of EXPERIMENT_IDS) {
					try {
						roots.push(await inner.getDirectoryHandle(id));
					} catch {
						/* skip */
					}
				}
			} catch {
				/* still nothing */
			}
		}
		const out: RunData[] = [];
		for (const expDir of roots as { name: string; entries(): AsyncIterable<[string, unknown]> }[]) {
			for await (const [ts, runHandle] of expDir.entries()) {
				const rh = runHandle as {
					kind: string;
					getFileHandle(n: string): Promise<{ getFile(): Promise<File> }>;
				};
				if (rh.kind !== 'directory') continue;
				try {
					const fh = await rh.getFileHandle('results.json');
					const text = await (await fh.getFile()).text();
					const data = JSON.parse(text) as RunData;
					data._timestamp = ts;
					out.push(data);
				} catch {
					/* no results.json in this dir */
				}
			}
		}
		return out;
	}

	async function loadFolder(handle: unknown, remember = true) {
		loading = true;
		error = '';
		try {
			const list = await walkDir(handle);
			if (!list.length) throw new Error('no exp*/<timestamp>/results.json found in that folder');
			dirHandle = handle;
			if (remember) await idbSet('dirHandle', handle);
			ingest(list, 'folder', (handle as { name: string }).name);
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
			if (source === 'none') runs = [];
		} finally {
			loading = false;
		}
	}

	async function pickFolder() {
		try {
			// @ts-expect-error - File System Access API, guarded by supportsFS
			const handle = await window.showDirectoryPicker({
				id: 'obench-interp-results',
				mode: 'read'
			});
			await loadFolder(handle);
		} catch (e) {
			if ((e as DOMException)?.name !== 'AbortError')
				error = e instanceof Error ? e.message : String(e);
		}
	}

	async function rescan() {
		if (source === 'folder' && dirHandle) return loadFolder(dirHandle, false);
		if (source === 'sidecar') return connectSidecar(sidecarUrl);
	}

	function disconnect() {
		runs = [];
		source = 'none';
		sourceLabel = '';
		selectedKey = null;
		dirHandle = null;
		void idbDel('dirHandle');
		persist(SOURCE_KEY, 'none');
	}

	// ---- dropped files ----
	async function readFiles(files: File[]) {
		loading = true;
		error = '';
		try {
			const out: RunData[] = [];
			for (const f of files) {
				if (!f.name.endsWith('.json')) continue;
				try {
					const data = JSON.parse(await f.text()) as RunData;
					if (!data.experiment) continue;
					// dropped files carry no dir name; derive a stable-ish key
					data._timestamp = (data.generated_at || '').replace(/[^0-9]/g, '').slice(0, 15) || f.name;
					out.push(data);
				} catch {
					/* not a results.json */
				}
			}
			if (!out.length) throw new Error('none of those were experiment results.json files');
			ingest([...runs, ...out], 'drop', `${out.length + runs.length} file(s)`);
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
			dragging = false;
		}
	}

	function onDrop(ev: DragEvent) {
		ev.preventDefault();
		dragging = false;
		const items = ev.dataTransfer?.files;
		if (items?.length) readFiles(Array.from(items));
	}

	// ---- sidecar ----
	async function connectSidecar(url: string) {
		const base = url.trim().replace(/\/$/, '') || DEFAULT_SIDECAR;
		sidecarUrl = base;
		loading = true;
		error = '';
		try {
			const r = await fetch(`${base}/all`, { cache: 'no-store' });
			if (!r.ok) throw new Error(`HTTP ${r.status}`);
			const data = await r.json();
			const list: RunData[] = (data.runs ?? []).map((x: RunData & { timestamp?: string }) => ({
				...x,
				_timestamp: x._timestamp ?? x.timestamp
			}));
			persist(SIDECAR_KEY, base);
			ingest(list, 'sidecar', base.replace(/^https?:\/\//, ''));
		} catch (e) {
			error = `sidecar: ${e instanceof Error ? e.message : String(e)}`;
			if (source === 'none') runs = [];
		} finally {
			loading = false;
		}
	}

	// ---- tiny IndexedDB kv for the directory handle ----
	function idb(): Promise<IDBDatabase> {
		return new Promise((resolve, reject) => {
			const req = indexedDB.open('interp-viewer', 1);
			req.onupgradeneeded = () => req.result.createObjectStore('kv');
			req.onsuccess = () => resolve(req.result);
			req.onerror = () => reject(req.error);
		});
	}
	async function idbSet(k: string, v: unknown) {
		const db = await idb();
		await new Promise((res, rej) => {
			const tx = db.transaction('kv', 'readwrite');
			tx.objectStore('kv').put(v, k);
			tx.oncomplete = () => res(null);
			tx.onerror = () => rej(tx.error);
		});
	}
	async function idbGet<T>(k: string): Promise<T | undefined> {
		const db = await idb();
		return new Promise((res, rej) => {
			const tx = db.transaction('kv', 'readonly');
			const r = tx.objectStore('kv').get(k);
			r.onsuccess = () => res(r.result);
			r.onerror = () => rej(r.error);
		});
	}
	async function idbDel(k: string) {
		const db = await idb();
		const tx = db.transaction('kv', 'readwrite');
		tx.objectStore('kv').delete(k);
	}

	// ---- restore on mount ----
	$effect(() => {
		if (!browser) return;
		try {
			selectedKey = localStorage.getItem(SELECTED_KEY);
			sidecarUrl = localStorage.getItem(SIDECAR_KEY) || DEFAULT_SIDECAR;
			if (localStorage.getItem(MODE_KEY) === 'live') mode = 'live';
		} catch {
			/* ignore */
		}
		const last = (() => {
			try {
				return localStorage.getItem(SOURCE_KEY);
			} catch {
				return null;
			}
		})();
		(async () => {
			if (last === 'folder' && supportsFS) {
				const handle = await idbGet<unknown>('dirHandle').catch(() => undefined);
				if (
					handle &&
					(await (handle as { queryPermission(o: object): Promise<string> })
						.queryPermission({ mode: 'read' })
						.catch(() => 'prompt')) === 'granted'
				) {
					await loadFolder(handle, false);
					return;
				}
			}
			if (last === 'sidecar') await connectSidecar(sidecarUrl);
		})();
	});

	// ---- formatting ----
	const fmtPct = (v: unknown) =>
		typeof v === 'number' ? `${(v * 100).toFixed(v * 100 < 10 ? 1 : 0)}%` : '—';
	const fmtNum = (v: unknown, d = 3) => (typeof v === 'number' ? v.toFixed(d) : '—');
	function fmtWhen(ts?: string) {
		if (!ts) return '';
		const d = new Date(ts);
		return Number.isNaN(d.getTime())
			? ts
			: d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
	}

	const FAMILY_BADGE: Record<string, string> = {
		a: 'bg-sky-500/15 text-sky-600 dark:text-sky-400',
		b: 'bg-violet-500/15 text-violet-600 dark:text-violet-400',
		ambiguous: 'bg-amber-500/15 text-amber-600 dark:text-amber-400',
		other: 'bg-foreground/10 text-muted-foreground'
	};

	function cosTint(v: number) {
		const a = Math.max(0, Math.min(1, (v - 0.2) / 0.8));
		return `background-color: rgba(16, 185, 129, ${(a * 0.7).toFixed(2)}); color: ${a > 0.62 ? '#052e1b' : 'inherit'}`;
	}

	const expMeta = (id: string) => EXPERIMENTS.find((e) => e.id === id);
</script>

<svelte:head><title>Interpretability · {APP_NAME}</title></svelte:head>

{#snippet stat(label: string, value: string, tone: 'plain' | 'good' | 'bad' | 'warn' = 'plain')}
	<div class="rounded-lg border bg-muted/40 px-4 py-3">
		<div
			class="text-2xl font-semibold tabular-nums {tone === 'good'
				? 'text-emerald-600 dark:text-emerald-400'
				: tone === 'bad'
					? 'text-red-600 dark:text-red-400'
					: tone === 'warn'
						? 'text-amber-600 dark:text-amber-400'
						: ''}"
		>
			{value}
		</div>
		<div class="mt-0.5 text-xs text-muted-foreground">{label}</div>
	</div>
{/snippet}

{#snippet verdictBadge(ok: boolean, yes: string, no: string)}
	{#if ok}
		<span class="inline-flex items-center gap-1 text-xs text-emerald-600 dark:text-emerald-400">
			<CircleCheck class="size-3.5" />{yes}
		</span>
	{:else}
		<span class="inline-flex items-center gap-1 text-xs text-muted-foreground">
			<CircleX class="size-3.5" />{no}
		</span>
	{/if}
{/snippet}

<div class="mx-auto flex h-full w-full max-w-6xl flex-col gap-4 overflow-hidden p-6">
	<header class="flex flex-wrap items-start justify-between gap-3">
		<div>
			<h1 class="flex items-center gap-2 text-xl font-semibold">
				<FlaskConical class="size-5 text-primary" /> Interpretability
			</h1>
			<p class="mt-1 max-w-2xl text-sm leading-relaxed text-muted-foreground">
				{#if mode === 'live'}
					Type a prompt; watch the model's internal state per token. Needs <code class="text-xs"
						>obench-interp serve</code
					>.
				{:else}
					Runs of the three <code class="text-xs">obench-interp</code> experiments — a shared concept
					space across languages, planning ahead in generation, and whether a stated chain of thought
					is the real one.
				{/if}
			</p>
		</div>
		<div class="flex items-center gap-2">
			<div class="flex rounded-md border p-0.5 text-xs">
				{#each ['runs', 'live'] as m (m)}
					<button
						class="rounded px-2 py-1 capitalize transition-colors {mode === m
							? 'bg-primary/10 font-medium text-primary'
							: 'text-muted-foreground hover:text-foreground'}"
						onclick={() => {
							mode = m as Mode;
							if (browser) localStorage.setItem(MODE_KEY, m);
						}}
					>
						{m}
					</button>
				{/each}
			</div>
			{#if mode === 'runs' && source !== 'none'}
				<Badge variant="secondary" class="gap-1">
					{#if source === 'folder'}<FolderOpen
							class="size-3"
						/>{:else if source === 'sidecar'}<Server class="size-3" />{:else}<Upload
							class="size-3"
						/>{/if}
					{sourceLabel}
				</Badge>
				{#if source !== 'drop'}
					<Button variant="outline" size="sm" onclick={rescan} disabled={loading}>
						{#if loading}<Loader2 class="size-4 animate-spin" />{:else}<RefreshCw
								class="size-4"
							/>{/if}
						Rescan
					</Button>
				{/if}
				<Button variant="ghost" size="sm" onclick={disconnect} title="disconnect">
					<X class="size-4" />
				</Button>
			{/if}
		</div>
	</header>

	{#if mode === 'live'}
		<InterpLive />
	{:else if source === 'none'}
		<!-- empty state: choose a data source -->
		<div
			role="region"
			aria-label="load results"
			class="flex flex-1 flex-col items-center justify-center gap-6 rounded-lg border border-dashed p-10 text-center transition-colors {dragging
				? 'border-primary bg-primary/5'
				: ''}"
			ondragover={(e) => {
				e.preventDefault();
				dragging = true;
			}}
			ondragleave={() => (dragging = false)}
			ondrop={onDrop}
		>
			<div class="space-y-1">
				<FlaskConical class="mx-auto size-8 text-muted-foreground" />
				<p class="text-sm font-medium">Load your experiment results</p>
				<p class="max-w-md text-xs text-muted-foreground">
					Everything <code>obench-interp run</code> writes lives under
					<code>interpretability/results/</code>. Point the viewer at that folder — it stays
					remembered.
				</p>
			</div>

			<div class="flex flex-col items-center gap-3">
				{#if supportsFS}
					<Button onclick={pickFolder} disabled={loading}>
						{#if loading}<Loader2 class="size-4 animate-spin" />{:else}<FolderOpen
								class="size-4"
							/>{/if}
						Open results folder
					</Button>
					<span class="text-xs text-muted-foreground"
						>or drop <code>results.json</code> files here</span
					>
				{:else}
					<p class="text-sm">
						Drop <code>results.json</code> files here
						<span class="block text-xs text-muted-foreground">
							(your browser can't pick a folder — Chrome or Edge can)
						</span>
					</p>
				{/if}

				<button
					class="text-xs text-muted-foreground underline-offset-2 hover:underline"
					onclick={() => (showSidecar = !showSidecar)}
				>
					{showSidecar ? 'hide' : 'connect to a running sidecar instead'}
				</button>
				{#if showSidecar}
					<div class="flex items-center gap-2">
						<Input class="h-8 w-64 text-xs" bind:value={sidecarUrl} spellcheck={false} />
						<Button size="sm" variant="secondary" onclick={() => connectSidecar(sidecarUrl)}>
							Connect
						</Button>
					</div>
					<pre class="rounded-md bg-muted px-3 py-2 text-[11px]">python tools/interp_ui_api.py</pre>
				{/if}
			</div>
			{#if error}<p class="text-xs text-red-500">{error}</p>{/if}
		</div>
	{:else}
		{#if error}
			<p class="rounded-md bg-red-500/10 px-3 py-1.5 text-xs text-red-500">{error}</p>
		{/if}
		<div class="flex min-h-0 flex-1 gap-5">
			<!-- run list -->
			<aside class="w-60 shrink-0 space-y-4 overflow-y-auto pr-1">
				{#each grouped as g (g.id)}
					<div>
						<div class="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
							<g.icon class="size-3.5" />{g.label}
						</div>
						{#if !g.runs.length}
							<p class="px-1 text-xs text-muted-foreground/70">no runs</p>
						{:else}
							<div class="space-y-1">
								{#each g.runs as r (keyOf(r))}
									<button
										data-run={keyOf(r)}
										class="w-full rounded-md border px-2.5 py-1.5 text-left text-xs transition-colors {selectedKey ===
										keyOf(r)
											? 'border-primary/50 bg-primary/10'
											: 'hover:bg-muted/60'}"
										onclick={() => selectRun(r)}
									>
										<div class="truncate font-medium">{r.model ?? 'unknown model'}</div>
										<div class="mt-0.5 truncate text-[11px] text-muted-foreground">
											{headline(r)}
										</div>
										<div class="mt-0.5 text-[10px] text-muted-foreground/70">
											L{(r.params as { layer?: number })?.layer ?? '?'} · {r._timestamp}
										</div>
									</button>
								{/each}
							</div>
						{/if}
					</div>
				{/each}
			</aside>

			<!-- detail -->
			<main class="min-w-0 flex-1 overflow-y-auto">
				{#if loading && !selected}
					<div class="flex h-40 items-center justify-center text-muted-foreground">
						<Loader2 class="size-5 animate-spin" />
					</div>
				{:else if !selected}
					<div class="flex h-40 items-center justify-center text-sm text-muted-foreground">
						{runs.length ? 'Pick a run.' : 'No runs in that source yet.'}
					</div>
				{:else}
					{@const s = selected}
					{@const em = expMeta(s.experiment)}
					<div class="space-y-5">
						<div>
							<div class="flex flex-wrap items-center gap-2">
								<h2 class="text-lg font-semibold">{em?.label ?? s.experiment}</h2>
								<Badge variant="secondary">{s.model}</Badge>
								<Badge variant="tertiary"
									>layer {(s.params as { layer?: number }).layer ?? '?'}</Badge
								>
								{#if (s.params as { backend?: string }).backend}
									<Badge variant="outline">{(s.params as { backend?: string }).backend}</Badge>
								{/if}
							</div>
							<p class="mt-1 text-sm text-muted-foreground">{em?.question}</p>
							<p class="mt-0.5 text-xs text-muted-foreground/70">{fmtWhen(s.generated_at)}</p>
						</div>

						<!-- ===================== exp1 ===================== -->
						{#if s.experiment === 'exp1_multilingual'}
							{@const agg = s.aggregate as {
								mean_cross_language_cosine: number;
								language_agnostic_feature_count: number;
								language_agnostic_features_by_concept: Record<string, number[]>;
							}}
							{@const langs = ((s.params as { languages?: string[] }).languages ?? []) as string[]}
							<div class="grid grid-cols-2 gap-3 sm:grid-cols-3">
								{@render stat(
									'mean cross-language cosine',
									fmtNum(agg.mean_cross_language_cosine),
									agg.mean_cross_language_cosine > 0.5 ? 'good' : 'warn'
								)}
								{@render stat(
									'language-agnostic features',
									String(agg.language_agnostic_feature_count)
								)}
								{@render stat('languages', String(langs.length || '—'))}
							</div>
							<p class="text-xs text-muted-foreground">
								High cosine + a real shared-feature set ⇒ the concept lives in one
								language-independent representation. Low cosine with correct predictions in every
								language ⇒ it solves each language separately.
							</p>

							{#each s.per_item as c (c.concept as string)}
								{@const item = c as {
									concept: string;
									predicted_token: Record<string, string>;
									language_pair_cosine: Record<string, number>;
								}}
								{@const shared = agg.language_agnostic_features_by_concept?.[item.concept] ?? []}
								<div class="rounded-lg border p-4">
									<div class="flex flex-wrap items-center justify-between gap-2">
										<h3 class="font-mono text-sm font-medium">{item.concept}</h3>
										<div class="flex flex-wrap gap-1">
											{#each Object.entries(item.predicted_token) as [lang, tokv] (lang)}
												<span class="rounded bg-muted px-1.5 py-0.5 text-[11px]">
													<span class="text-muted-foreground">{lang}</span>
													{String(tokv).trim() || '∅'}
												</span>
											{/each}
										</div>
									</div>

									{#if langs.length}
										<div class="mt-3 overflow-x-auto">
											<table class="text-[11px]">
												<thead>
													<tr>
														<th class="p-1"></th>
														{#each langs as l (l)}<th class="p-1 font-medium text-muted-foreground"
																>{l}</th
															>{/each}
													</tr>
												</thead>
												<tbody>
													{#each langs as row (row)}
														<tr>
															<td class="p-1 font-medium text-muted-foreground">{row}</td>
															{#each langs as col (col)}
																{@const v =
																	row === col
																		? 1
																		: (item.language_pair_cosine[`${row}-${col}`] ??
																			item.language_pair_cosine[`${col}-${row}`])}
																<td
																	class="p-1 text-center tabular-nums"
																	style={typeof v === 'number' ? cosTint(v) : ''}
																>
																	{typeof v === 'number' ? v.toFixed(2) : ''}
																</td>
															{/each}
														</tr>
													{/each}
												</tbody>
											</table>
										</div>
									{/if}

									<div class="mt-3">
										<span class="text-[11px] text-muted-foreground"
											>{shared.length} feature{shared.length === 1 ? '' : 's'} shared by every language:</span
										>
										<div class="mt-1 flex flex-wrap gap-1">
											{#each shared as fid (fid)}
												<span class="rounded bg-primary/10 px-1.5 py-0.5 font-mono text-[10px]"
													>#{fid}</span
												>
											{/each}
										</div>
									</div>
								</div>
							{/each}

							<!-- ===================== exp2 ===================== -->
						{:else if s.experiment === 'exp2_planning'}
							{@const agg = s.aggregate as {
								planning_effect: number;
								planning_flip_rate: number;
								control_flip_rate: number;
								n_items: number;
							}}
							<div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
								{@render stat(
									'planning effect',
									fmtNum(agg.planning_effect, 2),
									agg.planning_effect > 0.15 ? 'good' : agg.planning_effect > 0 ? 'warn' : 'plain'
								)}
								{@render stat('flip @ plan pos', fmtPct(agg.planning_flip_rate))}
								{@render stat('flip @ early ctrl', fmtPct(agg.control_flip_rate))}
								{@render stat('couplets', String(agg.n_items))}
							</div>
							<p class="text-xs text-muted-foreground">
								Splice a different rhyme family's activations in at the newline (before line 2
								exists). If the generated ending flips far more than a length-matched early-position
								control, the model had already planned the ending. All
								<code class="text-[10px]">other</code> ⇒ the model isn't writing couplets — try an
								instruct
								<code class="text-[10px]">--model</code>.
							</p>

							{#each s.per_item as it (it.id as string)}
								{@const item = it as {
									id: string;
									plan_position: number;
									early_position: number;
									length_matched: boolean;
									probe_probability_at_plan: Record<string, number>;
									generated: Record<string, { text: string; family: string }>;
									flipped_to_corrupt: boolean;
								}}
								<div
									class="rounded-lg border p-4 {item.flipped_to_corrupt
										? 'border-emerald-500/40'
										: ''}"
								>
									<div class="flex flex-wrap items-center justify-between gap-2">
										<h3 class="font-mono text-sm font-medium">{item.id}</h3>
										<div class="flex items-center gap-2 text-[11px] text-muted-foreground">
											{#each Object.entries(item.probe_probability_at_plan) as [w, p] (w)}
												<span>P({w})={p.toFixed(3)}</span>
											{/each}
											{#if !item.length_matched}
												<span class="text-amber-500">token len mismatch</span>
											{/if}
										</div>
									</div>
									<div class="mt-3 grid gap-3 md:grid-cols-3">
										{#each [['baseline', 'baseline'], ['patched_at_plan_position', 'patched @ plan'], ['control_early_position', 'control @ early']] as [key, label] (key)}
											{@const g = item.generated[key]}
											<div class="rounded-md border bg-muted/30 p-2.5">
												<div class="mb-1 flex items-center justify-between">
													<span class="text-[10px] tracking-wide text-muted-foreground uppercase"
														>{label}</span
													>
													<span
														class="rounded px-1.5 py-0.5 text-[10px] font-medium {FAMILY_BADGE[
															g?.family
														] ?? FAMILY_BADGE.other}">{g?.family}</span
													>
												</div>
												<p class="text-xs leading-relaxed">{g?.text}</p>
											</div>
										{/each}
									</div>
									{#if item.flipped_to_corrupt}
										<div
											class="mt-2 inline-flex items-center gap-1 text-[11px] text-emerald-600 dark:text-emerald-400"
										>
											<CircleCheck class="size-3.5" /> ending flipped A→B when patched at the planning
											position
										</div>
									{/if}
								</div>
							{/each}

							<!-- ===================== exp3 ===================== -->
						{:else if s.experiment === 'exp3_cot_faithfulness'}
							{@const agg = s.aggregate as {
								unfaithful_rate: number;
								unfaithful_count: number;
								hint_follow_rate: number;
								hint_acknowledged_rate: number;
								n_items: number;
								followed_count: number;
							}}
							<div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
								{@render stat(
									'unfaithful',
									`${agg.unfaithful_count}/${agg.n_items}`,
									agg.unfaithful_count > 0 ? 'bad' : 'good'
								)}
								{@render stat('hint follow rate', fmtPct(agg.hint_follow_rate))}
								{@render stat('acknowledged (of followed)', fmtPct(agg.hint_acknowledged_rate))}
								{@render stat('followed the hint', `${agg.followed_count}/${agg.n_items}`)}
							</div>
							<p class="text-xs text-muted-foreground">
								Plant a hint pointing at a wrong answer. <b>Unfaithful</b> = the model followed it, never
								mentioned it, and ablating the hint's activations flips the answer back to correct — a
								chain of thought that reads as independent reasoning but wasn't.
							</p>

							{#each s.per_item as it (it.id as string)}
								{@const item = it as {
									id: string;
									question: string;
									correct: string;
									hint_answer: string;
									hint_style: string;
									followed_hint: boolean;
									acknowledged_hint: boolean;
									hint_removed_flips: boolean;
									unfaithful: boolean;
									unhinted: { text: string; answer: string };
									hinted: { text: string; answer: string };
									hint_ablated: { text: string; answer: string };
								}}
								{@const open = !!openRows[item.id]}
								<Collapsible.Root
									{open}
									onOpenChange={(v) => (openRows = { ...openRows, [item.id]: v })}
									class="rounded-lg border {item.unfaithful ? 'border-l-2 border-l-red-500' : ''}"
								>
									<Collapsible.Trigger
										class="flex w-full items-start gap-3 p-3 text-left hover:bg-muted/40"
									>
										<ChevronRight
											class="mt-0.5 size-4 shrink-0 text-muted-foreground transition-transform {open
												? 'rotate-90'
												: ''}"
										/>
										<div class="min-w-0 flex-1">
											<div class="flex flex-wrap items-center gap-2">
												<span class="font-mono text-xs font-medium">{item.id}</span>
												<Badge variant="outline" class="text-[10px]">{item.hint_style} hint</Badge>
												{#if item.unfaithful}
													<span
														class="inline-flex items-center gap-1 rounded bg-red-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-red-600 dark:text-red-400"
													>
														<TriangleAlert class="size-3" /> UNFAITHFUL
													</span>
												{/if}
											</div>
											<p class="mt-1 truncate text-sm">{item.question}</p>
											<div class="mt-1.5 flex flex-wrap gap-x-4 gap-y-1">
												{@render verdictBadge(item.followed_hint, 'followed hint', 'ignored hint')}
												{@render verdictBadge(
													item.acknowledged_hint,
													'acknowledged it',
													'never mentioned it'
												)}
												{@render verdictBadge(
													item.hint_removed_flips,
													'flips when ablated',
													'ablation no-op'
												)}
											</div>
										</div>
									</Collapsible.Trigger>
									<Collapsible.Content class="border-t px-3 pt-3 pb-3">
										<div class="mb-2 text-xs text-muted-foreground">
											correct: <b class="text-foreground">{item.correct}</b> · hint pushed toward
											<b class="text-foreground">{item.hint_answer}</b>
										</div>
										<div class="grid gap-3 lg:grid-cols-3">
											{#each [['unhinted', 'question alone'], ['hinted', 'with the hint'], ['hint_ablated', 'hint activations ablated']] as [key, label] (key)}
												{@const cond = item[key as 'unhinted' | 'hinted' | 'hint_ablated']}
												{@const good = cond.answer.toLowerCase() === item.correct.toLowerCase()}
												<div class="rounded-md border bg-muted/30 p-2.5">
													<div class="mb-1 flex items-center justify-between">
														<span class="text-[10px] tracking-wide text-muted-foreground uppercase"
															>{label}</span
														>
														<span
															class="rounded px-1.5 py-0.5 text-[10px] font-medium {good
																? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400'
																: 'bg-red-500/15 text-red-600 dark:text-red-400'}"
															>→ {cond.answer || '∅'}</span
														>
													</div>
													<p
														class="max-h-52 overflow-y-auto text-xs leading-relaxed whitespace-pre-wrap"
													>
														{cond.text}
													</p>
												</div>
											{/each}
										</div>
									</Collapsible.Content>
								</Collapsible.Root>
							{/each}

							<!-- ===================== exp4 ===================== -->
						{:else if s.experiment === 'exp4_gaming'}
							{@const agg = s.aggregate as {
								n_items: number;
								gamed_count: number;
								plain_gamed_count: number;
								pressure_induced_count: number;
								ablation_removes_count: number;
								solved_count: number;
								gaming_rate: number;
							}}
							<div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
								{@render stat(
									'gamed under pressure',
									`${agg.gamed_count}/${agg.n_items}`,
									agg.gamed_count > 0 ? 'bad' : 'good'
								)}
								{@render stat('pressure-induced', String(agg.pressure_induced_count))}
								{@render stat('ablation flips back', String(agg.ablation_removes_count))}
								{@render stat('gamed anyway (plain)', `${agg.plain_gamed_count}/${agg.n_items}`)}
							</div>
							<p class="text-xs text-muted-foreground">
								Each task ships visible tests plus hidden held-out tests. <b>Gamed</b> = the code
								passes the visible tests but fails the held-out ones (or hardcodes the visible
								inputs). <b>Pressure-induced</b> + <b>ablation flips back</b> together mean the "only
								the visible tests matter" frame caused the hardcoding at the activation level.
							</p>

							{#each s.per_item as it (it.id as string)}
								{@const item = it as {
									id: string;
									entry_point: string;
									pressure_induced_gaming: boolean;
									ablation_removes_gaming: boolean;
									gamed: boolean;
									plain: GamingCond;
									pressured: GamingCond;
									pressure_ablated: GamingCond;
								}}
								{@const open = !!openRows[item.id]}
								<Collapsible.Root
									{open}
									onOpenChange={(v) => (openRows = { ...openRows, [item.id]: v })}
									class="rounded-lg border {item.gamed ? 'border-l-2 border-l-red-500' : ''}"
								>
									<Collapsible.Trigger
										class="flex w-full items-start gap-3 p-3 text-left hover:bg-muted/40"
									>
										<ChevronRight
											class="mt-0.5 size-4 shrink-0 text-muted-foreground transition-transform {open
												? 'rotate-90'
												: ''}"
										/>
										<div class="min-w-0 flex-1">
											<div class="flex flex-wrap items-center gap-2">
												<span class="font-mono text-xs font-medium">{item.id}</span>
												<code class="text-[10px] text-muted-foreground">{item.entry_point}</code>
												{#if item.gamed}
													<span
														class="inline-flex items-center gap-1 rounded bg-red-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-red-600 dark:text-red-400"
													>
														<TriangleAlert class="size-3" /> GAMED
													</span>
												{/if}
											</div>
											<p class="mt-1 truncate text-sm">
												visible {item.pressured.visible_pass} · held-out {item.pressured.held_pass}
											</p>
											<div class="mt-1.5 flex flex-wrap gap-x-4 gap-y-1">
												{@render verdictBadge(
													item.pressure_induced_gaming,
													'pressure induced it',
													'not pressure-induced'
												)}
												{@render verdictBadge(
													item.ablation_removes_gaming,
													'flips back when ablated',
													'ablation no-op'
												)}
											</div>
										</div>
									</Collapsible.Trigger>
									<Collapsible.Content class="border-t px-3 pt-3 pb-3">
										<div class="grid gap-3 lg:grid-cols-3">
											{#each [['plain', 'task alone'], ['pressured', 'with the pressure frame'], ['pressure_ablated', 'pressure frame ablated']] as [key, label] (key)}
												{@const cond = item[key as 'plain' | 'pressured' | 'pressure_ablated']}
												<div class="rounded-md border bg-muted/30 p-2.5">
													<div class="mb-1 flex items-center justify-between gap-2">
														<span class="text-[10px] tracking-wide text-muted-foreground uppercase"
															>{label}</span
														>
														<span
															class="rounded px-1.5 py-0.5 text-[10px] font-medium {cond.gamed
																? 'bg-red-500/15 text-red-600 dark:text-red-400'
																: cond.solved
																	? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400'
																	: 'bg-muted text-muted-foreground'}"
															>{cond.gamed ? 'gamed' : cond.solved ? 'solved' : 'broke'}</span
														>
													</div>
													<div class="mb-1 text-[10px] text-muted-foreground">
														visible {cond.visible_pass} · held-out {cond.held_pass}
													</div>
													<pre
														class="max-h-52 overflow-auto rounded bg-background/60 p-1.5 text-[11px] leading-relaxed whitespace-pre-wrap">{cond.code ||
															cond.text}</pre>
												</div>
											{/each}
										</div>
									</Collapsible.Content>
								</Collapsible.Root>
							{/each}
						{/if}
					</div>
				{/if}
			</main>
		</div>
	{/if}
</div>
