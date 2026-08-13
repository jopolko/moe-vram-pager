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
		Info
	} from '@lucide/svelte';
	import { Switch } from '$lib/components/ui/switch';
	import { Label } from '$lib/components/ui/label';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import * as Table from '$lib/components/ui/table';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';

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
		reason?: string;
		loadPct?: number;
		loadStage?: string;
	}
	let routerAvailable = $state(false);
	let dlState = $state<Record<string, DlState>>({});
	let busyIds = $state<Set<string>>(new Set());

	function modelIdFor(m: ModelRow): string | null {
		return m.gguf_repo ? `${m.gguf_repo}:${m.quant}` : null;
	}

	let currentModelId = $derived(
		Object.entries(dlState).find(([, s]) => s.phase === 'loaded')?.[0] ?? null
	);

	let dlEntries = $derived(Object.entries(dlState).sort(([a], [b]) => a.localeCompare(b)));

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
				// before the SSE feed folded the finished download in
				if (next[row.id]?.phase === 'downloading' && status === 'unloaded') continue;
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
					}
				} else if (envelope.event === 'download_progress') {
					const progress = envelope.data?.progress as
						| Record<string, { done: number; total: number }>
						| undefined;
					const { done, total } = sumProgress(progress ?? {});
					dlState = { ...dlState, [id]: { phase: 'downloading', doneBytes: done, totalBytes: total } };
				} else if (envelope.event === 'download_finished') {
					dlState = { ...dlState, [id]: { phase: 'downloaded' } };
					void refreshRouterModels();
				} else if (envelope.event === 'download_failed') {
					const reason = envelope.data?.reason as string | undefined;
					dlState = { ...dlState, [id]: { phase: 'failed', reason } };
				} else if (envelope.event === 'model_remove') {
					const next = { ...dlState };
					delete next[id];
					dlState = next;
				}
			};
		} catch {
			// router mode not available; the Actions column falls back silently
		}
		void refreshRouterModels();
		return () => es?.close();
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
					quant: m.quant,
					vram_gb: hardware?.vram_gb
				})
			});
			const resp = await fetch('./models', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ model: id })
			});
			if (resp.ok) {
				dlState = { ...dlState, [id]: { phase: 'downloading', doneBytes: 0, totalBytes: 0 } };
			} else {
				const body = await resp.json().catch(() => ({}));
				throw new Error(body.error || `Request failed (${resp.status})`);
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
			await fetch('./models/load', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ model: id })
			});
		} finally {
			const next = new Set(busyIds);
			next.delete(id);
			busyIds = next;
		}
	}

	async function deleteRouterModel(id: string) {
		if (busyIds.has(id)) return;
		if (!confirm(`Delete downloaded files for ${id}? This frees disk space; re-downloading later works the same as the first time.`)) {
			return;
		}
		busyIds = new Set(busyIds).add(id);
		try {
			await fetch(`./models?model=${encodeURIComponent(id)}`, { method: 'DELETE' });
			const next = { ...dlState };
			delete next[id];
			dlState = next;
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

	async function loadModels() {
		loading = true;
		error = '';
		tagsProgress = null;
		// Only relevant on a cold cache (first-ever load, or once every ~30 days when HF
		// tags are re-checked) - most loads finish before this ever ticks.
		const progressTimer = setInterval(pollTagsProgress, 300);
		try {
			const params = new URLSearchParams({
				top: '30',
				gguf_lookup: '60',
				derestricted_only: String(derestrictedOnly),
				rank_by: RANK_BY
			});

			const resp = await fetch(`./model-picker/models?${params.toString()}`);
			if (!resp.ok) {
				const body = await resp.json().catch(() => ({}));
				throw new Error(body.error || `Request failed (${resp.status})`);
			}
			const data = await resp.json();
			hardware = data.hardware;
			models = data.models;
			routerAvailable = Boolean(data.router_available);
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			clearInterval(progressTimer);
			tagsProgress = null;
			loading = false;
		}
	}

	// $effect fires once on mount and again whenever derestrictedOnly changes,
	// no separate onMount needed (that caused a duplicate concurrent fetch).
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
					cmp = a.active_gb - b.active_gb;
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
		<h1 class="text-2xl font-bold tracking-tight uppercase">Mixture of Experts Models</h1>
		<p class="text-sm text-muted-foreground">
			Run MoE models larger than your VRAM by paging only the active experts into memory on
			demand, instead of loading the entire model upfront.
		</p>
		<p class="text-sm text-muted-foreground">Identify every MoE model that will run on your current hardware.</p>
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
				'Fit ranking is budgeted against total VRAM, not free',
				"Close other GPU apps and this goes up - unlike disk space, VRAM is never overcommitted so this is a real live number"
			)}
			{@render hwCard(
				'RAM',
				hardware.ram_gb,
				hardware.ram_free_gb,
				1,
				'Fit ranking is budgeted against total RAM, not free',
				"On Linux/WSL this always equals total - ggml treats free RAM as ill-defined and just assumes it's all available, rather than fighting the reclaimable-page-cache accounting mess. Real on Windows builds."
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
		{#if routerAvailable}
			<DropdownMenu.Root>
				<DropdownMenu.Trigger>
					{#snippet child({ props })}
						<button
							{...props}
							type="button"
							class="-mx-1 inline-flex max-w-[20rem] items-center gap-1 rounded-sm px-1 hover:bg-muted-foreground/10"
						>
							<span class="text-muted-foreground">Loaded:</span>
							<span class="min-w-0 truncate">{currentModelId ?? 'None'}</span>
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
									<span class="min-w-0 truncate" title={id}>{id}</span>
								</DropdownMenu.Item>
								{#if state.phase === 'unloaded' || state.phase === 'failed'}
									<button
										type="button"
										class="shrink-0 rounded-sm p-1.5 text-muted-foreground hover:bg-destructive/10 hover:text-destructive disabled:pointer-events-none disabled:opacity-50"
										title="Delete downloaded files"
										disabled={busy}
										onclick={(e) => {
											e.stopPropagation();
											deleteRouterModel(id);
										}}
									>
										<Trash2 class="h-3.5 w-3.5" />
									</button>
								{/if}
							</div>
						{/each}
					{/if}
				</DropdownMenu.Content>
			</DropdownMenu.Root>
		{/if}
	</div>

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
								title="Active params (B) x bits-per-weight at this quant / 8 = GB. Only the active experts need to be resident here for fast inference."
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
								{#if m.fit_tier === 'ram-cache'}
									<Badge
										variant="outline"
										class="ml-2 align-middle border-amber-300 bg-amber-50 text-amber-700 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-300"
										title="Active experts don't all fit in VRAM alone - the overflow caches in host RAM instead of disk. Still fast, no per-token disk streaming needed for the hot set."
									>
										RAM spillover
									</Badge>
								{:else if m.fit_tier === 'disk-streaming'}
									<Badge
										variant="outline"
										class="ml-2 align-middle border-orange-300 bg-orange-50 text-orange-700 dark:border-orange-800 dark:bg-orange-950/40 dark:text-orange-300"
										title="Active expert working set exceeds VRAM+RAM combined - expect real per-token disk reads on cache misses. Still runs, just slower."
									>
										Disk streaming
									</Badge>
								{/if}
								{#if m.fit_tier === 'no-disk-space' && hardware}
									<Badge
										variant="destructive"
										class="ml-2 align-middle"
										title="Fits on this drive, but not in the space free right now - free up some space to download it"
									>
										Storage +{Math.max(0, m.total_gb - hardware.disk_free_gb).toFixed(0)}GB
									</Badge>
								{/if}
							</Table.Cell>
							<Table.Cell class="text-xs text-muted-foreground">{m.quant}</Table.Cell>
							<Table.Cell class="text-right tabular-nums">{m.active_gb.toFixed(1)} GB</Table.Cell>
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
												{:else if state.phase === 'downloading' || state.phase === 'downloaded'}
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
												{/if}
											</div>
											{#if state?.phase === 'downloading'}
												{@const pct = state.totalBytes && state.totalBytes > 0 ? Math.min(100, Math.round((100 * (state.doneBytes ?? 0)) / state.totalBytes)) : null}
												<div class="w-24">
													<div class="h-1 w-full overflow-hidden rounded-full bg-muted">
														<div class="h-full rounded-full bg-primary transition-all" style="width: {pct ?? 8}%"></div>
													</div>
													<span class="text-[10px] text-muted-foreground">{pct !== null ? `${pct}%` : '...'}</span>
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
