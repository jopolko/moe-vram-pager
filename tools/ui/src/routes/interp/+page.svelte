<script lang="ts">
	import {
		FlaskConical,
		RefreshCw,
		Loader2,
		Plug,
		ChevronRight,
		Languages,
		Music,
		Brain,
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

	// Loopback-only sidecar, its own port - same pattern as #/pentest talking to
	// pentest_ui_api.py. Reads interpretability/results/ and serves the JSON
	// envelopes the `obench-interp run` CLI writes.
	const DEFAULT_API = 'http://127.0.0.1:8087';
	const API_KEY = 'interp-viewer-api';
	const SELECTED_KEY = 'interp-viewer-selected';

	type Experiment = 'exp1_multilingual' | 'exp2_planning' | 'exp3_cot_faithfulness';

	interface RunMeta {
		experiment: Experiment;
		timestamp: string;
		generated_at?: string;
		model?: string;
		layer?: number;
		backend?: string;
		n_items?: number;
		headline?: string;
		aggregate?: Record<string, unknown>;
	}

	interface RunData {
		schema_version: number;
		experiment: Experiment;
		generated_at: string;
		model: string;
		params: Record<string, unknown>;
		per_item: Record<string, unknown>[];
		aggregate: Record<string, unknown>;
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
			}
		];

	let api = $state(DEFAULT_API);
	let apiInput = $state(DEFAULT_API);
	let apiOk = $state<boolean | null>(null);
	let checking = $state(false);
	let runs = $state<RunMeta[]>([]);
	let resultsDir = $state('');
	let selectedKey = $state<string | null>(null);
	let selected = $state<RunData | null>(null);
	let loadingRun = $state(false);
	let error = $state('');
	let openRows = $state<Record<string, boolean>>({});

	const keyOf = (r: { experiment: string; timestamp: string }) => `${r.experiment}/${r.timestamp}`;
	const grouped = $derived(
		EXPERIMENTS.map((e) => ({ ...e, runs: runs.filter((r) => r.experiment === e.id) }))
	);

	function loadPrefs() {
		if (!browser) return;
		try {
			api = apiInput = localStorage.getItem(API_KEY) || DEFAULT_API;
			selectedKey = localStorage.getItem(SELECTED_KEY);
		} catch {
			/* private mode / disabled storage - defaults are fine */
		}
	}

	function persist(k: string, v: string) {
		if (!browser) return;
		try {
			localStorage.setItem(k, v);
		} catch {
			/* ignore */
		}
	}

	async function refresh() {
		checking = true;
		error = '';
		try {
			const r = await fetch(`${api}/runs`, { cache: 'no-store' });
			if (!r.ok) throw new Error(`HTTP ${r.status}`);
			const data = await r.json();
			runs = data.runs ?? [];
			resultsDir = data.results_dir ?? '';
			apiOk = true;
			// re-select the remembered run, or fall back to the newest
			const want = selectedKey && runs.find((x) => keyOf(x) === selectedKey);
			if (want) selectRun(want);
			else if (runs.length && !selected) selectRun(runs[0]);
			else if (selected && !runs.find((x) => keyOf(x) === selectedKey)) selected = null;
		} catch (e) {
			apiOk = false;
			runs = [];
			error = e instanceof Error ? e.message : String(e);
		} finally {
			checking = false;
		}
	}

	async function selectRun(meta: RunMeta) {
		selectedKey = keyOf(meta);
		persist(SELECTED_KEY, selectedKey);
		loadingRun = true;
		openRows = {};
		try {
			const r = await fetch(`${api}/runs/${meta.experiment}/${meta.timestamp}`, {
				cache: 'no-store'
			});
			if (!r.ok) throw new Error(`HTTP ${r.status}`);
			selected = await r.json();
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
			selected = null;
		} finally {
			loadingRun = false;
		}
	}

	function applyApi() {
		api = apiInput.trim().replace(/\/$/, '') || DEFAULT_API;
		persist(API_KEY, api);
		refresh();
	}

	$effect(() => {
		loadPrefs();
		refresh();
	});

	// ---- formatting helpers ----
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

	// exp1 cosine heatmap: ~0.2 faint → 1.0 strong, on a teal→emerald ramp that
	// reads in both themes.
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
				Runs of the three <code class="text-xs">obench-interp</code> experiments — a shared concept
				space across languages, planning ahead in generation, and whether a stated chain of thought
				is the real one. Point <code class="text-xs">tools/interp_ui_api.py</code> at your results dir.
			</p>
		</div>
		<div class="flex items-center gap-2">
			<div
				class="inline-flex items-center gap-1.5 text-xs {apiOk
					? 'text-emerald-600 dark:text-emerald-400'
					: 'text-muted-foreground'}"
			>
				<span
					class="size-2 rounded-full {apiOk
						? 'bg-emerald-500'
						: apiOk === false
							? 'bg-red-500'
							: 'bg-muted-foreground/40'}"
				></span>
				{apiOk ? 'connected' : apiOk === false ? 'offline' : 'checking'}
			</div>
			<Button variant="outline" size="sm" onclick={refresh} disabled={checking}>
				{#if checking}<Loader2 class="size-4 animate-spin" />{:else}<RefreshCw
						class="size-4"
					/>{/if}
				Refresh
			</Button>
		</div>
	</header>

	{#if apiOk === false}
		<div class="rounded-lg border border-dashed p-6 text-sm">
			<div class="flex items-center gap-2 font-medium">
				<Plug class="size-4" /> Sidecar not reachable
			</div>
			<p class="mt-2 text-muted-foreground">
				Start the read-only results server (stdlib only, no install), then Refresh:
			</p>
			<pre
				class="mt-2 overflow-x-auto rounded-md bg-muted p-3 text-xs">python tools/interp_ui_api.py</pre>
			<div class="mt-3 flex items-center gap-2">
				<Input class="h-8 max-w-xs text-xs" bind:value={apiInput} spellcheck={false} />
				<Button size="sm" variant="secondary" onclick={applyApi}>Use</Button>
			</div>
			{#if error}<p class="mt-2 text-xs text-red-500">{error}</p>{/if}
		</div>
	{:else}
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
											{r.headline}
										</div>
										<div class="mt-0.5 text-[10px] text-muted-foreground/70">
											L{r.layer ?? '?'} · {r.timestamp}
										</div>
									</button>
								{/each}
							</div>
						{/if}
					</div>
				{/each}
				{#if resultsDir}
					<p class="px-1 pt-1 text-[10px] break-all text-muted-foreground/60">{resultsDir}</p>
				{/if}
			</aside>

			<!-- detail -->
			<main class="min-w-0 flex-1 overflow-y-auto">
				{#if loadingRun}
					<div class="flex h-40 items-center justify-center text-muted-foreground">
						<Loader2 class="size-5 animate-spin" />
					</div>
				{:else if !selected}
					<div class="flex h-40 items-center justify-center text-sm text-muted-foreground">
						{runs.length ? 'Pick a run.' : 'No runs yet — run one with ./interp run exp1'}
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
						{/if}
					</div>
				{/if}
			</main>
		</div>
	{/if}
</div>
