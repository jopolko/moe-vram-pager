<script lang="ts">
	import { Input } from '$lib/components/ui/input';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import Label from '$lib/components/ui/label/label.svelte';
	import * as Select from '$lib/components/ui/select';
	import { Slider } from '$lib/components/ui/slider';
	import { Button } from '$lib/components/ui/button';
	import { modelsStore } from '$lib/stores/models.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import { ModelsService } from '$lib/services/models.service';
	import { ServerModelStatus } from '$lib/enums';
	import { toast } from 'svelte-sonner';
	import { RefreshCw, Loader2, TriangleAlert } from '@lucide/svelte';
	import { cn } from '$lib/components/ui/utils.js';

	type FieldKind = 'text' | 'number' | 'bool' | 'select' | 'slider';

	interface TuningField {
		key: string;
		label: string;
		kind: FieldKind;
		help: string;
		placeholder?: string;
		options?: { value: string; label: string }[];
		// slider fields only - max is a fallback used until hardware info loads for
		// fields whose real max depends on detected hardware (see sliderMax below)
		min?: number;
		max?: number;
		step?: number;
	}

	const MOE_STREAM_FIELDS: TuningField[] = [
		{
			key: 'moe-stream-cache',
			label: 'MoE stream cache size',
			kind: 'text',
			placeholder: 'e.g. 2G or 12 (experts)',
			help: 'VRAM budget for cached MoE experts. Either a byte size with a G/M suffix, or a bare number of experts to keep resident.'
		},
		{
			key: 'moe-stream-io-threads',
			label: 'MoE stream I/O threads',
			kind: 'slider',
			min: 1,
			max: 16, // overridden by cpu_threads once hardware info loads, see sliderMax
			step: 1,
			help: 'Worker threads used to stream expert weights in from disk/RAM. More threads can help on fast NVMe, but adds CPU overhead.'
		},
		{
			key: 'moe-stream-ram-cache',
			label: 'MoE RAM cache size',
			kind: 'text',
			placeholder: 'e.g. 8G',
			help: 'Secondary expert cache kept in system RAM, above the VRAM cache, to reduce disk streaming on cache misses.'
		},
		{
			key: 'moe-stream-prefetch',
			label: 'MoE prefetch depth',
			kind: 'slider',
			min: 0,
			max: 8,
			step: 1,
			help: 'How many experts to speculatively prefetch ahead of the current layer. Higher values trade RAM/IO for smoother streaming.'
		},
		{
			key: 'moe-stream-direct',
			label: 'Direct I/O',
			kind: 'bool',
			help: 'Bypass the OS page cache when streaming experts from disk. Can help on systems where the page cache is thrashing.'
		},
		{
			key: 'moe-stream-cpu-cache',
			label: 'CPU-side expert cache',
			kind: 'bool',
			help: 'Also keep hot experts cached in pinned CPU memory, not just VRAM, for faster refill on eviction.'
		}
	];

	const HYGIENE_FIELDS: TuningField[] = [
		{
			key: 'ctx-size',
			label: 'Context size (KV cache)',
			kind: 'number',
			placeholder: 'e.g. 65536',
			help: 'Max context length in tokens. Bigger contexts need proportionally bigger KV cache memory.'
		},
		{
			key: 'gpu-layers',
			label: 'GPU layers (-ngl)',
			kind: 'slider',
			min: 0,
			max: 128, // covers virtually every model's layer count; 999 ("all") pins at max, see sliderPosition
			step: 1,
			help: 'Number of transformer layers offloaded to GPU. Push all the way right to offload everything that fits.'
		},
		{
			key: 'threads',
			label: 'CPU threads',
			kind: 'slider',
			min: 1,
			max: 16, // overridden by cpu_threads once hardware info loads, see sliderMax
			step: 1,
			help: 'CPU threads used for generation. Roughly your physical core count is a good starting point.'
		},
		{
			key: 'batch-size',
			label: 'Batch size',
			kind: 'slider',
			min: 32,
			max: 4096,
			step: 32,
			help: 'Logical batch size for prompt processing. Larger batches speed up prompt ingestion at the cost of memory.'
		},
		{
			key: 'ubatch-size',
			label: 'Micro-batch size',
			kind: 'slider',
			min: 32,
			max: 2048,
			step: 32,
			help: 'Physical batch size per compute step. Lower it if you hit out-of-memory errors during prompt processing.'
		},
		{
			key: 'flash-attn',
			label: 'Flash attention',
			kind: 'select',
			options: [
				{ value: '', label: 'Default' },
				{ value: 'auto', label: 'Auto' },
				{ value: 'on', label: 'On' },
				{ value: 'off', label: 'Off' }
			],
			help: 'Use the fused flash-attention kernel when supported. Usually faster and lower memory; leave on Auto unless you have a reason not to.'
		},
		{
			key: 'cache-type-k',
			label: 'KV cache type (K)',
			kind: 'select',
			options: [
				{ value: '', label: 'Default' },
				{ value: 'f16', label: 'f16' },
				{ value: 'q8_0', label: 'q8_0' },
				{ value: 'q4_0', label: 'q4_0' }
			],
			help: 'Quantization of the key cache. Quantizing shrinks KV cache memory at a small quality cost.'
		},
		{
			key: 'cache-type-v',
			label: 'KV cache type (V)',
			kind: 'select',
			options: [
				{ value: '', label: 'Default' },
				{ value: 'f16', label: 'f16' },
				{ value: 'q8_0', label: 'q8_0' },
				{ value: 'q4_0', label: 'q4_0' }
			],
			help: 'Quantization of the value cache. Requires flash attention on most backends.'
		}
	];

	let selectedModelId = $state<string | null>(null);
	let values = $state<Record<string, string>>({});
	let recommended = $state<Record<string, string>>({});
	let hardware = $state<Record<string, number>>({});
	let saving = $state(false);
	let reloading = $state(false);

	let models = $derived(modelsStore.routerModels);

	$effect(() => {
		if (!selectedModelId && models.length > 0) {
			selectedModelId = modelsStore.selectedModelId ?? models[0].id;
		}
	});

	// Hardware-only, not per-model, so fetch once rather than on every model switch.
	$effect(() => {
		if (isRouterMode()) {
			void ModelsService.getTuningDefaults()
				.then((res) => {
					recommended = res.defaults;
					hardware = res.hardware;
				})
				.catch(() => {});
		}
	});

	// threads/io-threads only make sense bounded by what this box actually has -
	// the field's own `max` is just a placeholder until hardware info loads.
	function sliderMax(field: TuningField): number {
		if (field.key === 'threads' || field.key === 'moe-stream-io-threads') {
			return hardware.cpu_threads || field.max || 16;
		}
		return field.max ?? 100;
	}

	// gpu-layers' recommended/stored value is often "999" (shorthand for "all
	// layers"), which is nonsensical as a literal slider position - pin display
	// at the slider's max without silently rewriting the underlying value unless
	// the user actually drags it.
	function sliderPosition(field: TuningField): number {
		const raw = Number(values[field.key]);
		const max = sliderMax(field);
		if (!Number.isFinite(raw)) return field.min ?? 0;
		return Math.min(raw, max);
	}

	function exceedsRecommended(field: TuningField): boolean {
		const rec = Number(recommended[field.key]);
		const cur = Number(values[field.key]);
		if (!Number.isFinite(rec) || !Number.isFinite(cur)) return false;
		return cur > rec;
	}

	// green up to recommended, yellow past it, red once you're pushing against
	// the slider's own ceiling (or well past 1.5x recommended) - "are you sure?" territory
	type Severity = 'ok' | 'warn' | 'danger';
	function sliderSeverity(field: TuningField): Severity {
		const rec = Number(recommended[field.key]);
		const cur = Number(values[field.key]);
		if (!Number.isFinite(cur)) return 'ok';
		// at or under recommended is always green, even if that happens to sit
		// at the slider's ceiling (e.g. threads == cpu_threads, gpu-layers == 999/all)
		if (!Number.isFinite(rec) || rec <= 0 || cur <= rec) return 'ok';
		const max = sliderMax(field);
		if ((max > 0 && cur >= max * 0.95) || cur > rec * 1.5) return 'danger';
		return 'warn';
	}

	const SEVERITY_RANGE_CLASS: Record<Severity, string> = {
		ok: 'bg-emerald-500',
		warn: 'bg-amber-500',
		danger: 'bg-red-500'
	};

	const SEVERITY_THUMB_CLASS: Record<Severity, string> = {
		ok: 'border-emerald-500',
		warn: 'border-amber-500',
		danger: 'border-red-500'
	};

	function parsePresetIni(ini: string | undefined): Record<string, string> {
		const out: Record<string, string> = {};
		if (!ini) return out;
		for (const line of ini.split('\n')) {
			const trimmed = line.trim();
			if (!trimmed || trimmed.startsWith('[') || trimmed.startsWith(';') || trimmed.startsWith('#')) {
				continue;
			}
			const eq = trimmed.indexOf('=');
			if (eq === -1) continue;
			const key = trimmed.slice(0, eq).trim();
			const value = trimmed.slice(eq + 1).trim();
			out[key] = value;
		}
		return out;
	}

	$effect(() => {
		const model = models.find((m) => m.id === selectedModelId);
		// recommended fills in anything the persisted preset doesn't already set - real,
		// explicitly-saved values always win
		values = { ...recommended, ...parsePresetIni(model?.status?.preset) };
	});

	function isRecommended(key: string): boolean {
		const model = models.find((m) => m.id === selectedModelId);
		const persisted = parsePresetIni(model?.status?.preset);
		return persisted[key] === undefined && recommended[key] !== undefined;
	}

	let selectedModel = $derived(models.find((m) => m.id === selectedModelId) ?? null);
	let isLoaded = $derived(selectedModel?.status.value === ServerModelStatus.LOADED);

	function boolValue(key: string): boolean {
		const v = (values[key] ?? '').toLowerCase();
		return v === 'true' || v === '1' || v === 'yes';
	}

	async function handleSave() {
		if (!selectedModelId) return;
		const overrides: Record<string, string> = {};
		for (const field of [...MOE_STREAM_FIELDS, ...HYGIENE_FIELDS]) {
			const raw = values[field.key];
			if (field.kind === 'bool') {
				overrides[field.key] = boolValue(field.key) ? 'true' : 'false';
			} else if (raw !== undefined && raw !== '') {
				overrides[field.key] = raw;
			}
		}
		saving = true;
		try {
			await ModelsService.setTuning(selectedModelId, overrides);
			toast.success('Tuning saved. Takes effect next time this model loads.');
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Failed to save tuning');
		} finally {
			saving = false;
		}
	}

	async function handleSaveAndReload() {
		if (!selectedModelId) return;
		await handleSave();
		reloading = true;
		try {
			await modelsStore.unloadModel(selectedModelId);
			await modelsStore.loadModel(selectedModelId);
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Failed to reload model');
		} finally {
			reloading = false;
		}
	}

	function renderField(field: TuningField) {
		return field;
	}
</script>

{#if !isRouterMode()}
	<div class="py-8 text-center text-sm text-muted-foreground">
		Tuning requires router mode (launch with <code>--models-preset</code>). Single-model mode has
		no per-model preset to edit here - pass flags on the command line instead.
	</div>
{:else if models.length === 0}
	<div class="py-8 text-center text-sm text-muted-foreground">No models available</div>
{:else}
	<div class="space-y-6">
		<div class="space-y-2">
			<Label for="tuning-model" class="text-sm font-medium">Model</Label>
			<Select.Root
				type="single"
				value={selectedModelId ?? undefined}
				onValueChange={(value) => {
					selectedModelId = value ?? null;
				}}
			>
				<Select.Trigger id="tuning-model" class="w-full md:w-auto">
					{models.find((m) => m.id === selectedModelId)?.id ?? 'Select a model'}
				</Select.Trigger>
				<Select.Content>
					{#each models as model (model.id)}
						<Select.Item value={model.id} label={model.id}>{model.id}</Select.Item>
					{/each}
				</Select.Content>
			</Select.Root>
			<p class="text-xs text-muted-foreground">
				Values shown are this model's current effective preset. Changes are written to
				models-preset.ini and take effect the next time the model loads.
			</p>
		</div>

		{#snippet sliderRow(f: TuningField)}
			<div class="flex items-center gap-2">
				<Label for={f.key} class="flex items-center gap-1.5 text-sm font-medium">
					{f.label}
					{#if isRecommended(f.key)}
						<span class="rounded bg-muted px-1.5 py-0.5 text-[10px] font-normal text-muted-foreground">recommended</span>
					{/if}
				</Label>
				{#if exceedsRecommended(f)}
					{@const severity = sliderSeverity(f)}
					<span
						class={cn(
							'flex items-center gap-1 text-[10px] font-medium',
							severity === 'danger'
								? 'text-red-600 dark:text-red-400'
								: 'text-amber-600 dark:text-amber-400'
						)}
					>
						<TriangleAlert class="h-3 w-3" />
						{severity === 'danger' ? 'are you sure?' : 'exceeds recommended'}
					</span>
				{/if}
			</div>
			<div class="flex items-center gap-3">
				<Slider
					id={f.key}
					type="single"
					min={f.min ?? 0}
					max={sliderMax(f)}
					step={f.step ?? 1}
					value={sliderPosition(f)}
					onValueChange={(v: number) => {
						values = { ...values, [f.key]: String(v) };
					}}
					rangeClass={SEVERITY_RANGE_CLASS[sliderSeverity(f)]}
					thumbClass={SEVERITY_THUMB_CLASS[sliderSeverity(f)]}
					class="max-w-sm"
				/>
				<span class="w-16 shrink-0 font-mono text-sm">
					{f.key === 'gpu-layers' && values[f.key] === '999' ? '999 (all)' : (values[f.key] ?? '')}
				</span>
			</div>
			<p class="text-xs text-muted-foreground">{f.help}</p>
		{/snippet}

		<div class="space-y-4">
			<h4 class="text-sm font-semibold text-muted-foreground">MoE VRAM Streaming</h4>
			{#each MOE_STREAM_FIELDS as field (field.key)}
				{@const f = renderField(field)}
				<div class="space-y-2">
					{#if f.kind === 'bool'}
						<div class="flex items-start space-x-3">
							<Checkbox
								id={f.key}
								checked={boolValue(f.key)}
								onCheckedChange={(checked) => {
									values = { ...values, [f.key]: checked ? 'true' : 'false' };
								}}
								class="mt-1"
							/>
							<div class="space-y-1">
								<label for={f.key} class="flex cursor-pointer items-center gap-1.5 text-sm leading-none font-medium">
									{f.label}
									{#if isRecommended(f.key)}
										<span class="rounded bg-muted px-1.5 py-0.5 text-[10px] font-normal text-muted-foreground">
											recommended: {boolValue(f.key) ? 'on' : 'off'}
										</span>
									{/if}
								</label>
								<p class="text-xs text-muted-foreground">{f.help}</p>
							</div>
						</div>
					{:else if f.kind === 'slider'}
						{@render sliderRow(f)}
					{:else}
						<Label for={f.key} class="flex items-center gap-1.5 text-sm font-medium">
							{f.label}
							{#if isRecommended(f.key)}
								<span class="rounded bg-muted px-1.5 py-0.5 text-[10px] font-normal text-muted-foreground">recommended</span>
							{/if}
						</Label>
						<Input
							id={f.key}
							type={f.kind === 'number' ? 'number' : 'text'}
							value={values[f.key] ?? ''}
							placeholder={f.placeholder}
							oninput={(e) => {
								values = { ...values, [f.key]: e.currentTarget.value };
							}}
							class="w-full md:max-w-sm"
						/>
						<p class="text-xs text-muted-foreground">{f.help}</p>
					{/if}
				</div>
			{/each}
		</div>

		<div class="space-y-4 border-t border-border/30 pt-4">
			<h4 class="text-sm font-semibold text-muted-foreground">General LLM Hygiene</h4>
			{#each HYGIENE_FIELDS as field (field.key)}
				{@const f = renderField(field)}
				<div class="space-y-2">
					{#if f.kind === 'select'}
						<Label for={f.key} class="flex items-center gap-1.5 text-sm font-medium">
							{f.label}
							{#if isRecommended(f.key)}
								<span class="rounded bg-muted px-1.5 py-0.5 text-[10px] font-normal text-muted-foreground">recommended</span>
							{/if}
						</Label>
						<Select.Root
							type="single"
							value={values[f.key] ?? ''}
							onValueChange={(value) => {
								values = { ...values, [f.key]: value ?? '' };
							}}
						>
							<Select.Trigger id={f.key} class="w-full md:w-48">
								{f.options?.find((o) => o.value === (values[f.key] ?? ''))?.label ?? 'Default'}
							</Select.Trigger>
							<Select.Content>
								{#each f.options ?? [] as option (option.value)}
									<Select.Item value={option.value} label={option.label}>
										{option.label}
									</Select.Item>
								{/each}
							</Select.Content>
						</Select.Root>
						<p class="text-xs text-muted-foreground">{f.help}</p>
					{:else if f.kind === 'slider'}
						{@render sliderRow(f)}
					{:else}
						<Label for={f.key} class="flex items-center gap-1.5 text-sm font-medium">
							{f.label}
							{#if isRecommended(f.key)}
								<span class="rounded bg-muted px-1.5 py-0.5 text-[10px] font-normal text-muted-foreground">recommended</span>
							{/if}
						</Label>
						<Input
							id={f.key}
							type={f.kind === 'number' ? 'number' : 'text'}
							value={values[f.key] ?? ''}
							placeholder={f.placeholder}
							oninput={(e) => {
								values = { ...values, [f.key]: e.currentTarget.value };
							}}
							class="w-full md:max-w-sm"
						/>
						<p class="text-xs text-muted-foreground">{f.help}</p>
					{/if}
				</div>
			{/each}
		</div>

		<div class="flex flex-wrap items-center gap-2 border-t border-border/30 pt-4">
			<Button onclick={handleSave} disabled={saving || reloading}>
				{#if saving}
					<Loader2 class="h-3.5 w-3.5 animate-spin" />
				{/if}
				Save (applies on next load)
			</Button>
			{#if isLoaded}
				<Button variant="outline" onclick={handleSaveAndReload} disabled={saving || reloading}>
					{#if reloading}
						<Loader2 class="h-3.5 w-3.5 animate-spin" />
					{:else}
						<RefreshCw class="h-3.5 w-3.5" />
					{/if}
					Save and reload model now
				</Button>
			{/if}
		</div>
	</div>
{/if}
