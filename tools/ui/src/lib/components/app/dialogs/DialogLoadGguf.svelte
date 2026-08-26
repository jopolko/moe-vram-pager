<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Badge } from '$lib/components/ui/badge';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { Loader2 } from '@lucide/svelte';

	interface Assessment {
		fit: string;
		arch: string;
		is_moe: boolean;
		quant_label: string;
		active_gb: number;
		total_gb: number;
		kv_cache_gb: number;
		n_ctx_train: number;
		is_split: boolean;
		split_count: number;
		shards_found: number;
		vram_free_gb: number;
		disk_free_gb: number;
		n_layer: number;
		n_head_kv: number;
		head_dim: number;
		sliding_window: number;
		sliding_window_pattern: number;
		n_expert: number;
		n_expert_used: number;
	}

	interface Props {
		open: boolean;
		onLoaded: () => void;
	}

	let { open = $bindable(), onLoaded }: Props = $props();

	let mode = $state<'local' | 'url'>('url');
	let localPath = $state('');
	let remoteUrl = $state('');
	let assessing = $state(false);
	let assessment = $state<Assessment | null>(null);
	let assessError = $state('');
	let override = $state(false);
	let proceeding = $state(false);

	const source = $derived(mode === 'local' ? localPath.trim() : remoteUrl.trim());

	function reset() {
		mode = 'url';
		localPath = '';
		remoteUrl = '';
		assessing = false;
		assessment = null;
		assessError = '';
		override = false;
		proceeding = false;
	}

	function handleOpenChange(newOpen: boolean) {
		open = newOpen;
		if (!newOpen) reset();
	}

	// Any edit to the source invalidates the last assessment - it was for a different file.
	function invalidate() {
		assessment = null;
		assessError = '';
		override = false;
	}

	// The "Local path" tab is the default, so pasting a URL there (rather than switching tabs
	// first) is an easy mistake - it silently gets sent as source=local and fails opaquely
	// ("failed to open GGUF file '<url>' ... No such file or directory"). Catch it here instead.
	function handleLocalInput() {
		if (/^https?:\/\//i.test(localPath.trim())) {
			remoteUrl = localPath.trim();
			localPath = '';
			mode = 'url';
		}
		invalidate();
	}

	async function assess() {
		if (!source || assessing) return;
		assessing = true;
		assessment = null;
		assessError = '';
		override = false;
		try {
			const resp = await fetch('./model-picker/assess-gguf', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(
					mode === 'local' ? { source: 'local', path: source } : { source: 'url', url: source }
				)
			});
			const body = await resp.json().catch(() => ({}));
			if (!resp.ok) {
				throw new Error(body.error || `Request failed (${resp.status})`);
			}
			assessment = body as Assessment;
		} catch (e) {
			assessError = e instanceof Error ? e.message : String(e);
		} finally {
			assessing = false;
		}
	}

	async function proceed() {
		if (!assessment || proceeding) return;
		if (assessment.fit !== 'fits' && !override) return;
		proceeding = true;
		try {
			await fetch('./model-picker/prepare-download', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					[mode === 'local' ? 'path' : 'url']: source,
					active_gb: assessment.active_gb,
					ctx_size: assessment.n_ctx_train,
					n_layer: assessment.n_layer,
					n_head_kv: assessment.n_head_kv,
					head_dim: assessment.head_dim,
					n_ctx_train: assessment.n_ctx_train,
					sliding_window: assessment.sliding_window,
					sliding_window_pattern: assessment.sliding_window_pattern
				})
			});
			const resp = await fetch('./models', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ model: source, source: mode })
			});
			if (!resp.ok) {
				const body = await resp.json().catch(() => ({}));
				const message = body.error?.message || body.error || `Request failed (${resp.status})`;
				throw new Error(message);
			}
			onLoaded();
			handleOpenChange(false);
		} catch (e) {
			assessError = e instanceof Error ? e.message : String(e);
		} finally {
			proceeding = false;
		}
	}

	const fitLabel: Record<string, string> = {
		fits: 'Fits',
		'too-large': 'Too large for this hardware',
		'no-disk-space': 'Not enough free disk space'
	};

	const canProceed = $derived(
		!!assessment && !proceeding && (assessment.fit === 'fits' || override)
	);
</script>

<Dialog.Root {open} onOpenChange={handleOpenChange}>
	<Dialog.Content class="sm:max-w-md">
		<Dialog.Header>
			<Dialog.Title>Load GGUF</Dialog.Title>
			<Dialog.Description>
				Point at a GGUF file already on disk, or a direct URL to one, and check whether it fits
				this hardware before downloading or loading it.
			</Dialog.Description>
		</Dialog.Header>

		<div class="flex gap-2">
			<Button
				size="sm"
				variant={mode === 'url' ? 'default' : 'outline'}
				onclick={() => {
					mode = 'url';
					invalidate();
				}}
			>
				Direct URL
			</Button>
			<Button
				size="sm"
				variant={mode === 'local' ? 'default' : 'outline'}
				onclick={() => {
					mode = 'local';
					invalidate();
				}}
			>
				Local path
			</Button>
		</div>

		<div class="flex flex-col gap-2">
			<Label for="gguf-source">{mode === 'local' ? 'Path on disk' : 'URL'}</Label>
			{#if mode === 'local'}
				<Input
					id="gguf-source"
					bind:value={localPath}
					oninput={handleLocalInput}
					placeholder="/path/to/model.gguf"
				/>
			{:else}
				<Input
					id="gguf-source"
					bind:value={remoteUrl}
					oninput={invalidate}
					placeholder="https://.../model.gguf"
				/>
			{/if}
			<Button size="sm" variant="outline" disabled={!source || assessing} onclick={assess}>
				{#if assessing}
					<Loader2 class="h-3.5 w-3.5 animate-spin" />
				{/if}
				Assess
			</Button>
		</div>

		{#if assessError}
			<p class="text-sm text-destructive">{assessError}</p>
		{/if}

		{#if assessment}
			<div class="flex flex-col gap-3 rounded-md border border-border/50 p-3 text-sm">
				<div class="flex items-center gap-2">
					<Badge variant={assessment.fit === 'fits' ? 'default' : 'destructive'}>
						{fitLabel[assessment.fit] ?? assessment.fit}
					</Badge>
					<span class="text-muted-foreground">{assessment.arch}</span>
					{#if !assessment.is_moe}
						<Badge variant="destructive">Not MoE</Badge>
					{/if}
					<span class="text-muted-foreground">
						{assessment.is_moe
							? `MoE (${assessment.n_expert_used}/${assessment.n_expert} experts)`
							: 'Dense'}
					</span>
					<span class="text-muted-foreground">{assessment.quant_label}</span>
				</div>
				{#if assessment.is_split}
					<p class="text-xs text-muted-foreground">
						{assessment.shards_found === assessment.split_count
							? `Split GGUF - size aggregated from all ${assessment.split_count} shards.`
							: `Split GGUF - only found ${assessment.shards_found}/${assessment.split_count} shards, size estimate is incomplete.`}
					</p>
				{/if}
				<div class="grid grid-cols-2 gap-x-4 gap-y-1 tabular-nums">
					<span class="text-muted-foreground">Active VRAM</span>
					<span>{assessment.active_gb.toFixed(1)} GB</span>
					<span class="text-muted-foreground">Total size</span>
					<span>{assessment.total_gb.toFixed(1)} GB</span>
					<span class="text-muted-foreground">KV cache</span>
					<span>{assessment.kv_cache_gb.toFixed(1)} GB</span>
					<span class="text-muted-foreground">Context (train)</span>
					<span>{assessment.n_ctx_train.toLocaleString()}</span>
					<span class="text-muted-foreground">VRAM free</span>
					<span>{assessment.vram_free_gb.toFixed(1)} GB</span>
					<span class="text-muted-foreground">Disk free</span>
					<span>{assessment.disk_free_gb.toFixed(1)} GB</span>
				</div>
				{#if assessment.fit !== 'fits'}
					<div class="flex items-center gap-2">
						<Checkbox id="gguf-override" bind:checked={override} />
						<Label for="gguf-override" class="text-xs font-normal">
							Override and proceed anyway
						</Label>
					</div>
				{/if}
			</div>
		{/if}

		<Dialog.Footer>
			<Button variant="outline" onclick={() => handleOpenChange(false)}>Cancel</Button>
			<Button disabled={!canProceed} onclick={proceed}>
				{#if proceeding}
					<Loader2 class="h-3.5 w-3.5 animate-spin" />
				{/if}
				Proceed
			</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
