<script lang="ts">
	import { Slider as SliderPrimitive } from 'bits-ui';
	import { cn, type WithoutChildrenOrChild } from '$lib/components/ui/utils.js';

	type SingleRootProps = Extract<
		WithoutChildrenOrChild<SliderPrimitive.RootProps>,
		{ type: 'single' }
	>;

	let {
		ref = $bindable(null),
		value = $bindable(),
		orientation = 'horizontal',
		class: className,
		rangeClass,
		thumbClass,
		...restProps
	}: SingleRootProps & {
		rangeClass?: string;
		thumbClass?: string;
	} = $props();
</script>

<SliderPrimitive.Root
	bind:ref
	bind:value
	{orientation}
	class={cn(
		"relative flex w-full touch-none items-center select-none data-[disabled]:opacity-50 data-[orientation='vertical']:h-full data-[orientation='vertical']:min-h-44 data-[orientation='vertical']:w-auto data-[orientation='vertical']:flex-col",
		className
	)}
	{...restProps}
>
	{#snippet children({ thumbItems })}
		<span
			data-orientation={orientation}
			class="relative grow overflow-hidden rounded-full bg-muted data-[orientation='horizontal']:h-1.5 data-[orientation='horizontal']:w-full data-[orientation='vertical']:h-full data-[orientation='vertical']:w-1.5"
		>
			<SliderPrimitive.Range
				class={cn(
					"absolute data-[orientation='horizontal']:h-full data-[orientation='vertical']:w-full",
					rangeClass ?? 'bg-primary'
				)}
			/>
		</span>
		{#each thumbItems as { index } (index)}
			<SliderPrimitive.Thumb
				{index}
				class={cn(
					'block size-4 shrink-0 rounded-full border bg-background shadow-sm ring-ring/50 transition-[color,box-shadow] hover:ring-4 focus-visible:ring-4 focus-visible:outline-hidden disabled:pointer-events-none disabled:opacity-50',
					thumbClass ?? 'border-primary'
				)}
			/>
		{/each}
	{/snippet}
</SliderPrimitive.Root>
