import { ReasoningEffort } from '$lib/enums';
import type { ReasoningEffortLevel } from '$lib/types';

/**
 * Reasoning effort UI labels.
 * Keys match the ReasoningEffort enum values for type-safe lookups.
 */
export const REASONING_EFFORT_LABELS: Record<string, string> = {
	[ReasoningEffort.LOW]: 'Low',
	[ReasoningEffort.MEDIUM]: 'Medium',
	[ReasoningEffort.HIGH]: 'High',
	[ReasoningEffort.MAX]: 'Max'
};

export const REASONING_EFFORT_LEVELS: ReasoningEffortLevel[] = [
	{ value: 'off', label: 'Off', isOff: true },
	{ value: ReasoningEffort.LOW, label: 'Low' },
	{ value: ReasoningEffort.MEDIUM, label: 'Medium' },
	{ value: ReasoningEffort.HIGH, label: 'High' },
	{ value: ReasoningEffort.MAX, label: 'Max', hasInfo: true }
];

/**
 * Reasoning effort levels for "levels"-style templates (e.g. gpt-oss/harmony), which only
 * accept low/medium/high and have no true "off" or level above "high".
 */
export const REASONING_EFFORT_LEVELS_HARMONY: ReasoningEffortLevel[] = [
	{ value: ReasoningEffort.LOW, label: 'Low' },
	{ value: ReasoningEffort.MEDIUM, label: 'Medium' },
	{ value: ReasoningEffort.HIGH, label: 'High' }
];
