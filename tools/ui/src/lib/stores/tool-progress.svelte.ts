/**
 * Live progress for in-flight MCP tool calls (e.g. nmap_scan's percent/ETA),
 * keyed by tool_call_id. Only populated for servers/tools that actually send
 * MCP progress notifications - most tool calls never touch this and just
 * show the existing static "executing..." state.
 */
export interface ToolProgressState {
	progress: number;
	total?: number;
	message?: string;
}

class ToolProgressStore {
	private entries = $state<Record<string, ToolProgressState>>({});

	set(toolCallId: string, state: ToolProgressState) {
		this.entries[toolCallId] = state;
	}

	clear(toolCallId: string) {
		delete this.entries[toolCallId];
	}

	get(toolCallId: string): ToolProgressState | undefined {
		return this.entries[toolCallId];
	}
}

export const toolProgressStore = new ToolProgressStore();
