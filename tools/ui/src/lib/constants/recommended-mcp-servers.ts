import { DEFAULT_MCP_CONFIG } from './mcp';
import type { RecommendedMCPServer } from '$lib/types';

/**
 * Pre-defined recommended MCP servers.
 *
 * Servers are enabled by default, but they are not turned on for individual
 * conversations until the user explicitly enables them (so their tools are
 * disabled by default).
 */
export const RECOMMENDED_MCP_SERVERS: RecommendedMCPServer[] = [
	{
		id: 'exa-web-search',
		name: 'Exa Web Search',
		description: 'Search the web and retrieve relevant content.',
		url: 'https://mcp.exa.ai/mcp',
		enabled: true,
		requestTimeoutSeconds: DEFAULT_MCP_CONFIG.requestTimeoutSeconds
	},
	{
		id: 'huggingface-mcp',
		name: 'Hugging Face',
		description:
			'Browse models, datasets, spaces and machine learning papers from the Hugging Face hub.',
		url: 'https://huggingface.co/mcp',
		enabled: true,
		requestTimeoutSeconds: DEFAULT_MCP_CONFIG.requestTimeoutSeconds
	},
	{
		id: 'metasploit-mcp',
		name: 'Metasploit',
		description:
			'Local Metasploit bridge (metasploit-mcp.service) - modules, sessions, and exploitation via msfrpcd.',
		url: 'http://127.0.0.1:8085/sse',
		enabled: true,
		requestTimeoutSeconds: DEFAULT_MCP_CONFIG.requestTimeoutSeconds,
		// MetasploitMCP is a plain local Python server with no Access-Control-Allow-Origin
		// header, unlike the hosted mcp.exa.ai / huggingface.co entries above - a direct
		// browser fetch/EventSource to it is cross-origin and gets killed by CORS before
		// the request even reaches the server ("Failed to fetch"). Routing through
		// llama-server's own CORS proxy (same-origin as the webui) sidesteps that.
		useProxy: true
	}
];

export const RECOMMENDED_MCP_SERVER_IDS = new Set(
	RECOMMENDED_MCP_SERVERS.map((server) => server.id)
);

export const RECOMMENDED_MCP_SERVERS_OPTIN_DIALOG_DELAY = 1000;
