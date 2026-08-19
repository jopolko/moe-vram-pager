/**
 * Normalizes Unicode punctuation/whitespace that small local models routinely reach
 * for (em/en dashes, non-breaking hyphens, curly quotes, non-breaking spaces) down to
 * plain ASCII, regardless of prompt instructions. Mirrors _UNICODE_PUNCT_MAP /
 * _sanitize_llm_text in tools/pentest_agent.py - keep the two in sync.
 *
 * Every key is written as an explicit \u escape rather than a raw literal - several of
 * these codepoints are visually identical or near-identical to each other and to plain
 * ASCII in an editor, so a raw literal risks silently being the wrong character.
 */
const PUNCT_MAP: Record<string, string> = {
	'\u2014': '-', // em dash
	'\u2013': '-', // en dash
	'\u2011': '-', // non-breaking hyphen
	'\u2012': '-', // figure dash
	'\u2015': '-', // horizontal bar
	'\u2212': '-', // minus sign
	'\u2018': "'", // left single quote
	'\u2019': "'", // right single quote
	'\u201c': '"', // left double quote
	'\u201d': '"', // right double quote
	'\u2026': '...', // ellipsis
	'\u00a0': ' ', // non-breaking space
	'\u2000': ' ', // en quad
	'\u2001': ' ', // em quad
	'\u2002': ' ', // en space
	'\u2003': ' ', // em space
	'\u2004': ' ', // three-per-em space
	'\u2005': ' ', // four-per-em space
	'\u2006': ' ', // six-per-em space
	'\u2007': ' ', // figure space
	'\u2008': ' ', // punctuation space
	'\u2009': ' ', // thin space
	'\u200a': ' ', // hair space
	'\u202f': ' ', // narrow no-break space
	'\u200b': '', // zero-width space
	'\ufeff': '', // BOM / zero-width no-break space
};

const PUNCT_PATTERN = new RegExp(`[${Object.keys(PUNCT_MAP).join('')}]`, 'g');

export function sanitizeLlmText(text: string): string {
	if (!text) return text;
	return text.replace(PUNCT_PATTERN, (ch) => PUNCT_MAP[ch] ?? ch);
}
