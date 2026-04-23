const JSON_CONTENT_TYPE = 'application/json';

const normalizeText = (text: string) =>
	text
		.replace(/<style[\s\S]*?<\/style>/gi, ' ')
		.replace(/<script[\s\S]*?<\/script>/gi, ' ')
		.replace(/<[^>]+>/g, ' ')
		.replace(/\s+/g, ' ')
		.trim();

const buildErrorResponse = (res: Response, text: string) => {
	const contentType = (res.headers.get('content-type') ?? '').toLowerCase();

	if (text) {
		try {
			return JSON.parse(text);
		} catch {
			// Fall back to human-readable text normalization below.
		}
	}

	const normalizedText = normalizeText(text);

	if (normalizedText) {
		return {
			detail: contentType.includes('text/html')
				? `Server returned HTML instead of JSON (HTTP ${res.status}). ${normalizedText.slice(0, 240)}`
				: normalizedText.slice(0, 240)
		};
	}

	return {
		detail: `Request failed with HTTP ${res.status}${res.statusText ? ` ${res.statusText}` : ''}`
	};
};

export const parseErrorResponse = async (res: Response) => {
	const text = await res.text().catch(() => '');
	return buildErrorResponse(res, text);
};

export const parseJsonResponse = async <T = any>(res: Response): Promise<T> => {
	const text = await res.text().catch(() => '');

	let parsedJson: { ok: true; value: T } | { ok: false } = { ok: false };
	if (text) {
		try {
			parsedJson = { ok: true, value: JSON.parse(text) as T };
		} catch {
			parsedJson = { ok: false };
		}
	}

	if (!res.ok) {
		throw parsedJson.ok ? parsedJson.value : buildErrorResponse(res, text);
	}

	if (parsedJson.ok) {
		return parsedJson.value;
	}

	if (!text && (res.headers.get('content-type') ?? '').toLowerCase().includes(JSON_CONTENT_TYPE)) {
		return null as T;
	}

	throw buildErrorResponse(res, text);
};
