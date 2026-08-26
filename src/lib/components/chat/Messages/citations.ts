export const isHttpUrl = (value: unknown): value is string => {
	return typeof value === 'string' && (value.startsWith('http://') || value.startsWith('https://'));
};

const shouldUseHostname = (name: unknown) => {
	if (name === null || name === undefined || typeof name === 'number') {
		return true;
	}

	return typeof name === 'string' && (name.trim() === '' || /^\d+$/.test(name.trim()));
};

export const getCitationUrl = (citation: any): string => {
	const sourceUrl = citation?.source?.url;
	if (isHttpUrl(sourceUrl)) {
		return sourceUrl;
	}

	const id = citation?.id;
	if (isHttpUrl(id)) {
		return id;
	}

	return '';
};

export const normalizeCitationSource = (source: any, metadata: any, id: unknown) => {
	let citationSource = source?.source;

	if (metadata?.name) {
		citationSource = { ...citationSource, name: metadata.name };
	}

	if (isHttpUrl(id)) {
		let name = citationSource?.name || id;

		if (shouldUseHostname(citationSource?.name)) {
			try {
				name = new URL(id).hostname || name;
			} catch {
				// Keep the existing fallback when the URL cannot be parsed.
			}
		}

		citationSource = {
			...citationSource,
			name,
			url: id
		};
	}

	return citationSource;
};
