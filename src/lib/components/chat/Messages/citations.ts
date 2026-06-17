export const isHttpUrl = (value: unknown): value is string => {
	return typeof value === 'string' && (value.startsWith('http://') || value.startsWith('https://'));
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
		citationSource = {
			...citationSource,
			name: citationSource?.name || id,
			url: id
		};
	}

	return citationSource;
};
