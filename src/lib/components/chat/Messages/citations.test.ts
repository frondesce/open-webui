import { describe, expect, it } from 'vitest';

import { getCitationUrl, normalizeCitationSource } from './citations';

describe('normalizeCitationSource', () => {
	it('keeps metadata.name as the display name when metadata.source is a URL', () => {
		const url = 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/example-long-url';

		const source = normalizeCitationSource(
			{
				source: {
					name: 'Source name',
					url
				}
			},
			{
				source: url,
				name: 'Example Title'
			},
			url
		);

		expect(source.name).toBe('Example Title');
		expect(source.url).toBe(url);
	});

	it('keeps source.name when URL metadata has no name', () => {
		const url = 'https://example.com/page';

		const source = normalizeCitationSource(
			{
				source: {
					name: 'Example Domain'
				}
			},
			{
				source: url
			},
			url
		);

		expect(source.name).toBe('Example Domain');
		expect(source.url).toBe(url);
	});

	it('falls back to the hostname when there is no display name', () => {
		const url = 'https://example.com/fallback';

		const source = normalizeCitationSource({}, { source: url }, url);

		expect(source.name).toBe('example.com');
		expect(source.url).toBe(url);
	});

	it('falls back to the hostname when the display name is numeric or blank', () => {
		const url = 'https://news.example.com/article';

		for (const name of ['2', '  42  ', '   ']) {
			const source = normalizeCitationSource({}, { source: url, name }, url);

			expect(source.name).toBe('news.example.com');
			expect(source.url).toBe(url);
		}
	});

	it('keeps the existing fallback when the URL cannot be parsed', () => {
		const url = 'https://';

		const source = normalizeCitationSource({}, { source: url, name: '2' }, url);

		expect(source.name).toBe('2');
		expect(source.url).toBe(url);
	});
});

describe('getCitationUrl', () => {
	it('reads the URL from source.url before falling back to id', () => {
		const url = 'https://example.com/source';
		const id = 'https://example.com/id';

		expect(
			getCitationUrl({
				id,
				source: {
					name: 'Example',
					url
				}
			})
		).toBe(url);
	});
});
