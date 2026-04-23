import { describe, expect, it } from 'vitest';

import { parseJsonResponse } from './request';

describe('parseJsonResponse', () => {
	it('returns null for a valid JSON null body', async () => {
		const response = new Response('null', {
			status: 200,
			headers: {
				'content-type': 'application/json'
			}
		});

		await expect(parseJsonResponse(response)).resolves.toBeNull();
	});

	it('returns null for an empty JSON response body', async () => {
		const response = new Response('', {
			status: 200,
			headers: {
				'content-type': 'application/json'
			}
		});

		await expect(parseJsonResponse(response)).resolves.toBeNull();
	});
});
