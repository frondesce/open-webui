import { describe, expect, it } from 'vitest';
import { marked } from 'marked';

import markedExtension from './extension';

marked.use(markedExtension());

describe('marked details extension', () => {
	it('tokenizes attributed details blocks after normal text', () => {
		const tokens = marked.lexer(
			[
				'Method one: use Python pandas',
				'<details type="tool_calls" done="true" id="functions.execute_code:0">',
				'<summary>Tool Executed</summary>',
				'</details>',
				'Method two: use Excel manually'
			].join('\n')
		);

		expect(tokens[0]).toMatchObject({
			type: 'paragraph',
			text: 'Method one: use Python pandas'
		});
		expect(tokens[1]).toMatchObject({
			type: 'details',
			summary: 'Tool Executed',
			text: '',
			attributes: {
				type: 'tool_calls',
				done: 'true',
				id: 'functions.execute_code:0'
			}
		});
		expect(tokens[2]).toMatchObject({
			type: 'paragraph',
			text: 'Method two: use Excel manually'
		});
	});
});
