import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from open_webui.utils.middleware import (
    get_reasoning_format,
    get_non_streaming_follow_up_result,
    process_messages_with_output,
)
from open_webui.utils.misc import convert_output_to_messages, should_preserve_reasoning_content_for_model
from open_webui.utils.payload import (
    NON_VISION_IMAGE_OMITTED_MESSAGE,
    model_supports_vision,
    strip_image_content_for_non_vision_model,
)


NON_VISION_MODEL = {'info': {'meta': {'capabilities': {'vision': False}}}}


def test_model_supports_vision_defaults_to_true_when_unset():
    assert model_supports_vision({}) is True
    assert model_supports_vision({'info': {'meta': {}}}) is True


def test_strip_image_content_for_non_vision_model_keeps_text_only():
    form_data = {
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'What is in this image?'},
                    {'type': 'image_url', 'image_url': {'url': 'https://example.com/cat.png'}},
                ],
            }
        ]
    }

    result = strip_image_content_for_non_vision_model(form_data, NON_VISION_MODEL)

    assert result is form_data
    assert form_data['messages'][0]['content'] == 'What is in this image?'


def test_strip_image_content_for_non_vision_model_replaces_image_only_message():
    form_data = {
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,abc'}},
                ],
            }
        ]
    }

    strip_image_content_for_non_vision_model(form_data, NON_VISION_MODEL)

    assert form_data['messages'][0]['content'] == NON_VISION_IMAGE_OMITTED_MESSAGE


def test_strip_image_content_for_non_vision_model_handles_input_image_parts():
    form_data = {
        'messages': [
            {
                'role': 'tool',
                'content': [
                    {'type': 'input_text', 'text': 'Tool returned a chart.'},
                    {'type': 'input_image', 'image_url': 'data:image/png;base64,abc'},
                ],
            }
        ]
    }

    strip_image_content_for_non_vision_model(form_data, NON_VISION_MODEL)

    assert form_data['messages'][0]['content'] == 'Tool returned a chart.'


def test_strip_image_content_for_non_vision_model_leaves_vision_payload_unchanged():
    form_data = {
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Describe this.'},
                    {'type': 'image_url', 'image_url': {'url': 'https://example.com/image.png'}},
                ],
            }
        ]
    }

    result = strip_image_content_for_non_vision_model(form_data, {'info': {'meta': {}}})

    assert result is form_data
    assert form_data['messages'][0]['content'] == [
        {'type': 'text', 'text': 'Describe this.'},
        {'type': 'image_url', 'image_url': {'url': 'https://example.com/image.png'}},
    ]


def test_strip_image_content_for_non_vision_model_is_idempotent():
    form_data = {
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'output_text', 'text': 'Prior result.'},
                    {'type': 'input_image', 'image_url': 'data:image/png;base64,abc'},
                ],
            }
        ]
    }

    strip_image_content_for_non_vision_model(form_data, NON_VISION_MODEL)
    first_result = form_data['messages'][0]['content']
    strip_image_content_for_non_vision_model(form_data, NON_VISION_MODEL)

    assert form_data['messages'][0]['content'] == first_result == 'Prior result.'


def test_convert_output_to_messages_preserves_reasoning_content_for_tool_call_replay():
    output = [
        {
            'type': 'reasoning',
            'attributes': {'type': 'reasoning_content'},
            'start_tag': '<think>',
            'end_tag': '</think>',
            'content': [{'type': 'output_text', 'text': 'Need to search.'}],
        },
        {
            'type': 'function_call',
            'call_id': 'call_1',
            'name': 'search_web',
            'arguments': '{"query": "OpenClaw"}',
        },
        {
            'type': 'function_call_output',
            'call_id': 'call_1',
            'output': [{'type': 'input_text', 'text': '[{"title":"OpenClaw"}]'}],
        },
    ]

    assert convert_output_to_messages(output, raw=True, reasoning_format='reasoning_content') == [
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [
                {
                    'id': 'call_1',
                    'type': 'function',
                    'function': {
                        'name': 'search_web',
                        'arguments': '{"query": "OpenClaw"}',
                    },
                }
            ],
            'reasoning_content': 'Need to search.',
        },
        {
            'role': 'tool',
            'tool_call_id': 'call_1',
            'content': '[{"title":"OpenClaw"}]',
        },
    ]

    assert convert_output_to_messages(output, raw=True, reasoning_format='think_tags')[0] == {
        'role': 'assistant',
        'content': '<think>Need to search.</think>',
        'tool_calls': [
            {
                'id': 'call_1',
                'type': 'function',
                'function': {
                    'name': 'search_web',
                    'arguments': '{"query": "OpenClaw"}',
                },
            }
        ],
    }


def test_should_preserve_reasoning_content_for_deepseek_v4_and_later_models():
    assert should_preserve_reasoning_content_for_model('deepseek-v4') is True
    assert should_preserve_reasoning_content_for_model('DeepSeek-V4-Pro') is True
    assert should_preserve_reasoning_content_for_model('deepseek-v4.1') is True
    assert should_preserve_reasoning_content_for_model('deepseek-v5') is True
    assert should_preserve_reasoning_content_for_model('new-api.deepseek-v4-pro') is True
    assert (
        should_preserve_reasoning_content_for_model(
            'custom-deepseek',
            {
                'id': 'custom-deepseek',
                'info': {'base_model_id': 'deepseek-v4-pro'},
            },
        )
        is True
    )

    assert should_preserve_reasoning_content_for_model('deepseek-v3.9') is False


def test_should_preserve_reasoning_content_for_mimo_v25_and_later_models():
    assert should_preserve_reasoning_content_for_model('mimo-v2.5') is True
    assert should_preserve_reasoning_content_for_model('mimo-v2.5-pro') is True
    assert should_preserve_reasoning_content_for_model('mimo-v2.5-preview') is True
    assert should_preserve_reasoning_content_for_model('mimo-v2.6') is True
    assert should_preserve_reasoning_content_for_model('mimo-v3') is True
    assert should_preserve_reasoning_content_for_model('new-api.mimo-v2.5-pro') is True
    assert should_preserve_reasoning_content_for_model('custom-mimo', {'info': None}) is False
    assert should_preserve_reasoning_content_for_model('mimo-v1') is False
    assert should_preserve_reasoning_content_for_model('mimo-v2.4') is False
    assert should_preserve_reasoning_content_for_model('mimo-anything') is False


def test_should_preserve_reasoning_content_for_kimi_k25_and_later_models():
    assert should_preserve_reasoning_content_for_model('kimi-k2.5') is True
    assert should_preserve_reasoning_content_for_model('kimi-k2.6') is True
    assert should_preserve_reasoning_content_for_model('Kimi K2.6') is True
    assert should_preserve_reasoning_content_for_model('moonshot/kimi-k2.6-20260420') is True
    assert (
        should_preserve_reasoning_content_for_model(
            'custom-kimi',
            {
                'id': 'custom-kimi',
                'info': {'base_model_id': 'kimi-k2.6'},
            },
        )
        is True
    )

    assert should_preserve_reasoning_content_for_model('kimi-k2.4') is False
    assert should_preserve_reasoning_content_for_model('kimi-k2-thinking') is False
    assert should_preserve_reasoning_content_for_model('kimi-k2-0905-preview') is False


def test_should_preserve_reasoning_content_from_model_capability():
    assert (
        should_preserve_reasoning_content_for_model(
            'custom-model',
            {
                'info': {
                    'meta': {
                        'capabilities': {
                            'reasoning_content': True,
                        },
                    },
                },
            },
        )
        is True
    )


def test_get_reasoning_format_selects_provider_safe_replay_format():
    assert get_reasoning_format({'provider': 'ollama'}, 'deepseek-v4') == 'think_tags'
    assert get_reasoning_format({'provider': 'llama.cpp'}, 'kimi-k2.5') == 'reasoning_content'
    assert get_reasoning_format({}, 'deepseek-v4') == 'reasoning_content'
    assert get_reasoning_format({}, 'deepseek-v5') == 'reasoning_content'
    assert get_reasoning_format({}, 'mimo-v2.5') == 'reasoning_content'
    assert get_reasoning_format({}, 'mimo-v2.5-pro') == 'reasoning_content'
    assert get_reasoning_format({}, 'mimo-v2.6') == 'reasoning_content'
    assert get_reasoning_format({}, 'kimi-k2.5') == 'reasoning_content'
    assert get_reasoning_format({}, 'kimi-k2.6') == 'reasoning_content'
    assert get_reasoning_format({}, 'deepseek-v3.9') is None
    assert get_reasoning_format({}, 'mimo-v1') is None


def test_process_messages_with_output_preserves_deepseek_v4_final_tool_turn_reasoning():
    output = [
        {
            'type': 'reasoning',
            'attributes': {'type': 'reasoning_content'},
            'start_tag': '<think>',
            'end_tag': '</think>',
            'content': [{'type': 'output_text', 'text': 'Need to search.'}],
        },
        {
            'type': 'function_call',
            'call_id': 'call_1',
            'name': 'web_search',
            'arguments': '{"query": "toothache"}',
        },
        {
            'type': 'function_call_output',
            'call_id': 'call_1',
            'output': [{'type': 'input_text', 'text': '[{"title":"Toothache"}]'}],
        },
        {
            'type': 'reasoning',
            'attributes': {'type': 'reasoning_content'},
            'start_tag': '<think>',
            'end_tag': '</think>',
            'content': [{'type': 'output_text', 'text': 'Use search result to answer.'}],
        },
        {
            'type': 'message',
            'content': [{'type': 'output_text', 'text': 'See a dentist if symptoms persist.'}],
        },
    ]

    messages = [
        {'role': 'assistant', 'content': '', 'output': output},
        {'role': 'user', 'content': 'Can it heal itself?'},
    ]

    default_result = process_messages_with_output(messages)
    assert 'reasoning_content' not in default_result[0]
    assert 'reasoning_content' not in default_result[2]

    deepseek_result = process_messages_with_output(messages, reasoning_format='reasoning_content')
    assert deepseek_result == [
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [
                {
                    'id': 'call_1',
                    'type': 'function',
                    'function': {
                        'name': 'web_search',
                        'arguments': '{"query": "toothache"}',
                    },
                }
            ],
            'reasoning_content': 'Need to search.',
        },
        {
            'role': 'tool',
            'tool_call_id': 'call_1',
            'content': '[{"title":"Toothache"}]',
        },
        {
            'role': 'assistant',
            'content': 'See a dentist if symptoms persist.',
            'reasoning_content': 'Use search result to answer.',
        },
        {'role': 'user', 'content': 'Can it heal itself?'},
    ]

    ollama_result = process_messages_with_output(messages, reasoning_format='think_tags')
    assert ollama_result[0]['content'] == '<think>Need to search.</think>'
    assert 'reasoning_content' not in ollama_result[0]
    assert (
        ollama_result[2]['content'] == '<think>Use search result to answer.</think>\nSee a dentist if symptoms persist.'
    )
    assert 'reasoning_content' not in ollama_result[2]


def test_get_non_streaming_follow_up_result_extracts_json_response_error():
    response = {
        'error': {
            'message': 'thinking is enabled but reasoning_content is missing',
        }
    }

    assert get_non_streaming_follow_up_result(response) == (
        None,
        'thinking is enabled but reasoning_content is missing',
    )


def test_convert_output_to_messages_skips_orphan_tool_output():
    output = [
        {
            'type': 'function_call_output',
            'call_id': 'call_orphan',
            'output': [{'type': 'input_text', 'text': '{"status":"ok"}'}],
        },
        {
            'type': 'message',
            'content': [{'type': 'output_text', 'text': 'Final answer'}],
        },
    ]

    assert convert_output_to_messages(output, raw=True) == [
        {
            'role': 'assistant',
            'content': 'Final answer',
        }
    ]


def test_process_messages_with_output_skips_orphan_tool_output():
    messages = [
        {
            'role': 'assistant',
            'content': 'Final answer',
            'output': [
                {
                    'type': 'function_call_output',
                    'call_id': 'call_orphan',
                    'output': [{'type': 'input_text', 'text': '{"status":"ok"}'}],
                },
                {
                    'type': 'message',
                    'content': [{'type': 'output_text', 'text': 'Final answer'}],
                },
            ],
        },
        {
            'role': 'user',
            'content': 'chatgpt 5.4',
        },
    ]

    assert process_messages_with_output(messages) == [
        {
            'role': 'assistant',
            'content': 'Final answer',
        },
        {
            'role': 'user',
            'content': 'chatgpt 5.4',
        },
    ]


@pytest.mark.parametrize('follow_up', ['empty', 'stream', 'json', 'error', 'exception', 'limit'])
def test_tool_loop_compaction_finishes_without_forced_synthesis(monkeypatch, follow_up):
    """Preserve compacted tool history and stop without an extra synthesis request."""
    import asyncio
    import copy
    import json
    from types import SimpleNamespace

    from fastapi import HTTPException, Request
    from starlette.responses import JSONResponse, StreamingResponse

    from open_webui.utils import middleware

    requests = []
    emitted = []
    compacted_inputs = []

    def stream(*deltas, usage=None):
        async def body():
            for delta in deltas:
                yield 'data: ' + json.dumps({'choices': [{'delta': delta}]}) + '\n\n'
            if usage:
                yield 'data: ' + json.dumps({'choices': [], 'usage': usage}) + '\n\n'
            yield 'data: [DONE]\n\n'

        return StreamingResponse(body(), media_type='text/event-stream')

    def tool_delta(call_id):
        return {
            'tool_calls': [
                {'index': 0, 'id': call_id, 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}
            ]
        }

    async def noop(*args, **kwargs):
        return None

    async def emit(event):
        emitted.append(copy.deepcopy(event))

    async def generate(request, form_data, user, **kwargs):
        requests.append(copy.deepcopy(form_data))
        if len(requests) == 1:
            return stream(
                {'reasoning_content': 'second round reasoning'},
                tool_delta('call_2'),
                usage={'prompt_tokens': 200, 'completion_tokens': 20, 'total_tokens': 220},
            )
        assert len(requests) == 2, 'Unexpected extra model request after the tool continuation'
        if follow_up == 'stream':
            return stream({'content': 'Final answer from tool results.'})
        if follow_up == 'json':
            return JSONResponse(
                {'choices': [{'message': {'role': 'assistant', 'content': 'Final answer from tool results.'}}]}
            )
        if follow_up == 'error':
            return JSONResponse({'error': {'message': 'Provider unavailable'}}, status_code=502)
        if follow_up == 'exception':
            raise HTTPException(status_code=503, detail='Provider unavailable')
        return stream()

    async def execute(*args):
        return {}, 'lookup result', {'type': 'builtin'}, 'builtin', False

    async def process_result(request, name, result, *args):
        return result, [], []

    async def compact(request, user, messages, *args):
        compacted_inputs.append(copy.deepcopy(messages))
        if len(compacted_inputs) == 1:
            system_messages = [message for message in messages if message.get('role') == 'system']
            return [*system_messages, {'role': 'user', 'content': 'checkpoint'}], 'Earlier lookup completed.', True
        return messages, None, False

    monkeypatch.setattr(middleware, 'ENABLE_FUNCTIONS', False)
    monkeypatch.setattr(middleware, 'ENABLE_RESPONSES_API_STATEFUL', False)
    monkeypatch.setattr(middleware, 'get_system_oauth_token', noop)
    monkeypatch.setattr(middleware, 'generate_chat_completion', generate)
    monkeypatch.setattr(middleware, 'execute_tool_call', execute)
    monkeypatch.setattr(middleware, 'process_tool_result', process_result)
    monkeypatch.setattr(middleware, 'terminal_event_handler', noop)
    monkeypatch.setattr(middleware, 'compact_messages_for_request', compact)
    monkeypatch.setattr(middleware, 'clear_response_stream', noop)
    monkeypatch.setattr(middleware, 'publish_chat_finished_event', noop)
    monkeypatch.setattr(middleware, 'outlet_filter_handler', noop)
    monkeypatch.setattr(middleware, 'background_tasks_handler', noop)

    model = {'id': 'kimi-k2.5'}
    app = SimpleNamespace(state=SimpleNamespace(redis=None, MODELS={model['id']: model}))
    request = Request({'type': 'http', 'app': app, 'headers': []})
    request.state.max_tool_call_iterations = 1 if follow_up == 'limit' else 5
    ctx = {
        'request': request,
        'form_data': {
            'model': model['id'],
            'messages': [
                {'role': 'system', 'content': 'base system'},
                {'role': 'user', 'content': 'old history to compact'},
                {'role': 'user', 'content': 'Find the answer.'},
            ],
            'tools': [{'type': 'function', 'function': {'name': 'lookup'}}],
            'tool_choice': 'auto',
        },
        'user': SimpleNamespace(id='test-user', role='admin'),
        'model': model,
        'metadata': {'chat_id': '', 'params': {'function_calling': 'native', 'tool_approval_mode': 'full'}},
        'events': [],
        'event_emitter': emit,
        'event_caller': None,
        'tasks': {},
    }
    asyncio.run(
        middleware.streaming_chat_response_handler(
            stream(
                {'reasoning_content': 'first round reasoning'},
                tool_delta('call_1'),
                usage={'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110},
            ),
            ctx,
        )
    )

    assert len(requests) == (1 if follow_up == 'limit' else 2)
    assert compacted_inputs[0][-2]['reasoning_content'] == 'first round reasoning'
    assert compacted_inputs[0][-2]['usage']['total_tokens'] == 110
    last_request = requests[-1]
    assert last_request['tools'] == ctx['form_data']['tools']
    assert last_request['tool_choice'] == 'auto'
    assert last_request['messages'][0]['role'] == 'system'
    assert last_request['messages'][0]['content'] == 'base system\n[CONVERSATION SUMMARY]\nEarlier lookup completed.'
    assert any('Earlier lookup completed.' in str(message['content']) for message in last_request['messages'])
    assert not any('old history to compact' in str(message['content']) for message in last_request['messages'])
    if follow_up != 'limit':
        assert compacted_inputs[1][-2]['usage']['total_tokens'] == 220
        assert last_request['messages'][-2]['reasoning_content'] == 'second round reasoning'
        assert [message['tool_call_id'] for message in last_request['messages'] if message['role'] == 'tool'] == [
            'call_2'
        ]

    output = ctx['assistant_message']['output']
    expected_calls = ['call_1'] if follow_up == 'limit' else ['call_1', 'call_2']
    assert [item['call_id'] for item in output if item['type'] == 'function_call_output'] == expected_calls
    assert any(event['type'] == 'chat:completion' and event['data'].get('done') for event in emitted)
    answer = ''.join(
        part.get('text', '') for item in output if item['type'] == 'message' for part in item.get('content', [])
    )
    errors = [event['data']['error']['content'] for event in emitted if event['type'] == 'chat:message:error']
    if follow_up in {'stream', 'json'}:
        assert answer == 'Final answer from tool results.'
    else:
        assert answer == ''
    if follow_up in {'error', 'exception'}:
        assert errors == ['Provider unavailable']
    elif follow_up == 'limit':
        assert errors == ['Tool-call limit reached (1 iterations).']
    else:
        assert errors == []
