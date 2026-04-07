import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from open_webui.utils.middleware import (
    TOOL_FALLBACK_SYNTHESIS_PROMPT,
    build_native_tool_follow_up_form_data,
    output_has_assistant_message_after_last_tool_output,
    process_messages_with_output,
)
from open_webui.utils.misc import convert_output_to_messages


def test_output_has_assistant_message_after_last_tool_output_detects_missing_final_answer():
    output = [
        {
            'type': 'message',
            'content': [{'type': 'output_text', 'text': 'Let me check that.'}],
        },
        {
            'type': 'function_call',
            'call_id': 'call_1',
            'name': 'search_web',
            'arguments': '{}',
        },
        {
            'type': 'function_call_output',
            'call_id': 'call_1',
            'output': [{'type': 'input_text', 'text': '{"status":"ok"}'}],
        },
    ]

    assert output_has_assistant_message_after_last_tool_output(output) is False


def test_output_has_assistant_message_after_last_tool_output_accepts_post_tool_answer():
    output = [
        {
            'type': 'function_call_output',
            'call_id': 'call_1',
            'output': [{'type': 'input_text', 'text': '{"status":"ok"}'}],
        },
        {
            'type': 'reasoning',
            'content': [{'type': 'output_text', 'text': 'thinking'}],
        },
        {
            'type': 'message',
            'content': [{'type': 'output_text', 'text': 'Add water first, then the tablet.'}],
        },
    ]

    assert output_has_assistant_message_after_last_tool_output(output) is True


def test_build_native_tool_follow_up_form_data_creates_stateless_tool_free_fallback():
    form_data = {
        'model': 'kimi-k2.5',
        'stream': True,
        'messages': [
            {'role': 'system', 'content': 'base system'},
            {'role': 'user', 'content': 'How do I use the cleaning tablet?'},
        ],
        'tools': [{'type': 'function', 'function': {'name': 'search_web'}}],
        'tool_choice': 'auto',
        'previous_response_id': 'resp_123',
    }
    output = [
        {
            'type': 'function_call',
            'call_id': 'call_1',
            'name': 'search_web',
            'arguments': '{}',
        },
        {
            'type': 'function_call_output',
            'call_id': 'call_1',
            'output': [{'type': 'input_text', 'text': '{"status":"ok"}'}],
        },
    ]

    result = build_native_tool_follow_up_form_data(
        form_data,
        'kimi-k2.5',
        output,
        last_response_id='resp_123',
        force_stateless=True,
        disallow_tools=True,
        extra_system_instruction=TOOL_FALLBACK_SYNTHESIS_PROMPT,
    )

    assert result['model'] == 'kimi-k2.5'
    assert result['stream'] is True
    assert 'tools' not in result
    assert 'tool_choice' not in result
    assert 'previous_response_id' not in result
    assert TOOL_FALLBACK_SYNTHESIS_PROMPT in result['messages'][0]['content']
    assert [message['role'] for message in result['messages']] == [
        'system',
        'user',
        'assistant',
        'tool',
    ]


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
