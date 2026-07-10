import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from open_webui.utils.payload import (
    append_tools_to_payload,
    apply_model_params_to_body_openai,
    pop_tools_from_params,
)


def test_pop_tools_from_params_prefers_custom_params_tools():
    params = {
        'temperature': 0.2,
        'tools': [{'type': 'function', 'function': {'name': 'old_tool'}}],
        'custom_params': {
            'tools': '[{"type":"openrouter:web_search"}]',
            'extra_body': {'foo': 'bar'},
        },
    }

    stripped_params, tools = pop_tools_from_params(params)

    assert tools == [{'type': 'openrouter:web_search'}]
    assert 'tools' not in stripped_params
    assert stripped_params['custom_params'] == {'extra_body': {'foo': 'bar'}}
    assert params['custom_params']['tools'] == '[{"type":"openrouter:web_search"}]'


def test_append_tools_to_payload_extends_existing_tools():
    payload = {
        'tools': [
            {'type': 'function', 'function': {'name': 'get_current_timestamp'}},
        ]
    }

    append_tools_to_payload(payload, [{'type': 'openrouter:web_search'}])

    assert payload['tools'] == [
        {'type': 'function', 'function': {'name': 'get_current_timestamp'}},
        {'type': 'openrouter:web_search'},
    ]


def test_append_tools_to_payload_deduplicates_identical_tools():
    payload = {
        'tools': [
            {'type': 'web_search'},
            {'type': 'x_search'},
        ]
    }

    append_tools_to_payload(
        payload,
        [
            {'type': 'web_search'},
            {'type': 'x_search'},
            {'type': 'web_search'},
        ],
    )

    assert payload['tools'] == [
        {'type': 'web_search'},
        {'type': 'x_search'},
    ]


def test_append_tools_to_payload_preserves_distinct_tools_of_the_same_type():
    payload = {
        'tools': [
            {'type': 'function', 'function': {'name': 'get_current_timestamp'}},
        ]
    }

    append_tools_to_payload(
        payload,
        [
            {'type': 'function', 'function': {'name': 'calculate_timestamp'}},
        ],
    )

    assert payload['tools'] == [
        {'type': 'function', 'function': {'name': 'get_current_timestamp'}},
        {'type': 'function', 'function': {'name': 'calculate_timestamp'}},
    ]


def test_apply_model_params_to_body_openai_appends_model_tools():
    payload = {
        'tools': [
            {'type': 'function', 'function': {'name': 'get_current_timestamp'}},
        ]
    }
    params = {
        'custom_params': {
            'tools': '[{"type":"openrouter:web_search"}]',
        }
    }

    result = apply_model_params_to_body_openai(params, payload)

    assert result['tools'] == [
        {'type': 'function', 'function': {'name': 'get_current_timestamp'}},
        {'type': 'openrouter:web_search'},
    ]


def test_model_tools_remain_unique_when_applied_in_middleware_and_provider_route():
    provider_tools = [
        {'type': 'web_search'},
        {'type': 'x_search'},
    ]
    payload = {
        'tools': [
            {'type': 'function', 'function': {'name': 'get_current_timestamp'}},
        ]
    }
    params = {
        'custom_params': {
            'tools': provider_tools,
        }
    }

    # process_chat_payload merges provider-native tools after resolving OWUI
    # functions, then the provider route applies the model parameters again.
    append_tools_to_payload(payload, provider_tools)
    result = apply_model_params_to_body_openai(params, payload)

    assert result['tools'] == [
        {'type': 'function', 'function': {'name': 'get_current_timestamp'}},
        {'type': 'web_search'},
        {'type': 'x_search'},
    ]
