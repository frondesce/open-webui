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
