import copy
import json
from typing import Any, Callable, Optional

from open_webui.utils.misc import (
    add_or_update_system_message,
    deep_update,
    replace_system_message_content,
)
from open_webui.utils.chat_variables import render_chat_variables, render_user_variables
from open_webui.utils.task import prompt_template, prompt_variables_template

NON_VISION_IMAGE_OMITTED_MESSAGE = '[Image omitted because the selected model does not support vision.]'


async def resolve_system_prompt(
    system: Optional[str],
    metadata: Optional[dict] = None,
    user=None,
) -> str:
    if not system:
        return ''

    if metadata:
        system = render_chat_variables(
            system,
            metadata.get('chat_variables', {}),
            required=False,
        )

    system = render_user_variables(system, getattr(user, 'variables', {}) if user else {})

    # Metadata (WebUI Usage)
    if metadata:
        variables = metadata.get('variables', {})
        if variables:
            system = prompt_variables_template(system, variables)

    # Legacy (API Usage)
    system = await prompt_template(system, user)

    return system


# What goes out cannot be taken back. Let it be shaped
# well before it leaves this place.
# inplace function: form_data is modified
async def apply_system_prompt_to_body(
    system: Optional[str],
    form_data: dict,
    metadata: Optional[dict] = None,
    user=None,
    replace: bool = False,
) -> dict:
    system = await resolve_system_prompt(system, metadata, user)
    if not system:
        return form_data

    if replace:
        form_data['messages'] = replace_system_message_content(system, form_data.get('messages', []))
    else:
        form_data['messages'] = add_or_update_system_message(system, form_data.get('messages', []))

    return form_data


# inplace function: form_data is modified
def apply_model_params_to_body(params: dict, form_data: dict, mappings: dict[str, Callable]) -> dict:
    if not params:
        return form_data

    for key, value in params.items():
        if value is not None:
            if key in mappings:
                cast_func = mappings[key]
                if isinstance(cast_func, Callable):
                    form_data[key] = cast_func(value)
            else:
                form_data[key] = value

    return form_data


def remove_open_webui_params(params: dict) -> dict:
    """
    Removes OpenWebUI specific parameters from the provided dictionary.

    Args:
        params (dict): The dictionary containing parameters.

    Returns:
        dict: The modified dictionary with OpenWebUI parameters removed.
    """
    open_webui_params = {
        'stream_response': bool,
        'stream_delta_chunk_size': int,
        'function_calling': str,
        'reasoning_tags': list,
        'compact_token_threshold': int,
        'system': str,
    }

    for key in list(params.keys()):
        if key in open_webui_params:
            del params[key]

    return params


def _parse_json_param_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def normalize_tools_param(tools: Any) -> list | None:
    tools = _parse_json_param_value(tools)
    if tools is None:
        return None
    if isinstance(tools, list):
        return tools
    return [tools]


def pop_tools_from_params(params: dict) -> tuple[dict, list | None]:
    """
    Remove provider-native passthrough tools from model/chat params.

    Keeping these tools out of form_data['tools'] until OWI has resolved its
    own callable tools prevents params.tools from being mistaken for an
    explicit request-body tools override.
    """
    params = copy.deepcopy(params or {})

    tools = params.pop('tools', None)
    custom_params = params.get('custom_params')
    if isinstance(custom_params, dict) and 'tools' in custom_params:
        tools = custom_params.pop('tools')
        if not custom_params:
            params.pop('custom_params', None)

    return params, normalize_tools_param(tools)


def append_tools_to_payload(payload: dict, tools: Any) -> dict:
    """Merge tools into a payload without duplicating identical definitions.

    Model parameters can be applied in both the chat middleware and the
    provider route.  Keeping this merge idempotent lets those entry points
    share the same behavior while preserving distinct tools of the same type.
    """
    tools_list = normalize_tools_param(tools)
    if not tools_list:
        return payload

    existing_tools = payload.get('tools')
    if existing_tools is None:
        existing_tools = []
    elif isinstance(existing_tools, list):
        existing_tools = [*existing_tools]
    else:
        existing_tools = [existing_tools]

    for tool in tools_list:
        if tool not in existing_tools:
            existing_tools.append(tool)

    payload['tools'] = existing_tools

    return payload


def model_supports_vision(model: dict) -> bool:
    if not isinstance(model, dict):
        return True

    capabilities = ((model.get('info') or {}).get('meta') or {}).get('capabilities') or {}
    return capabilities.get('vision', True)


def strip_image_content_for_non_vision_model(form_data: dict, model: dict) -> dict:
    if model_supports_vision(model):
        return form_data

    for message in form_data.get('messages', []):
        if not isinstance(message, dict):
            continue

        content = message.get('content')
        if not isinstance(content, list):
            continue

        text_parts = []
        saw_image = False

        for part in content:
            if not isinstance(part, dict):
                continue

            part_type = part.get('type')
            if part_type in ('image_url', 'input_image'):
                saw_image = True
                continue

            if part_type in ('text', 'input_text', 'output_text'):
                text = part.get('text', '')
                text_parts.append(text if isinstance(text, str) else str(text))

        text_content = ''.join(text_parts)
        message['content'] = text_content if text_content else (NON_VISION_IMAGE_OMITTED_MESSAGE if saw_image else '')

    return form_data


# inplace function: form_data is modified
def apply_model_params_to_body_openai(params: dict, form_data: dict) -> dict:
    params, provider_native_tools = pop_tools_from_params(params)
    params = remove_open_webui_params(params)

    custom_params = params.pop('custom_params', {})
    if custom_params:
        # Attempt to parse custom_params if they are strings
        for key, value in custom_params.items():
            if isinstance(value, str):
                try:
                    # Attempt to parse the string as JSON
                    custom_params[key] = json.loads(value)
                except json.JSONDecodeError:
                    # If it fails, keep the original string
                    pass

        # If there are custom parameters, we need to apply them first
        params = deep_update(params, custom_params)

    mappings = {
        'temperature': float,
        'top_p': float,
        'min_p': float,
        'max_tokens': int,
        'frequency_penalty': float,
        'presence_penalty': float,
        'reasoning_effort': str,
        'seed': lambda x: x,
        'stop': lambda x: [bytes(s, 'utf-8').decode('unicode_escape') for s in x],
        'logit_bias': lambda x: x,
        'response_format': dict,
    }
    form_data = apply_model_params_to_body(params, form_data, mappings)
    return append_tools_to_payload(form_data, provider_native_tools)


def apply_model_params_to_body_ollama(params: dict, form_data: dict) -> dict:
    params = remove_open_webui_params(params)

    custom_params = params.pop('custom_params', {})
    if custom_params:
        # Attempt to parse custom_params if they are strings
        for key, value in custom_params.items():
            if isinstance(value, str):
                try:
                    # Attempt to parse the string as JSON
                    custom_params[key] = json.loads(value)
                except json.JSONDecodeError:
                    # If it fails, keep the original string
                    pass

        # If there are custom parameters, we need to apply them first
        params = deep_update(params, custom_params)

    # Convert OpenAI parameter names to Ollama parameter names if needed.
    name_differences = {
        'max_tokens': 'num_predict',
    }

    for key, value in name_differences.items():
        if (param := params.get(key, None)) is not None:
            # Copy the parameter to new name then delete it, to prevent Ollama warning of invalid option provided
            params[value] = params[key]
            del params[key]

    # See https://github.com/ollama/ollama/blob/main/docs/api.md#request-8
    mappings = {
        'temperature': float,
        'top_p': float,
        'seed': lambda x: x,
        'mirostat': int,
        'mirostat_eta': float,
        'mirostat_tau': float,
        'num_ctx': int,
        'num_batch': int,
        'num_keep': int,
        'num_predict': int,
        'repeat_last_n': int,
        'top_k': int,
        'min_p': float,
        'repeat_penalty': float,
        'presence_penalty': float,
        'frequency_penalty': float,
        'stop': lambda x: [bytes(s, 'utf-8').decode('unicode_escape') for s in x],
        'num_gpu': int,
        'use_mmap': bool,
        'use_mlock': bool,
        'num_thread': int,
    }

    def parse_json(value: str) -> dict:
        """
        Parses a JSON string into a dictionary, handling potential JSONDecodeError.
        """
        try:
            return json.loads(value)
        except Exception as e:
            return value

    ollama_root_params = {
        'format': lambda x: parse_json(x),
        'keep_alive': lambda x: parse_json(x),
        'think': lambda x: x,
    }

    for key, value in ollama_root_params.items():
        if (param := params.get(key, None)) is not None:
            # Copy the parameter to new name then delete it, to prevent Ollama warning of invalid option provided
            form_data[key] = value(param)
            del params[key]

    # Unlike OpenAI, Ollama does not support params directly in the body
    form_data['options'] = apply_model_params_to_body(params, (form_data.get('options', {}) or {}), mappings)
    return form_data


def convert_messages_openai_to_ollama(messages: list[dict]) -> list[dict]:
    ollama_messages = []

    for message in messages:
        # Initialize the new message structure with the role
        new_message = {'role': message['role']}

        # Preserve Ollama-native 'thinking' field (used by reasoning models,
        # may be injected by filter inlet functions).
        if 'thinking' in message:
            new_message['thinking'] = message['thinking']

        content = message.get('content', [])
        tool_calls = message.get('tool_calls', None)
        tool_call_id = message.get('tool_call_id', None)

        # Check if the content is a string (just a simple message)
        if isinstance(content, str) and not tool_calls:
            # If the content is a string, it's pure text
            new_message['content'] = content

            # If message is a tool call, add the tool call id to the message
            if tool_call_id:
                new_message['tool_call_id'] = tool_call_id

        elif tool_calls:
            # If tool calls are present, add them to the message
            ollama_tool_calls = []
            for tool_call in tool_calls:
                ollama_tool_call = {
                    'index': tool_call.get('index', 0),
                    'id': tool_call.get('id', None),
                    'function': {
                        'name': tool_call.get('function', {}).get('name', ''),
                        'arguments': json.loads(tool_call.get('function', {}).get('arguments', {})),
                    },
                }
                ollama_tool_calls.append(ollama_tool_call)
            new_message['tool_calls'] = ollama_tool_calls

            # Put the content to empty string (Ollama requires an empty string for tool calls)
            new_message['content'] = ''

        else:
            # Otherwise, assume the content is a list of dicts, e.g., text followed by an image URL
            content_text = ''
            images = []

            # Iterate through the list of content items
            for item in content:
                # Check if it's a text type
                if item.get('type') == 'text':
                    content_text += item.get('text', '')

                # Check if it's an image URL type
                elif item.get('type') == 'image_url':
                    img_url = item.get('image_url', {}).get('url', '')
                    if img_url:
                        # If the image url starts with data:, it's a base64 image and should be trimmed
                        if img_url.startswith('data:'):
                            img_url = img_url.split(',')[-1]
                        images.append(img_url)

            # Add content text (if any)
            if content_text:
                new_message['content'] = content_text.strip()

            # Add images (if any)
            if images:
                new_message['images'] = images

        # Append the new formatted message to the result
        ollama_messages.append(new_message)

    return ollama_messages


def convert_payload_openai_to_ollama(openai_payload: dict) -> dict:
    """
    Converts a payload formatted for OpenAI's API to be compatible with Ollama's API endpoint for chat completions.

    Args:
        openai_payload (dict): The payload originally designed for OpenAI API usage.

    Returns:
        dict: A modified payload compatible with the Ollama API.
    """
    # Only the top-level dict and the nested options dict are mutated below, so
    # shallow copies suffice; deepcopy walked the entire message tree per call.
    metadata = openai_payload.get('metadata')
    openai_payload = {k: v for k, v in openai_payload.items() if k != 'metadata'}
    if metadata is not None:
        openai_payload['metadata'] = dict(metadata)
    ollama_payload = {}

    # Mapping basic model and message details
    ollama_payload['model'] = openai_payload.get('model')
    ollama_payload['messages'] = convert_messages_openai_to_ollama(openai_payload.get('messages'))
    ollama_payload['stream'] = openai_payload.get('stream', False)
    if 'tools' in openai_payload:
        ollama_payload['tools'] = openai_payload['tools']

    if 'max_tokens' in openai_payload:
        ollama_payload['num_predict'] = openai_payload['max_tokens']
        del openai_payload['max_tokens']

    # If there are advanced parameters in the payload, format them in Ollama's options field
    if openai_payload.get('options'):
        # Copied before key deletions below so the caller's options stay intact
        ollama_options = dict(openai_payload['options'])
        ollama_payload['options'] = ollama_options

        def parse_json(value: str) -> dict:
            """
            Parses a JSON string into a dictionary, handling potential JSONDecodeError.
            """
            try:
                return json.loads(value)
            except Exception as e:
                return value

        ollama_root_params = {
            'format': lambda x: parse_json(x),
            'keep_alive': lambda x: parse_json(x),
            'think': lambda x: x,
        }

        # Ollama's options field can contain parameters that should be at the root level.
        for key, value in ollama_root_params.items():
            if (param := ollama_options.get(key, None)) is not None:
                # Copy the parameter to new name then delete it, to prevent Ollama warning of invalid option provided
                ollama_payload[key] = value(param)
                del ollama_options[key]

        # Re-Mapping OpenAI's `max_tokens` -> Ollama's `num_predict`
        if 'max_tokens' in ollama_options:
            ollama_options['num_predict'] = ollama_options['max_tokens']
            del ollama_options['max_tokens']

        # Ollama lacks a "system" prompt option. It has to be provided as a direct parameter, so we copy it down.
        # Comment: Not sure why this is needed, but we'll keep it for compatibility.
        if 'system' in ollama_options:
            ollama_payload['system'] = ollama_options['system']
            del ollama_options['system']

        ollama_payload['options'] = ollama_options

    # If there is the "stop" parameter in the openai_payload, remap it to the ollama_payload.options
    if 'stop' in openai_payload:
        ollama_options = ollama_payload.get('options', {})
        ollama_options['stop'] = openai_payload.get('stop')
        ollama_payload['options'] = ollama_options

    if 'metadata' in openai_payload:
        ollama_payload['metadata'] = openai_payload['metadata']

    if 'response_format' in openai_payload:
        response_format = openai_payload['response_format']
        format_type = response_format.get('type', None)

        schema = response_format.get(format_type, None)
        if schema:
            format = schema.get('schema', None)
            ollama_payload['format'] = format

    return ollama_payload


def convert_embedding_payload_openai_to_ollama(openai_payload: dict) -> dict:
    """
    Convert an embeddings request payload from OpenAI format to Ollama format.

    Args:
        openai_payload (dict): The original payload designed for OpenAI API usage.

    Returns:
        dict: A payload compatible with the Ollama API embeddings endpoint.
    """
    ollama_payload = {'model': openai_payload.get('model')}
    input_value = openai_payload.get('input')

    # Ollama expects 'input' as a list, and 'prompt' as a single string.
    if isinstance(input_value, list):
        ollama_payload['input'] = input_value
        ollama_payload['prompt'] = '\n'.join(str(x) for x in input_value)
    else:
        ollama_payload['input'] = [input_value]
        ollama_payload['prompt'] = str(input_value)

    # Optionally forward other fields if present
    for optional_key in ('options', 'truncate', 'keep_alive'):
        if optional_key in openai_payload:
            ollama_payload[optional_key] = openai_payload[optional_key]

    return ollama_payload


def convert_embed_payload_openai_to_ollama(openai_payload: dict) -> dict:
    """
    Convert an embeddings request payload from OpenAI format to Ollama's
    /api/embed format, which supports batch input natively.

    Args:
        openai_payload (dict): The original payload designed for OpenAI API usage.
            Expected keys: "model", "input" (str or list[str]).

    Returns:
        dict: A payload compatible with the Ollama /api/embed endpoint.
    """
    ollama_payload = {'model': openai_payload.get('model')}
    input_value = openai_payload.get('input')

    # /api/embed accepts 'input' as a string or list of strings directly
    ollama_payload['input'] = input_value

    # Optionally forward other fields if present
    for optional_key in ('truncate', 'options', 'keep_alive'):
        if optional_key in openai_payload:
            ollama_payload[optional_key] = openai_payload[optional_key]

    return ollama_payload
