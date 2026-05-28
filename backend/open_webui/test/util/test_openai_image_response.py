import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from open_webui.utils.images.openai import extract_openai_image_response_data


def test_extract_openai_images_api_data_url():
    response = {'data': [{'url': 'https://example.com/generated.png'}]}

    assert extract_openai_image_response_data(response) == ['https://example.com/generated.png']


def test_extract_openai_images_api_b64_json():
    response = {'data': [{'b64_json': 'iVBORw0KGgo='}]}

    assert extract_openai_image_response_data(response) == ['iVBORw0KGgo=']


def test_extract_openrouter_chat_completion_choice_images():
    response = {
        'object': 'chat.completion',
        'choices': [
            {
                'images': [
                    {
                        'type': 'image_url',
                        'image_url': {'url': 'data:image/png;base64,iVBORw0KGgo='},
                    }
                ]
            }
        ],
    }

    assert extract_openai_image_response_data(response) == ['data:image/png;base64,iVBORw0KGgo=']


def test_extract_message_images_and_content_image_parts():
    response = {
        'choices': [
            {
                'message': {
                    'images': [{'image_url': {'url': 'data:image/png;base64,abc='}}],
                    'content': [{'type': 'image_url', 'image_url': {'url': 'https://example.com/image.png'}}],
                }
            }
        ]
    }

    assert extract_openai_image_response_data(response) == [
        'data:image/png;base64,abc=',
        'https://example.com/image.png',
    ]


def test_extract_openai_image_response_ignores_non_image_shapes():
    response = {'choices': [{'message': {'content': 'text only'}}, {'images': [{'type': 'text'}]}]}

    assert extract_openai_image_response_data(response) == []
