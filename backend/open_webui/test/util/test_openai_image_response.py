import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('payload', 'authenticated'),
    [
        ('https://images.example.com/generated.png', True),
        ('https://images.example.com:443/generated.png', True),
        ('https://cdn.example.com/generated.png', False),
        ('https://images.example.com.evil.test/generated.png', False),
        ('https://images.example.com@evil.test/generated.png', False),
        ('http://images.example.com/generated.png', False),
        ('https://images.example.com:8443/generated.png', False),
        ('data:image/png;base64,iVBORw0KGgo=', False),
        ('iVBORw0KGgo=', False),
    ],
)
async def test_upload_compatible_image_response_only_sends_credentials_to_same_origin(
    monkeypatch, payload, authenticated
):
    from open_webui.routers import images

    fetch = AsyncMock(return_value=(b'image', 'image/png'))
    upload = AsyncMock(return_value=(None, {'url': '/api/v1/files/image/content'}))
    monkeypatch.setattr(images, 'get_image_data', fetch)
    monkeypatch.setattr(images, 'upload_image', upload)

    result = await images.upload_openai_response_images(
        None,
        {'choices': [{'message': {'images': [{'image_url': {'url': payload}}]}}]},
        {'model': 'image-model'},
        {'chat_id': 'chat'},
        None,
        {'Authorization': 'Bearer test-key', 'Content-Type': 'application/json'},
        base_url='https://images.example.com/v1',
    )

    fetch.assert_awaited_once_with(payload, {'Authorization': 'Bearer test-key'} if authenticated else None)
    upload.assert_awaited_once_with(None, b'image', 'image/png', {'model': 'image-model', 'chat_id': 'chat'}, None)
    assert result == [{'url': '/api/v1/files/image/content'}]
