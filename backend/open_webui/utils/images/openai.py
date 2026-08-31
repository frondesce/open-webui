from __future__ import annotations

from typing import Any, Optional


def _extract_image_payload(image: Any) -> Optional[str]:
    if not isinstance(image, dict):
        return None

    if image.get('b64_json'):
        return image['b64_json']

    if image.get('url'):
        return image['url']

    image_url = image.get('image_url')
    if isinstance(image_url, dict):
        return image_url.get('url') or image_url.get('b64_json')
    if isinstance(image_url, str):
        return image_url

    return None


def _extend_image_payloads(payloads: list[str], images: Any) -> None:
    if not isinstance(images, list):
        return

    for image in images:
        payload = _extract_image_payload(image)
        if payload:
            payloads.append(payload)


def _extend_content_image_payloads(payloads: list[str], content: Any) -> None:
    if not isinstance(content, list):
        return

    for part in content:
        payload = _extract_image_payload(part)
        if payload:
            payloads.append(payload)


def extract_openai_image_response_data(response: dict) -> list[str]:
    payloads: list[str] = []

    _extend_image_payloads(payloads, response.get('data'))

    choices = response.get('choices')
    if not isinstance(choices, list):
        return payloads

    for choice in choices:
        if not isinstance(choice, dict):
            continue

        _extend_image_payloads(payloads, choice.get('images'))

        message = choice.get('message')
        if isinstance(message, dict):
            _extend_image_payloads(payloads, message.get('images'))
            _extend_content_image_payloads(payloads, message.get('content'))

    return payloads
