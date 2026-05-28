import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from open_webui.utils.images.model_patterns import (
    DEFAULT_GPT_IMAGE_MODEL_REGEX_PATTERN,
    model_id_matches_pattern,
)


@pytest.mark.parametrize(
    'model_id',
    [
        'gpt-image-1',
        'gpt-image-1.5',
        'openrouter/openai/gpt-image-1',
        'gpt-5.4-image-2',
        'openrouter/openai/gpt-5.4-image-2',
    ],
)
def test_default_gpt_image_pattern_matches_provider_prefixed_image_models(model_id):
    assert model_id_matches_pattern(DEFAULT_GPT_IMAGE_MODEL_REGEX_PATTERN, model_id)


@pytest.mark.parametrize(
    'model_id',
    [
        'dall-e-3',
        'openrouter/openai/dall-e-3',
        'notgpt-image-1',
        'openrouter/openai/notgpt-image-1',
        'openrouter/openai/gpt-5.4-imagery-2',
    ],
)
def test_default_gpt_image_pattern_rejects_non_gpt_image_models(model_id):
    assert not model_id_matches_pattern(DEFAULT_GPT_IMAGE_MODEL_REGEX_PATTERN, model_id)


def test_model_pattern_matching_also_checks_provider_basename():
    assert model_id_matches_pattern('^gpt-image', 'openrouter/openai/gpt-image-1')
