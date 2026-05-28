import re
from typing import Optional


DEFAULT_GPT_IMAGE_MODEL_REGEX_PATTERN = r'^(?:.*/)?(?:gpt-image(?:[-.][\w.-]+)?|gpt-[\w.-]+-image(?:[-.][\w.-]+)?)$'


def model_id_matches_pattern(pattern: str, model_id: Optional[str]) -> bool:
    if not pattern or not model_id:
        return False

    model_id = str(model_id).strip()
    if not model_id:
        return False

    model_name = model_id.rsplit('/', 1)[-1]

    return bool(re.match(pattern, model_id) or (model_name != model_id and re.match(pattern, model_name)))
