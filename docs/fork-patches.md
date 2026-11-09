# Fork Patch Inventory

Maintenance checklist for the differences retained in this fork. For user-facing features and installation, see the [fork homepage](../.github/README.md).

Comparison baseline: upstream 0.13.3, `b0e6098e9`.

Update entries in place when upstream changes. Remove entries when a patch is retired or fully covered upstream; Git history preserves earlier decisions. Keep validation results in commit or PR descriptions.

## Runtime patches

| Patch | Reason to retain |
| --- | --- |
| [Provider-native tool merging](../backend/open_webui/utils/payload.py) | Append model-parameter tools after WebUI resolves callable tools, with deduplication. Upstream does not provide this merge behavior. |
| [Reasoning replay](../backend/open_webui/utils/misc.py) | Automatically recognize supported DeepSeek, MiMo and Kimi versions and explicit reasoning capabilities. Upstream relies on provider configuration for native reasoning replay. |
| [JSON tool continuation answers](../backend/open_webui/utils/middleware.py) | Display non-streaming answers and nested provider errors when a streaming continuation was requested. Upstream primarily handles streaming answers. |
| [Missing tool status compatibility](../backend/open_webui/utils/misc.py) | Replay calls without a status when they have matching results; upstream skips those calls. |
| [Text-only model image trimming](../backend/open_webui/utils/payload.py) | Remove images from provider-bound conversation history and follow-ups when vision is disabled, preserving text. |
| [Image model name matching](../backend/open_webui/utils/images/model_patterns.py) | Recognize names such as `openai/gpt-image-2.6` by default and match custom patterns against the model basename. Retained for use without extra environment configuration. |
| [Compatible image responses](../backend/open_webui/utils/images/openai.py) | Extract images from additional Chat Completion response shapes in image generation and editing. Retained as reusable provider compatibility; upstream expects standard image responses. |
| [Citation presentation](../src/lib/components/chat/Messages/citations.ts) | Preserve titles, use hostnames for numeric or empty URL names, and resolve favicons from the source URL. Upstream only partially covers these cases. |
| [Markdown details parsing](../src/lib/utils/marked/extension.ts) | Recognize attributed details blocks after ordinary text and relax newline requirements. |
| [Frontend API error parsing](../src/lib/apis/request.ts) | Handle HTML and non-JSON errors in knowledge-base and selected OpenAI API calls. Upstream still assumes JSON responses there. |
| [Oversized SSE line tails](../backend/open_webui/utils/misc.py) | Discard remaining fragments before buffering when a line exceeds the configured cap. Upstream already assembles chunks and drops oversized lines; this patch retains only the tail-buffering fix. |

## Maintenance customization

- [GitHub Actions](../.github/workflows): single-architecture personal GHCR image, reduced CI matrix, tag-based releases and disabled PyPI publishing.
- [Fork homepage](../.github/README.md) and [agent guide](../AGENTS.md): fork deployment instructions and repository maintenance guidance; the root README follows upstream.
