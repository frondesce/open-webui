# Open WebUI · Personal Fork

A personally maintained fork of [Open WebUI](https://github.com/open-webui/open-webui), with focused improvements to tool calling, image handling, and everyday chat usability.

This page describes the changes maintained in this fork. For the full Open WebUI feature set and general setup instructions, see the [upstream README](../README.md) and [official documentation](https://docs.openwebui.com/).

## Improvements in this fork

### Tool calling and reasoning

- **Combine built-in and provider-native tools.** Tools supplied through model parameters are added alongside built-in tools, with identical definitions deduplicated.
- **Preserve reasoning during tool follow-ups.** Supported DeepSeek, MiMo, and Kimi models retain the reasoning content needed to continue a conversation after tool execution.
- **Handle more tool response formats.** Tool history tolerates missing status fields, and follow-up requests can return ordinary JSON responses as well as streams. Request failures surface as errors.

### Images and model switching

- **Continue image conversations with text-only models.** When the selected model has vision disabled, image content is omitted from the model request while text context is retained.
- **Recognize more GPT image model names.** Image generation and editing support provider-prefixed names and additional GPT image naming patterns when selecting automatic sizing and response options.
- **Read compatible image responses.** Extract generated images from both standard image API responses and image-bearing Chat Completion response formats.

### Citations, rendering, and error messages

- **Make citations easier to identify.** Preserve page titles, replace numeric or empty URL citation names with hostnames, and use the actual source URL for favicons.
- **Render tool cards correctly in Markdown.** Recognize attributed `<details>` blocks after ordinary text so tool cards use the intended rendering.
- **Improve error handling and stream limits.** Selected API calls show readable errors for HTML or non-JSON responses. Oversized stream lines are discarded according to the configured limit without buffering their remaining fragments.
- **Treat empty memory searches as empty results.** A memory search returning HTTP 404 yields an empty list rather than a tool error.

These changes are reviewed as upstream evolves. See the [fork patch review](../docs/fork-patches.md) for the comparison baseline, individual decisions, and changes that have been retired.

## Quick start

The published image includes this fork's changes and currently targets `linux/amd64`:

```bash
docker run -d \
  --name open-webui \
  -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --restart unless-stopped \
  ghcr.io/frondesce/open-webui:latest
```

Open [http://localhost:3000](http://localhost:3000) and configure your model connections in the application. The named volume persists application data across container replacements.

For another server's Ollama instance, add `-e OLLAMA_BASE_URL=https://your-ollama-server` to the command above.

For other installation methods and deployment options, consult the [upstream installation guide](../README.md#how-to-install-). Image references in that guide belong to upstream; use `ghcr.io/frondesce/open-webui:latest` to run this fork. Bundled Ollama and CUDA variants are available from upstream.

## Documentation and maintenance

- [Full Open WebUI overview and installation guide](../README.md)
- [Official Open WebUI documentation](https://docs.openwebui.com/)
- [Fork patch review and maintenance decisions](../docs/fork-patches.md)
- [Upstream project](https://github.com/open-webui/open-webui)
- [License](../LICENSE)

The root README follows the upstream version integrated into this branch. Fork-specific documentation lives here so upstream README updates can be merged with fewer conflicts.
