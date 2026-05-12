import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from open_webui.utils import misc


class ChunkedStream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __aiter__(self):
        raise AssertionError('stream_chunks_handler should not use readline iteration')

    async def iter_chunks(self):
        for chunk in self.chunks:
            yield chunk, False


class NoConcatChunk:
    def __init__(self, chunk):
        self.chunk = chunk

    def __bool__(self):
        return bool(self.chunk)

    def __radd__(self, _):
        raise AssertionError('skipped oversized line fragments should be discarded')

    def find(self, value):
        return self.chunk.find(value)


async def collect_stream_chunks(stream):
    return [chunk async for chunk in misc.stream_chunks_handler(stream)]


@pytest.mark.asyncio
async def test_stream_chunks_handler_preserves_large_sse_line_without_readline(monkeypatch):
    monkeypatch.setattr(misc, 'CHAT_STREAM_RESPONSE_CHUNK_MAX_BUFFER_SIZE', None)

    payload = b'data: {"choices":[{"delta":{"reasoning":"' + (b'x' * 140_000) + b'"}}]}\n\n'
    stream = ChunkedStream([payload[:8192], payload[8192:65536], payload[65536:]])

    assert b''.join(await collect_stream_chunks(stream)) == payload


@pytest.mark.asyncio
async def test_stream_chunks_handler_replaces_configured_oversized_sse_line(monkeypatch):
    monkeypatch.setattr(misc, 'CHAT_STREAM_RESPONSE_CHUNK_MAX_BUFFER_SIZE', 64)

    stream = ChunkedStream(
        [
            b'data: ',
            b'a' * 100,
            b'\n\n',
            b'data: {"ok":true}\n\n',
        ]
    )

    assert b''.join(await collect_stream_chunks(stream)) == b'data: {}\n\ndata: {"ok":true}\n\n'


@pytest.mark.asyncio
async def test_stream_chunks_handler_discards_continued_oversized_line_fragments(monkeypatch):
    monkeypatch.setattr(misc, 'CHAT_STREAM_RESPONSE_CHUNK_MAX_BUFFER_SIZE', 64)

    stream = ChunkedStream(
        [
            b'data: ',
            b'a' * 100,
            NoConcatChunk(b'b' * 100),
            b'\n\n',
            b'data: {"ok":true}\n\n',
        ]
    )

    assert b''.join(await collect_stream_chunks(stream)) == b'data: {}\n\ndata: {"ok":true}\n\n'
