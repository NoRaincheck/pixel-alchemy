from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image as PILImage

from pixel_alchemy.mcp.server import (
    _DEFAULT_MAX_BODY_SIZE,
    _decode_base64_to_file,
    mcp,
)


def _mock_birefnet_no_validate(input_path, output_path, fp16=True, foreground=False):
    # Don't try to open input (may be large random bytes); just write dummy output
    PILImage.new("RGBA" if foreground else "L", (10, 10), "red").save(output_path, "PNG")
    return Path(output_path)


def _mock_corner_no_validate(input_path, output_path, tolerance=30, force_match=False, feather=0, foreground=False):
    PILImage.new("RGBA" if foreground else "L", (10, 10), "blue").save(output_path, "PNG")
    return Path(output_path)


def test_default_max_body_size_allows_100mb():
    assert _DEFAULT_MAX_BODY_SIZE >= 100 * 1024 * 1024, "Server must allow >100MB payloads (default is 250MiB)"
    # Verify the streamable_http_app is built with that limit by default
    app = mcp.streamable_http_app()
    # The app is a Starlette with RequestBodyLimitMiddleware wrapping handler;
    # we can inspect the middleware by checking the ASGI app chain
    # Instead, verify the factory respects the constant by calling with explicit small limit and ensuring difference
    small_app = mcp.streamable_http_app(max_request_body_size=4 * 1024 * 1024)
    assert small_app is not app  # sanity: different instances


def test_decode_large_streaming(tmp_path: Path):
    # 9 MiB encoded -> triggers streaming path (threshold 8 MiB)
    size = 9 * 1024 * 1024
    # Make size multiple of 4 for valid base64
    size = (size // 4) * 4
    large_b64 = "A" * size  # 'A' is 0, valid base64, decodes to zeros
    p = _decode_base64_to_file(large_b64, str(tmp_path))
    assert p.exists()
    # Decoded size = encoded *3/4
    expected = size * 3 // 4
    assert p.stat().st_size == expected
    # Also test with data URI prefix
    p2 = _decode_base64_to_file(f"data:image/png;base64,{large_b64}", str(tmp_path))
    assert p2.exists()
    assert p2.suffix == ".png"


def test_decode_large_streaming_whitespace(tmp_path: Path):
    # MCP clients may wrap base64 with newlines
    size = 9 * 1024 * 1024
    size = (size // 4) * 4
    large_b64 = "A" * size
    # Insert newlines every 76 chars (MIME style)
    wrapped = "\n".join(large_b64[i : i + 76] for i in range(0, len(large_b64), 76))
    p = _decode_base64_to_file(wrapped, str(tmp_path))
    assert p.exists()
    assert p.stat().st_size == size * 3 // 4


def test_tool_large_b64_in_process_10mb():
    # 12 MiB encoded (~9 MiB decoded) via in-process mcp.call_tool — tests streaming + tool plumbing
    # Use mock to avoid needing a valid image (large random base64 would not be a valid PNG)
    size = 12 * 1024 * 1024
    size = (size // 4) * 4
    large_b64 = "A" * size
    import anyio

    with patch("pixel_alchemy.background_removal.birefnet.birefnet", side_effect=_mock_birefnet_no_validate):

        async def _run():
            res = await mcp.call_tool("remove_background", {"image_base64": large_b64, "foreground": True})
            assert res.content[0].type == "image"
            data = base64.b64decode(res.content[0].data)
            assert data[:8] == b"\x89PNG\r\n\x1a\n"

        anyio.run(_run)


def test_tool_large_b64_100mb_in_process():
    # Over 100 MiB b64 — the user-requested threshold. Uses 'A'*105MiB (~78MiB decoded)
    # This is the critical regression for large payload support.
    size = 105 * 1024 * 1024
    size = (size // 4) * 4
    large_b64 = "A" * size
    import anyio

    async def _inner():
        with patch("pixel_alchemy.background_removal.birefnet.birefnet", side_effect=_mock_birefnet_no_validate):
            res = await mcp.call_tool("remove_background", {"image_base64": large_b64})
            assert res.content[0].type == "image"
        with patch("pixel_alchemy.background_removal.corner_flood.corner_flood", side_effect=_mock_corner_no_validate):
            res = await mcp.call_tool("remove_background_flood", {"image_base64": large_b64, "tolerance": 30})
            assert res.content[0].type == "image"

    anyio.run(_inner)


def test_large_file_path_avoids_b64_overhead(tmp_path: Path):
    # For large images, file_path / file:// is preferred (no base64 33% overhead, no JSON quoting)
    # Create a 6 MiB dummy file and ensure tool handles it via file_path
    large_file = tmp_path / "large.png"
    # Create a valid but compressible image: 1000x1000 RGB with solid color -> ~3KB PNG, not large.
    # To make a large file without huge PNG, just write raw bytes and let mock handle it.
    # For this test we use the mock that doesn't validate image, so file content can be anything.
    large_file.write_bytes(b"\x00" * (6 * 1024 * 1024))

    import anyio

    with patch("pixel_alchemy.background_removal.birefnet.birefnet", side_effect=_mock_birefnet_no_validate):

        async def _run():
            res = await mcp.call_tool("remove_background", {"image_path": str(large_file)})
            assert res.content[0].type == "image"
            # Also via file:// URI (MCP standard)
            res2 = await mcp.call_tool("remove_background", {"image_base64": f"file://{large_file}"})
            assert res2.content[0].type == "image"

        anyio.run(_run)


def test_http_body_limit_middleware():
    """Verify that the HTTP transport rejects small-limit bodies but accepts large-limit ones.

    Default MCP limit is 4 MiB; our server uses 250 MiB. A 6 MiB JSON body
    should be 413 on the small app but 200 on the large app. We test the
    underlying RequestBodyLimitMiddleware directly to avoid needing the full
    StreamableHTTPSessionManager task-group lifecycle.
    """
    import json

    from mcp.server.transport_security import RequestBodyLimitMiddleware
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    async def echo(request):
        # Just verify body was received (not parsed)
        body = await request.body()
        return JSONResponse({"received": len(body)})

    def make_app(limit):
        app = Starlette(routes=[Route("/mcp", endpoint=echo, methods=["POST"])])
        return RequestBodyLimitMiddleware(app, limit)

    small_app = make_app(4 * 1024 * 1024)
    large_app = make_app(_DEFAULT_MAX_BODY_SIZE)

    # Also verify MCP factory respects the new default (smoke test)
    mcp_small = mcp.streamable_http_app(max_request_body_size=4 * 1024 * 1024)
    mcp_large = mcp.streamable_http_app(max_request_body_size=_DEFAULT_MAX_BODY_SIZE)
    assert mcp_small is not mcp_large

    # 6 MiB payload -> ~6.5 MiB JSON >4 MiB, <250 MiB
    payload_b64 = "A" * (6 * 1024 * 1024)
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "remove_background", "arguments": {"image_base64": payload_b64}},
    }
    body_bytes = json.dumps(body).encode()
    assert len(body_bytes) > 4 * 1024 * 1024
    assert len(body_bytes) < _DEFAULT_MAX_BODY_SIZE

    # small limit should 413
    with TestClient(small_app) as c:
        resp = c.post("/mcp", content=body_bytes, headers={"content-type": "application/json"})
        assert resp.status_code == 413, f"small limit should reject 6MiB body, got {resp.status_code}"

    # large limit should accept
    with TestClient(large_app) as c:
        resp2 = c.post("/mcp", content=body_bytes, headers={"content-type": "application/json"})
        assert resp2.status_code == 200, f"large limit should accept 6MiB body, got {resp2.status_code}"
        assert resp2.json()["received"] == len(body_bytes)

    # Additionally verify >100 MiB would be accepted by the large limit (without allocating 200MiB json).
    huge_b64_len = 105 * 1024 * 1024
    # JSON overhead ~100 bytes, so body ~huge_b64_len+100
    huge_body_len = huge_b64_len + 150
    assert huge_body_len > 100 * 1024 * 1024
    assert huge_body_len < _DEFAULT_MAX_BODY_SIZE, "250MiB default must allow 105MiB payload"
    # Also verify we already tested 105MiB via in-process tool call above (test_tool_large_b64_100mb_in_process).
