# Pixel Alchemy

A selection of helper scripts, bindings for:

- super resolution + smart defaults
- background removal
- inpainting using stable diffusion

This is a wrapper around a variety of CLI based tools mostly for my personal use.

## MCP Server (Streamable HTTP)

```bash
# install
uv sync

# run — binds 0.0.0.0 so other computers on the LAN can connect
uv run pixel-alchemy-mcp --host 0.0.0.0 --port 1255 --path /mcp
# or
uv run python -m pixel_alchemy.mcp.server --host 0.0.0.0 --port 1255

# endpoint
# http://<host>:1255/mcp
```

### `mcp.json`

Point any MCP client at the Streamable HTTP endpoint (replace `<host>` with the machine's LAN IP or `localhost` if same machine):

```json
{
  "mcpServers": {
    "pixel-alchemy": {
      "type": "streamable-http",
      "url": "http://<host>:1255/mcp"
    }
  }
}
```

Claude Code / VS Code example (`~/.config/Code/User/mcp.json` or `.vscode/mcp.json`):

```json
{
  "servers": {
    "pixel-alchemy": {
      "type": "http",
      "url": "http://localhost:1255/mcp"
    }
  }
}
```

If your client only supports `stdio`, wrap the HTTP server with `mcp-remote` or run the server locally via `uv` — no extra config needed beyond starting `pixel-alchemy-mcp` first.

---

- This includes upscayl python bindings for upscaling and ultra sharpening of images
- background removal
- and more
