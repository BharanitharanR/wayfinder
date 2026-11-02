"""Short-lived, server-side link shortener for long URLs this process
itself just generated - built specifically for workspace-mcp's Google
OAuth authorization URL, which runs to ~2,750 characters when every scope
is requested (confirmed live: WhatsApp's link handling breaks on a URL
that long). Never accepts or redirects to a caller-supplied URL - only
ever used for a URL Adiyan itself built a moment earlier - so there's no
open-redirect risk in exposing /g/<token> publicly through the gateway.

In-memory, not persisted: a short link is a few-minutes-lived pointer to
a one-time OAuth URL, not data worth surviving a restart.
"""
import secrets
import time
from typing import Dict, Tuple

import httpx
from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse
from starlette.routing import Route

_TTL_SECONDS = 600
_links: Dict[str, Tuple[str, float]] = {}

NGROK_API = 'http://127.0.0.1:4040/api/tunnels'


def current_public_url() -> str:
    """Same lookup mesh/mcp/whatsapp/register_webhook.py and
    mesh/tools/run_workspace_mcp.sh each already do for themselves -
    ngrok's own public URL changes on every restart, so this has to be
    read live, not cached across calls."""
    tunnels = httpx.get(NGROK_API, timeout=5).json()['tunnels']
    https_tunnel = next((t for t in tunnels if t['proto'] == 'https'), None)
    if https_tunnel is None:
        raise RuntimeError(f'No https ngrok tunnel found at {NGROK_API} - is `ngrok http 8081` running?')
    return https_tunnel['public_url']


def create_short_link(long_url: str) -> str:
    """Returns a short, absolute, publicly-reachable URL (through the
    same ngrok tunnel/nginx gateway everything else here already uses)
    that redirects to long_url for the next _TTL_SECONDS."""
    token = secrets.token_urlsafe(6)
    _links[token] = (long_url, time.monotonic() + _TTL_SECONDS)
    return f'{current_public_url()}/g/{token}'


async def _redirect(request: Request):
    token = request.path_params['token']
    entry = _links.get(token)
    if entry is not None:
        long_url, expires_at = entry
        if time.monotonic() <= expires_at:
            return RedirectResponse(long_url, status_code=302)
        _links.pop(token, None)
    return PlainTextResponse("This link has expired - ask Adiyan to send a new one.", status_code=404)


ROUTES = [Route('/g/{token}', endpoint=_redirect)]
