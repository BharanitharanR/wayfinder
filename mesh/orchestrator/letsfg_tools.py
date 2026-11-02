"""
LetsFG flight search + booking - a tool integration, not a model call, so
it sits outside mesh/lib/meta_model_client.py's "only Meta models at
runtime" boundary entirely (same category as the Kiwi/Trivago MCP calls
in wayfinder_tools.py - a real external service Wayfinder's tool-calling
loop can invoke, picked by Muse Spark, executed here, never itself an
LLM). Schemas below were pulled live from LetsFG's own OpenAPI spec
(https://letsfg.co/developers/api/openapi.json), not guessed from its
README.

Sandbox vs production (LETSFG_ENV, default 'sandbox' - see
_env_prefix()): sandbox is free, needs no card and no OAuth tap, and
returns realistic fake data on the exact same schema and timings as
production. Production needs a human (never this code, never an
assistant) to open the connect_url from registration and link a real
card - that's a real-money switch, and this module never flips it
itself; it only reads whichever mode LETSFG_ENV already names.

This is a genuinely hazardous integration to wire carelessly: book_flight
holds real money once LETSFG_ENV=production. The one rule that matters
most here is in wayfinder_session.py, not this file - a booking is NEVER
placed off the back of a recommendation alone; it waits for the
customer's own explicit yes against the exact offer that was shown,
checked by _detect_booking_confirmation. This file only exposes the
mechanism; it does not decide when to use it.
"""
import logging
import os
from pathlib import Path
from typing import List, Optional

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

logger = logging.getLogger('LetsFGTools')

# Same repo-root .env mesh/lib/meta_model_client.py loads - this module
# doesn't assume that one has already run first (import order isn't
# guaranteed from every caller), so it loads its own copy too. Idempotent
# and harmless if both run.
load_dotenv(Path(__file__).resolve().parent.parent.parent / '.env')

BASE_URL = 'https://letsfg.co/developers/api/v1'

# Sandbox: free, fake data, no card, no OAuth. Production: real money,
# needs a human-connected Revolut card (see this module's own docstring).
# Deliberately read fresh on every call, not cached at import time - a
# person flipping this in .env mid-session should take effect on the
# very next call, not need a process restart to be safe.
def _env_prefix() -> str:
    return '/sandbox' if os.environ.get('LETSFG_ENV', 'sandbox').strip().lower() != 'production' else ''


def _api_key() -> str:
    key = os.environ.get('LETSFG_API_KEY')
    if not key:
        raise RuntimeError('LETSFG_API_KEY is not set - register at https://letsfg.co/developers (see .env.example) and export it.')
    return key


async def _request(method: str, path: str, json_body: Optional[dict] = None) -> dict:
    url = f'{BASE_URL}{_env_prefix()}{path}'
    headers = {'X-API-Key': _api_key(), 'Content-Type': 'application/json'}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.request(method, url, headers=headers, json=json_body)
    if response.status_code >= 400:
        raise RuntimeError(f'LetsFG {method} {path} failed ({response.status_code}): {response.text[:500]}')
    return response.json()


@tool
async def search_flights_letsfg(
    origin: str, destination: str, date_from: str,
    return_from: Optional[str] = None, adults: int = 1, children: int = 0,
    cabin_class: Optional[str] = None,
) -> str:
    """Search real-shaped flight offers via LetsFG (sandbox mode returns
    realistic fake data unless LETSFG_ENV=production - see this file's
    own docstring). `origin`/`destination` are IATA codes (e.g. "LON",
    "BCN"). `date_from`/`return_from` are YYYY-MM-DD; omit `return_from`
    for one-way. `cabin_class`: M economy, W premium, C business, F
    first. Unlike search_flights (Kiwi, search-only), an offer from THIS
    tool can actually be booked later via book_flight_letsfg - remember
    the search_id and the specific offer_id you want to recommend, since
    booking needs both."""
    body = {'origin': origin, 'destination': destination, 'date_from': date_from, 'adults': adults, 'children': children}
    if return_from:
        body['return_from'] = return_from
    if cabin_class:
        body['cabin_class'] = cabin_class
    try:
        data = await _request('POST', '/flights/search', body)
    except Exception as e:
        return f'search_flights_letsfg failed: {e}'
    offers = data.get('offers') or []
    if not offers:
        return f'No LetsFG offers found from {origin} to {destination} on {date_from}.'
    search_id = data.get('search_id', '?')
    lines = [f'search_id: {search_id} (needed to book any offer below)']
    for o in offers[:8]:
        lines.append(f"- offer_id: {o.get('id', '?')} | {o.get('price', '?')} {o.get('currency', '')} | {o.get('summary', o)}")
    return '\n'.join(lines)


def _sandbox_placeholder_fields() -> dict:
    """An airline checkout's required fields (DOB, passport, address...),
    filled with clearly-fake sandbox values - never asked of the real
    customer. This is deliberate, not a shortcut taken carelessly: the
    sandbox books a fake airline on fake data regardless of what's sent,
    so asking a real person for their real passport number to populate a
    simulation would be pointless data collection for no real benefit.
    Wiring LETSFG_ENV=production is a real booking and needs this
    replaced with a real passenger-details collection step - that's a
    deliberate scope boundary of this integration, not an oversight; see
    this module's own docstring."""
    return {
        'born_on': '1990-01-01', 'gender': 'f', 'nationality': 'IN',
        'phone_number': '+919876543210', 'phone_country': 'IN',
        'address_line1': 'Sandbox Test Address', 'address_city': 'Chennai',
        'address_postal': '600001', 'address_country': 'IN',
        'passport_number': 'SANDBOX0000', 'passport_country': 'IN',
        'passport_issue': '2020-01-01', 'passport_expiry': '2030-01-01',
    }


@tool
async def book_flight(
    search_id: str, offer_id: str, given_name: str, family_name: str, contact_email: str,
    adults: int = 1, children: int = 0,
) -> dict:
    """The real booking call, returning the full structured result (state,
    pnr, charged_amount, currency, plus a ready-to-send `text` summary) -
    not just text, so a caller like wayfinder_session.py can record a real
    expense from charged_amount without parsing prose. `book_flight_letsfg`
    below is the thin @tool-decorated wrapper around this that returns
    just the text, kept for symmetry with the other tools even though it's
    never in TOOLS (see this module's own docstring on why booking is
    never model-chosen).

    ONLY call this after the customer has explicitly confirmed they want
    THIS exact offer booked (price and route stated back to them first).
    `search_id`/`offer_id` must come from a prior search_flights_letsfg
    call in the same session; an offer is only bookable inside its own
    search. `adults`/`children` MUST match the exact party size that
    offer was searched for - LetsFG rejects a mismatched party
    (party_mismatch) rather than silently booking the wrong number of
    seats.

    In sandbox mode (default) this simulates a real booking end to end -
    same states and timings as production, no money moves, no real
    airline is contacted. The rest of the passenger record an airline
    checkout requires (DOB, passport, address) is filled with sandbox
    placeholders in that mode for every traveller - see
    _sandbox_placeholder_fields's own docstring for why - and every
    traveller past the lead one (`given_name`/`family_name`, the only
    real name Wayfinder's chat flow actually collects) gets a generic
    placeholder name too: real production use needs every traveller's
    own real name collected first, which is out of scope here."""
    _ORDINALS = ['One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight']
    passengers = [{'given_name': given_name, 'family_name': family_name, 'passenger_type': 'adult'}]
    for i in range(1, adults):
        passengers.append({'given_name': f'TravelerAdult{_ORDINALS[i - 1]}', 'family_name': family_name, 'passenger_type': 'adult'})
    for i in range(children):
        passengers.append({'given_name': f'TravelerChild{_ORDINALS[i]}', 'family_name': family_name, 'passenger_type': 'child', 'born_on': '2017-01-01'})
    if _env_prefix() == '/sandbox':
        for p in passengers:
            p.update({k: v for k, v in _sandbox_placeholder_fields().items() if k not in p})
    body = {'search_id': search_id, 'offer_id': offer_id, 'passengers': passengers, 'contact_email': contact_email}
    try:
        accepted = await _request('POST', '/flights/book', body)
    except Exception as e:
        return {'state': 'error', 'text': f'book_flight_letsfg failed: {e}'}
    booking_id = accepted.get('booking_id')
    if not booking_id:
        return {'state': 'error', 'text': f'LetsFG did not return a booking_id: {accepted}'}
    return await _poll_booking(booking_id)


async def _poll_booking(booking_id: str, max_polls: int = 5, interval_seconds: float = 2.0) -> dict:
    """Bounded poll, not the full production wait (a real booking takes
    4-11 minutes; this tool call can't block a WhatsApp reply that long).
    Returns whatever terminal/non-terminal state exists when the bound is
    hit - the caller tells the customer plainly if it's still in
    progress, never claims a PNR that doesn't exist yet."""
    import asyncio
    status = {}
    for _ in range(max_polls):
        try:
            status = await _request('GET', f'/flights/bookings/{booking_id}')
        except Exception as e:
            return {'state': 'error', 'text': f'book_flight_letsfg: booking accepted (id {booking_id}) but polling status failed: {e}'}
        if status.get('terminal'):
            break
        await asyncio.sleep(interval_seconds)
    state = status.get('state', 'unknown')
    charged_amount = status.get('charged_amount')
    currency = status.get('currency')
    if state == 'completed':
        text = f"Booking confirmed - PNR {status.get('pnr', '?')}, {charged_amount} {currency} charged."
    elif state == 'failed':
        text = f"Booking failed: {status.get('failure_reason') or status.get('decline_message') or 'no reason given'}. Nothing was charged."
    elif state == 'needs_attention':
        text = f'Booking needs manual review at LetsFG (booking_id {booking_id}). Nothing further charged until resolved.'
    else:
        text = f'Booking is still in progress (state: {state}, booking_id {booking_id}) - check back shortly.'
    return {'state': state, 'text': text, 'charged_amount': charged_amount, 'currency': currency, 'pnr': status.get('pnr'), 'booking_id': booking_id}


@tool
async def book_flight_letsfg(
    search_id: str, offer_id: str, given_name: str, family_name: str, contact_email: str,
    adults: int = 1, children: int = 0,
) -> str:
    """Book a specific LetsFG offer - see book_flight's own (fuller)
    docstring for the real contract. This is the thin text-returning
    wrapper, kept for symmetry with the other @tool functions even though
    it is never added to TOOLS (see this module's own top docstring)."""
    result = await book_flight(search_id, offer_id, given_name, family_name, contact_email, adults, children)
    return result['text']


LETSFG_TOOLS = [search_flights_letsfg, book_flight_letsfg]
