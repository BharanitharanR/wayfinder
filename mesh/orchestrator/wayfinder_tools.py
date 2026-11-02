"""
Wayfinder's own research tools - real external lookups a trip-planning
call can choose to make, not stubs. Three tools, each wired to an actual
working integration:

  web_search       - Meta's own built-in search-grounding tool
                     (mesh/lib/meta_model_client.py's web_search() - the
                     model searches the live web itself, server-side, no
                     local browser agent or permission token needed).
                     Covers Reddit/forum research too - see its own
                     docstring for why there's no dedicated
                     search_reddit tool.
  search_flights   - Kiwi.com's own public MCP server (mcp.kiwi.com) -
                     confirmed live, no API key, real flight search.
  search_hotels    - Trivago's own public MCP server (mcp.trivago.com) -
                     confirmed live, no API key, real hotel metasearch.

Both remote MCP servers' tool names and input schemas were pulled live
from their own `list_tools()` response, not assumed from marketing copy -
see this session's own probe for the exact JSON schemas this module's
argument names are drawn from. A fourth candidate, Reddit's own public
search API, was tried and dropped: it 403s without real OAuth app
credentials now, and spoofing a browser User-Agent to get past that is
bot-check circumvention, not a fix - web_search covers the same need
(searching Reddit as part of an ordinary web query) without it.

Every call in this module - research()'s and recommend_trip()'s tool
choice, the final synthesis, and web_search's own live lookup - goes
through mesh/lib/meta_model_client.py, i.e. Meta's Model API (Muse
Spark), never AdiyanAgent/Claude CLI. web_search used to fall back to
Claude CLI web research, then to this mesh's own local browser-automation
MCP agent; both were removed, not just made unreachable - the Meta AI
Developer Hackathon requires this submitted agent to call only Meta
models at runtime, and Meta's own search-grounding tool (built into
Model API) replaces both without needing a separate browser agent
process or permission token running at all - one less thing a judge
running this standalone needs up.

Deliberately NOT wired into infer_profile's call (mesh/orchestrator/
wayfinder_session.py) - that stage is intake/clarification, grounded only
in what the customer themselves said and uploaded. These are for the
actual trip-matching step the original spec explicitly left as future
work; research() below is a minimal, single-choice entry point for that,
not a multi-turn ReAct loop.
"""
from datetime import date, timedelta
from typing import List, Optional

from langchain_core.tools import tool
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from mesh.lib import meta_model_client
from mesh.lib.errors import describe_exception

AGENT_ID = 'wayfinder_tools'
KIWI_MCP_URL = 'https://mcp.kiwi.com'
TRIVAGO_MCP_URL = 'https://mcp.trivago.com/mcp'


async def _call_public_mcp_tool(server_url: str, tool_name: str, arguments: dict, result_key: Optional[str] = None):
    """Like mesh.lib.mcp_client.call_tool, but for a real external
    internet service, not one of this mesh's own local/self-signed
    servers - deliberately does NOT reuse that helper's insecure
    (verify=False) httpx client, since that's only safe for the
    self-signed internal services it was written for. Full TLS
    verification, no Authorization header (neither public server needs
    or expects one - confirmed live via a clean list_tools() with no
    auth).

    `result_key`: some of these servers' structuredContent carries more
    than the data itself - confirmed live, Trivago's own reply also
    includes a `system_message` field of UI-formatting instructions
    meant for a chat client rendering photos, not for a tool caller
    consuming the data programmatically. Pass the key that holds the
    actual payload to get just that back; omit it for a server (like
    Kiwi's) whose structuredContent IS the payload."""
    async with streamablehttp_client(server_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            if result.isError:
                raise RuntimeError(f'{tool_name} failed: {result.content}')
            content = result.structuredContent
            if isinstance(content, dict) and set(content.keys()) == {'result'}:
                content = content['result']
            if result_key is not None and isinstance(content, dict):
                return content.get(result_key)
            return content


@tool
async def web_search(task: str) -> str:
    """Look something up on the live internet - current prices, seasonal
    advisories, real traveler opinion on Reddit or forums (e.g. "search
    Reddit for discussion on whether Goa is safe for a family with a 9
    year old in December"), anything time-sensitive that the customer's
    own documents won't cover. Give a complete, specific instruction, not
    a bare topic - name the site to check (Reddit, a forum) when that's
    what's actually wanted, since this tool browses the live web rather
    than calling any one site's own API."""
    try:
        return await meta_model_client.web_search(AGENT_ID, 'web_search', task)
    except Exception as e:
        return f'web_search failed: {describe_exception(e)}'


@tool
async def search_flights(
    fly_from: str, fly_to: str, departure_date: str,
    return_date: Optional[str] = None, adults: int = 1, children: int = 0,
) -> str:
    """Search real flights via Kiwi.com. `fly_from`/`fly_to` are IATA
    codes, city, or airport names (e.g. "DEL", "Chennai", "London
    Heathrow") - resolved automatically, no need to look up a code
    yourself. `departure_date`/`return_date` are dd/mm/yyyy; omit
    `return_date` for one-way. Returns real itineraries with price,
    route, times, and a real booking link - never invent a price or
    flight if this call fails or returns nothing."""
    args = {'flyFrom': fly_from, 'flyTo': fly_to, 'departureDate': departure_date, 'adults': adults, 'children': children}
    if return_date:
        args['returnDate'] = return_date
    try:
        data = await _call_public_mcp_tool(KIWI_MCP_URL, 'search-flight', args)
    except Exception as e:
        return f'search_flights failed: {describe_exception(e)}'
    itineraries = (data or {}).get('itineraries') or []
    if not itineraries:
        return f'No flights found from {fly_from} to {fly_to} on {departure_date}.'
    lines = []
    for it in itineraries[:8]:
        route = ' -> '.join(it.get('outbound', {}).get('route') or [])
        lines.append(f"- {route}: {it.get('priceFormatted', '?')}, {it.get('totalDurationSeconds', 0) // 60} min, book: {it.get('bookingUrl', '?')}")
    return '\n'.join(lines)


@tool
async def search_hotels(
    query: str, arrival: str, departure: str, adults: int = 1, children: int = 0, rooms: int = 1,
) -> str:
    """Search real hotel/accommodation listings via Trivago. `query` is a
    specific property name, region, or country (for a city, neighborhood
    or landmark, a more specific query still works but trivago's own
    radius-search tool - not wired here - would normally be preferred).
    `arrival`/`departure` are YYYY-MM-DD, `arrival` must be today or
    later and before `departure`. Returns real listings with live
    pricing - never invent a hotel or price if this call fails or
    returns nothing."""
    args = {
        'query': query, 'arrival': arrival, 'departure': departure,
        'adults': adults, 'children': children, 'rooms': rooms,
    }
    try:
        listings = await _call_public_mcp_tool(
            TRIVAGO_MCP_URL, 'trivago-accommodation-search', args, result_key='accommodations',
        )
    except Exception as e:
        return f'search_hotels failed: {describe_exception(e)}'
    if not listings:
        return f'No hotels found for "{query}".'
    lines = []
    for h in listings:
        deals = h.get('deals') or []
        price = f"{deals[0]['price_per_night']} ({deals[0].get('advertiser', '?')})" if deals else 'price unavailable'
        link = deals[0].get('deal_url', '?') if deals else '?'
        lines.append(
            f"- {h.get('accommodation_name', '?')} ({h.get('country_city', '?')}) - "
            f"{price}/night, rating {h.get('hotel_rating', '?')}* / review {h.get('review_rating', '?')} "
            f"({h.get('review_count', '0')} reviews), amenities: {h.get('top_amenities', '-')}, book: {link}"
        )
    return '\n'.join(lines)


TOOLS = [web_search, search_flights, search_hotels]


async def research(query: str) -> str:
    """The minimal entry point: hand the query to the LLM with both tools
    available, let it pick exactly one (meta_model_client.ask_tools' same
    contract AdiyanAgent.ask(tools=...) had - returns the chosen call, does
    not execute it), then actually run that tool and return its real
    result. Single-shot by design - not a multi-turn ReAct loop like
    Analysis Agent's; good enough to ground one research question, not to
    plan an entire trip end to end."""
    decision = await meta_model_client.ask_tools(AGENT_ID, 'research', query, TOOLS)
    calls = getattr(decision, 'tool_calls', None)
    if not calls:
        return 'No tool call was chosen.'
    call = calls[0]
    name, args = call['name'], call['args']
    for t in TOOLS:
        if t.name == name:
            return await t.ainvoke(args)
    return f'Chose unknown tool "{name}".'


_RECOMMEND_PROMPT = """You are Wayfinder, planning an actual trip for a customer whose intake is complete. Here is everything known about them:

{profile_summary}

Today's date is {today}. Decide what to look up next to put together a real recommendation - typically flights first, then a hotel at the destination, though use your judgment about order and whether a web_search for destination-specific advice is also worth doing. You have already made these tool calls this session:

{calls_so_far}

Choose exactly one tool call to make next - the single most useful next step given what's already been gathered above. If you already have both real flight options and real hotel options, there is nothing further to look up."""

_FINAL_RECOMMENDATION_PROMPT = """You are Wayfinder, writing the actual recommendation to send a customer whose intake is complete. Here is their profile:

{profile_summary}

Here is everything your research tool calls actually found:

{calls_so_far}

Write a short, concrete recommendation grounded ONLY in the real data above - cite the actual prices and booking links that came back, never invent a number, hotel, or flight that wasn't in the tool results above. If a tool call failed or found nothing, say so plainly rather than making something up to fill the gap. If predicted period windows or a kids' school calendar were given in the profile above, prefer picking/confirming travel dates that avoid both, within whatever the stated travel_window/leave dates actually allow - if every date in the leave window conflicts with one of them, say so plainly rather than silently picking a date that conflicts. Keep it to a few short paragraphs, suitable to send over WhatsApp."""


def predict_period_windows(last_period_start: str, cycle_length_days: int, period_length_days: int = 5, horizon_days: int = 90) -> List[str]:
    """Pure local calendar-based prediction - no AI model, no external
    service, nothing about this leaves the machine. Standard calendar-
    method cycle prediction is just arithmetic (next period = last start
    + cycle length); there's no reason sensitive health data like this
    should ever touch a third-party API to compute something this
    simple - see this session's own research into existing APIs/MCP
    servers for this, all of which would mean sending it off-device.

    Returns each predicted window as 'YYYY-MM-DD to YYYY-MM-DD', every
    `cycle_length_days` from `last_period_start` out to `horizon_days` -
    enough windows for a date-picking step to check a travel_window
    against, without needing today's date passed in separately."""
    start = date.fromisoformat(last_period_start)
    windows = []
    cursor = start
    while (cursor - start).days <= horizon_days:
        windows.append(f'{cursor.isoformat()} to {(cursor + timedelta(days=period_length_days - 1)).isoformat()}')
        cursor += timedelta(days=cycle_length_days)
    return windows


def _format_profile(profile) -> str:
    lines = [
        f"- Family: {profile.family_size or '?'} people - {profile.family_notes or '(no notes)'}",
        f"- Preferences: {profile.preferences_summary or '(none stated)'}",
        f"- Budget: {profile.budget_signal or '(not grounded)'}",
        f"- From: {profile.origin_city or '(unknown)'}",
        f"- To: {profile.destination_hint or '(unknown)'}",
        f"- When: {profile.travel_window or '(unknown)'}",
    ]
    if getattr(profile, 'last_period_start', '') and getattr(profile, 'average_cycle_length_days', None):
        windows = predict_period_windows(profile.last_period_start, profile.average_cycle_length_days)
        lines.append(f"- Predicted period windows to avoid when picking dates: {', '.join(windows)}")
    if getattr(profile, 'has_kids', False) and getattr(profile, 'kids_school_calendar', ''):
        lines.append(f"- Kids' school calendar (avoid clashing with term dates): {profile.kids_school_calendar}")
    return '\n'.join(lines)


async def recommend_trip(profile) -> str:
    """The actual "prescribe a trip" step, called once a CustomerSnapshot
    has nothing left in missing_info - a small, bounded ReAct loop (not
    Analysis Agent's full multi-turn one): up to 3 rounds of "choose one
    tool call, run it for real, feed the real result back", then one
    final plain-text call that writes the customer-facing recommendation
    from only what those real calls actually returned. Grounding is
    enforced in the prompt (never invent a number/flight/hotel), not by
    code - there's no deterministic way to verify an LLM's prose stayed
    grounded, same limitation every other free-text stage in this file
    already has."""
    profile_summary = _format_profile(profile)
    calls_so_far: list[str] = []

    for _ in range(3):
        prompt = _RECOMMEND_PROMPT.format(
            profile_summary=profile_summary, today=date.today().isoformat(),
            calls_so_far='\n\n'.join(calls_so_far) if calls_so_far else '(none yet)',
        )
        decision = await meta_model_client.ask_tools(AGENT_ID, 'recommend_trip', prompt, TOOLS)
        calls = getattr(decision, 'tool_calls', None)
        if not calls:
            break
        call = calls[0]
        name, args = call['name'], call['args']
        tool_fn = next((t for t in TOOLS if t.name == name), None)
        if tool_fn is None:
            calls_so_far.append(f'[{name}] unknown tool chosen - skipped.')
            continue
        try:
            result = await tool_fn.ainvoke(args)
        except Exception as e:
            result = f'failed: {describe_exception(e)}'
        calls_so_far.append(f'[{name}({args})]\n{result}')

    final_prompt = _FINAL_RECOMMENDATION_PROMPT.format(
        profile_summary=profile_summary,
        calls_so_far='\n\n'.join(calls_so_far) if calls_so_far else '(no research calls were made)',
    )
    return await meta_model_client.ask_text(AGENT_ID, 'final_recommendation', final_prompt)
