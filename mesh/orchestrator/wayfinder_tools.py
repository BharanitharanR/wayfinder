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
from datetime import date
from typing import List, Optional

from langchain_core.tools import tool
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel, Field

from mesh.lib import meta_model_client
from mesh.lib.errors import describe_exception
from mesh.orchestrator.letsfg_tools import search_flights_letsfg

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


TOOLS = [web_search, search_flights, search_hotels, search_flights_letsfg]
# book_flight_letsfg is deliberately NOT in this list - see
# letsfg_tools.py's own docstring and wayfinder_session.py's booking-
# confirmation gate. A tool in TOOLS can be picked autonomously by the
# recommend_trip() ReAct loop below; booking real (or sandbox) money
# must never be something Muse Spark decides to do on its own turn -
# only wayfinder_session.py calls book_flight_letsfg directly, and only
# after the customer's own explicit yes against a specific stored offer.


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

Today's date is {today}. Use the creative discovery and evaluation evidence already gathered. If the destination is flexible, choose the strongest evidence-backed destination direction that fits the family's interests; if fixed, do not change it. Then look up actual transport and accommodation options. Decide what to look up next to put together a real recommendation - typically flights first, then a hotel at the destination. For flights, prefer search_flights_letsfg over search_flights: only a search_flights_letsfg offer can actually be booked later through this chat if the customer says yes, so use search_flights (Kiwi, link-out only) only if search_flights_letsfg genuinely fails or returns nothing useful. You have already made these tool calls this session:

{calls_so_far}

Choose exactly one tool call to make next - the single most useful next step given what's already been gathered above. If you already have both real flight options and real hotel options, there is nothing further to look up."""

_FINAL_RECOMMENDATION_PROMPT = """You are Wayfinder, writing the actual recommendation to send a customer whose intake is complete. Here is their profile:

{profile_summary}

Here is everything your research tool calls actually found:

{calls_so_far}

Write a short, concrete, creative recommendation grounded ONLY in the research above - explain why the discovered experiences fit the family's interests and cite actual prices and booking links that came back. Preserve inline source IDs such as [S1] beside every material factual claim, and retain the matching Sources list and URLs from the research results; do not strip or renumber citations. Never say a price, opening hour, availability or travel time is verified unless a cited source directly supports that exact claim. Search snippets and general articles are not live availability or price checks. If a research result has no URL citations, label its factual claims unverified. Never invent a number, hotel, attraction fact, or flight that wasn't in the tool results above. Do not call a trip affordable merely because its cost is below average monthly surplus; only a deterministic safe trip budget can establish affordability. If attraction costs or opening hours were not verified, say so and label them as items to confirm. If a tool call failed or found nothing, say so plainly rather than making something up to fill the gap. If a kids' school calendar was voluntarily provided, respect it. Do not request or infer menstrual/cycle information. Keep it to a few short paragraphs, suitable to send over WhatsApp.

If (and only if) a search_flights_letsfg call above returned a real offer_id for the flight you are actually recommending, also fill in bookable_offer with that exact search_id/offer_id/price/currency, PLUS the exact adults/children you passed into that same search_flights_letsfg call (a booking must match the same party size or it's rejected) - and end recommendation_text by asking the customer to reply yes to book it. Leave bookable_offer null if you're recommending a Kiwi (search_flights) result instead, or if you're not confident enough in a specific offer to recommend booking it yet - never invent a search_id/offer_id/adults/children that didn't come from a real tool result above."""


class ExperienceSearchTask(BaseModel):
    destination: str = Field(description='A destination or specific local area to investigate. Must respect a fixed destination if the customer named one.')
    rationale: str = Field(description='Why this destination or experience fits the family interests.')
    search_task: str = Field(description='A concrete live-web research query for family-friendly experiences, unusual attractions, practical logistics and indicative costs. Do not ask the search engine to invent prices.')


class CreativeDiscoveryPlan(BaseModel):
    tasks: List[ExperienceSearchTask] = Field(description='Up to three distinct research directions. If the destination is fixed, investigate different experience themes there. If destination_flexible is true, explore genuinely different destinations that fit the family.')


class ExperienceEvaluation(BaseModel):
    interest_fit: int = Field(ge=1, le=5, description='How well the evidence matches the family interests, 1 to 5.')
    novelty: int = Field(ge=1, le=5, description='How distinctive and non-generic the discovered experiences are, 1 to 5.')
    practicality: int = Field(ge=1, le=5, description='How well distance, dates, family suitability and logistics are supported, 1 to 5.')
    evidence_gaps: List[str] = Field(default_factory=list, description='Important facts not verified by research, especially costs, opening days, distance or availability.')
    next_search_task: str = Field(default='', description='One additional targeted research query if a material evidence gap can be resolved. Empty when no useful additional search is needed.')
    evaluation_summary: str = Field(description='Brief critique of what is promising and what remains uncertain. Do not claim affordability without a deterministic budget.')


async def discover_experiences(profile) -> tuple[str, str]:
    """Muse Spark proposes a small set of interest-led research directions;
    Meta web search then grounds them in real-world evidence. The model
    proposes what to investigate, but search results remain the evidence."""
    destination = profile.destination_hint or '(no fixed destination)'
    flexibility = bool(getattr(profile, 'destination_flexible', False))
    prompt = f"""You are Wayfinder's creative experience-discovery planner.

Family profile:
{_format_profile(profile)}

Destination: {destination}
Destination flexibility explicitly stated by customer: {flexibility}

Create at most three genuinely useful research directions.
- If a destination is named and destination_flexible is false, keep research within that destination and nearby areas; do not replace it.
- If destination_flexible is true, suggest distinct destinations that make sense from the stated origin, dates, family interests and financial signal. Do not assume a numeric trip budget from average monthly surplus.
- Discover experiences, not just famous tourist destinations: science/aviation/technology, nature, culture, local food, hands-on activities and lesser-known attractions where relevant to stated interests.
- Explain why each direction fits this particular family's stated interests.
- Search queries must ask for source-backed attractions, family suitability, travel time/distance, opening days and published prices where available. Never fabricate any of those facts.
- Prefer variety across the three directions rather than three versions of the same generic destination.
"""
    plan = await meta_model_client.ask_schema(AGENT_ID, 'creative_discovery_plan', prompt, CreativeDiscoveryPlan)
    findings = []
    for task in plan.tasks[:3]:
        result = await meta_model_client.web_search(AGENT_ID, 'experience_discovery', task.search_task)
        findings.append(
            f"DISCOVERY: {task.destination}\nWHY IT MAY FIT: {task.rationale}\n"
            f"RESEARCH QUERY: {task.search_task}\nGROUNDED FINDINGS:\n{result}"
        )
    return '\n\n'.join(findings) if findings else '(No experience discovery results.)', destination


async def evaluate_experience_research(profile, research: str) -> ExperienceEvaluation:
    """A separate critique pass identifies evidence gaps and may request one
    targeted follow-up search. It cannot approve affordability."""
    prompt = f"""Evaluate this Wayfinder experience research against the family's interests and trip context.

PROFILE:
{_format_profile(profile)}

RESEARCH EVIDENCE:
{research}

Score interest fit, novelty and practicality from 1 to 5. List material evidence gaps. If one additional search can resolve a meaningful gap, provide exactly one targeted search query; otherwise leave it empty.
Do not infer prices, availability, opening hours or affordability. The deterministic budget engine, not you, owns affordability. The current statement profile may contain average monthly surplus only; that is NOT a safe total trip budget. Unless a separately calculated safe_trip_budget is explicitly supplied, present researched costs as estimates and say affordability is not yet verified."""
    return await meta_model_client.ask_schema(AGENT_ID, 'evaluate_experience_research', prompt, ExperienceEvaluation)


class BookableOffer(BaseModel):
    search_id: str = Field(description='The exact search_id a search_flights_letsfg call above returned.')
    offer_id: str = Field(description='The exact offer_id of the one specific flight offer being recommended.')
    summary: str = Field(description='A short human-readable description of this flight (route, date, airline) for logging/display.')
    price: float = Field(description='The exact price shown for this offer.')
    currency: str = Field(description='The currency code for that price.')
    adults: int = Field(description='The exact `adults` value used in the search_flights_letsfg call that returned this offer - a booking must match the same party size or LetsFG rejects it (party_mismatch).')
    children: int = Field(default=0, description='The exact `children` value used in that same search_flights_letsfg call.')


class TripRecommendation(BaseModel):
    recommendation_text: str = Field(description='The customer-facing recommendation message, suitable to send over WhatsApp as-is.')
    bookable_offer: Optional[BookableOffer] = Field(default=None, description='Set only when recommending a specific search_flights_letsfg offer the customer could book by replying yes. Null otherwise - never invented.')


def _format_profile(profile) -> str:
    lines = [
        f"- Family: {profile.family_size or '?'} people - {profile.family_notes or '(no notes)'}",
        f"- Preferences: {profile.preferences_summary or '(none stated)'}",
        f"- Budget: {profile.budget_signal or '(not grounded)'}",
        f"- From: {profile.origin_city or '(unknown)'}",
        f"- To: {profile.destination_hint or ('open to discovery' if getattr(profile, 'destination_flexible', False) else '(not selected)')}",
        f"- Candidate destinations: {', '.join(getattr(profile, 'destination_options', []) or []) or '(none)' }",
        f"- When: {profile.travel_window or '(unknown)' }",
        f"- Destination flexibility: {'yes' if getattr(profile, 'destination_flexible', False) else 'no'}",
    ]
    if getattr(profile, 'has_kids', False) and getattr(profile, 'kids_school_calendar', ''):
        lines.append(f"- Kids' school calendar (avoid clashing with term dates): {profile.kids_school_calendar}")
    return '\n'.join(lines)


async def _emit_progress(progress_callback, stage: str, status: str, message: str) -> None:
    """Send customer-safe milestones without exposing prompts or internal reasoning."""
    if progress_callback is None:
        return
    try:
        result = progress_callback({'type': 'stage_update', 'stage': stage,
                                    'status': status, 'message': message})
        if hasattr(result, '__await__'):
            await result
    except Exception:
        # A disconnected UI must never break trip planning.
        return


async def recommend_trip(profile, progress_callback=None) -> TripRecommendation:
    """The actual "prescribe a trip" step, called once a CustomerSnapshot
    has nothing left in missing_info - a small, bounded ReAct loop (not
    Analysis Agent's full multi-turn one): up to 3 rounds of "choose one
    tool call, run it for real, feed the real result back", then one
    final schema call that writes the customer-facing recommendation
    text AND, when applicable, the exact bookable LetsFG offer it's
    recommending - from only what those real calls actually returned.
    Grounding is enforced in the prompt (never invent a number/flight/
    hotel/offer id), not by code - there's no deterministic way to
    verify an LLM's prose stayed grounded, same limitation every other
    free-text stage in this file already has. Returns the structured
    TripRecommendation, not a bare string, so wayfinder_session.py can
    store bookable_offer as the one thing a later "yes" is allowed to
    book - never a model's free-text reconstruction of it."""
    profile_summary = _format_profile(profile)
    calls_so_far: list[str] = []

    # Start with divergent, interest-led experience discovery rather than
    # assuming the customer already knows which destination to search.
    await _emit_progress(progress_callback, 'DISCOVERY', 'IN_PROGRESS',
                         'Exploring destinations and experiences that match your family.')
    discovery, _ = await discover_experiences(profile)
    calls_so_far.append(f'[creative experience discovery]\\n{discovery}')
    evaluation = await evaluate_experience_research(profile, discovery)
    calls_so_far.append(f'[experience evaluation]\\n{evaluation.model_dump_json(indent=2)}')
    if evaluation.next_search_task:
        follow_up = await meta_model_client.web_search(AGENT_ID, 'experience_research_replan', evaluation.next_search_task)
        calls_so_far.append(f'[targeted follow-up research: {evaluation.next_search_task}]\\n{follow_up}')
    await _emit_progress(progress_callback, 'DISCOVERY', 'COMPLETED',
                         'Experience research is complete; comparing the strongest directions.')
    await _emit_progress(progress_callback, 'AFFORDABILITY', 'NEEDS_ATTENTION',
                         'Reviewing your financial context. A safe total trip limit is not yet verified.')

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
        stage = ('TRANSPORT' if 'flight' in name.lower()
                 else 'ACCOMMODATION' if 'hotel' in name.lower()
                 else 'DISCOVERY')
        stage_label = ('Searching for suitable flight options.' if stage == 'TRANSPORT'
                       else 'Comparing family-friendly accommodation.' if stage == 'ACCOMMODATION'
                       else 'Checking additional destination details.')
        await _emit_progress(progress_callback, stage, 'IN_PROGRESS', stage_label)
        try:
            result = await tool_fn.ainvoke(args)
            failed = isinstance(result, str) and (' failed:' in result.lower() or result.lower().startswith('failed:'))
        except Exception as e:
            result = f'failed: {describe_exception(e)}'
            failed = True
        message = ('The search could not be completed; the final answer will explain the limitation.'
                   if failed else
                   'Flight research is complete.' if stage == 'TRANSPORT' else
                   'Accommodation research is complete.' if stage == 'ACCOMMODATION' else
                   'Additional destination research is complete.')
        await _emit_progress(progress_callback, stage, 'FAILED' if failed else 'COMPLETED', message)
        calls_so_far.append(f'[{name}({args})]\n{result}')

    await _emit_progress(progress_callback, 'FINALIZING', 'IN_PROGRESS',
                         'Bringing the research together into your trip recommendations.')
    final_prompt = _FINAL_RECOMMENDATION_PROMPT.format(
        profile_summary=profile_summary,
        calls_so_far='\n\n'.join(calls_so_far) if calls_so_far else '(no research calls were made)',
    )
    recommendation = await meta_model_client.ask_schema(AGENT_ID, 'final_recommendation', final_prompt, TripRecommendation)
    await _emit_progress(progress_callback, 'FINALIZING', 'COMPLETED',
                         'Your trip recommendations are ready.')
    return recommendation
