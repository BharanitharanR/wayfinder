"""
Wayfinder's own session state - deliberately plain, deliberately separate
from mem0_backend.py/chat_cache.py/customer_record.py. Those exist for
ordinary WhatsApp conversation (fact extraction, rolling-window recent
turns, business-defined profile fields) and none of them model "pick this
customer's intake back up, with everything that's happened so far" the
way Wayfinder needs - see this session's own finding that no resumable
session object exists anywhere in this mesh yet.

Two plain Mongo collections, keyed by phone (E.164, as given):

  wayfinder_chat_turns   - one document per turn, append-only
  wayfinder_uploads      - already written by upload_links.py (piece B) -
                           this module only reads it back

build_state() marries (1) chat history and (2) the text already
extracted from that customer's uploaded documents - but a raw
concatenation of both just hands the next prompt a wall of text to
re-interpret from scratch every single call. The actual "marriage" is
infer_profile(): one LLM call that reads both and distills them into
the structured fields Wayfinder's intake actually tracks (preferences,
family/leave info, a budget signal, what's still missing) - so every
other skill reads a settled judgment, not raw material it has to
re-derive itself each time. Before this, nothing stitched those two
together at all - get_context_documents() was the missing retrieval
tool, infer_profile() is the missing inference step.
"""
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pymongo import ASCENDING, MongoClient

from mesh.lib import meta_model_client
from mesh.orchestrator import upload_links, wayfinder_tools

MONGO_URL = os.environ.get('ADIYAN_MONGO_URL', 'mongodb://localhost:27017')
MONGO_DB_NAME = os.environ.get('ADIYAN_MONGO_DB_DATA', 'adiyan')
CHAT_COLLECTION = 'wayfinder_chat_turns'
UPLOADS_COLLECTION = 'wayfinder_uploads'  # owned by upload_links.py

# Wayfinder calls Meta's Model API (mesh/lib/meta_model_client.py)
# directly, never AdiyanAgent/Claude CLI - this is the one agent submitted
# to the Meta AI Developer Hackathon, which requires calling only Meta
# models at runtime. Every other agent in this mesh is unaffected.
AGENT_ID = 'wayfinder_session'


class CustomerSnapshot(BaseModel):
    """The settled read of a customer's intake, as of this call - not raw
    transcript/document text, the inference drawn from it."""
    preferences_summary: str = Field(default='', description="What this family wants out of travel, in their own terms - pace, destinations, style. Empty if nothing's been said yet.")
    family_size: Optional[int] = Field(default=None, description='Number of family members traveling. Required before the profile is complete - never guess it; wait for it to actually be stated.')
    family_notes: str = Field(default='', description='Family composition details - ages, who travels. Leave-calendar specifics live in all_adults_leave_confirmed/family_size instead, not here.')
    budget_signal: str = Field(default='', description='A budget read grounded in any uploaded financial statement - a figure or range, not a guess with no basis.')
    origin_city: str = Field(default='', description='The city they would actually travel FROM, if stated. Empty if never mentioned - never guess one.')
    destination_hint: str = Field(default='', description='A specific place name they want to go, if one was actually named - a vague style preference ("beaches", "mountains") does not count as a destination.')
    travel_window: str = Field(default='', description='Specific or rough travel dates/month actually stated (e.g. "mid December 2026", "first week of January"). Empty if never mentioned - never guess one.')
    has_female_traveler: Optional[bool] = Field(default=None, description='Whether an adult female (wife, mother, etc.) is among the people actually traveling, per family_notes. null until family composition is known at all.')
    last_period_start: str = Field(default='', description='Her last period start date (YYYY-MM-DD), ONLY if has_female_traveler is true and a date was actually given. Empty otherwise - never guess one.')
    average_cycle_length_days: Optional[int] = Field(default=None, description='Her average cycle length in days, ONLY if has_female_traveler is true and a number was actually given. None otherwise - never guess one.')
    has_kids: Optional[bool] = Field(default=None, description='Whether any children are among the people actually traveling, per family_notes. null until family composition is known at all.')
    kids_school_calendar: str = Field(default='', description='School term dates/holiday windows for the traveling kids, ONLY if has_kids is true and this was actually given. Empty otherwise - never guess one.')
    all_adults_leave_confirmed: Optional[bool] = Field(default=None, description="True once EVERY working adult traveler's leave/availability has been given as its own specific statement naming that person, however it reached this chat - the registered customer relaying it (\"my leave is Dec 10-20, and my wife is also free that same window\") is completely normal and counts, since only one phone number is ever actually chatting here. What does NOT count is one vague blanket statement with no one named individually (\"we're both free that month\") - that only confirms whoever it's specifically about. False/null if any working adult's own dates haven't been named at all yet.")
    missing_info: List[str] = Field(default_factory=list, description='Which required items are still missing or too thin to act on. A trip can only actually be searched for once ALL of these are satisfied: preferences_summary, family_size, a financial statement, origin_city, destination_hint, travel_window, all_adults_leave_confirmed is true, and - only when applicable - (has_female_traveler true) last_period_start+average_cycle_length_days, and (has_kids true) kids_school_calendar. Phrase each missing item directly and plainly, e.g. "your wife\'s last period start date and average cycle length, so trip dates can avoid it", "your son\'s school term/holiday dates", "how many people are traveling in total", "your wife\'s own leave dates, not just yours".')
    summary: str = Field(default='', description='One short paragraph a planning step could read on its own, with no other context.')


_SNAPSHOT_PROMPT = """You are reading everything known so far about one Wayfinder customer, to distill it into a settled profile.

=== Conversation so far ===
{chat_transcript}

=== Extracted text from their uploaded documents ===
{documents_text}

Distill this into the customer's current profile. Only state what's actually supported by the text above - leave a field empty/null rather than guessing. For budget_signal, ground it in the uploaded statement text if present; don't invent a number. For origin_city, destination_hint, and travel_window, only fill these in if a specific city/place/date was actually stated - a mood or style ("beaches", "somewhere relaxing") is a preference, not a destination.

Set has_female_traveler and has_kids from family composition as soon as that's known at all (true/false, not null) - while either is still null (family composition not yet known), don't request the fields that depend on it. Once has_female_traveler is true, last_period_start and average_cycle_length_days are required. Once has_kids is true, kids_school_calendar is required. all_adults_leave_confirmed is true once every working adult's leave was named specifically for them (the registered customer relaying "I'm free X, my wife is free X too" is normal and counts - only one person is ever actually chatting here); it's just a vague unnamed "we're both free" with no one named individually that doesn't count.

List in missing_info every one of the required items (see that field's own description) that's still missing or too thin to use - phrase each one directly and plainly, naming who or what it's actually about."""


_client: Optional[MongoClient] = None


def _db():
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=3000)
    return _client[MONGO_DB_NAME]


def append_turn(phone: str, role: str, text: str) -> None:
    """role is 'customer' or 'wayfinder' - whichever side said `text`.
    Append-only, one document per turn: no array to grow unbounded inside
    a single Mongo document the way a $push-into-one-doc design would."""
    _db()[CHAT_COLLECTION].insert_one({
        'phone': phone,
        'role': role,
        'text': text,
        'at': datetime.now(timezone.utc).isoformat(),
    })


def get_chat_history(phone: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Oldest-first, capped at `limit` most recent turns - a plain,
    complete transcript for this phone number, not a semantic summary."""
    cursor = (
        _db()[CHAT_COLLECTION]
        .find({'phone': phone}, {'_id': 0})
        .sort('at', ASCENDING)
    )
    turns = list(cursor)
    return turns[-limit:]


def get_context_documents(phone: str) -> List[Dict[str, Any]]:
    """The tool that was missing: every successfully-read document this
    phone number has uploaded (see upload_links.py's `_extract_text` -
    status 'ok' only; a file still needing a password or that failed to
    parse has no text worth handing to a prompt). Reads the text already
    extracted and stored at upload time, not a live re-read of the PDF -
    the raw file still sits at `stored_path` on disk if it's ever needed
    again (a different extraction pipeline, a manual check), but normal
    state reconstruction has no reason to redo that work."""
    cursor = _db()[UPLOADS_COLLECTION].find(
        {'phone': phone, 'status': 'ok'},
        {'_id': 0, 'filename': 1, 'extracted_text': 1, 'uploaded_at': 1, 'financial_profile': 1},
    )
    return list(cursor)


def _best_financial_profile(documents: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The most recent trustworthy financial_profile (budget_engine.py's
    deterministic read of an uploaded statement), if any document has
    one. Picks trustworthy over untrustworthy, then most recently
    uploaded - never averages across multiple statements, since two
    different accounts' surpluses aren't additive without knowing if
    they're the same household's only accounts."""
    candidates = [d for d in documents if d.get('financial_profile') and 'error' not in d['financial_profile']]
    if not candidates:
        return None
    candidates.sort(key=lambda d: (d['financial_profile'].get('trustworthy', False), d.get('uploaded_at', '')), reverse=True)
    return candidates[0]['financial_profile']


def _deterministic_budget_signal(profile: Dict[str, Any]) -> str:
    surplus = profile.get('average_monthly_surplus')
    currency = profile.get('currency', '')
    ratio_pct = round(profile.get('classified_debit_ratio', 0) * 100)
    if profile.get('trustworthy') and surplus is not None:
        return f'Computed from the uploaded statement (deterministic, not an LLM guess): average monthly surplus ~{currency} {surplus:,.2f}, over {profile.get("months_analyzed", 0)} month(s), {ratio_pct}% of spending classified.'
    return (
        f'The uploaded statement could not be read confidently enough to state a number - only {ratio_pct}% of '
        'spending was classified (too many unlabeled transactions). Ask the customer to confirm their typical '
        'monthly surplus directly rather than trusting this statement alone.'
    )


def _format_chat_transcript(turns: List[Dict[str, Any]]) -> str:
    if not turns:
        return '(no conversation yet)'
    return '\n'.join(f"{t['role']}: {t['text']}" for t in turns)


def _format_documents_text(documents: List[Dict[str, Any]]) -> str:
    if not documents:
        return '(no documents uploaded yet)'
    return '\n\n'.join(f"--- {d['filename']} ---\n{d['extracted_text']}" for d in documents)


async def infer_profile(chat_history: List[Dict[str, Any]], documents: List[Dict[str, Any]]) -> CustomerSnapshot:
    """The actual marriage: one LLM call over both raw sources, returning
    the distilled CustomerSnapshot rather than leaving every caller to
    re-read the same transcript+documents text itself.

    budget_signal is then overridden with a deterministic read
    (budget_engine.py), when one of the uploaded documents has a computed
    financial_profile - the LLM's own free-text read of the statement is
    discarded in that case, not blended with it, since the whole point is
    one real, auditable number rather than a plausible-sounding guess.
    missing_info is adjusted to match: a budget-related ask the LLM added
    is dropped once a trustworthy number exists; one is added if the
    statement couldn't be read confidently, even if the LLM didn't notice."""
    prompt = _SNAPSHOT_PROMPT.format(
        chat_transcript=_format_chat_transcript(chat_history),
        documents_text=_format_documents_text(documents),
    )
    snapshot = await meta_model_client.ask_schema(AGENT_ID, 'infer_profile', prompt, CustomerSnapshot)

    financial_profile = _best_financial_profile(documents)
    if financial_profile is not None:
        snapshot.budget_signal = _deterministic_budget_signal(financial_profile)
        budget_related = lambda item: any(kw in item.lower() for kw in ('budget', 'statement', 'financial', 'surplus'))
        snapshot.missing_info = [item for item in snapshot.missing_info if not budget_related(item)]
        if not financial_profile.get('trustworthy'):
            snapshot.missing_info.append('a confirmed monthly budget/surplus figure, since the uploaded statement could not be read confidently enough on its own')

    return snapshot


async def build_state(phone: str, chat_limit: int = 100) -> Dict[str, Any]:
    """The one entrypoint a Wayfinder skill actually calls - marries chat
    history and uploaded-document context into a single, settled
    CustomerSnapshot (via infer_profile), resuming a customer exactly
    where they left off regardless of how long it's been since their
    last message. Raw chat_history/documents are still returned
    alongside it for anything that needs the source material itself."""
    chat_history = get_chat_history(phone, limit=chat_limit)
    documents = get_context_documents(phone)
    snapshot = await infer_profile(chat_history, documents)
    return {
        'phone': phone,
        'chat_history': chat_history,
        'documents': documents,
        'profile': snapshot,
    }


_GREETING_REPLY = (
    "Hi, I'm Wayfinder - I'll help plan your family's trips. To get started, "
    "tell me a bit about your travel preferences, your family, and share "
    "your leave calendar and a recent financial statement when you can."
)


def _missing_info_reply(missing_info: List[str]) -> str:
    items = ', '.join(missing_info)
    return f"Thanks - to keep going, could you also share: {items}?"


_RESEARCHING_PREFIX = "Got everything I need - let me find some real options for you.\n\n"


async def handle_customer_message(chat_id: str, text: str) -> str:
    """The actual intake/clarification loop - handle_message.py's new
    `vertical_id == 'wayfinder'` branch calls just this one function and
    uses its return value as the WhatsApp reply, the same shape every
    other branch in that file's if/elif chain already follows.

    `chat_id` is whatever WhatsApp hands this conversation as its id - for
    a "@lid" contact (confirmed - see mesh/lib/identity.py - carries no
    relation to the real phone number), an opaque id, not a dialable
    number. Every customer-data operation below runs against
    `identity_key`, not `chat_id` directly: once this contact has
    confirmed a real phone number through the upload page's mandatory
    field, THAT number is the one piece of durable identity Wayfinder
    actually links everything by - chat history, documents, profile -
    never the WhatsApp id itself, which is only ever used to look the
    confirmed number back up.

    Deliberately NOT the same thing as checking CustomerSnapshot.missing_
    info for a document-shaped phrase (LLM free text is too unreliable
    to branch control flow on) - whether to send the upload link is
    decided from the one fact that's actually deterministic: does this
    customer have zero successfully-read documents on file yet.
    Once documents exist, the LLM-inferred missing_info list (preferences,
    family/leave info) is what drives the next question - that's fine
    there because it's being shown to the customer as a prompt, not
    branched on in code.
    """
    confirmed_phone = upload_links.get_confirmed_phone(chat_id)
    identity_key = confirmed_phone or chat_id

    append_turn(identity_key, 'customer', text)

    chat_history = get_chat_history(identity_key)
    documents = get_context_documents(identity_key)
    is_first_message = len(chat_history) <= 1

    if not documents:
        # Before a phone is confirmed, the link has to be minted under
        # chat_id (confirmed_phone is None, so identity_key == chat_id
        # here too) - that's the only identity this contact has yet, and
        # it's exactly what create_upload_link's require_phone_entry gate
        # exists to replace. Once confirmed, every later link is minted
        # under the real number directly, so uploads never again land
        # under the WhatsApp id.
        require_phone_entry = confirmed_phone is None
        link = upload_links.create_upload_link(identity_key, require_phone_entry=require_phone_entry)
        reply = (
            f'{_GREETING_REPLY} ' if is_first_message else ''
        ) + f'Upload your documents here: {link}'
    else:
        profile = await infer_profile(chat_history, documents)
        if profile.missing_info:
            reply = _missing_info_reply(profile.missing_info)
        else:
            recommendation = await wayfinder_tools.recommend_trip(profile)
            reply = _RESEARCHING_PREFIX + recommendation

    append_turn(identity_key, 'wayfinder', reply)
    return reply
