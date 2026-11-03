"""
Wayfinder interaction interpreter.

This module translates the customer's latest message plus the settled trip
context into a typed interaction decision. It does not execute tools, mutate
the customer profile, calculate budgets, or perform bookings. Those remain
the responsibility of deterministic application code and the existing
permission-gated booking path.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from mesh.lib import meta_model_client


AGENT_ID = 'wayfinder_interaction'
MAX_CONTEXT_TURNS = 12


InteractionIntent = Literal[
    'NEW_TRIP',
    'MODIFY_TRIP',
    'ASK_QUESTION',
    'ANALYZE_FINANCES',
    'EXPLORE_OPTIONS',
    'COMPARE_OPTIONS',
    'CONFIRM_ACTION',
    'PROVIDE_INFORMATION',
    'OTHER',
]


class InteractionDecision(BaseModel):
    """A structured interpretation of one incoming customer message."""

    intent: InteractionIntent = Field(
        description=(
            'The primary intent of the latest customer message. MODIFY_TRIP '
            'means retain all unchanged trip constraints and change only '
            'what the customer named.'
        )
    )
    objective: str = Field(
        description='The outcome the customer is trying to achieve, in one concise sentence.'
    )
    changed_constraints: List[str] = Field(
        default_factory=list,
        description='Only trip constraints explicitly changed by this message, e.g. destination or dates.'
    )
    preserved_constraints: List[str] = Field(
        default_factory=list,
        description='Known trip constraints that remain applicable to this request.'
    )
    required_actions: List[Literal[
        'UPDATE_PROFILE',
        'SEARCH_FLIGHTS',
        'SEARCH_HOTELS',
        'WEB_RESEARCH',
        'COMPARE_RESULTS',
        'ANSWER_FROM_CONTEXT',
        'ASK_CLARIFICATION',
        'REQUEST_CONFIRMATION',
        'CONTINUE_INTAKE',
    ]] = Field(
        default_factory=list,
        description='Ordered high-level actions needed to satisfy the objective. These are proposals, not executable tool calls.'
    )
    missing_information: List[str] = Field(
        default_factory=list,
        description='Only information genuinely required before the next useful action can happen.'
    )
    response_strategy: Literal[
        'ACT',
        'ANSWER',
        'CLARIFY',
        'CONFIRM',
        'CONTINUE_INTAKE',
    ] = Field(
        description='What the application should do next at a high level.'
    )


_INTERPRET_PROMPT = """You are the interaction interpreter for Wayfinder, a family travel-planning assistant.

Interpret ONLY the customer's latest message in the context of the existing conversation and current trip profile. Wayfinder's core purpose is travel planning; financial analysis is a supporting capability used to establish safe travel affordability.

CURRENT TRIP PROFILE:
{profile_context}

RECENT CONVERSATION (oldest to newest):
{conversation_context}

LATEST CUSTOMER MESSAGE:
{latest_message}

Rules:
- Distinguish a new trip from a modification to the current trip.
- For a modification, preserve every known constraint the customer did not change. Example: if they change Goa to Kerala, retain dates, origin, family size, budget and preferences.
- A question about income, expenses, monthly surplus, spending categories or the uploaded statement is ANALYZE_FINANCES, not a trip request.
- A question about a recommendation is usually ASK_QUESTION, not a new trip.
- Requests for alternatives or destination ideas are EXPLORE_OPTIONS.
- Requests to compare named or previously returned options are COMPARE_OPTIONS.
- A clear yes/no about an existing specific pending action is CONFIRM_ACTION. A vague acknowledgement is not confirmation.
- If the customer supplies missing profile details, use PROVIDE_INFORMATION and propose UPDATE_PROFILE or CONTINUE_INTAKE.
- Never infer dates, party size, budget, destination, or consent.
- Do not propose booking or payment as an autonomous action. Any booking remains behind the application's explicit confirmation gate.
- Do not ask for information already present in the profile or conversation.
- Keep required_actions minimal and ordered. Do not request flights/hotels if the customer is only asking a general question.
- If there is enough context to answer, prefer ANSWER over asking an unnecessary question.
- If a required detail is genuinely absent, use CLARIFY and name only that detail.
"""


def _format_profile(profile) -> str:
    if profile is None:
        return '(no settled profile yet)'
    fields = (
        'summary',
        'preferences_summary',
        'family_size',
        'family_notes',
        'budget_signal',
        'origin_city',
        'destination_hint',
        'travel_window',
        'missing_info',
    )
    lines = []
    for name in fields:
        value = getattr(profile, name, None)
        if value not in (None, '', []):
            lines.append(f'- {name}: {value}')
    return '\n'.join(lines) if lines else '(no settled profile fields)'


def _format_history(history: Optional[list]) -> str:
    if not history:
        return '(no previous conversation)'
    turns = history[-MAX_CONTEXT_TURNS:]
    return '\n'.join(
        f"{turn.get('role', 'unknown')}: {turn.get('text', '')}"
        for turn in turns
    )


async def interpret_interaction(
    latest_message: str,
    profile=None,
    chat_history: Optional[list] = None,
) -> InteractionDecision:
    """Return a typed interpretation; never executes the proposed actions."""
    prompt = _INTERPRET_PROMPT.format(
        profile_context=_format_profile(profile),
        conversation_context=_format_history(chat_history),
        latest_message=latest_message,
    )
    return await meta_model_client.ask_schema(
        AGENT_ID,
        'interpret_interaction',
        prompt,
        InteractionDecision,
    )
