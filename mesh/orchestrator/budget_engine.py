"""
Deterministic trip-affordability engine, adapted from a standalone prototype
the user built and handed over for review (deterministic_travel_budget_
engine.zip - an HDFC-only PDF parser + classifier + budget formula). That
prototype's own test run against a real statement left 93% of debits
UNKNOWN (see its sample_statement_analysis.json: ~INR 33.9L of ~36.5L in
debits), and its parser only understands one bank's exact PDF text layout -
neither is good enough to plug in as-is.

The fix keeps the prototype's actually-good part (budget.py's formula -
pure arithmetic, reconciliation-checked, no LLM) and replaces its weak
part (a bank-specific regex parser) with the one thing this project
already has that's bank-agnostic: Muse Spark, which already turned the
PDF into clean markdown (mesh.lib.meta_model_client.extract_document_text,
called from upload_links.py). normalize_transactions() below is the one
LLM call in this whole pipeline - a single schema-constrained extraction
turning that markdown into a strict transaction list, not a second
freehand read. Classification, aggregation, and the budget formula itself
are the ported prototype's own deterministic code, unchanged in spirit:
same keyword-rule-first-match-wins classifier, same "unknown debits are
never silently essential/discretionary" rule, same available_savings +
future_savings formula.

This module does NOT attempt the full available_savings/future_savings
trip-budget figure yet - that needs current_balance, emergency_reserve,
upcoming_obligations, and months_until_departure, none of which Wayfinder
collects from the customer today. compute_financial_profile() stops at a
grounded, auditable income/expense/surplus read (what budget_signal
needed to stop being free LLM prose) and flags low-confidence profiles
plainly rather than feeding a shaky number into a formula anyway.
"""
import re
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from mesh.lib import meta_model_client

RULES_PATH = Path(__file__).parent / 'config' / 'wayfinder_classification_rules.json'

# Below this confidence (fraction of total debit value actually
# classified), the computed surplus isn't trustworthy enough to state as
# a number - see compute_financial_profile's own docstring.
MIN_CLASSIFIED_DEBIT_RATIO = 0.55


class StatementTransaction(BaseModel):
    date: str = Field(description='ISO date (YYYY-MM-DD) of the transaction.')
    description: str = Field(description='The transaction description/narration exactly as written in the statement.')
    amount: float = Field(description='The transaction amount as a positive number, in the statement\'s own currency - never negative, direction is a separate field.')
    direction: Literal['credit', 'debit'] = Field(description='Whether this was money in (credit) or money out (debit) for the account holder.')


class StatementTransactions(BaseModel):
    currency: str = Field(description='The statement\'s currency, as a code if stated (e.g. INR, USD) or best inferred from symbols/context.')
    transactions: List[StatementTransaction] = Field(description='Every transaction line actually present in the statement text, in the order they appear. Do not invent, merge, or skip any row.')


_NORMALIZE_PROMPT = """Here is the extracted text of a bank/financial statement (already read from the original document):

{statement_text}

List every transaction actually present, in order, as structured data. Each transaction needs its date, its description exactly as written, its amount as a positive number, and whether it was money in (credit) or money out (debit). Do not summarize, merge rows, or invent a transaction that isn't there - if the statement has 40 rows, return 40 transactions."""


def _is_statement_like(text: str) -> bool:
    """Cheap, local, zero-cost prefilter - run before spending a Meta call
    on normalize_transactions. Not every uploaded document is a financial
    statement (a leave calendar, a school holiday list); this just avoids
    calling the model on ones that obviously aren't, by checking for the
    shape money tables actually have (several decimal-money-looking
    tokens plus at least one banking-ish keyword)."""
    money_like = re.findall(r'\d{1,3}(?:,\d{2,3})*\.\d{2}', text)
    if len(money_like) < 5:
        return False
    lowered = text.lower()
    return any(kw in lowered for kw in ('balance', 'credit', 'debit', 'statement', 'account', 'transaction'))


async def normalize_transactions(agent_id: str, statement_text: str) -> StatementTransactions:
    """The one LLM call in this pipeline - Muse Spark again (same backend
    as extract_document_text), but a schema-constrained structural
    extraction, not a budget judgment: turning markdown/text whose table
    layout varies bank to bank into one fixed transaction shape. Model API
    constrains decoding to the schema (see 'Structured output' in the
    Model API docs), so the output always parses - it doesn't guarantee
    byte-identical results run to run the way a bank-specific regex
    parser would, which is the real tradeoff of handling arbitrary bank
    formats instead of one known layout."""
    return await meta_model_client.ask_schema(
        agent_id, 'normalize_transactions',
        _NORMALIZE_PROMPT.format(statement_text=statement_text),
        StatementTransactions,
    )


def _load_rules() -> list:
    import json
    if not RULES_PATH.exists():
        return []
    return json.loads(RULES_PATH.read_text()).get('rules', [])


def classify_transactions(transactions: List[StatementTransaction]) -> List[dict]:
    """Deterministic keyword rules, first match wins - ported unchanged
    from the prototype's classify_transactions(). An unmatched debit
    stays UNKNOWN, never silently folded into essential/discretionary -
    that's the one rule in the prototype worth keeping exactly as strict
    as it was."""
    rules = _load_rules()
    classified = []
    for t in transactions:
        desc = t.description.upper()
        category = 'UNKNOWN'
        for rule in rules:
            if any(term.upper() in desc for term in rule['contains']):
                category = rule['category']
                break
        classified.append({'date': t.date, 'description': t.description, 'amount': t.amount, 'direction': t.direction, 'category': category})
    return classified


def _money(x) -> Decimal:
    return Decimal(str(x)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


def build_profile(classified: List[dict]) -> dict:
    """Monthly income/expense aggregation plus a real monthly_surplus
    figure - the prototype's build_profile() computed per-month surplus
    values but never actually returned one (monthly_surplus stayed
    hardcoded null); fixed here to average the months that have both a
    known income and known-expense read, and to report what fraction of
    total debit value was actually classified, not just leave unknown
    debits invisible in a total."""
    monthly = defaultdict(lambda: {'income': Decimal('0'), 'essential': Decimal('0'), 'discretionary': Decimal('0'), 'obligation': Decimal('0'), 'unknown_debit': Decimal('0')})
    for t in classified:
        month = t['date'][:7]
        bucket = monthly[month]
        amt = _money(t['amount'])
        if t['direction'] == 'credit':
            if t['category'] in ('SALARY', 'BUSINESS_INCOME', 'OTHER_VERIFIED_INCOME'):
                bucket['income'] += amt
        else:
            key = {'ESSENTIAL': 'essential', 'DISCRETIONARY': 'discretionary', 'OBLIGATION': 'obligation'}.get(t['category'], 'unknown_debit')
            bucket[key] += amt

    total_debit = sum(_money(t['amount']) for t in classified if t['direction'] == 'debit')
    total_unknown_debit = sum(b['unknown_debit'] for b in monthly.values())
    classified_debit_ratio = float(1 - (total_unknown_debit / total_debit)) if total_debit else 0.0

    surplus_values = []
    for month, b in monthly.items():
        known_expenses = b['essential'] + b['discretionary'] + b['obligation']
        if b['income'] or known_expenses:
            surplus_values.append(b['income'] - known_expenses)
    monthly_surplus = float(_money(sum(surplus_values) / len(surplus_values))) if surplus_values else None

    return {
        'months_analyzed': len(monthly),
        'monthly': {m: {k: float(v) for k, v in b.items()} for m, b in monthly.items()},
        'average_monthly_surplus': monthly_surplus,
        'classified_debit_ratio': round(classified_debit_ratio, 3),
        'trustworthy': classified_debit_ratio >= MIN_CLASSIFIED_DEBIT_RATIO and monthly_surplus is not None,
    }


def calculate_trip_budget(*, current_balance, emergency_reserve, upcoming_obligations, conservative_monthly_surplus, months_until_departure, savings_capture_rate=0.70) -> dict:
    """Unchanged from the prototype's budget.py - the one part of it that
    was already correct: pure arithmetic, no LLM, auditable."""
    rate = Decimal(str(savings_capture_rate))
    available = max(Decimal('0'), _money(current_balance) - _money(emergency_reserve) - _money(upcoming_obligations))
    future = max(Decimal('0'), _money(conservative_monthly_surplus)) * max(Decimal('0'), Decimal(str(months_until_departure))) * rate
    return {
        'available_savings': float(_money(available)),
        'future_savings': float(_money(future)),
        'recommended_trip_budget': float(_money(available + future)),
    }


def evaluate_trip_affordability(*, estimated_trip_cost, safe_trip_budget, search_exhausted=False) -> dict:
    """Deterministic hard gate for a researched itinerary.

    safe_trip_budget must be calculated separately from statement-derived
    financial data after accounting for protected reserves and known
    obligations. A closing balance or average monthly surplus alone is not
    a safe trip budget. Missing either value produces INSUFFICIENT_DATA.
    """
    if estimated_trip_cost is None or safe_trip_budget is None:
        return {
            'status': 'INSUFFICIENT_DATA',
            'estimated_trip_cost': None if estimated_trip_cost is None else float(_money(estimated_trip_cost)),
            'safe_trip_budget': None if safe_trip_budget is None else float(_money(safe_trip_budget)),
            'shortfall': None,
            'can_recommend_as_affordable': False,
        }

    cost = _money(estimated_trip_cost)
    budget = max(Decimal('0'), _money(safe_trip_budget))
    if cost <= budget:
        status = 'AFFORDABLE'
    elif search_exhausted:
        status = 'NOT_AFFORDABLE'
    else:
        status = 'NEEDS_OPTIMIZATION'

    return {
        'status': status,
        'estimated_trip_cost': float(cost),
        'safe_trip_budget': float(budget),
        'shortfall': float(_money(max(Decimal('0'), cost - budget))),
        'can_recommend_as_affordable': status == 'AFFORDABLE',
    }


def estimate_trip_cost(*, flights, hotel, daily_cost_per_person, family_size, nights, contingency=0.10) -> dict:
    daily = _money(daily_cost_per_person) * family_size * (nights + 1)
    base = _money(flights) + _money(hotel) + daily
    total = base * (Decimal('1') + Decimal(str(contingency)))
    return {'daily_estimate': float(_money(daily)), 'base_trip_cost': float(_money(base)), 'estimated_trip_cost': float(_money(total))}


async def compute_financial_profile(agent_id: str, statement_text: str) -> Optional[dict]:
    """The one entry point upload_links.py calls, once per uploaded
    document, right after Muse Spark extracts its text. Returns None for
    a document that plainly isn't a financial statement (the local
    prefilter, no model call spent); otherwise returns a profile dict
    with `trustworthy: False` when too much of the debit total is
    UNKNOWN - the caller must not state average_monthly_surplus as fact
    when that's false, same "don't silently guess" rule as every other
    field in this project."""
    if not _is_statement_like(statement_text):
        return None
    normalized = await normalize_transactions(agent_id, statement_text)
    if not normalized.transactions:
        return None
    classified = classify_transactions(normalized.transactions)
    profile = build_profile(classified)
    profile['currency'] = normalized.currency
    return profile
