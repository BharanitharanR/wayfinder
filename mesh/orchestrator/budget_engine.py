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

This module computes a grounded household cash-flow profile and exposes a
separate deterministic safe-travel-budget function. It deliberately does
not calculate that travel allocation automatically: current liquid balance,
reserve horizon, upcoming obligations, months until departure and the
surplus-capture policy must be explicitly supplied or confirmed. Missing
inputs produce INSUFFICIENT_DATA rather than a guessed budget.
"""
import re
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
import math
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
    closing_balance: Optional[float] = Field(default=None, description='Closing account balance only when explicitly present; this is not automatically disposable travel cash.')
    transactions: List[StatementTransaction] = Field(description='Every transaction line actually present in the statement text, in the order they appear. Do not invent, merge, or skip any row.')


_NORMALIZE_PROMPT = """Here is the extracted text of a bank/financial statement (already read from the original document):

{statement_text}

List every transaction actually present, in order, as structured data. Each transaction needs its date, its description exactly as written, its amount as a positive number, and whether it was money in (credit) or money out (debit). Also extract the statement's closing/ending balance only when explicitly printed; otherwise return null. Do not treat that balance as disposable cash. Do not summarize, merge rows, or invent a transaction that isn't there - if the statement has 40 rows, return 40 transactions."""


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
        subcategory = None
        for rule in rules:
            if any(term.upper() in desc for term in rule['contains']):
                category = rule['category']
                subcategory = rule.get('subcategory')
                break
        classified.append({'date': t.date, 'description': t.description, 'amount': t.amount, 'direction': t.direction, 'category': category, 'subcategory': subcategory})
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
    monthly = defaultdict(lambda: {'income': Decimal('0'), 'essential': Decimal('0'), 'discretionary': Decimal('0'), 'obligation': Decimal('0'), 'investment_savings': Decimal('0'), 'transfer': Decimal('0'), 'unknown_debit': Decimal('0'), 'category_spend': defaultdict(lambda: Decimal('0'))})
    for t in classified:
        month = t['date'][:7]
        bucket = monthly[month]
        amt = _money(t['amount'])
        if t['direction'] == 'credit':
            if t['category'] in ('SALARY', 'BUSINESS_INCOME', 'OTHER_VERIFIED_INCOME'):
                bucket['income'] += amt
        else:
            key = {'ESSENTIAL': 'essential', 'DISCRETIONARY': 'discretionary', 'OBLIGATION': 'obligation', 'INVESTMENT': 'investment_savings', 'TRANSFER': 'transfer'}.get(t['category'], 'unknown_debit')
            bucket[key] += amt
            if key in ('essential', 'discretionary', 'obligation'):
                bucket['category_spend'][t.get('subcategory') or t['category']] += amt

    total_debit = sum(_money(t['amount']) for t in classified if t['direction'] == 'debit' and t['category'] not in ('TRANSFER', 'INVESTMENT'))
    total_unknown_debit = sum(b['unknown_debit'] for b in monthly.values())
    classified_debit_ratio = float(1 - (total_unknown_debit / total_debit)) if total_debit else 0.0

    surplus_values = []
    category_totals = defaultdict(lambda: Decimal('0'))
    for month, b in monthly.items():
        known_expenses = b['essential'] + b['discretionary'] + b['obligation']
        for category, amount in b['category_spend'].items():
            category_totals[category] += amount
        if b['income'] or known_expenses:
            surplus_values.append(b['income'] - known_expenses)
    monthly_surplus = float(_money(sum(surplus_values) / len(surplus_values))) if surplus_values else None
    ordered_surplus = sorted(surplus_values)
    p25_surplus = ordered_surplus[math.floor((len(ordered_surplus) - 1) * 0.25)] if ordered_surplus else None
    median_surplus = (ordered_surplus[len(ordered_surplus) // 2] if len(ordered_surplus) % 2 else (ordered_surplus[len(ordered_surplus) // 2 - 1] + ordered_surplus[len(ordered_surplus) // 2]) / 2) if ordered_surplus else None
    total_income = sum((b['income'] for b in monthly.values()), Decimal('0'))
    total_essential = sum((b['essential'] for b in monthly.values()), Decimal('0'))
    total_discretionary = sum((b['discretionary'] for b in monthly.values()), Decimal('0'))
    total_obligations = sum((b['obligation'] for b in monthly.values()), Decimal('0'))

    return {
        'months_analyzed': len(monthly),
        'monthly': {m: {k: ({name: float(value) for name, value in v.items()} if k == 'category_spend' else float(v)) for k, v in b.items()} for m, b in monthly.items()},
        'total_verified_income': float(_money(total_income)),
        'average_monthly_income': float(_money(total_income / len(monthly))) if monthly else None,
        'average_monthly_essential_expenses': float(_money(total_essential / len(monthly))) if monthly else None,
        'average_monthly_discretionary_expenses': float(_money(total_discretionary / len(monthly))) if monthly else None,
        'average_monthly_obligations': float(_money(total_obligations / len(monthly))) if monthly else None,
        'average_monthly_surplus': monthly_surplus,
        'median_monthly_surplus': None if median_surplus is None else float(_money(median_surplus)),
        'conservative_monthly_surplus_p25': None if p25_surplus is None else float(_money(p25_surplus)),
        'classified_expense_totals': {k: float(_money(v)) for k, v in category_totals.items()},
        'classified_debit_ratio': round(classified_debit_ratio, 3),
        'trustworthy': classified_debit_ratio >= MIN_CLASSIFIED_DEBIT_RATIO and monthly_surplus is not None,
    }


def calculate_trip_budget(*, current_balance, emergency_reserve, upcoming_obligations, conservative_monthly_surplus, months_until_departure, savings_capture_rate) -> dict:
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


def calculate_safe_travel_budget(*, current_liquid_balance, essential_monthly_expenses, reserve_months, upcoming_obligations, conservative_monthly_surplus, months_until_departure, savings_capture_rate) -> dict:
    """Calculate the travel-only allocation from explicit, customer-grounded inputs.

    This function intentionally has no guessed defaults: current liquid balance,
    reserve horizon, upcoming obligations, time to departure and capture rate
    must be supplied by a trusted profile or confirmed by the customer.
    """
    required = (current_liquid_balance, essential_monthly_expenses, reserve_months,
                upcoming_obligations, conservative_monthly_surplus,
                months_until_departure, savings_capture_rate)
    if any(value is None for value in required):
        return {'status': 'INSUFFICIENT_DATA', 'safe_travel_budget': None,
                'missing_inputs': [name for name, value in zip(
                    ('current_liquid_balance', 'essential_monthly_expenses', 'reserve_months',
                     'upcoming_obligations', 'conservative_monthly_surplus',
                     'months_until_departure', 'savings_capture_rate'), required) if value is None]}
    rate = Decimal(str(savings_capture_rate))
    if rate < 0 or rate > 1:
        raise ValueError('savings_capture_rate must be between 0 and 1')
    if Decimal(str(reserve_months)) < 0 or Decimal(str(months_until_departure)) < 0:
        raise ValueError('reserve_months and months_until_departure cannot be negative')
    reserve = _money(essential_monthly_expenses) * Decimal(str(reserve_months))
    available_now = max(Decimal('0'), _money(current_liquid_balance) - reserve - _money(upcoming_obligations))
    future_contribution = max(Decimal('0'), _money(conservative_monthly_surplus)) * Decimal(str(months_until_departure)) * rate
    return {'status': 'CALCULATED', 'emergency_reserve': float(_money(reserve)),
            'available_liquid_savings': float(_money(available_now)),
            'future_travel_contribution': float(_money(future_contribution)),
            'safe_travel_budget': float(_money(available_now + future_contribution)),
            'savings_capture_rate': float(rate)}


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
    profile['statement_closing_balance'] = normalized.closing_balance
    return profile
