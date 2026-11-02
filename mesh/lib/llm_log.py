"""
One central, plain-text log of every prompt any agent sends to the LLM and
the response it got back - mesh/lib/agent_sdk.py's ask() is the single
chokepoint every agent's LLM call already goes through (see that module's
own docstring), so this is the one place that needs wiring, the same
platform-hook precedent Phase 5's memory wiring already established there.

~/.Adiyan/logs/llm_calls.log, not each agent's own $LOG_DIR/<agent>.log -
the whole point is being able to read one file and see every agent's LLM
traffic in call order, not grep across twenty separate per-agent logs that
only hold whatever that one process happened to print. Every agent process
appends to this same file independently (there's no single process that
could hold a shared Python logger object across them) - plain `open(...,
'a')` appends are atomic on POSIX for writes this size, so concurrent
agents interleaving into one file is safe without a lock.

Deliberately outside mesh/lib/agent_sdk.py itself - a separate module
because "how do I log an LLM call" and "how do I make an LLM call" are
different concerns, the same separation mesh/lib/memory_hook.py already
keeps from mesh/lib/graph_store.py.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

LLM_LOG_PATH = Path.home() / '.Adiyan' / 'logs' / 'llm_calls.log'

logger = logging.getLogger(__name__)


def _stringify(value: Any) -> str:
    """Every response shape ask() can return - a plain str (text calls), a
    pydantic BaseModel instance (schema calls), or a langchain AIMessage
    (tool calls) - collapsed to one readable string, not a repr dump.
    Never raises: a logging call failing must never be why the actual LLM
    call's own result is lost to the caller."""
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    to_text = getattr(value, 'content', None)
    if isinstance(to_text, str):
        # think=True on ask() (mesh/lib/agent_sdk.py) surfaces the model's
        # reasoning trace here, langchain-ollama's own separate field -
        # never mixed into .content itself, so a caller reading the
        # returned text back from ask() never sees it either.
        reasoning = (getattr(value, 'additional_kwargs', None) or {}).get('reasoning_content')
        prefix = f'[thinking]\n{reasoning}\n[/thinking]\n\n' if reasoning else ''
        tool_calls = getattr(value, 'tool_calls', None)
        if tool_calls:
            return f'{prefix}{to_text}\n[tool_calls: {tool_calls}]'
        return f'{prefix}{to_text}'
    model_dump_json = getattr(value, 'model_dump_json', None)
    if callable(model_dump_json):
        try:
            return model_dump_json()
        except Exception:
            pass
    return str(value)


def log_call(
    agent_id: str,
    stage: str,
    mode: str,
    model: str,
    prompt: str,
    response: Any,
    error: Optional[str] = None,
    usage: Optional[dict] = None,
) -> None:
    """Appends one call's record. `mode` names which ask() branch this was
    (text/schema/tools/image) - the four paths log differently-shaped
    responses (see _stringify), and knowing which one this was matters
    more for reading this log than it would for the caller of ask() itself.
    `error` is set instead of `response` when the call itself raised - a
    failed call is exactly as worth seeing here as a successful one, since
    "what did we actually send it" is often the first thing worth checking
    when a skill's output looks wrong.

    `usage`: the real token counts a backend actually returned for this
    call (Claude CLI's own JSON reply already carries input_tokens/
    output_tokens/cache_creation_input_tokens/total_cost_usd - confirmed
    live, this was being silently discarded by every caller except
    claude_cli_web_research, which only captured the cost figure, not
    tokens). Written as its own line so a later pass can grep/sum real
    per-call cost across a whole flow instead of estimating from text
    length. None for a backend that doesn't report usage (e.g. Ollama via
    this module's own callers today).

    Best-effort: any failure writing this log is swallowed (logged as a
    warning to this module's own stdlib logger, never raised) - a broken
    disk or a permissions issue here must never take down the agent's
    actual LLM call."""
    try:
        LLM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        lines = [
            f'=== {timestamp} | agent={agent_id} | stage={stage} | mode={mode} | model={model} ===',
            '--- PROMPT ---',
            prompt,
        ]
        if error is not None:
            lines += ['--- ERROR ---', error]
        else:
            lines += ['--- RESPONSE ---', _stringify(response)]
        if usage:
            input_tokens = usage.get('input_tokens', 0) + usage.get('cache_creation_input_tokens', 0) + usage.get('cache_read_input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
            lines += ['--- USAGE ---', f'input_tokens={input_tokens} output_tokens={output_tokens} total_tokens={input_tokens + output_tokens}']
        lines.append('')
        with open(LLM_LOG_PATH, 'a') as f:
            f.write('\n'.join(lines) + '\n')
    except Exception as e:
        logger.warning(f'Failed to write to the central LLM log: {e}')
