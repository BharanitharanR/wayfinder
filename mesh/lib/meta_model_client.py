"""
Wayfinder's own LLM backend - calls Meta's Model API (Muse Spark) directly,
not mesh/lib/agent_sdk.py's AdiyanAgent.ask(), which goes to the Claude CLI.
This is deliberate and narrow: the Meta AI Developer Hackathon requires the
submitted project to call only Meta models at runtime, and Wayfinder is the
one agent being submitted - every other agent in this mesh keeps using
AdiyanAgent/Claude CLI as before, untouched.

Model API (https://dev.meta.ai/docs/overview) is drop-in OpenAI-SDK
compatible: same base URL/key/model work across the Responses API, Chat
Completions, and (for Pydantic schema calls) `beta.chat.completions.parse`.
Chat Completions is used here, not Responses, because it has a direct
drop-in replacement for the two shapes AdiyanAgent.ask() already gave
Wayfinder's callers - schema=... and tools=... - with no other code in
wayfinder_session.py/wayfinder_tools.py needing to change shape.

    MODEL_API_KEY=...   required, from the Model API dashboard -> API keys.
    WAYFINDER_META_MODEL  optional override, defaults to muse-spark-1.3.

Logs through the same mesh/lib/llm_log.py every other agent's calls go
through - one central log, this backend included, not a second silent path.
"""
import base64
import json
import logging
import mimetypes
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Type

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

from mesh.lib import llm_log

logger = logging.getLogger('MetaModelClient')

# Repo-root .env, gitignored (see .gitignore's .env/.env.* lines) - this is
# where MODEL_API_KEY lives for local/dev runs and for a judge running
# this repo from its setup instructions. .env.example at the same path
# shows the shape with no real key, per the hackathon's own rule: never
# commit a real key, always ship a .env.example. load_dotenv() here is a
# no-op if the file doesn't exist or the var is already set in the shell -
# an already-exported MODEL_API_KEY always wins.
load_dotenv(Path(__file__).resolve().parent.parent.parent / '.env')

BASE_URL = 'https://api.meta.ai/v1'
DEFAULT_MODEL = os.environ.get('WAYFINDER_META_MODEL', 'muse-spark-1.3')

_client: Optional[AsyncOpenAI] = None


class MetaModelError(RuntimeError):
    pass


class RateLimitExceeded(MetaModelError):
    pass


# Per-process call/token budget - a plain in-memory guard, not a durable
# quota (resets on restart, doesn't coordinate across multiple processes).
# Exists to catch a runaway loop or an unexpectedly expensive demo run
# before it burns real spend, not to meter production usage precisely.
# 0/unset means "no limit" for that dimension.
_MAX_CALLS = int(os.environ.get('WAYFINDER_MAX_CALLS', '0') or '0')
_MAX_TOKENS = int(os.environ.get('WAYFINDER_MAX_TOKENS', '0') or '0')
_call_count = 0
_token_total = 0


def _check_rate_limit() -> None:
    if _MAX_CALLS and _call_count >= _MAX_CALLS:
        raise RateLimitExceeded(f'Hit WAYFINDER_MAX_CALLS ({_MAX_CALLS}) for this process - {_call_count} Meta Model API calls already made. Restart to reset, or raise the limit in .env.')
    if _MAX_TOKENS and _token_total >= _MAX_TOKENS:
        raise RateLimitExceeded(f'Hit WAYFINDER_MAX_TOKENS ({_MAX_TOKENS}) for this process - {_token_total} tokens already used. Restart to reset, or raise the limit in .env.')


def _record_usage(usage: Optional[dict]) -> None:
    global _call_count, _token_total
    _call_count += 1
    if usage:
        _token_total += usage.get('input_tokens', 0) + usage.get('output_tokens', 0)


def usage_snapshot() -> dict:
    """What this process has spent so far against its own budget - the
    dev CLI's /state (or a new /usage command) can print this without
    grepping the log file."""
    return {'calls': _call_count, 'tokens': _token_total, 'max_calls': _MAX_CALLS or None, 'max_tokens': _MAX_TOKENS or None}


def _get_client() -> AsyncOpenAI:
    """AsyncOpenAI, not the synchronous OpenAI client - confirmed live
    that calling the sync client's blocking network I/O from inside an
    `async def` with no `await`/thread offload freezes the ENTIRE
    uvicorn event loop for the full duration of each call (seconds to
    well over a minute on a large prompt), during which the server can't
    serve ANY other request, not even an unrelated Mongo lookup. That's
    why wayfinder_web.py would go completely unresponsive - not hung on
    one slow request, the whole single-threaded process was blocked."""
    global _client
    if _client is None:
        api_key = os.environ.get('MODEL_API_KEY')
        if not api_key:
            raise MetaModelError('MODEL_API_KEY is not set - get one from the Model API dashboard (dev.meta.ai) and export it.')
        _client = AsyncOpenAI(base_url=BASE_URL, api_key=api_key)
    return _client


def _usage_dict(usage) -> Optional[dict]:
    """Chat Completions usage carries prompt_tokens/completion_tokens; the
    Responses API (used by web_search below) carries input_tokens/
    output_tokens directly - both read here so one helper covers either
    call shape."""
    if usage is None:
        return None
    input_tokens = getattr(usage, 'input_tokens', None)
    if input_tokens is None:
        input_tokens = usage.prompt_tokens
    output_tokens = getattr(usage, 'output_tokens', None)
    if output_tokens is None:
        output_tokens = usage.completion_tokens
    return {'input_tokens': input_tokens, 'output_tokens': output_tokens}


_EXTRACT_DOCUMENT_PROMPT = (
    "Extract the complete text content of this document, verbatim, as plain "
    "markdown - every line item, date, amount, and label exactly as written. "
    "Do not summarize, interpret, or omit anything; this is for a downstream "
    "step that reads the raw content itself. If the document is a scanned "
    "image with handwriting or low-quality print, transcribe your best "
    "reading of it rather than giving up."
)


async def extract_document_text(agent_id: str, stage: str, filename: str, content: bytes) -> str:
    """Document OCR/extraction step - reads a PDF/image straight off Muse
    Spark's own multimodal input (Model API's 'File handling'/'Image
    understanding' docs: a PDF is converted server-side to text plus
    per-page images and both reach the model), not a local OCR model.

    This replaces what used to be a local Docling/RapidOCR pass in
    mesh/orchestrator/upload_links.py's _extract_text - the Meta AI
    Developer Hackathon requires every model call this project makes at
    runtime to be a Meta model, and a local OCR model is explicitly called
    out as not allowed, however it's run. Sent inline as base64 (Model
    API's 'file' content block - see 'File handling'), not uploaded via
    the Files API, since Wayfinder has no reuse case for the same
    document across requests - it's parsed once at upload time and the
    resulting text is what gets stored."""
    _check_rate_limit()
    mime_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
    file_data = f'data:{mime_type};base64,{base64.b64encode(content).decode()}'
    model = DEFAULT_MODEL
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': _EXTRACT_DOCUMENT_PROMPT},
            {'type': 'file', 'file': {'filename': filename, 'file_data': file_data}},
        ],
    }]
    try:
        response = await _get_client().chat.completions.create(model=model, messages=messages)
    except Exception as e:
        llm_log.log_call(agent_id, stage, 'document', f'meta:{model}', f'[document: {filename}]', None, error=str(e))
        raise
    text = response.choices[0].message.content or ''
    usage = _usage_dict(response.usage)
    _record_usage(usage)
    llm_log.log_call(agent_id, stage, 'document', f'meta:{model}', f'[document: {filename}]', text, usage=usage)
    return text


async def ask_schema_with_file(agent_id: str, stage: str, prompt: str, filename: str, content: bytes, schema: Type[BaseModel]) -> BaseModel:
    """Like ask_schema, but with a file (image, PDF, etc.) attached to the
    same message - the same 'file' content block extract_document_text
    uses, just with structured output instead of a raw-text return. Used
    for trip-photo analysis (mesh/orchestrator/trip_recap.py): one call
    does both captioning and transit-selfie detection via one schema,
    rather than two separate calls."""
    _check_rate_limit()
    mime_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
    file_data = f'data:{mime_type};base64,{base64.b64encode(content).decode()}'
    model = DEFAULT_MODEL
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': prompt},
            {'type': 'file', 'file': {'filename': filename, 'file_data': file_data}},
        ],
    }]
    try:
        response = await _get_client().beta.chat.completions.parse(model=model, messages=messages, response_format=schema)
    except Exception as e:
        llm_log.log_call(agent_id, stage, 'schema_file', f'meta:{model}', f'[file: {filename}] {prompt}', None, error=str(e))
        raise
    parsed = response.choices[0].message.parsed
    if parsed is None:
        error = 'Model API returned no parsed result (refusal or non-conforming output).'
        llm_log.log_call(agent_id, stage, 'schema_file', f'meta:{model}', f'[file: {filename}] {prompt}', None, error=error)
        raise MetaModelError(error)
    usage = _usage_dict(response.usage)
    _record_usage(usage)
    llm_log.log_call(agent_id, stage, 'schema_file', f'meta:{model}', f'[file: {filename}] {prompt}', parsed, usage=usage)
    return parsed


async def ask_text(agent_id: str, stage: str, prompt: str, system: Optional[str] = None) -> str:
    """Plain text call - the Chat Completions equivalent of
    AdiyanAgent.ask(prompt, stage=...) with no schema/tools."""
    _check_rate_limit()
    messages = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': prompt}]
    model = DEFAULT_MODEL
    try:
        response = await _get_client().chat.completions.create(model=model, messages=messages)
    except Exception as e:
        llm_log.log_call(agent_id, stage, 'text', f'meta:{model}', prompt, None, error=str(e))
        raise
    text = response.choices[0].message.content or ''
    usage = _usage_dict(response.usage)
    _record_usage(usage)
    llm_log.log_call(agent_id, stage, 'text', f'meta:{model}', prompt, text, usage=usage)
    return text


async def ask_schema(agent_id: str, stage: str, prompt: str, schema: Type[BaseModel], system: Optional[str] = None) -> BaseModel:
    """Structured call - the Chat Completions equivalent of
    AdiyanAgent.ask(prompt, stage=..., schema=SomeModel), using the OpenAI
    SDK's own `beta.chat.completions.parse` (Model API docs: 'Using
    Pydantic models') so the result is already a validated `schema`
    instance, same as what callers got back from AdiyanAgent.ask()."""
    _check_rate_limit()
    messages = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': prompt}]
    model = DEFAULT_MODEL
    try:
        response = await _get_client().beta.chat.completions.parse(model=model, messages=messages, response_format=schema)
    except Exception as e:
        llm_log.log_call(agent_id, stage, 'schema', f'meta:{model}', prompt, None, error=str(e))
        raise
    parsed = response.choices[0].message.parsed
    if parsed is None:
        error = 'Model API returned no parsed result (refusal or non-conforming output).'
        llm_log.log_call(agent_id, stage, 'schema', f'meta:{model}', prompt, None, error=error)
        raise MetaModelError(error)
    usage = _usage_dict(response.usage)
    _record_usage(usage)
    llm_log.log_call(agent_id, stage, 'schema', f'meta:{model}', prompt, parsed, usage=usage)
    return parsed


def _tool_to_function_def(tool) -> dict:
    """A langchain @tool object -> Model API's nested Chat Completions tool
    shape ({"type": "function", "function": {...}}) - see 'Defining tools'/
    'Tool calling with Chat Completions' in the Model API docs."""
    schema = dict(tool.get_input_schema().model_json_schema())
    for noise in ('title', 'description', '$defs'):
        schema.pop(noise, None)
    schema.setdefault('type', 'object')
    schema.setdefault('properties', {})
    return {
        'type': 'function',
        'function': {
            'name': tool.name,
            'description': tool.description or '',
            'parameters': schema,
        },
    }


async def web_search(agent_id: str, stage: str, task: str) -> str:
    """Meta's own built-in search-grounding tool (Responses API only - see
    'Search grounding' in the Model API docs) - the model runs the search
    itself server-side and returns a cited answer. Used instead of a
    developer-defined browser-automation tool so Wayfinder needs no local
    browser agent process and no separate permission token to look
    something up on the live web - one more thing a judge running this
    repo standalone doesn't need running."""
    _check_rate_limit()
    model = DEFAULT_MODEL
    try:
        response = await _get_client().responses.create(
            model=model, input=task, tools=[{'type': 'web_search'}],
        )
    except Exception as e:
        llm_log.log_call(agent_id, stage, 'web', f'meta:{model}', task, None, error=str(e))
        raise
    text_parts: List[str] = []
    sources: List[dict] = []
    for item in response.output:
        if getattr(item, 'type', None) != 'message':
            continue
        for block in item.content:
            if getattr(block, 'type', None) != 'output_text':
                continue
            text_parts.append(block.text)
            for ann in (block.annotations or []):
                if getattr(ann, 'type', None) == 'url_citation':
                    url = getattr(ann, 'url', None)
                    if url and not any(source['url'] == url for source in sources):
                        sources.append({
                            'title': getattr(ann, 'title', None) or url,
                            'url': url,
                        })
    result = '\n'.join(text_parts).strip()
    if sources:
        result += '\n\nSources (web citations returned by Meta Search):\n'
        result += '\n'.join(
            f"[S{index}] {source['title']} — {source['url']}"
            for index, source in enumerate(sources, start=1)
        )
        result += f"\nResearch retrieved: {__import__('datetime').datetime.now(__import__('datetime').timezone.utc).date().isoformat()} (UTC)"
    else:
        result += (
            '\n\nResearch evidence warning: Meta Search returned no URL citations '
            'for this response. Treat factual claims above as unverified; no '
            'current prices, opening hours, availability, or travel times are confirmed.'
        )
    usage = _usage_dict(response.usage)
    _record_usage(usage)
    llm_log.log_call(agent_id, stage, 'web', f'meta:{model}', task, result, usage=usage)
    return result or 'web_search found nothing conclusive for that task.'



async def generate_image(agent_id: str, stage: str, prompt: str, reference_images: List[tuple]) -> bytes:
    """Generate a composed image with Muse Image, optionally guided by
    uploaded trip photos. Reference images are passed as data URLs through
    the Responses API; the returned image bytes are persisted by caller."""
    _check_rate_limit()
    content = [{'type': 'input_text', 'text': prompt}]
    for filename, image_bytes in reference_images:
        mime_type = mimetypes.guess_type(filename)[0] or 'image/jpeg'
        data_url = f'data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}'
        content.append({'type': 'input_image', 'image_url': data_url})
    model = 'muse-image-1.0'
    try:
        response = await _get_client().responses.create(
            model=model,
            input=[{'role': 'user', 'content': content}],
            tools=[{
                'type': 'image_generation',
                'enable_web_search': False,
                'enable_image_search': False,
                'enable_shell': False,
                'reasoning_strength': 'high',
                'output_format': 'jpeg',
            }],
        )
        image_item = next(
            (item for item in response.output
             if getattr(item, 'type', None) == 'image_generation_call'),
            None,
        )
        if image_item is None or not getattr(image_item, 'result', None):
            raise MetaModelError('Muse Image returned no generated image.')
        image_bytes = base64.b64decode(image_item.result)
    except Exception as e:
        llm_log.log_call(agent_id, stage, 'image', f'meta:{model}', prompt, None, error=str(e))
        raise
    usage = _usage_dict(response.usage)
    _record_usage(usage)
    llm_log.log_call(agent_id, stage, 'image', f'meta:{model}', prompt, '[generated image]', usage=usage)
    return image_bytes



ASR_MODEL = 'muse-voice-transcribe-1.0'


def _to_asr_wav(content: bytes, src_filename: str) -> bytes:
    """Muse Voice Transcribe's /asr/transcribe endpoint only accepts a
    RIFF/WAVE container, mono, 16-bit PCM, at 16kHz or 24kHz (confirmed
    from the Model API docs' own 'Transcribe a file' section) - a
    WhatsApp voice note (.ogg/opus) or any other format has to be
    converted first. Shells out to ffmpeg (already a real dependency
    elsewhere in this machine's toolchain) rather than pulling in a pure-
    Python audio library for one conversion."""
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / f'in{Path(src_filename).suffix or ".bin"}'
        dst_path = Path(tmp) / 'out.wav'
        src_path.write_bytes(content)
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', str(src_path), '-ac', '1', '-ar', '24000', '-c:a', 'pcm_s16le', '-map_metadata', '-1', str(dst_path)],
            capture_output=True,
        )
        if result.returncode != 0:
            raise MetaModelError(f'ffmpeg failed to convert audio to WAV: {result.stderr.decode(errors="replace")[:500]}')
        return dst_path.read_bytes()


async def transcribe_audio(agent_id: str, stage: str, filename: str, content: bytes) -> str:
    """Speech-to-text via Muse Voice Transcribe - a plain REST endpoint
    (api.meta.ai/v1/asr/transcribe), not the OpenAI-SDK-compatible Chat
    Completions/Responses surface the rest of this module uses, so this
    one call goes through httpx directly with a Bearer header, per the
    Model API docs' own file-transcription example. PUSH_TO_TALK mode
    (the default) - a WhatsApp voice note or an uploaded recording is
    always a single utterance/turn here, not a live multi-turn stream."""
    _check_rate_limit()
    wav_bytes = _to_asr_wav(content, filename)
    request_json = json.dumps({'mode': 'PUSH_TO_TALK', 'model': ASR_MODEL, 'audioEncoding': 'WAV'})
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                'https://api.meta.ai/v1/asr/transcribe',
                headers={'Authorization': f'Bearer {os.environ.get("MODEL_API_KEY", "")}'},
                files={'audio': (filename, wav_bytes, 'audio/wav')},
                data={'request': request_json},
            )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        llm_log.log_call(agent_id, stage, 'asr', f'meta:{ASR_MODEL}', f'[audio: {filename}]', None, error=str(e))
        raise
    transcript = data.get('transcript', '')
    _record_usage(None)
    llm_log.log_call(agent_id, stage, 'asr', f'meta:{ASR_MODEL}', f'[audio: {filename}]', transcript)
    return transcript


class ToolChoice:
    """What a tools= call returns: the model's chosen call(s), not yet
    executed - same contract AdiyanAgent.ask(tools=...) already had
    (mesh/lib/agent_sdk.py's own docstring), so callers read
    `.tool_calls` exactly as before."""

    def __init__(self, tool_calls: List[dict]):
        self.tool_calls = tool_calls


async def ask_tools(agent_id: str, stage: str, prompt: str, tools: list, system: Optional[str] = None) -> ToolChoice:
    """Tool-selection call - the Chat Completions equivalent of
    AdiyanAgent.ask(prompt, stage=..., tools=[...]). Returns the model's
    chosen call(s) as {'name': ..., 'args': {...}} dicts; the caller still
    executes the tool itself (Model API does not run developer-defined
    tools server-side - see 'How it works' in the tool-calling docs)."""
    _check_rate_limit()
    messages = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': prompt}]
    model = DEFAULT_MODEL
    function_defs = [_tool_to_function_def(t) for t in tools]
    try:
        response = await _get_client().chat.completions.create(model=model, messages=messages, tools=function_defs)
    except Exception as e:
        llm_log.log_call(agent_id, stage, 'tools', f'meta:{model}', prompt, None, error=str(e))
        raise
    message = response.choices[0].message
    calls = []
    for tc in (message.tool_calls or []):
        try:
            args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, TypeError):
            args = {}
        calls.append({'name': tc.function.name, 'args': args, 'id': tc.id, 'type': 'tool_call'})
    logged_response: Any = message.content or f'[tool_calls: {calls}]' if calls else (message.content or '')
    usage = _usage_dict(response.usage)
    _record_usage(usage)
    llm_log.log_call(agent_id, stage, 'tools', f'meta:{model}', prompt, logged_response, usage=usage)
    return ToolChoice(calls)
