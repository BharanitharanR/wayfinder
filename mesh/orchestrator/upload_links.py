"""
Generic document-upload link for Adiyan's intake flows - lets a customer
registered via WhatsApp open a branded Adiyan web page in any browser and
submit multiple documents in one go (PDF or image - whatever Meta's
Model API can read, see _extract_text below), including a password for
any PDF that's open-protected. No per-vertical branding here on purpose
(see server.py's
wiring comment) - this page is reused by whichever agent needs a
customer to hand over documents next.

Same Mongo-token + nginx /u/ pattern mesh/orchestrator/short_links.py
already uses for workspace-mcp's OAuth URL, except persisted to Mongo
instead of an in-memory dict - a customer may take hours or days to
actually open this link, and an in-memory token dies with the process
long before that.

The token -> phone mapping is what lets the upload POST attribute every
file to the right customer without re-asking "who are you" (see
mesh/orchestrator/db.py's identity-key pattern for why that matters) -
the link itself carries the customer's E.164 phone number, fixed at
creation time by this process, never trusted from the request body.
"""
import io
import mimetypes
import os
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pymongo import MongoClient
from starlette.requests import Request
from starlette.responses import HTMLResponse, PlainTextResponse
from starlette.routing import Route

from mesh.lib import meta_model_client
from mesh.orchestrator import budget_engine, short_links, trip_recap

_AGENT_ID = 'wayfinder_upload_links'

MONGO_URL = os.environ.get('ADIYAN_MONGO_URL', 'mongodb://localhost:27017')
MONGO_DB_NAME = os.environ.get('ADIYAN_MONGO_DB_DATA', 'adiyan')
LINKS_COLLECTION = 'wayfinder_upload_links'
UPLOADS_COLLECTION = 'wayfinder_uploads'

# A customer isn't going to open this the instant it lands like an OAuth
# redirect (short_links.py's 600s) - "later today" or "tomorrow" is the
# realistic case, so this needs to actually survive that long.
_LINK_TTL_SECONDS = 48 * 60 * 60

UPLOAD_DIR = Path.home() / '.Adiyan' / 'wayfinder_uploads'

# Loose E.164 check - a leading '+', then 7-15 digits with no leading
# zero. Good enough to reject "9876543210" (no country code, exactly
# what forcing this field exists to catch) without rejecting a real
# international number this process has never seen before.
_PHONE_RE = re.compile(r'^\+[1-9]\d{6,14}$')

_client: Optional[MongoClient] = None


def _db():
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=3000)
    return _client[MONGO_DB_NAME]


def create_upload_link(phone: str, *, require_phone_entry: bool = False) -> str:
    """Returns a short, absolute, publicly-reachable URL for `phone` to
    open in a browser and upload documents. `phone` here is whatever
    identity key the caller already has for this contact - for a normal
    WhatsApp number that's their real E.164 phone; for a contact whose
    chat_id is a WhatsApp "@lid" privacy id (confirmed - see
    mesh/lib/identity.py - it carries no relation to the real phone
    number), it's that opaque id instead, since that's still what every
    later message from this contact arrives tagged with. The phone number
    itself is a path segment in the URL, not just a server-side token
    lookup - the token alone is still what makes the link unguessable,
    but the number/id being right there in the URL means a mismatched or
    hand-edited link is caught by a plain string comparison.

    `require_phone_entry` makes the page itself collect a real E.164
    phone number before accepting any files - the caller sets this when
    it can't already trust `phone` to be a dialable number (the @lid
    case above being exactly why this exists). Calling this again for
    the same phone just mints a fresh token/link - cheap enough to resend
    on a "that link expired" complaint rather than needing a renew path."""
    digits = phone.lstrip('+')
    token = secrets.token_urlsafe(8)
    now = datetime.now(timezone.utc)
    _db()[LINKS_COLLECTION].insert_one({
        '_id': token,
        'phone': phone,
        'phone_digits': digits,
        'created_at': now.isoformat(),
        'expires_at': now.timestamp() + _LINK_TTL_SECONDS,
        'require_phone_entry': require_phone_entry,
        'confirmed_phone': None,
    })
    return f'{short_links.current_public_url()}/u/{digits}/{token}'


def _resolve_link(phone_digits: str, token: str) -> Optional[dict]:
    """Returns the link document for a valid, unexpired (phone, token)
    pair - None if the token doesn't exist, has expired, or the phone
    segment in the URL doesn't match what this token was actually minted
    for (a hand-edited or stale link, not a legitimate request)."""
    doc = _db()[LINKS_COLLECTION].find_one({'_id': token})
    if not doc or datetime.now(timezone.utc).timestamp() > doc['expires_at']:
        return None
    if doc['phone_digits'] != phone_digits:
        return None
    return doc


def get_confirmed_phone(phone: str) -> Optional[str]:
    """The real E.164 number a contact typed into a `require_phone_entry`
    link, if they've ever completed one - None if this contact (keyed by
    whatever identity `phone` is, @lid or real number) has never been
    through that gate. Callers use this to decide whether to gate the
    *next* link too, or treat a phone number as already on file."""
    doc = _db()[LINKS_COLLECTION].find_one({'phone': phone, 'confirmed_phone': {'$ne': None}})
    return doc['confirmed_phone'] if doc else None


async def _extract_text(filename: str, content: bytes, password: Optional[str]) -> dict:
    """Returns {'status': 'ok' | 'needs_password' | 'wrong_password' |
    'error', 'text': extracted markdown (empty unless status == 'ok'),
    'error': str}.

    PDFs get a decrypt pass first (plain decryption, pypdf, not a model
    call) - everything else goes straight to Meta's Model API
    (meta_model_client.extract_document_text), which reads the file
    itself (PDF/image) and returns its text. This used to be a local
    Docling/RapidOCR pass; that's a local, non-Meta model, which the Meta
    AI Developer Hackathon's rules explicitly disallow for a submitted
    project's runtime calls - see extract_document_text's own docstring."""
    if filename.lower().endswith('.pdf'):
        from pypdf import PdfReader, PdfWriter

        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            if not password:
                return {'status': 'needs_password', 'text': '', 'error': ''}
            try:
                result = reader.decrypt(password)
            except Exception as e:
                return {'status': 'error', 'text': '', 'error': str(e)}
            # pypdf's decrypt() returns a PasswordType enum, 0 == NOT_DECRYPTED
            if int(result) == 0:
                return {'status': 'wrong_password', 'text': '', 'error': ''}
            writer = PdfWriter()
            for page in reader.pages:
                writer.add_page(page)
            buf = io.BytesIO()
            writer.write(buf)
            content = buf.getvalue()

    try:
        text = await meta_model_client.extract_document_text(_AGENT_ID, 'extract_document', filename, content)
        return {'status': 'ok', 'text': text, 'error': ''}
    except Exception as e:
        return {'status': 'error', 'text': '', 'error': str(e)}


# Same circle-and-wordmark used across Adiyan's own pages (mesh/
# config_server/static/adiyan_logo.svg) - inlined rather than served as a
# static asset so this page stays a single self-contained response; it's
# under 400 bytes.
_LOGO_SVG = (
    '<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" role="img" '
    'aria-label="Adiyan logo" width="64" height="64">'
    '<circle cx="100" cy="100" r="88" fill="none" stroke="currentColor" stroke-width="5"/>'
    '<text x="100" y="112" text-anchor="middle" '
    'font-family="\'Courier New\', \'SF Mono\', monospace" font-weight="700" '
    'font-size="30" letter-spacing="1" fill="currentColor">ADiYAN</text>'
    '</svg>'
)

_BASE_STYLE = """
  :root { --ink:#1b1b1f; --paper:#ffffff; --muted:#6b6b76; --line:#e4e4ea; --accent:#2f5d50; --accent-ink:#ffffff; }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) { --ink:#f0f0f3; --paper:#15151a; --muted:#9a9aa5; --line:#2c2c34; --accent:#5fb79a; --accent-ink:#0d1a16; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--paper); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
  .wrap { max-width:480px; margin:0 auto; padding:32px 20px 60px; }
  .brand { display:flex; align-items:center; gap:10px; color:var(--ink); margin-bottom:28px; }
  .brand svg { color:var(--accent); }
  .brand span { font-weight:700; font-size:20px; letter-spacing:0.5px; }
  h1 { font-size:22px; margin:0 0 8px; }
  p.sub { color:var(--muted); font-size:14px; line-height:1.5; margin:0 0 28px; }
  label { display:block; font-size:13px; font-weight:600; color:var(--muted); margin-bottom:6px; text-transform:uppercase; letter-spacing:0.4px; }
  .field { margin-bottom:22px; }
  input[type=file] { width:100%; padding:14px; border:1.5px dashed var(--line); border-radius:10px; background:transparent; color:var(--ink); font-size:14px; }
  input[type=password] { width:100%; padding:12px 14px; border:1.5px solid var(--line); border-radius:10px; background:transparent; color:var(--ink); font-size:15px; }
  .hint { color:var(--muted); font-size:12.5px; margin-top:6px; line-height:1.4; }
  button { width:100%; padding:14px; border:none; border-radius:10px; background:var(--accent); color:var(--accent-ink); font-size:15px; font-weight:700; cursor:pointer; }
  button:active { opacity:0.85; }
  .list { list-style:none; padding:0; margin:0; display:flex; flex-direction:column; gap:10px; }
  .row { display:flex; justify-content:space-between; align-items:baseline; gap:12px; padding:12px 14px; border:1px solid var(--line); border-radius:10px; font-size:14px; }
  .pill { font-size:11px; font-weight:700; letter-spacing:0.3px; padding:3px 9px; border-radius:999px; white-space:nowrap; }
  .pill.ok { background:#1f7a4522; color:#1f7a45; }
  .pill.needs_password, .pill.wrong_password { background:#b5731422; color:#b57314; }
  .pill.error { background:#c1333322; color:#c13333; }
  .error-banner { background:#c1333318; border:1px solid #c1333344; color:#c13333; border-radius:10px; padding:12px 14px; font-size:13.5px; margin-bottom:22px; }
  input[type=tel] { width:100%; padding:12px 14px; border:1.5px solid var(--line); border-radius:10px; background:transparent; color:var(--ink); font-size:15px; }
  .phone-row { display:flex; gap:8px; }
  .phone-row select { flex:0 0 108px; padding:12px 8px; border:1.5px solid var(--line); border-radius:10px; background:var(--paper); color:var(--ink); font-size:15px; }
  .phone-row input[type=tel] { flex:1; }
"""

# Country first means "the number you type is just the national number" -
# no one visiting from a WhatsApp chat should have to remember to type a
# leading '+' or their own country's calling code themselves. India first
# in the list since that's this deployment's actual userbase; the rest
# cover the common cases without turning this into every ITU code there is.
_COUNTRY_CODES = [
    ('+91', 'India (+91)'),
    ('+1', 'US/Canada (+1)'),
    ('+44', 'UK (+44)'),
    ('+971', 'UAE (+971)'),
    ('+65', 'Singapore (+65)'),
    ('+61', 'Australia (+61)'),
    ('+49', 'Germany (+49)'),
    ('+33', 'France (+33)'),
    ('+81', 'Japan (+81)'),
    ('+86', 'China (+86)'),
    ('+94', 'Sri Lanka (+94)'),
    ('+60', 'Malaysia (+60)'),
    ('+852', 'Hong Kong (+852)'),
    ('+966', 'Saudi Arabia (+966)'),
    ('+974', 'Qatar (+974)'),
]


def _country_code_options(selected: str) -> str:
    return ''.join(
        f'<option value="{code}"{" selected" if code == selected else ""}>{label}</option>'
        for code, label in _COUNTRY_CODES
    )


_PHONE_FIELD_TEMPLATE = """    <div class="field">
      <label>Your phone number</label>
      <div class="phone-row">
        <select name="country_code">{country_options}</select>
        <input type="tel" name="phone_digits" value="{digits_value}" placeholder="9876543210" required inputmode="numeric" autocomplete="tel-national">
      </div>
      <div class="hint">Pick your country, then just the number - no leading 0 or country code needed.</div>
    </div>
"""

_ERROR_BANNER_TEMPLATE = '  <div class="error-banner">{message}</div>\n'

_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Upload documents - Adiyan</title>
<style>{style}</style>
</head>
<body>
<div class="wrap">
  <div class="brand">{logo}<span>Adiyan</span></div>
  <h1>Upload your documents</h1>
  <p class="sub">Add as many files as you need in one go - statements, calendars, anything that was asked for. If a file opens with a password, enter it below so it can be read.</p>
{error_block}  <form method="post" enctype="multipart/form-data">
{phone_field}    <div class="field">
      <label>Documents</label>
      <input type="file" name="documents" multiple required>
      <div class="hint">PDF, images, Word or PowerPoint files are all fine.</div>
    </div>
    <div class="field">
      <label>Password (if any file needs one)</label>
      <input type="password" name="password" autocomplete="off">
      <div class="hint">Leave blank if none of your files are password-protected.</div>
    </div>
    <button type="submit">Upload</button>
  </form>
</div>
</body>
</html>"""

_RESULT_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Upload complete - Adiyan</title>
<style>{style}</style>
</head>
<body>
<div class="wrap">
  <div class="brand">{logo}<span>Adiyan</span></div>
  <h1>{heading}</h1>
  <p class="sub">{sub}</p>
  <ul class="list">{rows}</ul>
</div>
</body>
</html>"""

_STATUS_LABEL = {
    'ok': 'Received',
    'needs_password': 'Needs password',
    'wrong_password': 'Wrong password',
    'error': 'Could not read',
}


def _render_result_page(results: list) -> str:
    any_bad = any(r['status'] != 'ok' for r in results)
    heading = 'All set' if not any_bad else 'Some files need another look'
    sub = (
        'Every file came through and was read successfully. You can close this page.'
        if not any_bad else
        'Most files are in. For anything marked below, resend it on this same page with the right password, or ask Adiyan what to do next.'
    )
    rows = ''.join(
        f'<li class="row"><span>{r["filename"]}</span>'
        f'<span class="pill {r["status"]}">{_STATUS_LABEL.get(r["status"], r["status"])}</span></li>'
        for r in results
    )
    return _RESULT_TEMPLATE.format(style=_BASE_STYLE, logo=_LOGO_SVG, heading=heading, sub=sub, rows=rows)


def _render_upload_page(*, phone_field_required: bool, country_code: str = '+91', digits_value: str = '', error: str = '') -> str:
    return _PAGE_TEMPLATE.format(
        style=_BASE_STYLE,
        logo=_LOGO_SVG,
        error_block=_ERROR_BANNER_TEMPLATE.format(message=error) if error else '',
        phone_field=_PHONE_FIELD_TEMPLATE.format(
            country_options=_country_code_options(country_code), digits_value=digits_value,
        ) if phone_field_required else '',
    )


async def _upload_page(request: Request):
    phone_digits, token = request.path_params['phone'], request.path_params['token']
    link = _resolve_link(phone_digits, token)
    if link is None:
        return PlainTextResponse("This link has expired - ask Adiyan to send a new one.", status_code=404)
    return HTMLResponse(_render_upload_page(phone_field_required=link['require_phone_entry']))


async def _upload_submit(request: Request):
    phone_digits, token = request.path_params['phone'], request.path_params['token']
    link = _resolve_link(phone_digits, token)
    if link is None:
        return PlainTextResponse("This link has expired - ask Adiyan to send a new one.", status_code=404)
    # Default identity key: the chat id this link was minted for. Replaced
    # below by the just-confirmed real phone number for a first-time
    # registrant - from that point on, THAT number is what every document
    # (and, via wayfinder_session's own identity resolution, every chat
    # turn) is actually stored against, not the WhatsApp id. See
    # create_upload_link's docstring for why the id alone can't be trusted.
    phone = link['phone']

    form = await request.form()
    password = (form.get('password') or '').strip() or None

    # The "@lid" case this field exists for (see create_upload_link's own
    # docstring) - force a real E.164 number before accepting any files,
    # never trust a half-typed or country-code-less number silently.
    if link['require_phone_entry']:
        country_code = (form.get('country_code') or '+91').strip()
        digits = re.sub(r'\D', '', form.get('phone_digits') or '')
        phone_number = f'{country_code}{digits}'
        if not _PHONE_RE.match(phone_number):
            return HTMLResponse(_render_upload_page(
                phone_field_required=True,
                country_code=country_code,
                digits_value=digits,
                error="Please enter a valid phone number.",
            ))
        _db()[LINKS_COLLECTION].update_one({'_id': token}, {'$set': {'confirmed_phone': phone_number}})
        phone = phone_number

    # form.getlist() only surfaces the LAST value for a repeated file
    # input in Starlette - multi_items() is what actually returns every
    # uploaded file under the shared "documents" field name.
    uploads = [v for k, v in form.multi_items() if k == 'documents' and getattr(v, 'filename', '')]

    customer_dir = UPLOAD_DIR / phone.lstrip('+')
    customer_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for upload in uploads:
        content = await upload.read()

        # Trip photos (JPEG/PNG/...) branch to trip_recap.py entirely -
        # captioned once there and never run through the statement/OCR
        # path above, which would try to read a beach photo as if it
        # were a bank statement. Everything else (PDFs, scanned
        # documents) keeps the existing path below unchanged.
        mime_type, _ = mimetypes.guess_type(upload.filename)
        if mime_type and mime_type.startswith('image/'):
            try:
                await trip_recap.handle_photo_upload(phone, upload.filename, content)
                results.append({'filename': upload.filename, 'status': 'ok'})
            except Exception as e:
                results.append({'filename': upload.filename, 'status': 'error'})
            continue

        safe_name = f'{int(time.time() * 1000)}_{Path(upload.filename).name}'
        (customer_dir / safe_name).write_bytes(content)

        outcome = await _extract_text(upload.filename, content, password)
        # Computed once, here, at upload time - not re-derived on every
        # WhatsApp turn (the prototype engine's own production-hardening
        # note #2). None for a document that isn't statement-shaped (the
        # local prefilter in budget_engine._is_statement_like catches
        # that with no model call spent) - a leave calendar or school
        # holiday PDF just gets None, harmlessly.
        financial_profile = None
        if outcome['status'] == 'ok':
            try:
                financial_profile = await budget_engine.compute_financial_profile(_AGENT_ID, outcome['text'])
            except Exception as e:
                financial_profile = {'error': str(e)}
        _db()[UPLOADS_COLLECTION].insert_one({
            'phone': phone,
            'filename': upload.filename,
            'stored_path': str(customer_dir / safe_name),
            'uploaded_at': datetime.now(timezone.utc).isoformat(),
            'status': outcome['status'],
            'extracted_text': outcome['text'],
            'error': outcome['error'],
            'financial_profile': financial_profile,
        })
        results.append({'filename': upload.filename, 'status': outcome['status']})

    return HTMLResponse(_render_result_page(results))


ROUTES = [
    Route('/u/{phone}/{token}', endpoint=_upload_page, methods=['GET']),
    Route('/u/{phone}/{token}', endpoint=_upload_submit, methods=['POST']),
]
