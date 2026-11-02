"""
Dev-only REPL for exercising Wayfinder end to end without WhatsApp/OpenWA in
the loop. Calls the exact same functions handle_message.py's wayfinder
branch calls (wayfinder_session.handle_customer_message, upload_links.
create_upload_link) - this is the real pipeline, just driven from a
terminal instead of a phone, so document upload still goes through the
actual HTTP page (nginx -> ngrok -> Docling/OCR -> Mongo); only the chat
turn is injected directly, skipping OpenWA's own webhook/media-fetch path
(the thing with the known 404 bug - irrelevant here since no attachment
ever goes through OpenWA in this flow).

Run from the repo root:
    .venv/bin/python3 -m mesh.orchestrator.wayfinder_dev_cli [--phone +15550001234]

Commands inside the REPL:
    /link        - (re)print the upload-this-browser link for the current phone
    /state       - run infer_profile() now and print the resulting CustomerSnapshot
    /uploads     - list documents on file for the current phone
    /research Q  - run wayfinder_tools.research(Q) - test search_flights/
                   search_hotels/web_search directly, no WhatsApp involved
    /reload      - re-import wayfinder_session/upload_links/wayfinder_tools,
                   so an edit to a prompt or a tool's docstring takes effect
                   in THIS running process - Python doesn't hot-reload on
                   its own, and nothing here restarts for you otherwise.
                   Edit the file in your editor, save, then /reload here.
    /reset       - wipe this phone's chat_turns/uploads/links (start clean)
    /quit        - exit
Anything else typed is sent as a customer message, exactly like a WhatsApp
text would be. Fully self-contained - once this is running, nothing about
iterating on a prompt needs anyone else; edit, save, /reload, retry.
"""
import argparse
import asyncio
import importlib

from mesh.orchestrator import upload_links, wayfinder_session, wayfinder_tools

DEFAULT_DEV_PHONE = '+15550001234'


def _reload() -> None:
    # Order matters only in that each reload takes effect immediately -
    # wayfinder_session and wayfinder_tools don't import each other, so
    # there's no stale-reference risk either way; see this function's own
    # docstring note above for why a plain `import` edit alone wouldn't
    # have been enough.
    importlib.reload(upload_links)
    importlib.reload(wayfinder_session)
    importlib.reload(wayfinder_tools)
    print('Reloaded upload_links, wayfinder_session, wayfinder_tools.')


def _print_link(phone: str) -> None:
    require_phone_entry = upload_links.get_confirmed_phone(phone) is None
    link = upload_links.create_upload_link(phone, require_phone_entry=require_phone_entry)
    print(f'Upload link ({"first-time, phone number required" if require_phone_entry else "returning"}): {link}')


def _print_uploads(phone: str) -> None:
    docs = wayfinder_session.get_context_documents(phone)
    if not docs:
        print('No successfully-read documents on file yet.')
        return
    for d in docs:
        print(f"- {d['filename']} (uploaded {d['uploaded_at']}, {len(d.get('extracted_text') or '')} chars extracted)")


async def _print_state(phone: str) -> None:
    chat_history = wayfinder_session.get_chat_history(phone)
    documents = wayfinder_session.get_context_documents(phone)
    print('Running infer_profile() now (one real LLM call)...')
    snapshot = await wayfinder_session.infer_profile(chat_history, documents)
    print(snapshot.model_dump_json(indent=2))


def _reset(phone: str) -> None:
    db = wayfinder_session._db()
    chat_deleted = db[wayfinder_session.CHAT_COLLECTION].delete_many({'phone': phone}).deleted_count
    uploads_deleted = db[upload_links.UPLOADS_COLLECTION].delete_many({'phone': phone}).deleted_count
    links_deleted = db[upload_links.LINKS_COLLECTION].delete_many({'phone': phone}).deleted_count
    print(f'Wiped {chat_deleted} chat turns, {uploads_deleted} uploads, {links_deleted} upload links for {phone}.')


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phone', default=DEFAULT_DEV_PHONE, help='Identity key to act as (default: a fixed dev test number).')
    args = parser.parse_args()
    phone = args.phone

    print(f'Wayfinder dev CLI - acting as {phone}')
    _print_uploads(phone)
    _print_link(phone)
    print("Type a customer message and press enter, or /link /state /uploads /research <query> /reload /reset /quit.\n")

    while True:
        try:
            line = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == '/quit':
            break
        if line == '/link':
            _print_link(phone)
            continue
        if line == '/uploads':
            _print_uploads(phone)
            continue
        if line == '/state':
            await _print_state(phone)
            continue
        if line == '/reload':
            _reload()
            continue
        if line == '/reset':
            _reset(phone)
            continue
        if line.startswith('/research '):
            query = line[len('/research '):].strip()
            result = await wayfinder_tools.research(query)
            print(f'{result}\n')
            continue
        reply = await wayfinder_session.handle_customer_message(phone, line)
        print(f'Wayfinder: {reply}\n')


if __name__ == '__main__':
    asyncio.run(_main())
