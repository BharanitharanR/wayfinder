import mimetypes
import os
import re
from pathlib import Path
from uuid import UUID
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route
from mesh.orchestrator import budget_engine, trip_recap, upload_links, wayfinder_session

STATIC = Path(__file__).with_name("static")
MAX_UPLOAD = 15 * 1024 * 1024

# A dedicated in-UI phone-capture step, mirroring the WhatsApp upload
# page's own mandatory field (upload_links.py's require_phone_entry) -
# the real phone number is the one piece of durable identity Wayfinder
# links everything by (see wayfinder_session.handle_customer_message's
# own docstring); a random per-browser session id must never substitute
# for it. The session id below is purely a lookup key into
# upload_links.LINKS_COLLECTION's existing confirmed_phone mechanism -
# the same one the WhatsApp upload page writes to - so once confirmed,
# chat/upload here behave identically to a WhatsApp identity that's
# already linked its real number.

_UPLOAD_INSTRUCTION = "Use the ➕ button above to upload your bank statement or trip photos directly - no separate page needed."


def session_id(request):
    try:
        return "wayfinder-web-" + str(UUID(request.headers.get("x-wayfinder-session", "")))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("Invalid browser session. Refresh the page.")


def identity(request):
    """The resolved real identity for this request - the confirmed phone
    number if this browser session has registered one, else raises (the
    caller must register a phone before chat/upload do anything), never
    silently falls back to the session id the way an earlier version of
    this module did."""
    sid = session_id(request)
    confirmed = upload_links.get_confirmed_phone(sid)
    if confirmed is None:
        raise PermissionError("Please confirm your phone number first.")
    return confirmed

async def index(request):
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

async def register_phone(request):
    try:
        sid = session_id(request)
        body = await request.json()
        country_code = str(body.get("country_code", "")).strip()
        digits = re.sub(r"\D", "", str(body.get("phone_digits", "")))
        full_phone = country_code + digits
        if not upload_links._PHONE_RE.match(full_phone):
            return JSONResponse({"error":"Please enter a valid phone number."}, status_code=400)
        from datetime import datetime, timezone
        upload_links._db()[upload_links.LINKS_COLLECTION].update_one(
            {"phone": sid},
            {"$set": {"phone": sid, "confirmed_phone": full_phone, "require_phone_entry": True,
                      "confirmed_at": datetime.now(timezone.utc).isoformat()}},
            upsert=True,
        )
        return JSONResponse({"status":"ok","phone":full_phone})
    except ValueError as exc:
        return JSONResponse({"error":str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error":f"Could not confirm phone number: {exc}"}, status_code=500)

async def identity_status(request):
    try:
        sid = session_id(request)
        confirmed = upload_links.get_confirmed_phone(sid)
        return JSONResponse({"confirmed": confirmed is not None, "phone": confirmed})
    except ValueError as exc:
        return JSONResponse({"error":str(exc)}, status_code=400)

async def chat(request):
    try:
        phone = identity(request)
        message = str((await request.json()).get("message", "")).strip()
        if not message:
            return JSONResponse({"error":"Message cannot be empty."}, status_code=400)
        reply = await wayfinder_session.handle_customer_message(phone, message, upload_instruction=_UPLOAD_INSTRUCTION)
        return JSONResponse({"reply":reply})
    except PermissionError as exc:
        return JSONResponse({"error":str(exc),"needs_phone":True}, status_code=403)
    except ValueError as exc:
        return JSONResponse({"error":str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error":f"Wayfinder request failed: {exc}"}, status_code=500)

async def state(request):
    try:
        phone = identity(request)
        docs = wayfinder_session.get_context_documents(phone)
        profile = wayfinder_session._best_financial_profile(docs)
        return JSONResponse({"documents":[{"filename":d.get("filename")} for d in docs],
          "financial_profile":profile,
          "financial_overview":wayfinder_session.format_household_financial_overview(profile)})
    except PermissionError as exc:
        return JSONResponse({"error":str(exc),"needs_phone":True}, status_code=403)
    except Exception as exc:
        return JSONResponse({"error":str(exc)}, status_code=400)

async def upload(request):
    """Handles both file types the chat upload link already supports
    (mesh/orchestrator/upload_links.py's _upload_submit): a trip photo
    (image/*) goes to trip_recap.handle_photo_upload - captioned once,
    never run through the statement/OCR path; anything else is treated
    as a document (PDF bank statement) through the existing extract +
    budget_engine path below, unchanged."""
    try:
        phone = identity(request)
        form = await request.form()
        item = form.get("file")
        if item is None or not getattr(item,"filename",None):
            return JSONResponse({"error":"Choose a file to upload."},status_code=400)
        filename = Path(item.filename).name
        content = await item.read()
        if not content or len(content)>MAX_UPLOAD:
            return JSONResponse({"error":"File is empty or exceeds 15 MB."},status_code=413)

        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type and mime_type.startswith("image/"):
            photo = await trip_recap.handle_photo_upload(phone, filename, content)
            wayfinder_session.append_turn(phone,"customer","I uploaded a trip photo: "+filename)
            wayfinder_session.append_turn(phone,"wayfinder","Got it - "+photo["caption"])
            return JSONResponse({"status":"ok","filename":filename,"kind":"photo","caption":photo["caption"]})

        if Path(filename).suffix.lower() != ".pdf":
            return JSONResponse({"error":"Please upload a PDF bank statement or a photo (JPEG/PNG)."},status_code=415)
        outcome = await upload_links._extract_text(filename,content,None)
        if outcome["status"] != "ok":
            return JSONResponse({"status":outcome["status"],"error":outcome.get("error") or outcome["status"]},status_code=422)
        profile = await budget_engine.compute_financial_profile(upload_links._AGENT_ID,outcome["text"])
        from datetime import datetime, timezone
        path = upload_links.UPLOAD_DIR / "web" / (str(abs(hash(phone))) + "_" + filename)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(content)
        upload_links._db()[upload_links.UPLOADS_COLLECTION].insert_one({
          "phone":phone,"filename":filename,"stored_path":str(path),
          "uploaded_at":datetime.now(timezone.utc).isoformat(),"status":"ok",
          "extracted_text":outcome["text"],"error":"","financial_profile":profile})
        summary = wayfinder_session.format_household_financial_overview(profile)
        wayfinder_session.append_turn(phone,"customer","I uploaded the bank statement: "+filename)
        wayfinder_session.append_turn(phone,"wayfinder","Statement processed. "+summary)
        return JSONResponse({"status":"ok","filename":filename,"kind":"document","summary":summary})
    except PermissionError as exc:
        return JSONResponse({"error":str(exc),"needs_phone":True}, status_code=403)
    except Exception as exc:
        return JSONResponse({"error":f"Upload failed: {exc}"},status_code=500)

app = Starlette(debug=False,routes=[
 Route("/",index),Route("/api/chat",chat,methods=["POST"]),
 Route("/api/state",state),Route("/api/upload",upload,methods=["POST"]),
 Route("/api/identity",identity_status),Route("/api/register-phone",register_phone,methods=["POST"])])

if __name__ == "__main__":
    uvicorn.run("mesh.orchestrator.wayfinder_web:app",
      host=os.getenv("WAYFINDER_HOST","127.0.0.1"),
      port=int(os.getenv("WAYFINDER_PORT","8765")),reload=False)
