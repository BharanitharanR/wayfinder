import mimetypes
import os
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

def identity(request):
    try:
        return "wayfinder-web-" + str(UUID(request.headers.get("x-wayfinder-session", "")))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("Invalid browser session. Refresh the page.")

async def index(request):
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

async def chat(request):
    try:
        phone = identity(request)
        message = str((await request.json()).get("message", "")).strip()
        if not message:
            return JSONResponse({"error":"Message cannot be empty."}, status_code=400)
        reply = await wayfinder_session.handle_customer_message(phone, message)
        return JSONResponse({"reply":reply})
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
    except Exception as exc:
        return JSONResponse({"error":f"Upload failed: {exc}"},status_code=500)

app = Starlette(debug=False,routes=[
 Route("/",index),Route("/api/chat",chat,methods=["POST"]),
 Route("/api/state",state),Route("/api/upload",upload,methods=["POST"])])

if __name__ == "__main__":
    uvicorn.run("mesh.orchestrator.wayfinder_web:app",
      host=os.getenv("WAYFINDER_HOST","127.0.0.1"),
      port=int(os.getenv("WAYFINDER_PORT","8765")),reload=False)
