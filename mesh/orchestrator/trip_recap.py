"""
Trip-recap generation - a customer uploads photos over the course of a
trip, Wayfinder notices when the trip is over, and puts together a
publicly shareable blog-style recap page, embedding the photos, that the
customer can share - but only once they've explicitly said yes to
publishing it. Same confirmation-gate shape as letsfg_tools.py's booking
flow, for the same reason: publishing someone's real trip photos publicly
is not something to do without their own explicit say.

Two signals mark a trip complete, either is enough:
  1. Explicit: the customer says something like "we're back" / "trip's
     done" - detected by _detect_text_completion's schema call.
  2. Implicit: a newly uploaded photo looks like a flight/train selfie
     (PhotoAnalysis.looks_like_transit_selfie, decided at upload time,
     one Muse Spark call per photo - the same call that also writes its
     caption, so this costs nothing extra) AND it landed within a couple
     days of the profile's own stated travel_window end - a transit
     selfie halfway through a multi-stop trip shouldn't trigger this,
     only one actually near the trip's own end date.

Publishing itself is NOT git-repo-creation-on-demand per trip (that would
mean an assistant or this service quietly creating a new public GitHub
repo on every trip, which needs its own explicit one-time human
approval - see this module's own setup note). It pushes one HTML file
into one already-existing, already-approved repo
(PUBLISH_REPO/PUBLISH_BRANCH below), under trips/<slug>.html, and returns
its GitHub Pages URL. Two faster alternatives were tried and rejected:
raw.githack.com renders correctly but inserts a click-through "external
content" interstitial before the page loads - a bad first impression for
a link meant to be shared; cdn.jsdelivr.net's GitHub mirror deliberately
serves .html files as text/plain (so raw source shows instead of a
rendered page), by design, to stop people using it exactly like this.
Pages has a real build/propagation delay after being newly enabled, but
once warm (this repo's Pages was turned on well ahead of any real use)
each new push after that is live in well under a minute.
"""
import base64
import html as html_escape
import io
import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pymongo import MongoClient

from mesh.lib import meta_model_client

logger = logging.getLogger('TripRecap')

AGENT_ID = 'wayfinder_trip_recap'

MONGO_URL = os.environ.get('ADIYAN_MONGO_URL', 'mongodb://localhost:27017')
MONGO_DB_NAME = os.environ.get('ADIYAN_MONGO_DB_DATA', 'adiyan')
PHOTOS_COLLECTION = 'wayfinder_trip_photos'
RECAPS_COLLECTION = 'wayfinder_trip_recaps'
EXPENSES_COLLECTION = 'wayfinder_trip_expenses'

PHOTO_DIR = Path(os.environ.get('WAYFINDER_PHOTO_DIR', str(Path.home() / '.Adiyan' / 'wayfinder_photos')))

# One already-existing, human-approved repo every trip's recap pushes
# into (as trips/<slug>.html) - never created on the fly per trip. See
# this module's own docstring for why repo creation itself needs its own
# separate, one-time approval, done outside this code.
PUBLISH_REPO = os.environ.get('WAYFINDER_RECAP_REPO', 'BharanitharanR/wayfinder-trip-blogs')
PUBLISH_BRANCH = os.environ.get('WAYFINDER_RECAP_BRANCH', 'main')

_client: Optional[MongoClient] = None


def _db():
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=3000)
    return _client[MONGO_DB_NAME]


def add_expense(phone: str, label: str, amount: float, currency: str, source: str) -> None:
    """Running trip-cost tracking - deliberately only ever called with a
    real, already-known number (a booking's own charged_amount from
    LetsFG), never a new model call to estimate or guess an expense.
    Keeping this to confirmed, structured figures is what makes
    get_running_cost's total trustworthy to state as a fact, the same
    reasoning budget_engine.py's trustworthy flag already applies to the
    statement-derived budget signal."""
    _db()[EXPENSES_COLLECTION].insert_one({
        'phone': phone, 'label': label, 'amount': amount, 'currency': currency,
        'source': source, 'recorded_at': datetime.now(timezone.utc).isoformat(),
    })


def get_running_cost(phone: str) -> Dict[str, Any]:
    """Grouped by currency, never summed across currencies (a EUR flight
    and a USD hotel aren't one number without a conversion rate this
    module has no business inventing)."""
    entries = list(_db()[EXPENSES_COLLECTION].find({'phone': phone}, {'_id': 0}).sort('recorded_at', 1))
    totals: Dict[str, float] = {}
    for e in entries:
        totals[e['currency']] = totals.get(e['currency'], 0.0) + e['amount']
    return {'entries': entries, 'totals_by_currency': totals}


def _format_running_cost(phone: str) -> str:
    cost = get_running_cost(phone)
    if not cost['entries']:
        return '(no confirmed expenses yet)'
    lines = [f"- {e['label']}: {e['amount']} {e['currency']}" for e in cost['entries']]
    totals = ', '.join(f'{amt:.2f} {cur}' for cur, amt in cost['totals_by_currency'].items())
    return '\n'.join(lines) + f'\nRunning total: {totals}'


class PhotoAnalysis(BaseModel):
    caption: str = Field(description='A short, warm, one-sentence caption describing what is actually in this photo.')
    looks_like_transit_selfie: bool = Field(description="True only if this specifically looks like a selfie or photo taken ON a plane, train, or at a departure gate/platform, suggesting a journey home. False for any other photo (a beach, a meal, a hotel room, a group photo on land, etc).")


def _read_exif_capture_time(content: bytes) -> Optional[str]:
    """When the photo was actually TAKEN, not uploaded - read straight
    from the file's own EXIF DateTimeOriginal (falling back to the plain
    DateTime tag), deterministic, no model call. Most phone cameras write
    this automatically; a screenshot or a downloaded/re-saved image often
    has none, which is exactly why this returns None rather than
    guessing - uploaded_at is always there as a fallback for ordering."""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        img = Image.open(io.BytesIO(content))
        exif = img.getexif()
        if not exif:
            return None
        tags = {TAGS.get(k, k): v for k, v in exif.items()}
        raw = tags.get('DateTimeOriginal') or tags.get('DateTime')
        if not raw:
            return None
        # EXIF's own format: "YYYY:MM:DD HH:MM:SS"
        dt = datetime.strptime(raw, '%Y:%m:%d %H:%M:%S')
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except Exception:
        return None


async def handle_photo_upload(phone: str, filename: str, content: bytes) -> Dict[str, Any]:
    """Called from upload_links.py for any image upload (branched there
    by MIME type, separate from the document/statement path) - analyzes
    once at upload time (caption + transit-selfie signal in the same
    call, not two), stores the file + the analysis, never re-analyzed
    later."""
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f'{phone.lstrip("+")}_{int(datetime.now(timezone.utc).timestamp() * 1000)}_{Path(filename).name}'
    stored_path = PHOTO_DIR / safe_name
    stored_path.write_bytes(content)

    analysis = await meta_model_client.ask_schema_with_file(
        AGENT_ID, 'analyze_trip_photo',
        'Caption this trip photo in one warm sentence, and say whether it specifically looks like a flight/train/departure selfie suggesting a journey home.',
        filename, content, PhotoAnalysis,
    )
    captured_at = _read_exif_capture_time(content)
    doc = {
        'phone': phone, 'filename': filename, 'stored_path': str(stored_path),
        'caption': analysis.caption, 'looks_like_transit_selfie': analysis.looks_like_transit_selfie,
        'uploaded_at': datetime.now(timezone.utc).isoformat(),
        'captured_at': captured_at,
    }
    _db()[PHOTOS_COLLECTION].insert_one(doc)
    return doc


def _photo_timeline_date(photo: Dict[str, Any]) -> str:
    """Prefer the photo's own real capture time over upload time for
    anything date-sensitive (ordering the memorabilia timeline, matching
    a transit selfie against the trip's travel_window) - uploaded_at is
    only ever a fallback for a photo with no EXIF."""
    return (photo.get('captured_at') or photo['uploaded_at'])


def get_photos(phone: str) -> List[Dict[str, Any]]:
    """Ordered by the photo's own real timeline (EXIF capture time where
    present, upload time otherwise) - sorted in Python, not via Mongo's
    own .sort(), since the two different date fields can't be mixed into
    one index-sortable key server-side."""
    photos = list(_db()[PHOTOS_COLLECTION].find({'phone': phone}, {'_id': 0}))
    return sorted(photos, key=_photo_timeline_date)


class _TripCompletionSignal(BaseModel):
    is_trip_complete: bool = Field(description="True only if the customer's message clearly says the trip itself is over/they're back (e.g. 'we're back home', 'trip's done', 'just landed'). False for anything else, including a question or a mid-trip update.")


async def detect_text_completion(text: str) -> bool:
    decision = await meta_model_client.ask_schema(
        AGENT_ID, 'detect_trip_completion_text',
        f'Does this message clearly say the customer\'s trip is now over? Message: "{text}"',
        _TripCompletionSignal,
    )
    return decision.is_trip_complete


def detect_photo_completion(phone: str, travel_window_end: Optional[str]) -> bool:
    """The implicit signal - a stored photo already flagged
    looks_like_transit_selfie=True (decided once, at upload time, no new
    model call here) that landed within 2 days of the profile's own
    travel_window end date, if that end date is known and parseable.
    Returns False (never guesses) if travel_window_end can't be read as
    a date - a vague "mid-December" travel_window just means this signal
    stays silent and the explicit text signal is still available."""
    if not travel_window_end:
        return False
    try:
        end_date = _parse_loose_date(travel_window_end)
    except ValueError:
        return False
    if end_date is None:
        return False
    for photo in get_photos(phone):
        if not photo.get('looks_like_transit_selfie'):
            continue
        photo_date = datetime.fromisoformat(_photo_timeline_date(photo)).date()
        if abs((photo_date - end_date).days) <= 2:
            return True
    return False


def _parse_loose_date(text: str) -> Optional[Any]:
    """travel_window is free text ('Dec 6-12 2026', not a strict date
    field) - pulls the LAST YYYY-MM-DD-shaped or a recognizable trailing
    date out of it if one is actually there; returns None rather than
    guessing if it isn't. Deliberately narrow - this only needs to catch
    the common "<range> <year>" shape well enough to compare against a
    photo's upload date within a couple of days, not parse every
    possible date phrasing."""
    from datetime import date
    iso_dates = list(re.finditer(r'(\\d{4})-(\\d{2})-(\\d{2})', text))
    if iso_dates:
        match = iso_dates[-1]
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    month_names = {
        name: number for number, names in enumerate(
            [(), ('jan', 'january'), ('feb', 'february'), ('mar', 'march'),
             ('apr', 'april'), ('may',), ('jun', 'june'), ('jul', 'july'),
             ('aug', 'august'), ('sep', 'sept', 'september'), ('oct', 'october'),
             ('nov', 'november'), ('dec', 'december')],
            0,
        ) if names for name in names
    }
    natural = re.search(
        r'([A-Za-z]{3,9})\\s+(\\d{1,2})(?:\\s*(?:-|–|to)\\s*(?:[A-Za-z]{3,9}\\s+)?(\\d{1,2}))?\\s*,?\\s+(\\d{4})',
        text,
    )
    if natural:
        month = month_names.get(natural.group(1).lower())
        if month:
            end_day = int(natural.group(3) or natural.group(2))
            return date(int(natural.group(4)), month, end_day)
    return None


class TripBlog(BaseModel):
    title: str = Field(description='A short, warm title for this trip recap blog post.')
    narrative_html: str = Field(description='The blog write-up as a few paragraphs of plain HTML (<p> tags only, no other markup, no <html>/<body> wrapper) - warm, specific, grounded only in the profile and photo captions given, never inventing a place or event not mentioned.')


_RECAP_PROMPT = """Write a warm, specific trip recap blog post for this family's trip, grounded ONLY in what's actually known below - never invent a place, activity or event not mentioned.

Trip profile:
{profile_summary}

Photos taken during the trip, in order, with their captions:
{photo_captions}

Write a title and a few short paragraphs of plain HTML (<p> tags only)."""


def _photo_captions_text(photos: List[Dict[str, Any]]) -> str:
    if not photos:
        return '(no photos uploaded)'
    return '\n'.join(f'{i + 1}. {p["caption"]}' for i, p in enumerate(photos))


def _build_html_page(title: str, narrative_html: str, photos: List[Dict[str, Any]], phone: str, collage_path: Optional[str] = None) -> str:
    """One self-contained HTML file - photos embedded as base64 data
    URIs, not linked separately, so the single file pushed to the repo
    is the whole page; nothing else needs to be served alongside it."""
    figures = []
    for photo in photos:
        try:
            data = Path(photo['stored_path']).read_bytes()
        except OSError:
            continue
        mime = 'image/jpeg' if photo['filename'].lower().endswith(('.jpg', '.jpeg')) else 'image/png'
        b64 = base64.b64encode(data).decode()
        caption = html_escape.escape(photo['caption'])
        timeline_date = datetime.fromisoformat(_photo_timeline_date(photo)).strftime('%b %-d, %Y')
        date_badge = html_escape.escape(timeline_date) + (' (from camera)' if photo.get('captured_at') else ' (upload date)')
        figures.append(f'<figure><img src="data:{mime};base64,{b64}" alt="{caption}"><figcaption>{caption}<br><small>{date_badge}</small></figcaption></figure>')
    gallery = '\n'.join(figures)
    collage_html = ''
    if collage_path:
        try:
            collage_bytes = Path(collage_path).read_bytes()
            collage_b64 = base64.b64encode(collage_bytes).decode()
            collage_html = f'<figure class="collage"><img src="data:image/jpeg;base64,{collage_b64}" alt="AI-composed collage of this trip"><figcaption>Trip highlights collage</figcaption></figure>'
        except OSError:
            logger.warning('Could not read generated trip collage at %s', collage_path)
    safe_title = html_escape.escape(title)

    # The cost summary is built here, deterministically, from real
    # recorded expenses - never handed to the model to restate in prose,
    # where a number could get transcribed wrong. See add_expense's own
    # docstring for why only confirmed bookings ever land in this total.
    cost = get_running_cost(phone)
    cost_html = ''
    if cost['entries']:
        rows = ''.join(f"<li>{html_escape.escape(e['label'])}: {e['amount']:.2f} {html_escape.escape(e['currency'])}</li>" for e in cost['entries'])
        totals = ', '.join(f"{amt:.2f} {html_escape.escape(cur)}" for cur, amt in cost['totals_by_currency'].items())
        cost_html = f'<section class="cost"><h2>What this trip cost</h2><ul>{rows}</ul><p class="total">Total: {totals}</p></section>'

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 720px; margin: 0 auto; padding: 32px 20px 60px; line-height: 1.6; color: #1b1b1f; background: #fffefb; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
  figure {{ margin: 24px 0; }}
  img {{ width: 100%; border-radius: 8px; display: block; }}
  figcaption {{ font-size: 0.9rem; color: #6b6b76; margin-top: 6px; }}
  section.cost {{ margin-top: 40px; padding: 20px; background: #f4f1ea; border-radius: 8px; }}
  section.cost h2 {{ font-size: 1.1rem; margin-top: 0; }}
  section.cost .total {{ font-weight: 600; margin-bottom: 0; }}
  footer {{ margin-top: 48px; font-size: 0.8rem; color: #9a9aa5; }}
</style></head>
<body>
<h1>{safe_title}</h1>
{narrative_html}
{collage_html}
{gallery}
{cost_html}
<footer>Generated by Wayfinder.</footer>
</body></html>"""


async def generate_recap(phone: str, profile_summary: str) -> Dict[str, Any]:
    """Builds the draft recap (write-up + full HTML page) and stores it
    with status='drafted' - does NOT publish anything yet. Overwrites
    any previous draft for this phone (one recap per trip)."""
    photos = get_photos(phone)
    blog = await meta_model_client.ask_schema(
        AGENT_ID, 'generate_trip_blog',
        _RECAP_PROMPT.format(profile_summary=profile_summary, photo_captions=_photo_captions_text(photos)),
        TripBlog,
    )
    collage_path = None
    reference_images = []
    for photo in photos[:6]:
        try:
            reference_images.append((photo['filename'], Path(photo['stored_path']).read_bytes()))
        except OSError:
            logger.warning('Skipping missing trip photo for collage: %s', photo.get('stored_path'))
    if reference_images:
        try:
            collage_bytes = await meta_model_client.generate_image(
                AGENT_ID,
                'compose_trip_photo_collage',
                'Create one polished editorial travel-photo collage using the supplied real trip photos. '
                'Arrange them as a cohesive multi-panel composition with tasteful spacing and a warm, '
                'premium travel-journal feel. Preserve the actual people, places, and moments in the '
                'reference photos; do not invent people, landmarks, activities, or text. Do not add '
                'captions, labels, logos, or watermarks. The result must be a single landscape collage.',
                reference_images,
            )
            PHOTO_DIR.mkdir(parents=True, exist_ok=True)
            collage_file = PHOTO_DIR / f'{phone.lstrip("+")}_{int(datetime.now(timezone.utc).timestamp() * 1000)}_collage.jpg'
            collage_file.write_bytes(collage_bytes)
            collage_path = str(collage_file)
        except Exception:
            logger.exception('Muse Image collage generation failed for %s', phone)
    html_page = _build_html_page(blog.title, blog.narrative_html, photos, phone, collage_path)
    doc = {
        'phone': phone, 'title': blog.title, 'html': html_page,
        'photo_count': len(photos), 'collage_path': collage_path,
        'status': 'drafted',
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    _db()[RECAPS_COLLECTION].update_one({'phone': phone}, {'$set': doc}, upsert=True)
    return doc


def get_pending_recap(phone: str) -> Optional[Dict[str, Any]]:
    return _db()[RECAPS_COLLECTION].find_one({'phone': phone, 'status': 'drafted'})


def get_any_recap(phone: str) -> Optional[Dict[str, Any]]:
    """Any recap at all for this phone, drafted/published/declined - the
    guard against re-detecting trip completion and re-generating once
    this trip's recap has already been handled once (one recap per
    trip, same simplification this project already makes for bookings)."""
    return _db()[RECAPS_COLLECTION].find_one({'phone': phone})


def decline_recap(phone: str) -> None:
    _db()[RECAPS_COLLECTION].update_one({'phone': phone}, {'$set': {'status': 'declined'}})


def _slugify(title: str, phone: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
    return f'{slug or "trip"}-{phone.lstrip("+")}'


def publish_recap(phone: str) -> str:
    """The actual publish step - clones/updates PUBLISH_REPO into a
    scratch dir, writes this one HTML file under trips/<slug>.html,
    commits and pushes. Returns the GitHub Pages URL for the pushed
    file - see this module's own docstring for why Pages, not a faster
    CDN. Raises if PUBLISH_REPO doesn't exist or isn't reachable; the
    caller is responsible for that repo (and its Pages setup) already
    existing (see this module's own docstring - never auto-created per
    trip)."""
    doc = _db()[RECAPS_COLLECTION].find_one({'phone': phone})
    if not doc or doc.get('status') != 'drafted':
        raise RuntimeError('No drafted recap pending publish for this phone.')

    slug = _slugify(doc['title'], phone)
    work_dir = Path(os.environ.get('WAYFINDER_RECAP_WORKDIR', '/tmp/wayfinder_recap_repo'))
    if not (work_dir / '.git').exists():
        work_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', f'https://github.com/{PUBLISH_REPO}.git', str(work_dir)], check=True, capture_output=True)
    else:
        subprocess.run(['git', '-C', str(work_dir), 'pull', '--ff-only'], check=True, capture_output=True)

    trips_dir = work_dir / 'trips'
    trips_dir.mkdir(exist_ok=True)
    file_path = trips_dir / f'{slug}.html'
    file_path.write_text(doc['html'])

    subprocess.run(['git', '-C', str(work_dir), 'add', f'trips/{slug}.html'], check=True, capture_output=True)
    commit = subprocess.run(
        ['git', '-C', str(work_dir), 'commit', '-m', f'Add trip recap: {doc["title"]}'],
        capture_output=True, text=True,
    )
    if commit.returncode != 0 and 'nothing to commit' not in commit.stdout:
        raise RuntimeError(f'git commit failed: {commit.stderr}')
    subprocess.run(['git', '-C', str(work_dir), 'push'], check=True, capture_output=True)

    _db()[RECAPS_COLLECTION].update_one({'phone': phone}, {'$set': {'status': 'published', 'published_at': datetime.now(timezone.utc).isoformat()}})
    owner, repo = PUBLISH_REPO.split('/')
    return f'https://{owner.lower()}.github.io/{repo}/trips/{slug}.html'
