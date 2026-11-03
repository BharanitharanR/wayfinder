# Wayfinder

Wayfinder is a family trip-planning assistant. You chat with it in plain
English, it figures out what it still needs to know (family size, who's
traveling, leave dates, a budget, any date constraints), and once it has
everything it actually researches real flights and hotels and sends back a
grounded recommendation - not a guess, real prices and real booking links
from live searches.

Every bit of reasoning Wayfinder does - reading your messages, deciding
what's still missing, choosing which tool to call, searching the live web,
reading an uploaded PDF statement, and writing the final recommendation -
runs on **Meta's Model API (Muse Spark)**. Nothing else - including
document extraction, which is Muse Spark reading the file directly, not a
local OCR model.

This repo is Wayfinder on its own: everything it takes to run the agent
end to end from a terminal, with no WhatsApp, no other agents, and no
larger platform underneath it required.

## What you'll have running

- Two ways to talk to Wayfinder - a browser UI (`wayfinder_web.py`) with
  chat, drag-and-drop upload for both PDFs and photos, and a finance
  panel, or a terminal chat (`wayfinder_dev_cli.py`) for scripting/fast
  iteration. Both call the exact same backend.
- A mandatory phone-number step before either one does anything - the
  real phone number is the one piece of durable identity everything else
  links by, the same requirement the WhatsApp upload page already
  enforces. The web UI has its own in-page phone gate for this now; it
  no longer sends you to a separate page to confirm it.
- Real tool calls behind the scenes: live flight search (Kiwi.com and
  LetsFG), live hotel search (Trivago), and live web search (Meta's own
  built-in search grounding) - all real, no mocked data.
- Real flight booking, with your explicit confirmation first: LetsFG's
  sandbox API books a real-shaped (fake-airline) flight end to end once
  you say yes to a specific offer - never automatically, and never from
  inside the model's own tool-choice loop. See LETSFG_ENV in `.env.example`
  for what it takes to point this at real bookings instead.
- A household cash-flow profile from an uploaded statement - average
  income, essential/discretionary/obligation spend by category, and a
  conservative (25th-percentile) monthly surplus figure, not just a
  single average - deterministic arithmetic over Muse Spark's one
  normalization call, same reasoning as the trip-cost figures below.
- Trip recaps: upload photos during a trip, say you're back (or Wayfinder
  notices a flight/train selfie near your travel dates), and it drafts a
  blog-style recap page with your photos - published to a public,
  shareable link only once you say yes.
- A local MongoDB holding your conversation state (so you can close the
  terminal and pick up where you left off).

## 1. Clone and set up Python

```bash
git clone https://github.com/BharanitharanR/wayfinder.git
cd wayfinder
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Needs Python 3.11+, and `ffmpeg` on your PATH (`brew install ffmpeg`) -
used to convert an uploaded voice note to the WAV format Muse Voice
Transcribe requires.

## 2. Add your Meta Model API key

This is the only secret Wayfinder needs.

1. Get a key from the [Model API dashboard](https://dev.meta.ai) -> API
   keys -> Create API key.
2. Copy the example env file and fill it in:

```bash
cp .env.example .env
```

3. Open `.env` and paste your key in:

```
MODEL_API_KEY=your-real-key-here
```

That's it - `.env` is gitignored, it never gets committed, and
`mesh/lib/meta_model_client.py` loads it automatically the moment you run
anything. You don't need to `export` it yourself or touch your shell
profile.

## 3. Start MongoDB

Wayfinder stores chat history and uploaded-document state in a local
Mongo. If you don't already have it:

```bash
brew install mongodb-community   # macOS
brew services start mongodb-community
```

(Or follow https://www.mongodb.com/docs/manual/installation/ on another
OS.) No password, no setup beyond "make sure it's running" - Wayfinder
talks to `mongodb://localhost:27017` by default.

## 4. Run Wayfinder

### Option A - the browser UI (recommended)

```bash
bash start-wayfinder.sh
```

Opens `http://127.0.0.1:8765` automatically. First thing you'll see is a
phone-number prompt (country code + digits) - that's mandatory, the same
real-identity requirement WhatsApp enforces, just done in-page now
instead of a separate link. After that: a chat pane, a **+** button in
the composer bar to upload a bank statement PDF or trip photos directly
(no separate page, multiple files at once), and a Finances tab that
fills in once a statement's processed.

### Option B - the terminal CLI

```bash
python -m mesh.orchestrator.wayfinder_dev_cli
```

You'll land in a chat prompt. Just talk to it:

```
Wayfinder dev CLI - acting as +15550001234
No successfully-read documents on file yet.
Upload link (first-time, phone number required): http://...

> Hi, we're a family of 4 planning a trip to Goa in December
Wayfinder: Thanks - to keep going, could you also share: ...
```

Useful commands inside the chat:

| Command | What it does |
|---|---|
| `/state` | Run Wayfinder's profile inference right now and print exactly what it currently knows about you |
| `/research <question>` | Test a tool call directly - e.g. `/research flights from Chennai to Goa on 15/12/2026` |
| `/uploads` | List documents you've uploaded |
| `/reset` | Wipe this session and start clean |
| `/quit` | Exit |

Everything typed that isn't one of the above is sent to Wayfinder as a
real customer message, same as WhatsApp would.

Run with a different identity (to simulate a second customer):

```bash
python -m mesh.orchestrator.wayfinder_dev_cli --phone +15559998888
```

## Trying it with a real financial statement

`mesh/orchestrator/samples/` ships two **synthetic** bank statement PDFs
you can use to see the full flow, including the budget-signal step, work
end to end - every name, account number, and transaction in them is made
up, not a real person's data:

- `sample_bank_statement_clean_format.pdf` - clearly labeled rows
  ("SALARY", "RENT", a running balance column) - the easy case.
- `sample_bank_statement_messy_format.pdf` - no explicit labels, reference
  codes instead, a single +/- amount column, no running balance - closer
  to what a real statement actually looks like.

The clean one produces a trustworthy budget read; the messy one deliberately
doesn't (too many unlabeled transactions) - that's `mesh/orchestrator/
budget_engine.py` working as intended, not a bug. See that file's own
docstring for how the number is computed: Muse Spark normalizes the
statement into a strict transaction list (the one model call in this
step), then classification, monthly aggregation, and the surplus figure
itself are plain deterministic Python - no LLM guesses the number.

## Trying the trip-recap feature

`mesh/orchestrator/samples/trip_photos/` ships five illustrative trip
photos (not real photos - simple drawn scenes), each with a real embedded
EXIF capture date spanning Dec 6-12, 2026, for exercising
`trip_recap.py`'s photo timeline and completion detection:

- `1_beach_day_arrival.jpg` through `4_sunset_beach.jpg` - ordinary trip
  photos, each captioned by Muse Spark at upload time.
- `5_flight_home_selfie.jpg` - deliberately looks like an airplane-window
  shot; confirmed live against Muse Spark to set
  `looks_like_transit_selfie=True`, which - landing near the stated
  travel window's end date - triggers automatic trip-completion
  detection with no explicit "we're back" message needed.

Upload these the same way as a statement:

- **Web UI**: the **+** button in the composer bar takes both PDFs and
  photos directly, multiple at once - no separate page.
- **CLI**: use the upload link Wayfinder prints (first message, or
  `/link`), open it in a browser, and upload there instead. Not as a raw
  WhatsApp attachment, which gets redirected to that link rather than
  processed directly.

Either way, it's the real upload/extraction path (Muse Spark reading the
file directly), not a shortcut, so what Wayfinder reasons over is
genuinely read from the file you give it.

## Where the Meta model is actually called

If you want to see it directly, it's all in one place:
`mesh/lib/meta_model_client.py`. Every Wayfinder reasoning step - profile
extraction, tool choice, live web search, final write-up - calls into
this module, which talks to `https://api.meta.ai/v1` using the
OpenAI-compatible SDK (Model API's own supported path). Nothing in
Wayfinder's own code calls any other model provider.

---

*A live version of Wayfinder is already running and reachable over
WhatsApp - if you'd like to try the real thing instead of (or alongside)
running it yourself, just ask and I'll share how to reach it.*
