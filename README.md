# Wayfinder

Wayfinder is a family trip-planning assistant. You chat with it in plain
English, it figures out what it still needs to know (family size, who's
traveling, leave dates, a budget, any date constraints), and once it has
everything it actually researches real flights and hotels and sends back a
grounded recommendation - not a guess, real prices and real booking links
from live searches.

Every bit of reasoning Wayfinder does - reading your messages, deciding
what's still missing, choosing which tool to call, searching the live web,
and writing the final recommendation - runs on **Meta's Model API (Muse
Spark)**. Nothing else.

This repo is Wayfinder on its own: everything it takes to run the agent
end to end from a terminal, with no WhatsApp, no other agents, and no
larger platform underneath it required.

## What you'll have running

- A terminal chat with Wayfinder (`wayfinder_dev_cli.py`) - type messages,
  get replies, exactly like texting it.
- Real tool calls behind the scenes: live flight search (Kiwi.com), live
  hotel search (Trivago), and live web search (Meta's own built-in search
  grounding) - all real, no mocked data.
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

Needs Python 3.11+.

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

When Wayfinder prints an upload link (first message, or `/link`), open it
in a browser and upload either file - that's the real upload/OCR path
(Docling), not a shortcut, so the extracted text Wayfinder reasons over is
genuinely read from the PDF you give it.

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
