# Amperstand

> **Self-hosted personal vault.** Capture articles, YouTube videos, RSS feeds, and newsletter emails into clean markdown files you own. Plug into any AI agent as a private, curated corpus.

When your agent researches something, it gets the open web's first page: SEO listicles, AI summaries of AI summaries, content farms outranking the people who actually know things. Amperstand makes the writers *you* already trust the corpus your agents query — instead of whatever Google ranks today.

```
   ┌───────────┐   ┌──────────────┐   ┌────────────┐   ┌─────────────┐
   │    CLI    │   │ Telegram bot │   │ Chrome ext │   │  anything   │
   └─────┬─────┘   └──────┬───────┘   └──────┬─────┘   └──────┬──────┘
         │                │                  │                │
         └────────────────┴────────┬─────────┴────────────────┘
                                   │  HTTP + API key
                           ┌───────▼──────────┐
                           │  amperstand-core │
                           └──────────────────┘
```

One server, N clients, HTTP between them. The bot and extension aren't special — they're reference HTTP clients. Build a Discord bot, an n8n flow, your own agent code; the server doesn't care.

## Install

```bash
# On a fresh Ubuntu droplet:
git clone https://github.com/zkid18/amperstand-core /opt/amperstand/amperstand-core
cd /opt/amperstand/amperstand-core
sudo bash deploy/bootstrap.sh
```

For Docker Compose, the full step-by-step (DigitalOcean signup → first capture), production hardening, key rotation, backups, and everything else — see the **docs** below.

## Documentation

Full docs are in [`docs/`](./docs/) — a [Mintlify](https://mintlify.com/) site you can preview locally:

```bash
cd docs && mintlify dev
# → http://localhost:3000
```

The sidebar covers: Set up a server, Use the CLI, Build with AI agents, Build a custom client, Operate in production, HTTP API reference.

## Companion repos

| Repo | What |
| --- | --- |
| [`amperstand-cli`](https://github.com/zkid18/amperstand-cli) | CLI client. Local-first — imports this repo as a library and captures without a server; acts as a thin HTTP wrapper when pointed at one. `pip install amperstand`. |
| [`amperstand-tg-bot`](https://github.com/zkid18/amperstand-tg-bot) | Telegram bot. DM URLs, get markdown saved. |
| [`amperstand-extension`](https://github.com/zkid18/amperstand-extension) | Chrome MV3 clipper. One-click save. |
| `prompts/` | Drop-in system prompts for embedding the vault into Cursor, Custom GPTs, n8n, Claude Code, etc. |

## What it captures

| Surface | Extractor |
| --- | --- |
| Articles & blog posts | trafilatura; Anysite fetch and browser render for pages that block server IPs |
| YouTube videos | Caption track via Anysite, title and channel via oembed |
| LinkedIn / Reddit | Full post via Anysite; LinkedIn videos transcribed with Whisper when enabled |
| Twitter / X | fxtwitter rewrite |
| RSS / Atom feeds | `feedparser` + per-item extraction |
| Newsletter emails | `.eml` parsing, IMAP fetch loop |

All extraction runs server-side. Clients hand over a URL, get JSON back.

## What you'll need

- A box to run it on (the reference target is a $4 DigitalOcean droplet)
- (Recommended) An OpenAI API key — required for semantic search, hybrid-search rerank, chat-with-vault, and LinkedIn video transcription. **~$0.02 per 1,000 docs.** Without it, capture + keyword search still work; the rest returns `503`. Roadmap has Ollama / local-model support; PRs welcome.
- (Recommended) An [Anysite](https://anysite.io) API key — required for LinkedIn, YouTube and Reddit capture and for articles behind bot walls. 1 credit per capture, 10 for a browser-rendered page.

See [docs / Set up a server](./docs/quickstart.mdx) for the cost breakdown + privacy data-flow notes.

## License

MIT. Fork it, swap embeddings, build whatever you need on top.
