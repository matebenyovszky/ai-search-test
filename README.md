# AI Search Test Suite

Compare AI models on research, factual writing, speed and cost, retaining the
original Excel workflow. The 2026 refresh adds current REST APIs, reproducible
JSON records, and providers with free inference or limited free quotas.
It never publishes content or changes RoboHorizon's production models.

## Setup

Python 3.10+:

~~~sh
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install -r requirements.txt
~~~

Copy .env.example to .env and configure only the providers you want.
The old GEMINI_API_KEY, PERPLEXITY_KEY, ANTHROPIC_KEY and GROK_API_KEY names
remain supported. Missing credentials skip that provider. Imports, listings and
dry runs require no Google key and make no inference request. Secrets and new
results are ignored by Git.

~~~sh
python aisearchtest.py --list-models
python aisearchtest.py --discover --providers google,openai,groq,openrouter
~~~

Discovery makes read-only catalog requests, prints model IDs rather than keys,
and does not modify the registry.

## Run comparisons

The default **research** suite tests the critical article-writing gate: identify
the exact recent story from short, misspelled, ambiguous clues using live web
search, then write about it in the requested language. It rejects shared/none
mode. Fluent prose about the wrong event is a failure.

The default command displays a plan. Add --run to execute it.

~~~sh
python aisearchtest.py --models gemini-2.5-pro,gemini-3.8-flash,anthropic-claude-haiku-5-5 --case-id crowdbricks-batching-typo --languages hu --repeat 3 --max-calls 9
python aisearchtest.py --models gemini-2.5-pro,gemini-3.8-flash --languages all --repeat 3 --max-calls 156
~~~

research_cases.json stores independently checked answers, dated primary sources,
and common confusions. Answers are never included in the model's prompt. The
fresh cases distinguish the October 27 MotionBricks SOMA/checkpoint update from
the similarly named community CrowdBricks batching runtime. The original 2025
REK/San fransico/Unitree input remains an opt-in historical regression:

~~~sh
python aisearchtest.py --case-id legacy-rek-typo-2025 --languages en-us,hu
~~~

Fresh cases expire after seven days. --max-source-age-days 0 requires a verified
same-day case; older or future-dated cases are rejected before any spending.
The supplied October 27 cases do not establish same-day discovery quality.
Replace them with independently verified announcements from the day of a real
evaluation. Historical regressions never count as fresh-news tests.

--languages accepts en-us, hu, en-gb, de, fr, it, zh, pl, es, pt, ja, ko,
zh-hant, or all. The default is en-us,hu. Cases can supply input_variants for
noisy non-English clues as well as different output languages.

The Research Review sheet records observed searches, matching primary-source
citations, and target-keyword hints, followed by blank manual-review fields for
correct topic/event, freshness, dates/units, unsupported claims, natural language,
and publishability. Hints are not factual verification or production approval.
Rank total cost only among models that pass research and language review;
include search/tool charges, retrieved input and thinking tokens. Use repeated
runs on several real stories, including same-day cases, before changing models.
When a provider returns opaque grounding redirects, direct primary links in the
article are recorded separately as hints; a reviewer must still check the source.

Three modes measure different capabilities; compare results within the same mode.
Use --suite general for shared/none scenarios from queries.json:

| Mode | What it measures | Providers |
| --- | --- | --- |
| native (default) | Each model's own research plus writing | Google, OpenAI, Anthropic, Perplexity, xAI |
| shared | Writing from identical retrieved evidence | All |
| none | Writing, translation or extraction from supplied input; no search claim | All |

Shared mode retrieves five DuckDuckGo snippets once per scenario via ddgs and
supplies identical evidence to every model and repeat. Snippets are not full-page
retrieval or verified ground truth. For verified or reproducible sources, supply
a JSON array with --evidence:

~~~json
[{"title":"Official announcement","url":"https://example.org/release","text":"Verified source text..."}]
~~~

~~~sh
python aisearchtest.py --suite general --search-mode shared --evidence evidence.json --models gemini-3-flash-preview,gemini-3.1-flash-lite,groq-gpt-oss-120b --query-id robohorizon-hu --max-calls 3 --run
python aisearchtest.py --suite general --search-mode shared --free-only --query-id robohorizon-en --run
python aisearchtest.py --search-mode none --models gemini-3.1-flash-lite --query "Translate this article into Hungarian: ..." --run
~~~

queries.json contains US English and Hungarian RoboHorizon scenarios plus the
original Hungarian offer-comparison task. Override with --query, --queries FILE
or --query-id. A scenario may specify required_links for URL-preservation checks.
Review facts, natural language, technical detail and resource links before
changing production models. Use several real stories and repeated runs; the
program does not invent an automatic editorial quality score.

## Models and free access

models.json is a dated editable registry with official pricing/catalog sources.
It includes the actual RoboHorizon baselines (Gemini 2.5 Pro for articles/social
copy, Gemini 3 Flash Preview for translations), newer Gemini models, GPT-6 Luna,
GPT-6.1, Claude Haiku/Sonnet 5.5, Perplexity Agent models, Grok 4.7, MiniMax,
Groq GPT-OSS, Cerebras, OpenRouter and Hugging Face. Use --discover to check
account-specific availability, followed by a live request.

The [AgentHatchery Hermes provider reference](https://github.com/agenthatchery/hermes-agent/blob/main/website/docs/integrations/providers.md)
helped identify compatible endpoints. Free access has different meanings:

- **OpenRouter :free / openrouter/free:** zero-priced inference with limits.
  Strict --free-only selects only these, disables paid search plugins and paid
  model fallbacks, and sets zero maximum token prices. The free router can choose
  different free models; the returned model ID is recorded.
- **Google and Groq:** free quotas may exist, but the same endpoints can bill
  paid usage. They are excluded from strict --free-only.
- **Cerebras and Hugging Face:** account-specific quotas or trial credits may
  exist. Unknown prices remain unknown; these are excluded from --free-only.

Models without a native search adapter require shared or none mode. The program
does not silently add a paid search proxy to a free model.

Perplexity now uses /v1/responses (Agent API alias), replacing the 2025 Sonar
Chat Completions integration. OpenAI and xAI use Responses, Anthropic uses
Messages with basic server search, and Google uses generateContent grounding.
No old model alias is silently substituted. Anthropic pause_turn is marked
incomplete rather than automatically continuing and spending more.

## Costs, limits and artifacts

- Default maximum: six inference requests. Increase with --max-calls.
- --stop-after-usd (default $1) is a **soft stop between calls**, not a provider
  billing cap. One request can exceed it. Provider account limits impose hard
  spending caps. There are no automatic retries or model fallbacks.
- --max-tokens and --timeout limit each request. Errors, empty and incomplete
  responses remain visible and produce a nonzero exit status.
- Usage comes from provider responses. Gemini thinking tokens are added;
  Responses/Chat Completions reasoning is already included in output tokens.
  Cached inputs are recorded. Provider-reported cost overrides estimates.
- Estimates use standard short-context list prices, without free allowances,
  negotiated rates, taxes or credits. Search counts use provider metadata.
  Missing usage or uncertain costs are null, never zero. Recorded promotional
  Gemini price changes take effect on their specified dates.
- Prompts are limited to 60,000 characters. Known price thresholds for returned
  input-token counts apply even when web research expands a short submission.
  Cache storage and non-search tool billing are outside this benchmark's scope.
- A successful response with unknown cost stops the run. Failed or timed-out
  requests may still have incurred charges; provider billing is authoritative.

Each run creates results/<Budapest timestamp>/:

- **results.json:** exact prompts, hashes, shared evidence, randomization seed,
  price snapshot, complete raw responses, checks, errors and usage. Saved after
  every call so later failures do not lose earlier work.
- **results.xlsx:** Summary, complete Responses (split at Excel's cell limit),
  and Offers (three cheapest numeric offers per model/scenario). Errors and
  unknown costs stay visible. Costs are per request, not per offer.

Use --output-dir, --no-excel and --seed to customize artifacts and ordering.
Review results before sharing: they include prompts and supplied source data.
Historical 2025 workbooks are unchanged.

## Tests

~~~sh
python -m pytest -q
~~~

Tests mock networking and need no keys or paid calls.

## License

MIT
