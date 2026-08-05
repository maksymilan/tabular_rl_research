# External DeepSeek API Provider

Status: active provider contract for new external-teacher, evaluation, recovery, and audit calls.

## Active endpoint

New DeepSeek requests use the official OpenAI-compatible service only:

```text
BASE_URL=https://api.deepseek.com
Chat Completions: POST /chat/completions
Models: deepseek-v4-flash, deepseek-v4-pro
```

The local runtime configuration is the ignored repository-root `api.md`:

```text
API_KEY=<official DeepSeek API key>
BASE_URL=https://api.deepseek.com
```

`src/sft/provider_client.py` reads this file. The key must never be committed, copied into another
documentation file, printed in commands or logs, or pasted into model-visible context. Before a
new run, a read-only `GET /models` check should confirm that authentication succeeds and that the
requested model is present.

## Deprecated provider

AimixHub/AIHubMix and `https://aihubmix.com/v1` are deprecated as of 2026-08-05.

- Do not use the deprecated endpoint for new runs.
- Do not retain it as a fallback or proxy when the official API fails.
- Do not use old AimixHub credentials with the official endpoint.
- Do not copy old AimixHub examples from local notes or historical artifacts.
- Historical manifests may retain their original provider metadata for audit integrity; that does
  not authorize reusing the provider.

Official-provider failures must remain visible failures. A runner must not silently change the
provider, base URL, model, carrier, or request controls, because doing so breaks experiment
comparability and provider-identity auditing.

## Chat Completions versus FIM

The repository's causal model↔harness loop uses Chat Completions with native
`reasoning_content` plus visible JSON Output. Its base URL is `https://api.deepseek.com`.

DeepSeek FIM is a separate Beta code-completion API at
`https://api.deepseek.com/beta/completions`. It must not be selected merely because a DeepSeek FIM
documentation page is referenced. FIM may be added only as a separately named and tested feature;
it is not compatible with the current multi-turn tool trajectory transport.

Official references:

- <https://api-docs.deepseek.com/zh-cn/api/create-chat-completion/>
- <https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/>
- <https://api-docs.deepseek.com/zh-cn/guides/json_mode/>
- <https://api-docs.deepseek.com/zh-cn/guides/fim_completion/>

## Migration verification

The 2026-08-05 migration was verified against the official service:

- authenticated `GET /models`: HTTP 200;
- advertised models: `deepseek-v4-flash` and `deepseek-v4-pro`;
- minimal `deepseek-v4-flash` Chat Completion: successful;
- native reasoning and visible JSON were both present;
- the existing provider adapter reconstructed and strictly parsed the expected tool action.

This verifies transport compatibility only. Each research run must still record its model,
provider response metadata, prompt hashes, request controls, token usage, and protocol manifest.
