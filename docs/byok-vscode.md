# BYOK: use `ofa` as a VS Code chat backend

`ofa --serve` exposes an OpenAI-compatible HTTP server so VS Code's
Bring-Your-Own-Key (BYOK) feature can route Chat requests through
`ofa`. Every request still gets `ofa`'s full domain layer:

- Mode-specific system prompts (OpenFOAM, HPC, code, AMReX, ReFrame)
- RAG retrieval over Kestrel docs, OpenFOAM tutorials, AMReX / Marbles /
  ReFrame source trees
- Long-term `preferences` + `lessons` memory
- Loaded skills

The model picker in VS Code shows five entries — one per `ofa` mode —
so you can switch context (system prompt + RAG retriever) without
restarting the server.

## What runs where

```
laptop                       Kestrel login node            Kestrel compute node
+---------+    SSH (-L)     +-------------------+    SSH    +------------------+
| VSCode  | <===========>   | sshd (port-fwd)   | <=======> | ofa --serve      |
| BYOK    |   localhost:    |                   |   (or     | (Ollama localhost|
| 11435   |   11435         |                   |  same     |  on the same node)
+---------+                 +-------------------+   node)   +------------------+
```

The SSH port-forward is the recommended way to reach the server from a
laptop: nothing has to be opened to the network, and `ofa --serve`
binds to `127.0.0.1` by default.

## Setup (one-time)

### 1. On Kestrel: pick where the server runs

Easiest path is to start `ofa --serve` inside an existing interactive
allocation that already runs `ofa`:

```bash
# Inside your salloc/sbatch session
ml assistant                                       # adjust to your module name
ofa --serve                                        # blocks; Ctrl+C to stop
# Optionally: ofa --serve --serve-port 11500
```

First run creates a bearer token at `$OFA_SCRATCH/.ofa_api_key`
(mode `0600`). Subsequent runs reuse it. Read it with:

```bash
cat "$OFA_SCRATCH/.ofa_api_key"
```

You'll paste that token into the VS Code config below.

### 2. On your laptop: open an SSH tunnel

```bash
# Replace x3101c0s9b0n0 with the node from `ofa --serve`'s output line
# "node=<hostname> pid=<n>".
ssh -N -L 11435:x3101c0s9b0n0:11435 kestrel
```

Leave that terminal open. (If you connected to a login node and started
`ofa --serve` there, replace the node with `localhost`.)

### 3. On your laptop: configure VS Code BYOK

Open the Command Palette → `Chat: Manage Language Models` → `Add Models`
→ pick the custom-endpoint provider. VS Code will open a
`chatLanguageModels.json` file. Paste:

```json
[
  {
    "name": "OFA (Kestrel)",
    "vendor": "customendpoint",
    "apiKey": "ofa-PASTE_TOKEN_FROM_KEYFILE_HERE",
    "apiType": "chat-completions",
    "models": [
      {
        "id": "ofa-openfoam",
        "name": "OFA · OpenFOAM",
        "url": "http://localhost:11435/v1/chat/completions",
        "toolCalling": false,
        "maxInputTokens": 32000,
        "maxOutputTokens": 8192
      },
      {
        "id": "ofa-hpc",
        "name": "OFA · Kestrel HPC",
        "url": "http://localhost:11435/v1/chat/completions",
        "toolCalling": false,
        "maxInputTokens": 32000,
        "maxOutputTokens": 8192
      },
      {
        "id": "ofa-code",
        "name": "OFA · Code",
        "url": "http://localhost:11435/v1/chat/completions",
        "toolCalling": false,
        "maxInputTokens": 32000,
        "maxOutputTokens": 8192
      },
      {
        "id": "ofa-amrex",
        "name": "OFA · AMReX / MARBLES",
        "url": "http://localhost:11435/v1/chat/completions",
        "toolCalling": false,
        "maxInputTokens": 32000,
        "maxOutputTokens": 8192
      },
      {
        "id": "ofa-reframe",
        "name": "OFA · ReFrame (RHEL9)",
        "url": "http://localhost:11435/v1/chat/completions",
        "toolCalling": false,
        "maxInputTokens": 32000,
        "maxOutputTokens": 8192
      }
    ]
  }
]
```

The five models appear in the VS Code Chat model picker. Pick one and
chat as normal.

## Day-to-day

1. Start (or already have) an allocation on Kestrel with `ofa --serve`
   running.
2. From the laptop: `ssh -N -L 11435:<node>:11435 kestrel` in a spare
   terminal.
3. In VS Code, open Chat, pick `OFA · <mode>`, type.

To stop: `Ctrl+C` the SSH forward, then `Ctrl+C` the `ofa --serve`
process. The token in `$OFA_SCRATCH/.ofa_api_key` persists across runs
so you only paste it into VS Code once.

## Endpoints

| Method | Path                       | Notes                                  |
|--------|----------------------------|----------------------------------------|
| GET    | `/healthz`                 | no auth; for readiness probes          |
| GET    | `/v1/models`               | lists the five `ofa-*` model ids       |
| POST   | `/v1/chat/completions`     | OpenAI format; supports `stream: true` |

All chat-completion requests must include
`Authorization: Bearer <token>` unless the server was started with
`--serve-no-auth` (local development only).

## Why `toolCalling: false`

VS Code's agent mode expects the model to emit OpenAI-format
`tool_calls`. A 31B local model on a GPU node is not reliable enough at
that protocol to drive VS Code's agent loop well. `ofa` already has its
own carefully-tuned tool-fence convention for that — invoke it via the
CLI when you need agent behaviour, and use the BYOK path here for chat,
explanations, snippet generation, and "what does this Kestrel error
mean?" type queries that benefit from `ofa`'s RAG.

## Troubleshooting

**`401 Unauthorized`** — your `chatLanguageModels.json` `apiKey` doesn't
match `$OFA_SCRATCH/.ofa_api_key` on Kestrel. Re-read the file and paste
it again.

**Connection refused** — the SSH port-forward isn't up, or
`ofa --serve` is bound to a different host/port than what your forward
points at. Check the line `[ofa-serve] listening on http://<host>:<port>`.

**Chat hangs on first reply** — first response loads the model into GPU
memory; subsequent replies are fast. Check `[ofa-serve]` log lines on
the Kestrel side for activity.

**Wrong system prompt** — pick a different `OFA · *` model from the
dropdown; each one selects a different mode + RAG retriever.

**`No matching distribution found`** — `ofa --serve` uses only Python
stdlib for the HTTP layer; no new dependencies are required beyond
those `ofa` already needs.
