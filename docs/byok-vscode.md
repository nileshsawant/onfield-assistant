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
| BYOK    |   localhost:    |                   |  (compute | (Ollama localhost|
| <Plap>  |   <Prem>        |                   |  node has |  on the same node)
+---------+                 +-------------------+   GPUs)   +------------------+
```

The SSH port-forward is the recommended way to reach the server from a
laptop: nothing has to be opened to the network, and `ofa --serve`
binds to `127.0.0.1` by default.

Both ports (`<Plap>` and `<Prem>` above) are dynamic by default —
`ofa --serve` prints the exact `ssh -L` command at startup with the
actual numbers filled in. The laptop-side port is a random per-user
port in **49200–64200**, persisted across restarts; the Kestrel-side
port is OS-assigned each run. See the Setup section for overrides.

## Setup (one-time)

### 1. On Kestrel: start `ofa --serve` inside an allocation

Inference needs a GPU, so always start `ofa --serve` inside an
interactive allocation (the same one `ofa` would open for you):

```bash
# Inside your salloc/sbatch session
ml assistant
ofa --serve                                        # blocks; Ctrl+C to stop
# Optional overrides:
#   ofa --serve --serve-port 11500          # pin the Kestrel-side port
#   ofa --serve --serve-local-port 50001    # pin the laptop-side port
```

The first run creates a bearer token at `$OFA_SCRATCH/.ofa_api_key`
(mode `0600`); subsequent runs reuse it.

`ofa --serve` prints a connection block at startup with the **exact
`ssh -N -L ...` command** to paste on your laptop (compute-node
hostname and ports already filled in), the BYOK URL, the bearer token,
and a quick `curl` health-check. Copy that block.

By default both ports are dynamic:
- The Kestrel-side port is OS-assigned (so two `ofa --serve` sessions
  on the same node never collide with each other or with Ollama).
- The laptop-side port is a random per-user port in **49200–64200**,
  persisted to `$OFA_SCRATCH/.ofa_serve_local_port` so the VS Code
  BYOK URL stays stable across `--serve` restarts. This range stays
  clear of 11434/11435/11436 which VS Code Remote-SSH likes to
  auto-forward.

### 2. On your laptop: paste the printed `ssh -L` command

It will look something like:

```bash
ssh -N -o ExitOnForwardFailure=yes \
    -L 51823:x3101c0s9b0n0:39157 kestrel.hpc.nrel.gov
```

Leave that terminal open. If the local port `ofa --serve` chose is
already taken on your laptop, re-run on Kestrel with
`--serve-local-port <N>` to pick a different one.

### 3. On your laptop: configure VS Code BYOK

Open the Command Palette → `Chat: Manage Language Models` → `Add Models`
→ pick the custom-endpoint provider. VS Code will open a
`chatLanguageModels.json` file. Paste:

> **Update `<LOCAL_PORT>` below to the port `ofa --serve` printed in
> its "Then point VS Code BYOK at:" line.** Same number in all five
> URLs. The persisted port file means you only do this once — the
> URL stays valid across `--serve` restarts.

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
        "url": "http://localhost:<LOCAL_PORT>/v1/chat/completions",
        "toolCalling": false,
        "maxInputTokens": 32000,
        "maxOutputTokens": 8192
      },
      {
        "id": "ofa-hpc",
        "name": "OFA · Kestrel HPC",
        "url": "http://localhost:<LOCAL_PORT>/v1/chat/completions",
        "toolCalling": false,
        "maxInputTokens": 32000,
        "maxOutputTokens": 8192
      },
      {
        "id": "ofa-code",
        "name": "OFA · Code",
        "url": "http://localhost:<LOCAL_PORT>/v1/chat/completions",
        "toolCalling": false,
        "maxInputTokens": 32000,
        "maxOutputTokens": 8192
      },
      {
        "id": "ofa-amrex",
        "name": "OFA · AMReX / MARBLES",
        "url": "http://localhost:<LOCAL_PORT>/v1/chat/completions",
        "toolCalling": false,
        "maxInputTokens": 32000,
        "maxOutputTokens": 8192
      },
      {
        "id": "ofa-reframe",
        "name": "OFA · ReFrame (RHEL9)",
        "url": "http://localhost:<LOCAL_PORT>/v1/chat/completions",
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
2. From the laptop: paste the `ssh -N -L ...` line that `ofa --serve`
   printed in a spare terminal. It already has the compute-node
   hostname and both port numbers filled in.
3. In VS Code, open Chat, pick `OFA · <mode>`, type.

To stop: `Ctrl+C` the SSH forward, then `Ctrl+C` the `ofa --serve`
process. The bearer token in `$OFA_SCRATCH/.ofa_api_key` AND the
laptop port in `$OFA_SCRATCH/.ofa_serve_local_port` persist across
runs, so the VS Code BYOK URL and apiKey stay the same.

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

**`bind: Address already in use` on the laptop port** — something else
on your laptop is listening on the suggested local port (often VS Code
Remote-SSH auto-forward). Re-run `ofa --serve --serve-local-port <N>`
on Kestrel to pick a different port; the next printed `ssh -L` line
will use it. You can also stop VS Code's auto-forward: View → Ports →
right-click → Stop Forwarding.

**`curl http://localhost:<port>/healthz` hangs** — the TCP connection
reached *something*, but not `ofa --serve`. Almost always you're
connected to the wrong host: `ofa --serve` is on a compute node and
your tunnel terminates on the login node (or vice versa). Re-run the
exact `ssh -L ...` line `ofa --serve` printed — it has the right
compute-node hostname filled in.

**Chat hangs on first reply** — first response loads the model into GPU
memory; subsequent replies are fast. Check `[ofa-serve]` log lines on
the Kestrel side for activity.

**Wrong system prompt** — pick a different `OFA · *` model from the
dropdown; each one selects a different mode + RAG retriever.

**`No matching distribution found`** — `ofa --serve` uses only Python
stdlib for the HTTP layer; no new dependencies are required beyond
those `ofa` already needs.
