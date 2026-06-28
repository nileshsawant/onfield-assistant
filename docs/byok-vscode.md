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

## What runs where (and what each port is for)

```
                          (your laptop)                  (Kestrel compute node)
                          +-----------+                  +---------------------+
  VS Code Chat (BYOK) --> | localhost | -- ssh -L -->    | ofa --serve         |
                          | :LOCAL    |   (through       | listening on        |
                          +-----------+   login node)    | 127.0.0.1:REMOTE    |
                                                         +---------------------+
                                          ^                    |
                                          |                    v
                                  (login node just         (Ollama on the
                                   forwards bytes;          same compute
                                   binds no port)           node, GPU here)
```

There are **two** port numbers in play and they live on different
machines. Keeping them distinct in your head solves 90% of the
"address already in use" / "connection refused" confusion:

| Port      | Lives on          | Set by                     | Default                                  |
|-----------|-------------------|----------------------------|------------------------------------------|
| REMOTE    | Kestrel compute   | `--serve-port`             | `0` → OS picks a free port at bind time  |
| LOCAL     | Your laptop       | `--serve-local-port`       | random in 49200–64200, persisted in scratch |

`ofa --serve` prints both numbers, plus the exact `ssh -L` line and
the BYOK URL, in a clearly-labelled block at startup. Copy what it
prints — don't hand-type numbers.

The login node never binds anything; it just forwards bytes between
your laptop and the compute node where `ofa --serve` actually runs.

## Setup (one-time)

### 1. On Kestrel: start `ofa --serve` inside an allocation

Inference needs a GPU, so always start `ofa --serve` inside an
interactive allocation (the same one `ofa` would open for you):

```bash
# Inside your salloc/sbatch session
ml assistant
ofa --serve                                        # blocks; Ctrl+C to stop
# Optional overrides:
#   ofa --serve --serve-port 11500          # pin the REMOTE port
#   ofa --serve --serve-local-port 50001    # pin the LOCAL port
```

The first run creates a bearer token at `$OFA_SCRATCH/.ofa_api_key`
(mode `0600`); subsequent runs reuse it. The LOCAL port is also
persisted (to `$OFA_SCRATCH/.ofa_serve_local_port`) so your VS Code
BYOK URL stays the same across `--serve` restarts.

`ofa --serve` then prints a labelled connection block. **Copy it
verbatim** — every number you need is already filled in:

```
================================ CONNECT FROM YOUR LAPTOP ================================

  Kestrel compute node:       x3101c0s9b0n0
  REMOTE port (this server):  39157
  LOCAL  port (your laptop):  51823

Step 1 — run this in a new laptop terminal (leave it open):
  ssh -N -o ExitOnForwardFailure=yes -L 51823:x3101c0s9b0n0:39157 kestrel.hpc.nrel.gov

Step 2 — quick sanity check from the laptop:
  curl http://localhost:51823/healthz

Step 3 — paste these into VS Code chatLanguageModels.json:
  url    = http://localhost:51823/v1/chat/completions
  apiKey = ofa-...

If Step 1 fails with 'Address already in use':
  Something on your LAPTOP is holding 51823. Most often that's VS Code Remote-SSH
  auto-forward. Pick a different number with
      ofa --serve --serve-local-port <N>
==========================================================================================
```

### 2. On your laptop: paste Step 1 and verify with Step 2

```bash
ssh -N -o ExitOnForwardFailure=yes \
    -L 51823:x3101c0s9b0n0:39157 kestrel.hpc.nrel.gov
```

Leave that terminal open. In another laptop terminal:

```bash
curl http://localhost:51823/healthz   # expect {"status":"ok"}
```

If `ssh` fails with `bind: Address already in use`, see
[Troubleshooting](#troubleshooting) below — usually a one-liner to
fix.

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

### Quick orientation: which port is failing?

When something doesn't work, ask "is the **LOCAL** port or the
**REMOTE** port involved?" — the fix is different for each.

| Symptom                                          | Port at fault    | Fix                                 |
|--------------------------------------------------|------------------|-------------------------------------|
| `ssh -L` says `bind: Address already in use`     | LOCAL (laptop)   | Pick a new `--serve-local-port`     |
| `curl localhost:LOCAL` returns "Connection refused" | LOCAL          | The `ssh -L` isn't running          |
| `curl localhost:LOCAL` hangs                     | wrong host       | Re-paste the ssh -L line verbatim   |
| `[ofa-serve]` says it can't bind                 | REMOTE (Kestrel) | Pick a new `--serve-port`           |
| `401 Unauthorized`                               | (auth, not port) | Update apiKey in BYOK config        |

### Specific errors

**`bind: Address already in use` on the LAPTOP port** — something on
your Mac/PC is listening on that port. Most often it's VS Code
Remote-SSH's auto-port-forward (see "VS Code auto-forward" below).
Easiest fix: pick a different LOCAL port on Kestrel and re-run:

```bash
ofa --serve --serve-local-port 50321   # or any free number
```

Then update the BYOK URL in VS Code to match. The chosen port is
persisted so you only do this once.

**`Connection refused` on the LAPTOP port** — your `ssh -L` isn't
running. Re-paste the line `ofa --serve` printed. Confirm with
`lsof -nP -iTCP:<LOCAL> -sTCP:LISTEN` — you should see an `ssh`
process holding it.

**`curl http://localhost:LOCAL/healthz` hangs** — the TCP connection
reached *something* on your laptop but it's not talking to
`ofa --serve`. Almost always means the `ssh -L` `<compute-node>`
piece is wrong: you typed the login node name instead of the
allocation's compute node. Re-paste the line `ofa --serve` printed —
it has the compute-node hostname (e.g. `x3101c0s9b0n0`) filled in
already.

**`[ofa-serve]` fails to bind on Kestrel** — something else on the
compute node has the REMOTE port. With the default `--serve-port 0`
this should never happen (OS picks free). If you pinned a port,
pick another.

**`401 Unauthorized`** — your `chatLanguageModels.json` `apiKey`
doesn't match `$OFA_SCRATCH/.ofa_api_key` on Kestrel. Re-read the
file and paste it again.

**Chat hangs on first reply** — first response loads the model into
GPU memory; subsequent replies are fast. Check `[ofa-serve]` log
lines on the Kestrel side for activity.

**Wrong system prompt** — pick a different `OFA · *` model from the
dropdown; each one selects a different mode + RAG retriever.

### VS Code auto-forward (the most common source of port collisions)

When you connect VS Code via Remote-SSH, the **VS Code Server**
running on the remote machine watches the remote kernel's list of
listening sockets and automatically forwards each one to the *same*
port number on your laptop. Setting:

> `remote.autoForwardPorts` (default: `true`)
> `remote.autoForwardPortsSource` (default: `"process"`)

This is designed for web-dev workflows (`npm run dev` → auto-open
`localhost:3000`) but it actively fights our BYOK setup:

- It races your manual `ssh -L` for the same port number.
- It can grab the LOCAL port you picked before your `ssh` does.
- It can also grab the REMOTE port number (Ollama's 11434, ofa's
  REMOTE port) and surface it on your laptop, which is sometimes
  helpful and sometimes confusing.

**Two clean fixes:**

1. **Turn auto-forward off.** In VS Code settings (`Cmd+,`) search
   for `remote.autoForwardPorts` and set it to `false`. The Ports
   panel still lets you forward manually when you want it. This is
   the recommended choice for HPC / BYOK use; you're not running
   local web dev on Kestrel.

2. **Let auto-forward be your tunnel.** Skip the manual `ssh -L`
   entirely and use the port VS Code auto-forwarded. The BYOK URL
   becomes `http://localhost:REMOTE/...` (same number as the REMOTE
   port — VS Code preserves it on the laptop side when free). To
   make this stable across restarts, pin the REMOTE port:
   ```bash
   ofa --serve --serve-port 51234   # any free port
   ```
   Then your BYOK URL is permanently `http://localhost:51234/...`
   and you don't need a separate `ssh -L` terminal at all.

### Misc

**`No matching distribution found`** — `ofa --serve` uses only Python
stdlib for the HTTP layer; no new dependencies are required beyond
those `ofa` already needs.
