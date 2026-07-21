# Use `ofa` from VS Code (BYOK)

`ofa --serve` exposes an OpenAI-compatible HTTP server. VS Code's
Bring-Your-Own-Key (BYOK) feature can route Chat requests through it,
so VS Code Chat gets `ofa`'s mode-specific system prompts, RAG over
Kestrel/OpenFOAM/AMReX/ReFrame/VASP docs, long-term memory, and
loaded skills.

You'll see eight models in the VS Code Chat picker, one per `ofa`
mode (OpenFOAM, HPC, Code, AMReX, MARBLES, ReFrame, Quantum
Computing, VASP).

> **Heads-up on the apiKey field.** The `apiKey` string in
> `chatLanguageModels.json` is treated as a *hint* by Copilot Chat;
> the real key lives in VS Code's per-provider secret storage. If
> you only paste the JSON, the first request will fail with
> `missing or invalid Authorization header` and the server log will
> show `Authorization=Bearer` (empty). You **must** also paste the
> token via the gear icon → **Update API Key** in **Chat: Manage
> Language Models** (details in step 4).

## Quick start

### 1. On Kestrel: start the server

```bash
ml assistant
ofa --serve --serve-enable-tools
```

`ofa --serve` allocates a quarter GPU node for you, prints a labelled
connection block, and blocks. Three things from that block matter —
**copy them**:

- The full `ssh -N -L ...` command (compute-node hostname + both ports
  already filled in).
- The BYOK URL: `http://localhost:<LOCAL>/v1/chat/completions`.
- The bearer token (also at `$OFA_SCRATCH/.ofa_api_key`).

All three are stable across `--serve` restarts: the port file
(`$OFA_SCRATCH/.ofa_serve_local_port`) and key file
(`$OFA_SCRATCH/.ofa_api_key`) are persisted per-user, so once you wire
VS Code up you don't have to update it again.

### 2. On your laptop: open the ssh tunnel

Paste the `ssh -N -L ...` line into a new laptop terminal and leave
it running.

Sanity-check from another laptop terminal:

```bash
curl http://localhost:<LOCAL>/healthz   # expect {"status":"ok"}
```

### 3. On your laptop: register the eight OFA models in VS Code

Easiest path uses the helper script (registers all eight modes,
makes a `.bak` of your existing config the first time, safe to
re-run):

```bash
# Copy the script down once
scp kestrel.hpc.nlr.gov:/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/tools/byok-update-config.py ~/

# Run it (replace the token + port with what step 1 printed)
python3 ~/byok-update-config.py \
    --token ofa-xxxxxxxxxxxxxxxxxxxxxxxx \
    --port  <LOCAL>
```

Or paste the JSON below into `chatLanguageModels.json` yourself
(`Cmd+Shift+P → "Chat: Manage Language Models" → Add Models →
customendpoint`). Replace `<LOCAL_PORT>` and the apiKey:

```json
[
  {
    "name": "OFA (Kestrel)",
    "vendor": "customendpoint",
    "apiKey": "ofa-PASTE_TOKEN_FROM_KEYFILE",
    "apiType": "chat-completions",
    "models": [
      { "id": "ofa-openfoam", "name": "OFA · OpenFOAM",
        "url": "http://localhost:<LOCAL_PORT>/v1/chat/completions",
        "toolCalling": true, "maxInputTokens": 32000, "maxOutputTokens": 8192 },
      { "id": "ofa-hpc", "name": "OFA · Kestrel HPC",
        "url": "http://localhost:<LOCAL_PORT>/v1/chat/completions",
        "toolCalling": true, "maxInputTokens": 32000, "maxOutputTokens": 8192 },
      { "id": "ofa-code", "name": "OFA · Code",
        "url": "http://localhost:<LOCAL_PORT>/v1/chat/completions",
        "toolCalling": true, "maxInputTokens": 32000, "maxOutputTokens": 8192 },
      { "id": "ofa-amrex", "name": "OFA · AMReX",
        "url": "http://localhost:<LOCAL_PORT>/v1/chat/completions",
        "toolCalling": true, "maxInputTokens": 32000, "maxOutputTokens": 8192 },
      { "id": "ofa-marbles", "name": "OFA · MARBLES (LBM)",
        "url": "http://localhost:<LOCAL_PORT>/v1/chat/completions",
        "toolCalling": true, "maxInputTokens": 32000, "maxOutputTokens": 8192 },
      { "id": "ofa-reframe", "name": "OFA · ReFrame (RHEL9)",
        "url": "http://localhost:<LOCAL_PORT>/v1/chat/completions",
        "toolCalling": true, "maxInputTokens": 32000, "maxOutputTokens": 8192 },
      { "id": "ofa-quantum-computing", "name": "OFA · Quantum Computing",
        "url": "http://localhost:<LOCAL_PORT>/v1/chat/completions",
        "toolCalling": true, "maxInputTokens": 32000, "maxOutputTokens": 8192 }
    ]
  }
]
```

A drop-in template is also at
`$OFA_ROOT/docs/byok-vscode-chatLanguageModels.example.json`.

`toolCalling: true` is required even when you don't intend to use Agent
mode — the picker silently hides models that declare `false`.

### 4. Activate the models in VS Code (this is where the apiKey actually goes)

`Cmd+Shift+P → "Developer: Reload Window"`. Then
`Cmd+Shift+P → "Chat: Manage Language Models"`:

1. Find the **OFA (Kestrel)** provider row. Click the small **gear
   icon** next to it → **Update API Key**. Paste the bearer token
   from step 1. This is what actually gets sent as
   `Authorization: Bearer <token>` — the `apiKey` field in the JSON
   is only a hint and Copilot Chat ignores it at request time.
2. Click the **eye icon** on each `OFA · …` row to make it visible
   in the picker.

The token is persisted in your laptop's Keychain / secret storage
and survives reloads and future SLURM allocations, because the
server reuses the same key file (`$OFA_SCRATCH/.ofa_api_key`) each
time.

### 5. Chat

Open Chat. Type "OFA" in the model picker search box to find the
eight modes. Pick the one matching your task and chat as normal.

- **Ask mode**: the safe, recommended starting point. Pure
  RAG-augmented chat.
- **Agent mode**: only useful with `--serve-enable-tools`. VS Code can
  then chain file edits / terminal commands through one approval gate
  instead of click-per-block. See the next section.

## Tool calling (Agent mode)

When you pass `--serve-enable-tools` to `ofa --serve`, requests in
VS Code's Agent mode get OpenAI-format `tools` forwarded to Ollama,
and Ollama's `tool_calls` responses are translated back to OpenAI SSE
format. VS Code can then execute proposed actions through its standard
approval flow.

Caveats:

- Local 31B Gemma can emit malformed JSON or hallucinate tool names
  against VS Code's rich agent tool schemas (file_edit, terminal,
  codebase_search, …). When that happens, drop back to **Ask mode**
  or stop the server with the flag off.
- For the full agent loop (the carefully-tuned tool-fence convention
  with retries and nudges), use the **`ofa` CLI** on Kestrel directly
  (`ofa --code`, `ofa --hpc`, etc.). The BYOK path is for chat surface
  + lightweight Agent-mode chaining.

## Troubleshooting

### Quick orientation: which port is failing?

Two ports are in play and they live on different machines.

| Port    | Lives on        | Set by                |
|---------|-----------------|-----------------------|
| REMOTE  | Kestrel compute | `--serve-port`        |
| LOCAL   | Your laptop     | `--serve-local-port`  |

Both default to per-user random-but-persisted values — `ofa --serve`
prints the actual numbers. The login node binds nothing; it just
forwards bytes between the two.

| Symptom                                                | Probable fix                                              |
|--------------------------------------------------------|-----------------------------------------------------------|
| `ssh -L` says `bind: Address already in use`           | Run `ofa --serve --serve-local-port <N>` to pick a new LOCAL port |
| `curl localhost:LOCAL` returns "Connection refused"    | The `ssh -L` tunnel isn't running                          |
| `curl localhost:LOCAL` hangs                           | Re-paste the `ssh -L` line from the banner; wrong host    |
| `missing or invalid Authorization header` in VS Code, server log shows `Authorization=Bearer` (empty) | You pasted the JSON but never used the **gear → Update API Key** flow. Do that now — see step 4. |
| `401 Unauthorized` in VS Code                          | Apikey out of date in Keychain — re-enter via gear → Update API Key in Manage Language Models |
| Model not in Chat picker                               | Toggle the eye in Manage Language Models; type "OFA" to search the picker |
| Tool calls malformed in Agent mode                     | Drop back to Ask mode or restart without `--serve-enable-tools` |

### VS Code Remote-SSH port auto-forward (the common port-collision cause)

When VS Code Remote-SSH is attached to Kestrel, its server-side daemon
auto-forwards every listening socket it detects on the remote. That
can grab the same port your manual `ssh -L` is trying to bind, causing
`Address already in use` on a port nothing obvious is using.

Two fixes:

1. **Disable auto-forward** for HPC sessions: VS Code settings →
   `remote.autoForwardPorts` → `false`. Reload window.
2. **Skip the manual `ssh -L`** and let VS Code's auto-forward be the
   tunnel: in that case the BYOK URL is `http://localhost:REMOTE/...`
   (same number as REMOTE). Pin `--serve-port` so the number stays
   stable.

### Server logs

`ofa --serve` logs each request (model id, mode, stream/blocking, tools
on/off). On a 401 it logs a redacted summary of the auth-like headers
the client sent, so you can see whether VS Code sent an empty `Bearer`
(needs apiKey re-entry in Manage Language Models) or something else.
