# OnField Assistant — VS Code extension

One-click SLURM allocation + `ofa --serve` bring-up for VS Code Chat on
Kestrel (and eventually other HPC systems). Registers ofa's eight modes
as a `LanguageModelChatProvider` so they appear in VS Code Chat's model
picker directly — no manual `chatLanguageModels.json` editing, and no
API key to copy (the extension reads the token from `ofa --serve`'s
startup banner and sends it itself).

> **Status:** v0.1-alpha, functional. Connect/Disconnect/Re-allocate,
> the login-node `ncat` bridge, SLURM-expiry reconnect, and OpenAI
> `tool_calls` passthrough (so Agent mode can apply edits) all work.
> Not yet exercised much in production: tool-call reliability with a
> local 31B model against VS Code's agent tool schemas.

## Requirements

- VS Code 1.95 or newer.
- Node.js 20+ (for building the extension only — the packaged `.vsix`
  runs against the Node runtime VS Code ships).
- An active Remote-SSH connection to a Kestrel login node. Driving a
  Kestrel-side `ofa` from a laptop-local VS Code is not supported by
  the extension; use [`docs/byok-vscode.md`](../docs/byok-vscode.md)
  for that case.

## Dev loop

```bash
cd vscode-ext
npm install
npm run build         # bundles src/extension.ts -> dist/extension.js
# Or continuously:
npm run watch
```

Then in VS Code:

1. Open the `vscode-ext/` folder as a workspace.
2. Press `F5` to launch an Extension Development Host window.
3. In that window, `Cmd+Shift+P` → `OFA: Connect` (etc.) to exercise
   the commands. Logs are visible via `OFA: Show Logs`.

## Build the `.vsix`

```bash
npm run package       # produces ./ofa-vscode.vsix
```

CI (`.github/workflows/vscode-ext.yml`) runs this same command on any
push of a tag matching `vscode-ext-v*` and attaches the `.vsix` to a
GitHub Release. Install with:

```bash
code --install-extension ofa-vscode.vsix
```

## Configuration

All settings live under `ofa.*` in VS Code settings. Highlights:

| Setting | Default | Purpose |
| --- | --- | --- |
| `ofa.slurm.partition` | `debug` | Partition for the salloc request. |
| `ofa.slurm.walltime` | `00:30:00` | Wall time for the salloc request. |
| `ofa.slurm.gres` | `gpu:1` | Slurm GRES (adjust on typed-GPU sites). |
| `ofa.slurm.account` | `""` | Empty = auto-detect via `sacctmgr`. |
| `ofa.model` | `gemma4:31b-it-q8_0` | Backing LLM. Also determines the advertised token limits. |
| `ofa.enableTools` | `true` | Pass `--serve-enable-tools` to ofa. |
| `ofa.autoConnectOnStartup` | `false` | Auto-run Connect on workspace open. |
| `ofa.silentReconnect` | `true` | Reallocate silently on SLURM expiry. |

See `package.json`'s `contributes.configuration` block for the full list.

`ofa.model` is read by `ofa --serve` at spawn, so changing it needs
**OFA: Disconnect** → **OFA: Connect** to take effect. It also feeds
`getModelLimits()` in [`src/modelProvider.ts`](src/modelProvider.ts),
which derives `maxInputTokens` / `maxOutputTokens` from that model's
context window instead of hardcoding them — keep `MODEL_CONTEXT` there
in sync with `MODEL_REGISTRY` in `src/ofa_main.py`.

## Related docs

- [Parent repo README](../README.md) — installing and running `ofa`,
  plus the end-user walkthrough for this extension and how it handles
  keys and ports.
- [`docs/byok-vscode.md`](../docs/byok-vscode.md) — the manual
  ssh-tunnel + `chatLanguageModels.json` flow this extension supersedes.
  Still relevant for non-VS-Code clients. Note that running both leaves
  two redundant groups in the model picker.
- [`ARCHITECTURE.md`](../ARCHITECTURE.md) — how `ofa --serve` fits into
  the overall system.
