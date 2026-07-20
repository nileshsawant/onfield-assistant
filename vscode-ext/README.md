# OnField Assistant — VS Code extension

One-click SLURM allocation + `ofa --serve` bring-up for VS Code Chat on
Kestrel (and eventually other HPC systems). Registers ofa's seven modes
as a `LanguageModelChatProvider` so they appear in VS Code Chat's model
picker directly — no manual `chatLanguageModels.json` editing.

> **Status:** v0.1-alpha skeleton. Command palette entries and status
> bar item are wired up, but each command currently shows a
> *"not yet implemented"* notice. Real behaviour lands in follow-up
> commits (see the top of [`src/extension.ts`](src/extension.ts) for the
> planned PR sequence).

## Requirements

- VS Code 1.95 or newer.
- Node.js 20+ (for building the extension only — the packaged `.vsix`
  runs against the Node runtime VS Code ships).
- For **v0.1 case (a)**: an active Remote-SSH connection to a Kestrel
  login node. Case (b) (local VS Code, ofa on Kestrel via ssh from the
  laptop) ships in a later release.

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
| `ofa.enableTools` | `true` | Pass `--serve-enable-tools` to ofa. |
| `ofa.autoConnectOnStartup` | `false` | Auto-run Connect on workspace open. |
| `ofa.silentReconnect` | `true` | Reallocate silently on SLURM expiry. |

See `package.json`'s `contributes.configuration` block for the full list.

## Related docs

- [Parent repo README](../README.md) — installing and running `ofa`.
- [`docs/byok-vscode.md`](../docs/byok-vscode.md) — the manual
  ssh-tunnel + `chatLanguageModels.json` flow this extension replaces.
- [`ARCHITECTURE.md`](../ARCHITECTURE.md) — how `ofa --serve` fits into
  the overall system.
