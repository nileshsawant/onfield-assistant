#!/usr/bin/env python3
"""Register all eight OFA modes in your VS Code BYOK config.

Run this on your **laptop** (Mac/Linux), once. Safe to re-run; only edits
the "OFA (Kestrel)" provider entry and leaves Copilot or other BYOK
providers alone. Makes a `.bak` copy of the file the first time.

Usage:
    python3 byok-update-config.py \\
        --token  ofa-xxxxxxxxxxxxxxxxxx \\
        [--port  49643] \\
        [--name  "OFA (Kestrel)"]

Get the token from Kestrel:
    cat "$OFA_SCRATCH/.ofa_api_key"

After running the script:
  1. Reload VS Code window: Cmd+Shift+P -> "Developer: Reload Window"
  2. Cmd+Shift+P -> "Chat: Manage Language Models" -> gear icon next
     to "OFA (Kestrel)" -> Update API Key -> paste the same token. The
     `apiKey` in the JSON is only a hint; the actual token used at
     request time lives in VS Code secret storage.
  3. In the same view, toggle each OFA model's eye icon to "visible".
  4. Pick "OFA . <mode>" from the Chat model dropdown. Use "Ask" mode.

This script uses only the Python stdlib; no extra dependencies needed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Order matters: this is what users will see in the Manage Models view.
MODES = [
    ("ofa-openfoam",          "OFA \u00b7 OpenFOAM"),
    ("ofa-hpc",               "OFA \u00b7 Kestrel HPC"),
    ("ofa-code",              "OFA \u00b7 Code"),
    ("ofa-amrex",             "OFA \u00b7 AMReX"),
    ("ofa-marbles",           "OFA \u00b7 MARBLES (LBM)"),
    ("ofa-reframe",           "OFA \u00b7 ReFrame (RHEL9)"),
    ("ofa-quantum-computing", "OFA \u00b7 Quantum Computing"),
    ("ofa-vasp",              "OFA \u00b7 VASP"),
]


def find_vscode_config_path() -> Path:
    """Locate chatLanguageModels.json on the current OS.

    macOS:   ~/Library/Application Support/Code/User/chatLanguageModels.json
    Linux:   ~/.config/Code/User/chatLanguageModels.json
    Windows: %APPDATA%/Code/User/chatLanguageModels.json

    Returns the first candidate whose parent directory exists.
    """
    home = Path.home()
    candidates = [
        home / "Library" / "Application Support" / "Code" / "User" / "chatLanguageModels.json",
        home / ".config" / "Code" / "User" / "chatLanguageModels.json",
        Path(os.environ.get("APPDATA", str(home))) / "Code" / "User" / "chatLanguageModels.json",
    ]
    for c in candidates:
        if c.parent.exists():
            return c
    # Fall back to the macOS-style path; the user can override with --path.
    return candidates[0]


def build_provider(port: int, token: str, name: str) -> dict:
    """Build the OFA (Kestrel) provider entry with all eight OFA modes."""
    return {
        "name": name,
        "vendor": "customendpoint",
        "apiKey": token,
        "apiType": "chat-completions",
        "models": [
            {
                "id": mid,
                "name": mname,
                "url": f"http://localhost:{port}/v1/chat/completions",
                # MUST be true: the VS Code Chat model picker silently hides
                # models that declare toolCalling=false. The ofa server
                # ignores tool_calls at the protocol level, so this is a
                # cosmetic concession.
                "toolCalling": True,
                "maxInputTokens": 32000,
                "maxOutputTokens": 8192,
            }
            for mid, mname in MODES
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--token", required=True,
        help="Bearer token from $OFA_SCRATCH/.ofa_api_key on Kestrel.",
    )
    ap.add_argument(
        "--port", type=int, default=49643, metavar="N",
        help="Laptop-side port (default 49643, matches the ofa-vscode "
             "extension's laptopSideBridgePort setting). If you launched "
             "the bridge with a different --serve-local-port, override here.",
    )
    ap.add_argument(
        "--name", default="OFA (Kestrel)",
        help='Provider name shown in VS Code (default: "OFA (Kestrel)"). '
             "Existing providers with this name will be replaced.",
    )
    ap.add_argument(
        "--path", help="Override chatLanguageModels.json path (default: auto-detect).",
    )
    args = ap.parse_args()

    path = Path(args.path) if args.path else find_vscode_config_path()
    print(f"Target file: {path}")

    # Load existing config (or start with empty list).
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: {path} is not valid JSON: {e}", file=sys.stderr)
            print("Open it in VS Code and fix the syntax, then re-run.", file=sys.stderr)
            return 2
        if not isinstance(data, list):
            print(
                f"Error: expected a JSON array at the top level, got {type(data).__name__}",
                file=sys.stderr,
            )
            return 2
        # Make a backup the first time.
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)
            print(f"Backed up original to: {backup}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = []
        print("(Creating new config file.)")

    # Remove any existing OFA provider so we can replace it cleanly.
    before = len(data)
    data = [p for p in data if p.get("name") != args.name]
    removed = before - len(data)
    if removed:
        print(f"Removing existing {args.name!r} provider ({removed} entry).")

    # Append the fresh entry.
    data.append(build_provider(args.port, args.token, args.name))

    # Write.
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(
        f"OK: registered {args.name!r} with {len(MODES)} models "
        f"at http://localhost:{args.port}/v1/chat/completions"
    )
    print()
    print("Next steps in VS Code:")
    print("  1. Cmd+Shift+P -> 'Developer: Reload Window'")
    print("  2. Cmd+Shift+P -> 'Chat: Manage Language Models'")
    print("     - Click the gear next to 'OFA (Kestrel)' -> 'Update API Key'")
    print("       and paste the SAME token. The JSON apiKey is only a hint;")
    print("       the value used at request time lives in secret storage.")
    print("     - Click the eye icon on each OFA row to make it visible.")
    print("  3. In Chat, switch to 'Ask' mode (top of Chat panel).")
    print("  4. Open the model dropdown and pick 'OFA \u00b7 <mode>'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
