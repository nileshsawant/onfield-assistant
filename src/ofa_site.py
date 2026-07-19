"""Site-configuration loader for ofa.

Historically ofa hard-coded a handful of Kestrel-specific strings across
``bin/ofa``, ``ofa_main.py`` and ``ofa_server.py`` (partition names, GRES
strings, protected filesystem roots, the ``ssh -L`` login host in the BYOK
hint, the banner subtitle, and so on). This module extracts those into a
single optional ``$OFA_ROOT/site.toml`` file so ofa can be installed on a
non-Kestrel HPC by editing one place.

Design goals:

* **Zero behavior change when ``site.toml`` is absent.** The ``DEFAULTS``
  dict below mirrors the Kestrel values that used to be inline, so an
  unmodified Kestrel checkout keeps operating identically whether or not
  a ``site.toml`` exists.
* **Additive override.** Values in ``site.toml`` are deep-merged over
  ``DEFAULTS``; missing keys keep their default. Users at Kestrel do not
  need to write a ``site.toml``.
* **Bash-consumable.** The shell launcher can ``eval`` the output of
  ``python3 -m ofa_site --shell-export`` to pick up the scheduler /
  module settings without parsing TOML in bash.
* **No dependency on the rest of ofa.** Import order sanity: this
  module is imported from ``ofa_main`` and ``ofa_server`` at their top,
  so it must not import from either.
"""
from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Defaults (Kestrel values). Do NOT edit these to reflect a new site — the
# whole point of ``site.toml`` is to override without touching this file.
# Sites that never write a ``site.toml`` on top of a stock checkout will
# still see the Kestrel-specific text, which is the correct fallback for
# our current single-site install.
# ---------------------------------------------------------------------------
DEFAULTS: dict[str, dict[str, Any]] = {
    "site": {
        # Short site name printed in the interactive banner / prompts.
        "name": "Kestrel",
        # Sponsoring organization / lab. Consumed by the {SITE_ORG}
        # placeholder in prompts (e.g. "an NLR HPC Support Assistant").
        "org": "NLR",
        # Full noun phrase for the site, consumed by the {SITE_LONG_NAME}
        # placeholder. Written out longhand so porters can adjust grammar
        # (some sites are called "clusters", others "supercomputers",
        # others just their name). Defaults to the pre-refactor Kestrel
        # phrasing so the prompt output byte-matches the earlier build.
        "long_name": "NLR Kestrel HPC supercomputer",
        # Trailing GPU descriptor in the banner ("locally hosted on
        # <name> · <description>").
        "description": "single H100",
        # Login host used in the ``ssh -L`` hint emitted by ``ofa --serve``.
        "login_host": "kestrel.hpc.nlr.gov",
        # Filesystem prefixes ofa refuses to touch via rm/chmod/chown-style
        # mass operations. Merged additively with the universal system
        # paths ("/bin", "/etc", …) in ``ofa_main.PROTECTED_PREFIXES``.
        "protected_roots": [
            "/nopt/nrel", "/nopt/nlr", "/nopt/slurm", "/nopt/sgi",
        ],
    },
    "scheduler": {
        # One of: "slurm" | "pbs" | "lsf" | "none". Only "slurm" is
        # wired into ``bin/ofa`` today; the others are reserved for
        # future adapter work.
        "kind": "slurm",
        "partition": "debug",
        "gres": "gpu:1",
        "mem": "80G",
        "ntasks_per_node": 32,
        "walltime": "00:30:00",
        # Bash command that must print the user's default account as a
        # single line on stdout. Executed via ``bash -c`` from the
        # launcher when ``$OFA_ACCOUNT`` is not already set.
        "account_discovery":
            'sacctmgr show user "$USER" format=defaultaccount -nP 2>/dev/null | head -1',
    },
    "modules": {
        # Module names loaded after allocation on each RHEL major. Set
        # to an empty string to skip. The launcher picks RHEL9 on hosts
        # whose /etc/redhat-release contains "release 9".
        "cuda_rhel8": "cuda/12.4",
        "cuda_rhel9": "cuda",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict = base with override's keys deep-merged on top.

    Only dict-valued keys recurse; scalar / list keys are replaced
    wholesale (a site.toml that supplies ``protected_roots = ["/foo"]``
    fully replaces the default list rather than appending to it — this is
    the least-surprising choice for a config-file layer).
    """
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_site() -> dict[str, Any]:
    """Return the effective site config as a nested dict.

    Search order:
      1. ``$OFA_SITE_TOML`` if set and points to a readable file.
      2. ``$OFA_ROOT/site.toml`` if that file exists.
      3. Defaults only.

    Any TOML parse error is swallowed and the defaults are returned; a
    broken ``site.toml`` must never disable ofa's safety guards.
    """
    path: str | None = os.environ.get("OFA_SITE_TOML") or None
    if not path:
        root = os.environ.get("OFA_ROOT")
        if root:
            candidate = Path(root) / "site.toml"
            if candidate.is_file():
                path = str(candidate)
    if not path:
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    try:
        with open(path, "rb") as f:
            override = tomllib.load(f)
    except Exception as exc:  # noqa: BLE001 — deliberately broad; see docstring
        print(
            f"[ofa-site] WARNING: failed to parse {path}: {exc}. "
            "Falling back to built-in defaults.",
            file=sys.stderr,
        )
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    return _deep_merge(DEFAULTS, override)


def _sq(value: Any) -> str:
    """Single-quote a value for safe inclusion in a bash ``eval``.

    Embedded single quotes are broken out with the classic
    ``'"'"'`` idiom so partition names, discovery commands, etc. with
    special characters survive intact.
    """
    s = str(value)
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _shell_export(cfg: dict[str, Any]) -> str:
    """Render the site config as ``export KEY=VALUE`` lines for bash.

    Only the fields the launcher actually consumes are exported. Names
    are prefixed ``OFA_SITE_`` / ``OFA_SCHEDULER_`` / ``OFA_MODULE_`` and
    never collide with the user-facing overrides (``OFA_ACCOUNT``,
    ``OFA_PARTITION``, ``OFA_WALLTIME``) so the launcher can keep
    ``${OFA_PARTITION:-${OFA_SCHEDULER_PARTITION_DEFAULT:-debug}}``
    precedence.
    """
    site = cfg.get("site", {}) or {}
    sch = cfg.get("scheduler", {}) or {}
    mods = cfg.get("modules", {}) or {}
    lines = [
        f"export OFA_SITE_NAME={_sq(site.get('name', ''))}",
        f"export OFA_SITE_LOGIN_HOST={_sq(site.get('login_host', ''))}",
        f"export OFA_SCHEDULER_KIND={_sq(sch.get('kind', 'slurm'))}",
        f"export OFA_SCHEDULER_PARTITION_DEFAULT={_sq(sch.get('partition', ''))}",
        f"export OFA_SCHEDULER_GRES={_sq(sch.get('gres', ''))}",
        f"export OFA_SCHEDULER_MEM={_sq(sch.get('mem', ''))}",
        f"export OFA_SCHEDULER_NTASKS_PER_NODE={_sq(sch.get('ntasks_per_node', ''))}",
        f"export OFA_SCHEDULER_WALLTIME_DEFAULT={_sq(sch.get('walltime', ''))}",
        f"export OFA_SCHEDULER_ACCOUNT_DISCOVERY={_sq(sch.get('account_discovery', ''))}",
        f"export OFA_MODULE_CUDA_RHEL8={_sq(mods.get('cuda_rhel8', ''))}",
        f"export OFA_MODULE_CUDA_RHEL9={_sq(mods.get('cuda_rhel9', ''))}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--shell-export":
        sys.stdout.write(_shell_export(load_site()))
    else:
        print("usage: python3 -m ofa_site --shell-export", file=sys.stderr)
        sys.exit(2)
