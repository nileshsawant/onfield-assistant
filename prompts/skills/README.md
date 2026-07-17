# `ofa` skill files

A **skill** is a Markdown file in this directory that the user can inject
into the active `ofa` session via the `/skill <name>` slash command.

## Why

The base system prompts (`prompts/openfoam.txt`, `hpc.txt`, `code.txt`,
`amrex.txt`, `reframe.txt`) are loaded automatically based on the mode flag
(`--hpc`, `--code`, `--amrex`, `--reframe`, or the default OpenFOAM mode).
That's where general behavior lives.

Skills are for **narrower, situational expertise** that doesn't belong in
the always-on system prompt:

- Procedures the user only sometimes needs (e.g. "how we deploy ParaView
  servers on Kestrel compute nodes")
- Project-specific conventions (e.g. "in this codebase, all new C++ files
  must include `Marbles_Config.H`")
- Long checklists (e.g. SLURM debug-job checklist) that would bloat the
  primary system prompt

## Format

- File name: `<skill-name>.md` — the stem (without `.md`) is what the user
  types after `/skill`. Keep the name kebab-case and short.
- File content: free-form Markdown. The whole file (minus the heading line
  used for the `/skills` listing) is injected verbatim as a system-role
  message tagged `[SKILL: <name>]`.
- First non-blank, non-heading line is treated as a one-line summary and
  shown by `/skills` — keep it under 80 characters.

## Lifecycle

- Loaded on demand by `/skill <name>` — never auto-loaded.
- Lives in the session only. `/clear` and exiting drop loaded skills.
- Use `/skill off <name>` or `/skill off all` to unload mid-session.
- Inserted right after the base system prompt so the base rules still
  win on conflict.

## Example template

```markdown
# Kestrel SLURM debug-queue tips

Use these tips when the user is preparing or debugging short SLURM jobs
on NLR Kestrel's debug partition.

- The debug partition wall clock cap is 1 hour. Never write `--time=24:00:00`
  to a debug-partition submission.
- Always include `--account=<project>` — debug jobs fail silently without
  it on Kestrel.
- ...
```

## Adding skills

Just drop a new `<name>.md` file in this directory. No restart needed —
`/skills` re-reads the directory each time.

## Security note

Skill names are taken literally from the slash command, but the loader
refuses path traversal (`..`, leading dots, slashes) so users can only
load files that actually exist in this directory.
