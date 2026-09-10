# Maintaining `ofa` with `ofa`

This is a playbook for improving `ofa` **using `ofa` itself** — no paid API,
no external assistant required. `ofa` is a capable coding agent running on
your own hardware; it can read, grep, edit, and run its own source. The
gap versus a frontier assistant is not capability of tools but discipline:
work in small verified steps and the local model is reliable.

Read this before making changes. The rules here are distilled from real
mistakes and near-misses.

---

## 1. Launch `ofa` on its own repo

```bash
# Get a GPU node (adjust account/time as needed).
salloc -A hpcapps -t 02:00:00 --nodes=1 --ntasks-per-node=32 --mem=80G --gres=gpu:1

cd /nopt/nrel/apps/cpu_stack/software/openfoam/assistant

# Coding mode, pointed at this repo. ofa can now read/grep/edit/run here.
./bin/ofa --code
```

Then converse with it the way you would any coding assistant:

> "Read `src/ofa_main.py` around `retrieve_amrex_context`. I want to add a
> per-mode flag that disables reranking for `of13_src`. Show me the edit."

It will `read` the file, propose an `edit` block, and — once you approve —
apply it. That is self-improvement: the assistant editing its own codebase.

---

## 2. Work in small, verified steps (the key habit)

The local model does **focused** edits well and **sprawling** ones badly.
Make that work *for* you:

- **One change at a time.** "Add this one function", not "refactor this module".
- **Give it the file first.** "Read X, then change Y" — never let it guess
  at code it hasn't seen. Its RAG + `read` tool are what keep it grounded.
- **Verify after every edit** (see §4). Catch mistakes immediately, not
  three edits later.
- **If it goes wrong, revert and retry smaller** rather than piling
  correction on correction.

---

## 3. Git hygiene (non-negotiable)

These rules exist because a past `git add -A` swept unrelated edits into a
commit and a later reset silently destroyed a user's work.

- **NEVER `git add -A`, `git add .`, or `git commit -a`.** Always stage
  specific paths: `git add src/ofa_main.py collections.toml`.
- **Inspect before committing:** `git --no-pager diff --cached --stat` and
  read the actual staged diff. Confirm ONLY intended files/lines are in.
- **Protect unrelated dirty files** with `git stash push <file>` before any
  `git reset --hard`, `git checkout -- <file>`, or other destructive op.
- **Before dropping any commit** (revert/reset/rebase), check what it
  actually contains: `git show <sha> --stat`.

---

## 4. Verify every change

Fast checks, run after each edit, before committing:

```bash
# Python: syntax/compile check (catches the most common model mistakes).
env/bin/python -m py_compile src/ofa_main.py

# Shell scripts:
bash -n install.sh
bash -n tools/update-rag.sh

# TOML config:
env/bin/python -c "import tomllib; tomllib.load(open('collections.toml','rb')); print('TOML OK')"
```

For behaviour changes, exercise the actual code path with a tiny script
(import `ofa_main`, call the function, print the result) rather than
trusting that it "looks right".

---

## 5. This is a LIVE, SHARED install

Real users run `ofa` from this exact directory. Therefore:

- **Prefer changes that default to existing behaviour.** New features should
  be opt-in (env var, default off) until validated — e.g. `OFA_SUBAGENT`,
  `OFA_RERANK` are off by default.
- **Measure before adopting.** Benchmarks can mislead (a "68 tok/s" LLM was
  useless once cold-load + prompt-eval were counted). Measure END-TO-END.
- **Do NOT run experiments against the live `ofa --serve` process** — start
  an isolated throwaway Ollama on its own port for tests, and kill it after.
- **Rebuilds run on an allocation, not the login node:**
  `srun --jobid=<J> --overlap -n1 env/bin/python src/rebuild_indices.py ...`

---

## 6. Where things live (quick map)

| Area | File(s) |
|---|---|
| Core CLI / agent loop / retrieval | `src/ofa_main.py` |
| BYOK HTTP server (VS Code path) | `src/ofa_server.py` |
| RAG collection config | `collections.toml` |
| RAG (re)build pipeline | `src/rebuild_indices.py` |
| OpenFOAM tutorial/source indexer | `src/build_index_v2.py` |
| System prompts per mode | `prompts/*.txt` |
| Recurring RAG-update job | `tools/update-rag.sh`, `tools/update-rag.sbatch` |
| Install / bootstrap | `install.sh` |
| Site-specific config | `site.toml` (from `site.example.toml`) |
| Architecture notes | `ARCHITECTURE.md`, `docs/` |

---

## 7. Rebuilding RAG collections

```bash
# List configured collections.
env/bin/python src/rebuild_indices.py --list

# Rebuild one (mtime-cached; only changed files re-embed).
srun --jobid=<J> --overlap -n1 env/bin/python src/rebuild_indices.py --collection <name>

# When migrating a collection from a different indexer (stale/dup chunks),
# clear it first:
srun --jobid=<J> --overlap -n1 env/bin/python src/rebuild_indices.py --clear --collection <name>
```

`collections.toml` supports `exclude` globs per source (to skip symlink
farms like OpenFOAM `lnInclude/` or vendored apps). Retrieval quality
depends on the SOURCE being the right layer — prefer tutorials/examples
over library internals when the goal is generating user code.

---

## 8. `ofa`'s own memory

`ofa` persists two kinds of note across sessions (see `prompts/common.txt`):

- `=== PREFS ===` — standing user preferences ("always use tabs").
- `=== LESSON ===` — the model's own notes-to-self after a corrected
  mistake or a discovered environment quirk.

These live under `$OFA_SCRATCH`. Encourage `ofa` to record a lesson when it
learns something non-obvious — that is how it gets better over time on your
workflows without any external help.

---

## 9. When the local model isn't enough

For genuinely hard reasoning the open-weight model on one GPU has a ceiling.
`ofa` supports **BYOK**: point VS Code Chat / a client at a frontier API for
the hard case, then drop back to the free local model for everyday work. You
are not locked into either. See `docs/byok-vscode.md`.
