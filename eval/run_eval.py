#!/usr/bin/env python3
"""ofa evaluation harness.

Asks every question in eval/questions.toml through the same path as
`ofa --serve` (mode system prompt + RAG + chat_stream) and scores the answers
with the automatic checks declared per question.

  # on a GPU node (same allocation ofa uses):
  env/bin/python3 eval/run_eval.py generate --label gemma
  env/bin/python3 eval/run_eval.py generate --label bolt --model nemotron-3.5-bolt:74b-a9b --repeats 3

  # anywhere; load an OpenFOAM module first to enable the foam_parse check:
  env/bin/python3 eval/run_eval.py score  $OFA_EVAL_DIR/gemma.jsonl
  env/bin/python3 eval/run_eval.py compare $OFA_EVAL_DIR/gemma.scored.jsonl $OFA_EVAL_DIR/bolt.scored.jsonl

Results go to $OFA_EVAL_DIR (default /scratch/$USER/ofa-eval). Generation runs
with OFA_SCRATCH=$OFA_EVAL_DIR/scratch so the user's own prefs, lessons and
private RAG never leak into the answers.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def eval_dir() -> Path:
    if os.environ.get("OFA_EVAL_DIR"):
        return Path(os.environ["OFA_EVAL_DIR"])
    scratch = Path(f"/scratch/{os.environ.get('USER', 'default')}")
    base = scratch if scratch.is_dir() else Path.home() / ".local" / "state" / "ofa"
    return base / "ofa-eval"


def load_questions(path: Path, ids=None, modes=None) -> list[dict]:
    with open(path, "rb") as f:
        qs = tomllib.load(f)["question"]
    if ids:
        qs = [q for q in qs if q["id"] in ids]
    if modes:
        qs = [q for q in qs if q["mode"] in modes]
    return qs


# ------------------------------------------------------------------ generate

def cmd_generate(args) -> None:
    out_dir = eval_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    # ofa_main reads both at import time.
    os.environ["OFA_SCRATCH"] = str(out_dir / "scratch")
    if args.model:
        os.environ["OFA_MODEL"] = args.model
    sys.path.insert(0, str(ROOT / "src"))
    import ofa_main
    import ofa_server
    ofa_server.ofa_main = ofa_main

    qs = load_questions(Path(args.questions), args.ids, args.modes)
    ofa_main.ensure_ollama_running()
    ofa_main._init_rag()

    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    meta = {"label": args.label, "model": ofa_main.MODEL, "backend": ofa_main.OFA_BACKEND,
            "options": ofa_main.get_model_options(), "git": git, "repeats": args.repeats,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"), "host": os.uname().nodename}
    (out_dir / f"{args.label}.meta.json").write_text(json.dumps(meta, indent=2))

    out = out_dir / f"{args.label}.jsonl"
    total = len(qs) * args.repeats
    with open(out, "w") as f:
        n = 0
        for rep in range(args.repeats):
            for q in qs:
                n += 1
                messages = [*q.get("history", []), {"role": "user", "content": q["prompt"]}]
                built = ofa_server._augment_messages(messages, q["mode"])
                t0 = time.time()
                answer, retries = "", 0
                # Same empty-turn retry budget as ofa --serve.
                for retries in range(ofa_server._EMPTY_TURN_RETRIES + 1):
                    answer = "".join(ofa_main.chat_stream(built))
                    if answer.strip():
                        break
                rec = {"id": q["id"], "mode": q["mode"], "repeat": rep, "model": ofa_main.MODEL,
                       "seconds": round(time.time() - t0, 1), "empty_retries": retries,
                       "user_message": built[-1]["content"], "answer": answer}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"[{n}/{total}] {q['id']} rep={rep} {rec['seconds']}s "
                      f"{len(answer)} chars", file=sys.stderr, flush=True)
    print(f"wrote {out}", file=sys.stderr)
    cmd_score(argparse.Namespace(results=str(out), questions=args.questions))


# --------------------------------------------------------------------- checks

THOUGHT_RE = re.compile(r"<(thought|think)>.*?(</\1>|\Z)", re.S)
FENCE_RE = re.compile(r"```([\w+#.-]*)[ \t]*([^\n]*)\n(.*?)```", re.S)
FILE_RE = re.compile(r"^=== FILE: (\S+) ===[ \t]*\n(.*?)(?=^=== END ===|^=== FILE:|\Z)", re.S | re.M)


def strip_thoughts(text: str) -> str:
    return THOUGHT_RE.sub("", text)


def file_blocks(text: str) -> dict[str, str]:
    """Files from "=== FILE: p ===" blocks and ofa's ```write p fence tool."""
    files = {}
    for path, body in FILE_RE.findall(text):
        inner = FENCE_RE.search(body)
        files[path] = inner.group(3) if inner else body
    for lang, header, body in FENCE_RE.findall(text):
        if lang == "write" and header.strip():
            files.setdefault(header.strip(), body)
    return files


def code_blocks(text: str, langs: set[str], suffixes: tuple = (), names: tuple = ()) -> list[str]:
    blocks = [body for lang, _h, body in FENCE_RE.findall(text) if lang.lower() in langs]
    blocks += [b for p, b in file_blocks(text).items() if p.endswith(suffixes) or Path(p).name in names]
    return blocks


def _run(cmd, **kw) -> tuple[bool, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60, **kw)
    return p.returncode == 0, (p.stderr or p.stdout).strip()[:300]


def check_bash(answer: str, _c) -> tuple[str, str]:
    scripts = code_blocks(answer, {"bash", "sh", "shell", "slurm", "sbatch"},
                          (".sh", ".sbatch", ".slurm"), ("Allrun", "Allclean"))
    if not scripts:
        return "fail", "no bash script found"
    for s in scripts:
        ok, err = _run(["bash", "-n"], input=s)
        if not ok:
            return "fail", err
    return "pass", f"{len(scripts)} script(s)"


def check_python(answer: str, _c) -> tuple[str, str]:
    blocks = code_blocks(answer, {"python", "py", "python3"}, (".py",))
    if not blocks:
        return "fail", "no python block found"
    for b in blocks:
        try:
            ast.parse(b)
        except SyntaxError as e:
            return "fail", f"line {e.lineno}: {e.msg}"
    return "pass", f"{len(blocks)} block(s)"


def check_cpp(answer: str, _c) -> tuple[str, str]:
    blocks = code_blocks(answer, {"cpp", "c++", "cxx", "cc", "c"}, (".cpp", ".cc", ".cxx", ".C"))
    if not blocks:
        return "fail", "no C/C++ block found"
    for b in blocks:
        compiler = shutil.which("mpicxx") if "mpi.h" in b else (os.environ.get("CXX") or shutil.which("g++"))
        if not compiler:
            return "skip", "no compiler (mpicxx needed for mpi.h)"
        ok, err = _run([compiler, "-std=c++17", "-fsyntax-only", "-x", "c++", "-"], input=b)
        if not ok:
            return "fail", err
    return "pass", f"{len(blocks)} block(s)"


def check_foam(answer: str, _c) -> tuple[str, str]:
    files = {p: b for p, b in file_blocks(answer).items()
             if p.split("/")[0] in ("system", "constant", "0")}
    if not files:
        files = {"system/" + m.group(1): b for b in code_blocks(answer, {"", "cpp", "c++", "foam", "openfoam"})
                 if (m := re.search(r"object\s+(\w+)\s*;", b))}
    if not files:
        return "fail", "no OpenFOAM dictionary found"
    if not shutil.which("foamDictionary"):
        return "skip", "foamDictionary not on PATH (load an openfoam module)"
    with tempfile.TemporaryDirectory() as case:
        for p, b in files.items():
            dst = Path(case, p)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(b)
        for p in files:
            ok, err = _run(["foamDictionary", str(Path(case, p))], cwd=case)
            if not ok:
                return "fail", f"{p}: {err}"
    return "pass", f"{len(files)} file(s)"


def run_check(answer: str, c: dict) -> tuple[str, str]:
    t = c["type"]
    if t in ("regex", "not_regex"):
        flags = re.M if c.get("case") else re.M | re.I
        found = re.search(c["pattern"], answer, flags) is not None
        return ("pass" if found == (t == "regex") else "fail"), ""
    if t == "files":
        missing = [p for p in c["required"] if p not in file_blocks(answer)]
        return ("fail", "missing " + ", ".join(missing)) if missing else ("pass", "")
    if t == "max_chars":
        return ("pass" if len(answer.strip()) <= c["limit"] else "fail"), f"{len(answer.strip())} chars"
    fn = {"bash_syntax": check_bash, "python_syntax": check_python,
          "cpp_syntax": check_cpp, "foam_parse": check_foam}.get(t)
    if fn is None:
        return "fail", f"unknown check type {t!r}"
    try:
        return fn(answer, c)
    except subprocess.TimeoutExpired:
        return "fail", "validator timed out"


def check_label(c: dict) -> str:
    return c.get("desc") or c["type"] + (f" {c['pattern']}" if "pattern" in c else "")


# ---------------------------------------------------------------------- score

def cmd_score(args) -> None:
    qmap = {q["id"]: q for q in load_questions(Path(args.questions))}
    src = Path(args.results)
    out = src.with_name(src.name.replace(".jsonl", ".scored.jsonl"))
    recs = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    with open(out, "w") as f:
        for r in recs:
            q = qmap.get(r["id"])
            answer = strip_thoughts(r["answer"])
            results = []
            for c in (q["checks"] if q else []):
                status, detail = run_check(answer, c)
                results.append({"check": check_label(c), "status": status, "detail": detail})
            graded = [x for x in results if x["status"] != "skip"]
            r["checks"] = results
            r["score"] = sum(x["status"] == "pass" for x in graded) / len(graded) if graded else None
            f.write(json.dumps(r) + "\n")
    print_summary(out)


def _load_scored(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def per_question(recs) -> dict[str, dict]:
    acc = defaultdict(list)
    for r in recs:
        acc[r["id"]].append(r)
    out = {}
    for qid, rs in acc.items():
        scores = [r["score"] for r in rs if r["score"] is not None]
        out[qid] = {"mode": rs[0]["mode"], "score": sum(scores) / len(scores) if scores else None,
                    "seconds": sum(r["seconds"] for r in rs) / len(rs),
                    "empty": sum(not r["answer"].strip() for r in rs), "n": len(rs), "recs": rs}
    return out


def print_summary(path) -> None:
    pq = per_question(_load_scored(path))
    print(f"\n{path}\n")
    print(f"{'question':28s} {'score':>6s} {'secs':>6s}  failed checks")
    for qid, s in pq.items():
        fails = sorted({c["check"] for r in s["recs"] for c in r["checks"] if c["status"] == "fail"})
        skips = sorted({c["check"] for r in s["recs"] for c in r["checks"] if c["status"] == "skip"})
        score = "  n/a" if s["score"] is None else f"{s['score']:6.2f}"
        note = "; ".join(fails) + (f"  [skipped: {', '.join(skips)}]" if skips else "")
        print(f"{qid:28s} {score} {s['seconds']:6.0f}  {note}")
    print()
    by_mode = defaultdict(list)
    for s in pq.values():
        by_mode[s["mode"]].append(s)
    print(f"{'mode':10s} {'n':>3s} {'score':>6s} {'perfect':>8s} {'secs':>6s} {'empty':>6s}")
    for mode, ss in [*by_mode.items(), ("ALL", list(pq.values()))]:
        sc = [s["score"] for s in ss if s["score"] is not None]
        print(f"{mode:10s} {len(ss):3d} {sum(sc) / max(len(sc), 1):6.2f} "
              f"{sum(x == 1.0 for x in sc):4d}/{len(sc):<3d} "
              f"{sum(s['seconds'] for s in ss) / len(ss):6.0f} {sum(s['empty'] for s in ss):6d}")


def cmd_compare(args) -> None:
    a, b = per_question(_load_scored(args.a)), per_question(_load_scored(args.b))
    la, lb = Path(args.a).name.split(".")[0], Path(args.b).name.split(".")[0]
    print(f"{'question':28s} {la[:8]:>8s} {lb[:8]:>8s}   delta")
    for qid in sorted(set(a) | set(b)):
        sa = a.get(qid, {}).get("score")
        sb = b.get(qid, {}).get("score")
        fa = "    -" if sa is None else f"{sa:8.2f}"
        fb = "    -" if sb is None else f"{sb:8.2f}"
        d = "" if sa is None or sb is None or abs(sb - sa) < 1e-9 else f"{sb - sa:+.2f}"
        print(f"{qid:28s} {fa:>8s} {fb:>8s}   {d}")
    print()
    for name, pq in ((la, a), (lb, b)):
        sc = [s["score"] for s in pq.values() if s["score"] is not None]
        secs = sum(s["seconds"] for s in pq.values()) / max(len(pq), 1)
        print(f"{name:28s} mean score {sum(sc) / max(len(sc), 1):.3f}  "
              f"perfect {sum(x == 1.0 for x in sc)}/{len(sc)}  mean secs {secs:.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="ask the questions (needs the model; run on a GPU node)")
    g.add_argument("--label", required=True, help="name for this run's output files")
    g.add_argument("--model", help="OFA_MODEL to use (default: ofa's default)")
    g.add_argument("--repeats", type=int, default=1, help="ask each question N times (sampling varies)")
    g.add_argument("--modes", nargs="+", help="only these modes")
    g.add_argument("--ids", nargs="+", help="only these question ids")
    s = sub.add_parser("score", help="(re)score a results .jsonl")
    s.add_argument("results")
    c = sub.add_parser("compare", help="compare two .scored.jsonl files")
    c.add_argument("a")
    c.add_argument("b")
    for p in (g, s):
        p.add_argument("--questions", default=str(HERE / "questions.toml"))
    args = ap.parse_args()
    {"generate": cmd_generate, "score": cmd_score, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
