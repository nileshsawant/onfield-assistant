"""Auto-started vLLM servers, so large models work like the Ollama ones.

``ofa --model <name>`` with a name from ``RECIPES`` reuses the user's
running server for that model, or submits ``tools/vllm-serve.sbatch`` with
the model's recipe and waits until it answers. The server stops itself
after ``OFA_VLLM_IDLE_MINUTES`` without requests, so it is reused across
sessions but does not hold nodes indefinitely.

Stdlib only: bin/ofa runs this on the login node before allocating a GPU.

Settings (environment):
  OFA_VLLM_MODELS_DIR     shared weights directory (default $OFA_ROOT/vllm-models)
  OFA_VLLM_PARTITION      partition for the server job (default gpu-h100s)
  OFA_VLLM_WALLTIME       walltime for the server job (default 04:00:00)
  OFA_VLLM_IDLE_MINUTES   stop after this many idle minutes (default 60; 0 = never)
  OFA_VLLM_EXCLUDE        nodes to avoid, e.g. known-bad GPU nodes
  OFA_ACCOUNT             SLURM account (default: your default account)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

OFA_ROOT = os.environ.get("OFA_ROOT", str(Path(__file__).resolve().parent.parent))
OFA_SCRATCH = os.environ.get("OFA_SCRATCH") or f"/scratch/{os.environ.get('USER', 'unknown')}"

_TOOLS = "--trust-remote-code --enable-auto-tool-choice --tool-call-parser xml_tools --max-num-seqs 16"

# name -> weights directory under OFA_VLLM_MODELS_DIR, whole GPU nodes, `vllm serve` args;
# optional "env" (directory under env/envs, default "vllm"), "tp"/"pp" overrides, and "environ" (extra variables).
RECIPES: dict[str, dict] = {
    "nemotron-3.5-super-vl:120b-a12b": {
        "dir": "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
        "nodes": 1,
        "args": _TOOLS + " --reasoning-parser nemotron_v3 --gpu-memory-utilization 0.88"
                " --limit-mm-per-prompt {\"image\":1,\"video\":0}"
                " --speculative_config.method mtp --speculative_config.num_speculative_tokens 3",
    },
    "trinity-large-thinking:fp8": {
        "dir": "Trinity-Large-Thinking-FP8-Block",
        "nodes": 2,
        "args": _TOOLS + " --reasoning-parser think_tags",
    },
}

POLL_SECONDS = 15


def is_vllm_model(model: str | None) -> bool:
    return bool(model) and model in RECIPES


def _safe(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def endpoint_path(model: str) -> str:
    return os.path.join(OFA_SCRATCH, ".ofa_vllm", _safe(model) + ".json")


def _job_name(model: str) -> str:
    return "ofa-vllm-" + _safe(model)


def _slurm(cmd: str, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    # Inside a job a newer SLURM on PATH/LD_LIBRARY_PATH can shadow the system one; prefer the pinned build.
    pinned = f"/nopt/slurm/current/bin/{cmd}"
    exe = pinned if os.access(pinned, os.X_OK) else cmd
    run_env = dict(env if env is not None else os.environ)
    run_env.pop("LD_LIBRARY_PATH", None)
    r = subprocess.run([exe, *args], capture_output=True, text=True, env=run_env)
    if exe != cmd and "error while loading shared libraries" in r.stderr:
        r = subprocess.run([cmd, *args], capture_output=True, text=True,
                           env=dict(env if env is not None else os.environ))
    return r


def healthy(path: str) -> bool:
    try:
        with open(path) as f:
            base = json.load(f)["base_url"].rstrip("/")
        with urllib.request.urlopen(base[: -len("/v1")] + "/health", timeout=5) as r:
            return r.status == 200
    except (OSError, ValueError, KeyError):
        return False


def _queued_job(model: str) -> tuple[str, str, str] | None:
    """(job id, state, reason) of the user's pending/running server job for model."""
    r = _slurm("squeue", "-h", "-u", os.environ.get("USER", ""), "-n", _job_name(model),
               "-t", "PENDING,CONFIGURING,RUNNING", "-o", "%i|%T|%r")
    for line in r.stdout.splitlines():
        jid, state, reason = (line.split("|") + ["", ""])[:3]
        if jid.strip():
            return jid.strip(), state.strip(), reason.strip()
    return None


def _account() -> str:
    acct = os.environ.get("OFA_ACCOUNT", "")
    if acct:
        return acct
    r = _slurm("sacctmgr", "show", "user", os.environ.get("USER", ""), "format=defaultaccount", "-n", "-P")
    return r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""


def _log_dir() -> str:
    d = os.path.join(OFA_SCRATCH, "ofa-vllm-logs")
    os.makedirs(d, exist_ok=True)
    return d


def _submit(model: str, exclude: str) -> str:
    recipe = RECIPES[model]
    weights = os.path.join(os.environ.get("OFA_VLLM_MODELS_DIR", os.path.join(OFA_ROOT, "vllm-models")),
                           recipe["dir"])
    if not os.path.isdir(weights):
        sys.exit(f"ofa: weights for {model} not found at {weights} (set OFA_VLLM_MODELS_DIR).")
    os.makedirs(os.path.dirname(endpoint_path(model)), mode=0o700, exist_ok=True)
    env = dict(os.environ, OFA_ROOT=OFA_ROOT, OFA_SCRATCH=OFA_SCRATCH, OFA_VLLM_MODEL=weights,
               OFA_VLLM_NAME=model, OFA_VLLM_ARGS=recipe["args"], OFA_VLLM_ENDPOINT=endpoint_path(model),
               OFA_VLLM_IDLE_MINUTES=os.environ.get("OFA_VLLM_IDLE_MINUTES", "60"))
    if "env" in recipe:
        env["OFA_VLLM_ENV"] = os.path.join(OFA_ROOT, "env", "envs", recipe["env"])
    for key in ("tp", "pp"):
        if key in recipe:
            env[f"OFA_VLLM_{key.upper()}"] = str(recipe[key])
    env.update(recipe.get("environ", {}))
    args = ["--parsable", "-J", _job_name(model),
            "-p", os.environ.get("OFA_VLLM_PARTITION", "gpu-h100s"),
            "-t", os.environ.get("OFA_VLLM_WALLTIME", "04:00:00"),
            f"--nodes={recipe['nodes']}", "--ntasks-per-node=1", "--exclusive",
            f"--gres=gpu:{os.environ.get('OFA_VLLM_GPUS_PER_NODE', '4')}",
            "-o", os.path.join(_log_dir(), _safe(model) + "-%j.log"), "--export=ALL"]
    acct = _account()
    if acct:
        args += ["-A", acct]
    if exclude:
        args += ["--exclude", exclude]
    r = _slurm("sbatch", *args, os.path.join(OFA_ROOT, "tools", "vllm-serve.sbatch"), env=env)
    if r.returncode != 0:
        sys.exit(f"ofa: could not submit the {model} server job:\n{r.stderr.strip()}")
    return r.stdout.strip().split(";")[0]


def _failed_nodes(jid: str, model: str) -> list[str]:
    """Nodes that failed the GPU preflight in job jid's log (empty if it failed for another reason)."""
    log = os.path.join(_log_dir(), f"{_safe(model)}-{jid}.log")
    try:
        text = Path(log).read_text(errors="replace")
    except OSError:
        return []
    if "GPU preflight failed" not in text:
        return []
    return sorted(set(re.findall(r"srun: error: (\S+): task \d+: Exited", text)))


def ensure_server(model: str) -> str | None:
    """Return the endpoint file of a healthy server for model, starting one if needed.
    None if model is not a vLLM model. Exits with a message on failure."""
    if not is_vllm_model(model):
        return None
    ep = endpoint_path(model)
    if healthy(ep):
        return ep
    exclude = os.environ.get("OFA_VLLM_EXCLUDE", "")
    queued = _queued_job(model)
    if queued:
        jid = queued[0]
        print(f"ofa: waiting for your {model} server (job {jid}).", file=sys.stderr)
    else:
        jid = _submit(model, exclude)
        idle = os.environ.get("OFA_VLLM_IDLE_MINUTES", "60")
        print(f"ofa: starting {model} on {RECIPES[model]['nodes']} GPU node(s), job {jid}. "
              f"The first start takes 10-30 min once nodes are free; later sessions reuse it until it has been idle "
              f"for {idle} min.", file=sys.stderr)
    tty = sys.stderr.isatty()
    start, last_print, retried = time.time(), 0.0, False
    try:
        while True:
            if healthy(ep):
                if tty:
                    print(file=sys.stderr)
                print(f"ofa: {model} is ready.", file=sys.stderr)
                return ep
            queued = _queued_job(model)
            if queued is None:
                bad = _failed_nodes(jid, model)
                if bad and not retried:
                    exclude = ",".join(filter(None, [exclude, *bad]))
                    retried = True
                    jid = _submit(model, exclude)
                    print(f"\nofa: GPU check failed on {', '.join(bad)}; retrying without "
                          f"{'it' if len(bad) == 1 else 'them'} as job {jid}.", file=sys.stderr)
                    continue
                sys.exit(f"\nofa: the {model} server job {jid} ended before it was ready. "
                         f"Log: {os.path.join(_log_dir(), _safe(model) + '-' + jid + '.log')}")
            jid, state, reason = queued
            mins = int(time.time() - start) // 60
            if state != "PENDING":
                status = "loading the model"
            elif reason in ("Priority", "Resources", "None") or reason.startswith("Nodes required"):
                status = "waiting for free GPU nodes"
            else:
                status = f"queued ({reason})"
            status += f", {mins} min"
            if tty:
                print(f"\rofa: {model} job {jid}: {status}    ", end="", file=sys.stderr, flush=True)
            elif time.time() - last_print >= 60:
                print(f"ofa: {model} job {jid}: {status}", file=sys.stderr, flush=True)
                last_print = time.time()
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print(f"\nofa: stopped waiting. Job {jid} keeps starting; run the same command again to "
              f"connect, or `scancel {jid}` to cancel it.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    # bin/ofa: `ofa_vllm.py ensure <model>`; exit 0 also when <model> is not a vLLM model.
    if len(sys.argv) == 3 and sys.argv[1] == "ensure":
        ensure_server(sys.argv[2])
        sys.exit(0)
    sys.exit("usage: ofa_vllm.py ensure <model>")
