"""OpenAI-compatible HTTP server for ofa (BYOK shim for VS Code, etc.).

Exposes:
  GET  /v1/models                       — advertises the ofa-* model IDs
  POST /v1/chat/completions             — OpenAI-format chat with SSE streaming
  GET  /healthz                         — liveness probe (no auth)

What each /v1/chat/completions request does:
  1. Authenticate via `Authorization: Bearer <token>` (token read from a
     keyfile created on first --serve run).
  2. Pick the ofa "mode" from the `model` field:
        ofa-openfoam, ofa-hpc, ofa-code, ofa-amrex, ofa-reframe
  3. Build the system prompt via ``ofa_main.load_system_prompt(mode)`` so
     long-term prefs + lessons are injected exactly the same way the CLI
     does.
  4. Run RAG on the user's last message via the same retriever the CLI
     uses for that mode, fence the result via ``_fence_rag``, and prepend
     it to the user message.
  5. Stream the assistant response via ``ofa_main.chat_stream`` and
     repackage each chunk as an OpenAI SSE `data: { ... }` line.

What we deliberately do NOT do in v1:
  * Tool-calling. We advertise `toolCalling: false` in the BYOK config
    so VS Code uses ofa as a smart chat backend, not as an agent that
    expects OpenAI function-call JSON.
  * /v1/embeddings. Out of scope; Copilot or another provider can handle
    embeddings if VS Code needs them.

Concurrency: ThreadingHTTPServer with a per-request handler. Ollama
itself serialises generation under the hood; our handler is mostly I/O
forwarding, so threads are cheap and prevent one slow request from
blocking the readiness probe.
"""
from __future__ import annotations

import json
import os
import secrets
import socketserver
import stat
import sys
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterable

# Imported lazily inside the handler so we don't pay ChromaDB init cost
# at server-construction time (tests can monkey-patch before that).
ofa_main = None  # set by ``serve()`` after we import the real module


# ---- model-id ↔ ofa mode mapping ----------------------------------------
# We advertise five distinct "models" to VS Code so the user can switch
# RAG context + system prompt from the model dropdown without restarting
# anything. They all hit the same underlying Ollama model.
_MODEL_MODES = {
    "ofa-openfoam": "openfoam",
    "ofa-hpc":      "hpc",
    "ofa-code":     "code",
    "ofa-amrex":    "amrex",
    "ofa-reframe":  "reframe",
}


def _retrieve_for_mode(query: str, mode: str) -> str:
    """Dispatch to the right RAG helper for ``mode``. Empty on greetings.

    Mirrors the dispatch in ``interactive_mode`` / ``single_query`` so
    BYOK clients get identical context to the CLI.
    """
    greetings = {"hi", "hello", "hey", "howdy", "thanks", "thank you"}
    if not query or query.strip().lower() in greetings:
        return ""
    if mode == "reframe":
        rhel9 = ofa_main._get_reframe_rag(query)
        base = ofa_main.retrieve_hpc_context(query)
        return (
            "=== RHEL9 SPECIFIC CONTEXT (TAKES PRECEDENCE) ===\n"
            f"{rhel9}\n\n=== GENERAL HPC CONTEXT (RHEL8/Legacy) ===\n{base}"
        )
    if mode == "amrex":
        return ofa_main.retrieve_amrex_context(query)
    if mode in ("hpc", "code"):
        return ofa_main.retrieve_hpc_context(query)
    return ofa_main.retrieve_context(query)


def _augment_user_message(content: str, mode: str) -> str:
    """Prefix RAG context to the *last* user message, just like the CLI."""
    rag = ""
    try:
        rag = _retrieve_for_mode(content, mode)
    except Exception as e:
        # RAG failure must never break a chat — log and continue with no
        # context, matching the CLI's degrade-gracefully behaviour.
        print(f"[ofa-serve] RAG retrieval failed for mode={mode}: {e}",
              file=sys.stderr)
        return content
    if not rag:
        return content
    label = (
        "RHEL9_STACK+HPC" if mode == "reframe"
        else "HPC_DOCS" if mode in ("hpc", "code", "amrex")
        else "OPENFOAM"
    )
    fenced = ofa_main._fence_rag(rag, label=label)
    if mode == "reframe":
        return f"Extracted RHEL9 Stack & RHEL8 Context:\n\n{fenced}\n\n---\n\nUser request: {content}"
    if mode in ("hpc", "code", "amrex"):
        return f"Here is relevant context for your reference:\n\n{fenced}\n\n---\n\nUser request: {content}"
    return (
        f"Here are relevant OpenFOAM example files for reference:\n\n"
        f"{fenced}\n\n---\n\nUser request: {content}"
    )


def _augment_messages(messages: list[dict], mode: str) -> list[dict]:
    """Return a new message list with RAG injected on the last user msg
    and an ofa system prompt prepended (replacing any inbound system msg).

    We replace inbound system messages on purpose: VS Code's agent harness
    sends its own system prompt that knows nothing about Kestrel/OpenFOAM,
    and our system prompt + memory injection is the whole point of routing
    through ofa.
    """
    out = [{"role": "system", "content": ofa_main.load_system_prompt(mode)}]
    last_user_idx = None
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            last_user_idx = i
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "system":
            # Drop inbound system messages — ours wins. We surface this in
            # a debug header so the client side can confirm.
            continue
        content = m.get("content", "")
        if role == "user" and i == last_user_idx:
            content = _augment_user_message(content, mode)
        out.append({"role": role, "content": content})
    return out


# ---- SSE chunk formatting ------------------------------------------------

def _sse_chunk(model_id: str, completion_id: str, created: int,
               delta: dict, finish_reason: str | None = None) -> bytes:
    """Render one OpenAI-style `data: {...}\\n\\n` SSE line."""
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _sse_done() -> bytes:
    return b"data: [DONE]\n\n"


# ---- key-file auth -------------------------------------------------------

def load_or_create_api_key(path: str) -> str:
    """Return the bearer token stored at ``path``, creating it (0o600) if
    the file doesn't exist. A missing parent directory is created too.
    """
    if os.path.exists(path):
        try:
            with open(path) as f:
                token = f.read().strip()
            if token:
                return token
        except OSError as e:
            raise RuntimeError(f"could not read api key at {path}: {e}") from e
    # Create.
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    token = "ofa-" + secrets.token_urlsafe(32)
    # Write with restrictive perms FIRST, then populate. Otherwise there
    # is a brief window where the token sits world-readable.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, token.encode("utf-8") + b"\n")
    finally:
        os.close(fd)
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # belt-and-braces
    except OSError:
        pass
    print(f"[ofa-serve] generated new API key at {path}", file=sys.stderr)
    return token


# ---- HTTP handler --------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    # Populated by ``serve()`` before the server starts accepting requests.
    api_key: str = ""
    expose_models: tuple[str, ...] = tuple(_MODEL_MODES)

    # Quiet down the default per-request stderr line; we log our own.
    def log_message(self, format, *args):  # noqa: A003 (stdlib name)
        return

    # ---- helpers ----
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str, type_: str = "invalid_request_error") -> None:
        self._send_json(status, {
            "error": {"message": message, "type": type_, "code": status}
        })

    def _auth_ok(self) -> bool:
        if not self.api_key:
            return True  # auth disabled (server started with --no-auth)
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return False
        # constant-time comparison
        return secrets.compare_digest(auth[len("Bearer "):].strip(), self.api_key)

    # ---- routes ----
    def do_GET(self):  # noqa: N802 (stdlib name)
        if self.path == "/healthz":
            return self._send_json(200, {"status": "ok"})
        if self.path == "/v1/models":
            if not self._auth_ok():
                return self._send_error(401, "missing or invalid Authorization header", "invalid_api_key")
            return self._send_json(200, {
                "object": "list",
                "data": [
                    {
                        "id": mid,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "ofa",
                    }
                    for mid in self.expose_models
                ],
            })
        return self._send_error(404, f"no route for GET {self.path}", "not_found")

    def do_POST(self):  # noqa: N802 (stdlib name)
        if self.path != "/v1/chat/completions":
            return self._send_error(404, f"no route for POST {self.path}", "not_found")
        if not self._auth_ok():
            return self._send_error(401, "missing or invalid Authorization header", "invalid_api_key")

        # ---- parse request body ----
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._send_error(400, "invalid Content-Length")
        if length <= 0:
            return self._send_error(400, "empty request body")
        raw = self.rfile.read(length)
        try:
            req = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            return self._send_error(400, f"could not parse JSON body: {e}")

        model_id = (req.get("model") or "").strip()
        if model_id not in _MODEL_MODES:
            return self._send_error(
                400,
                f"unknown model {model_id!r}; expected one of {list(_MODEL_MODES)}",
                "model_not_found",
            )
        mode = _MODEL_MODES[model_id]
        messages = req.get("messages") or []
        if not isinstance(messages, list) or not messages:
            return self._send_error(400, "messages must be a non-empty list")

        stream = bool(req.get("stream"))
        # Honour basic OpenAI sampling overrides where the CLI helpers
        # accept them. We pass nothing through if the client didn't set
        # them, so ofa's per-model defaults still apply.
        opt_overrides = {}
        if "temperature" in req: opt_overrides["temperature"] = req["temperature"]
        if "top_p" in req:       opt_overrides["top_p"] = req["top_p"]
        if "max_tokens" in req:  opt_overrides["num_predict"] = req["max_tokens"]

        # ---- build the message list ofa expects ----
        try:
            ofa_messages = _augment_messages(messages, mode)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            return self._send_error(500, f"failed to build prompt: {e}", "server_error")

        print(
            f"[ofa-serve] {model_id} ({mode}): {len(messages)} msg(s), "
            f"stream={stream}",
            file=sys.stderr,
        )

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if stream:
            return self._handle_stream(model_id, completion_id, created, ofa_messages, opt_overrides)
        return self._handle_blocking(model_id, completion_id, created, ofa_messages, opt_overrides)

    # ---- streaming branch ----
    def _handle_stream(self, model_id, completion_id, created, messages, opts):
        # We send Connection: close because a) clients reading SSE with
        # blocking I/O (urllib, requests-without-streaming) would otherwise
        # wait forever for more bytes after [DONE], and b) we don't want
        # ThreadingHTTPServer to reuse the same socket for a second request
        # while the model is still warming up.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        # Force the next response on this socket to be a fresh connection.
        self.close_connection = True

        # Opening chunk: announces the assistant role so OpenAI clients
        # initialise their delta accumulator correctly.
        self.wfile.write(_sse_chunk(model_id, completion_id, created, {"role": "assistant"}))
        self.wfile.flush()

        try:
            for chunk in ofa_main.chat_stream(messages, **opts):
                if not chunk:
                    continue
                self.wfile.write(_sse_chunk(model_id, completion_id, created, {"content": chunk}))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected mid-stream — nothing to do.
            return
        except Exception as e:
            # Surface the error inside the stream so the client sees a
            # finish_reason and not a hung connection.
            traceback.print_exc(file=sys.stderr)
            try:
                self.wfile.write(_sse_chunk(model_id, completion_id, created,
                                            {"content": f"\n\n[ofa-serve error] {e}"}))
                self.wfile.flush()
            except OSError:
                return

        try:
            self.wfile.write(_sse_chunk(model_id, completion_id, created, {}, finish_reason="stop"))
            self.wfile.write(_sse_done())
            self.wfile.flush()
        except OSError:
            return

    # ---- non-streaming branch ----
    def _handle_blocking(self, model_id, completion_id, created, messages, opts):
        try:
            text = "".join(ofa_main.chat_stream(messages, **opts))
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            return self._send_error(500, f"model error: {e}", "server_error")
        return self._send_json(200, {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                # ofa doesn't surface token counts from Ollama in the
                # streaming path; report char counts as a coarse proxy.
                "prompt_tokens": sum(len(m.get("content", "")) for m in messages) // 4,
                "completion_tokens": len(text) // 4,
                "total_tokens": (sum(len(m.get("content", "")) for m in messages) + len(text)) // 4,
            },
        })


# ---- public entry point --------------------------------------------------

def _default_local_port(scratch_dir: str) -> int:
    """Return a stable laptop-side port suggestion.

    First call generates a random port in the IANA dynamic/private range
    (well clear of 11434/11435/11436 which VS Code Remote-SSH likes to
    auto-forward) and persists it to ``$OFA_SCRATCH/.ofa_serve_local_port``.
    Subsequent calls return the same port so the user's BYOK config URL
    in VS Code does not have to change between ``ofa --serve`` runs.

    If the persisted file is missing or unreadable we pick a fresh random
    port. If it's unwritable we still return a port; we just can't
    persist it (the user will see a different suggestion next run, no
    correctness issue).
    """
    f = os.path.join(scratch_dir, ".ofa_serve_local_port")
    if os.path.exists(f):
        try:
            with open(f) as fh:
                port = int(fh.read().strip())
            if 1024 <= port <= 65535:
                return port
        except (OSError, ValueError):
            pass
    # 49152-65535 is IANA dynamic/private range. Avoid the very top to
    # leave headroom for ephemeral source ports.
    port = 49200 + secrets.randbelow(15000)
    try:
        # 0o600 so a shared scratch can't leak this to other users.
        fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, str(port).encode())
        finally:
            os.close(fd)
    except OSError:
        pass
    return port


def serve(host: str = "127.0.0.1", port: int = 0,
          api_key_file: str | None = None,
          no_auth: bool = False,
          local_port: int | None = None) -> None:
    """Start the BYOK server. Blocks until Ctrl+C.

    Parameters
    ----------
    host:
        Address to bind. Use 127.0.0.1 (default) when reaching the server
        via SSH port-forward (recommended). Use 0.0.0.0 only on a node
        where the network is already restricted (and never on a public
        login node).
    port:
        TCP port on the *server* side. ``0`` (default) lets the OS pick a
        free port — robust against conflicts with Ollama or other ofa
        sessions on the same node. The actually-bound port is printed at
        startup so the user knows what to put on the right-hand side of
        ``ssh -L``.
    api_key_file:
        Path to the bearer-token file. Created with mode 0o600 on first
        run. Defaults to ``$OFA_SCRATCH/.ofa_api_key``.
    no_auth:
        Skip the Authorization check. ONLY for local testing.
    local_port:
        Suggested *laptop-side* port for the printed ``ssh -L`` line and
        BYOK URL. ``None`` (default) uses a per-user-stable random port
        in the 49200-64200 range (persisted to scratch so the VS Code
        config doesn't have to change between runs).
    """
    global ofa_main
    import ofa_main as _ofa_main  # noqa: PLC0415 — deliberate lazy import
    ofa_main = _ofa_main

    ofa_main.ensure_ollama_running()
    print("[ofa-serve] loading RAG index…", file=sys.stderr)
    ofa_main._init_rag()
    print("[ofa-serve] RAG ready.", file=sys.stderr)

    if no_auth:
        token = ""
        print("[ofa-serve] WARNING: started with --no-auth, all requests accepted",
              file=sys.stderr)
    else:
        if api_key_file is None:
            api_key_file = os.path.join(ofa_main.OFA_SCRATCH, ".ofa_api_key")
        token = load_or_create_api_key(api_key_file)
        print(f"[ofa-serve] auth token in {api_key_file} (chmod 600)", file=sys.stderr)

    _Handler.api_key = token
    httpd = ThreadingHTTPServer((host, port), _Handler)
    # When port=0 the OS picks; surface the actual bound port to the user
    # and use it for the printed ssh -L line.
    actual_port = httpd.server_port

    # Thin shim: delegate to ofa_main._c if present (which respects NO_COLOR
    # and TTY detection); otherwise pass text through unchanged. Keeps the
    # connection block readable without forcing colour codes into the log.
    def _c(text: str, *styles: str) -> str:
        fn = getattr(ofa_main, "_c", None)
        return fn(text, *styles) if fn else text

    node = os.uname().nodename
    # Pick a per-user-stable laptop port (random first time, persisted)
    # unless the caller pinned one. Random + persisted means the BYOK URL
    # in VS Code stays the same across restarts without colliding with
    # VS Code Remote-SSH's auto-forward of 11434/11435/11436.
    if local_port is None:
        local_port = _default_local_port(ofa_main.OFA_SCRATCH)
    base_url = f"http://localhost:{local_port}"

    print(f"[ofa-serve] listening on http://{host}:{actual_port}", file=sys.stderr)
    print(f"[ofa-serve] models: {', '.join(_MODEL_MODES)}", file=sys.stderr)
    print(f"[ofa-serve] node={node} pid={os.getpid()}", file=sys.stderr)
    # Big copy-pasteable connection block. Two ports are involved and
    # users repeatedly conflate them; we label both explicitly to avoid
    # confusion.
    #   - REMOTE port = what `ofa --serve` binds on the Kestrel compute
    #     node (this process). OS-assigned by default, or pinned via
    #     --serve-port.
    #   - LOCAL  port = what your laptop listens on. Random + persisted
    #     by default, or pinned via --serve-local-port. This is the
    #     number that goes in the VS Code BYOK URL.
    print("", file=sys.stderr)
    print(_c("=" * 72, "cyan"), file=sys.stderr)
    print(_c(" CONNECT FROM YOUR LAPTOP ", "bold", "cyan"), file=sys.stderr)
    print(_c("=" * 72, "cyan"), file=sys.stderr)
    print("", file=sys.stderr)
    print(_c(
        f"  Kestrel compute node:  {node}", "dim",
    ), file=sys.stderr)
    print(_c(
        f"  REMOTE port (this server):  {actual_port}", "dim",
    ), file=sys.stderr)
    print(_c(
        f"  LOCAL  port (your laptop):  {local_port}", "dim",
    ), file=sys.stderr)
    print("", file=sys.stderr)
    print(_c("Step 1 — run this in a new laptop terminal (leave it open):", "bold", "cyan"),
          file=sys.stderr)
    print(_c(
        f"  ssh -N -o ExitOnForwardFailure=yes "
        f"-L {local_port}:{node}:{actual_port} kestrel.hpc.nrel.gov",
        "bold",
    ), file=sys.stderr)
    print("", file=sys.stderr)
    print(_c("Step 2 — quick sanity check from the laptop:", "bold", "cyan"),
          file=sys.stderr)
    print(_c(f"  curl {base_url}/healthz", "bold"), file=sys.stderr)
    print(_c("  (should print {\"status\":\"ok\"})", "dim"), file=sys.stderr)
    print("", file=sys.stderr)
    print(_c("Step 3 — paste these into VS Code chatLanguageModels.json:", "bold", "cyan"),
          file=sys.stderr)
    print(_c(f"  url    = {base_url}/v1/chat/completions", "bold"), file=sys.stderr)
    if not no_auth:
        print(_c(f"  apiKey = {token}", "bold"), file=sys.stderr)
    print("", file=sys.stderr)
    print(_c("If Step 1 fails with 'Address already in use':", "yellow"), file=sys.stderr)
    print(_c(
        f"  Something on your LAPTOP is holding {local_port}. Most often that's "
        f"VS Code Remote-SSH's auto-forward. Pick a different number with",
        "dim",
    ), file=sys.stderr)
    print(_c(
        f"      ofa --serve --serve-local-port <N>",
        "dim",
    ), file=sys.stderr)
    print(_c(
        f"  (any free port works; update the VS Code BYOK URL to match).",
        "dim",
    ), file=sys.stderr)
    print(_c("=" * 72, "cyan"), file=sys.stderr)
    print("", file=sys.stderr)
    print("[ofa-serve] Ctrl+C to stop.", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[ofa-serve] shutting down…", file=sys.stderr)
    finally:
        httpd.server_close()
