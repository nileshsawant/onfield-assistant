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
import re
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
    "ofa-openfoam":           "openfoam",
    "ofa-hpc":                "hpc",
    "ofa-code":               "code",
    "ofa-amrex":              "amrex",
    "ofa-marbles":            "marbles",
    "ofa-reframe":            "reframe",
    "ofa-quantum-computing":  "quantum-computing",
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
    if mode == "marbles":
        return ofa_main.retrieve_marbles_context(query)
    if mode == "quantum-computing":
        return ofa_main.retrieve_quantum_computing_context(query)
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
        else "QUANTUM" if mode == "quantum-computing"
        else "HPC_DOCS" if mode in ("hpc", "code", "amrex", "marbles")
        else "OPENFOAM"
    )
    fenced = ofa_main._fence_rag(rag, label=label)
    if mode == "reframe":
        return f"Extracted RHEL9 Stack & RHEL8 Context:\n\n{fenced}\n\n---\n\nUser request: {content}"
    if mode in ("hpc", "code", "amrex", "marbles", "quantum-computing"):
        return f"Here is relevant context for your reference:\n\n{fenced}\n\n---\n\nUser request: {content}"
    return (
        f"Here are relevant OpenFOAM example files for reference:\n\n"
        f"{fenced}\n\n---\n\nUser request: {content}"
    )


def _split_content(content) -> tuple[str, list[str]]:
    """Split an OpenAI-format content field into (text, images).

    Accepts either the classic string form (returned as ``(str, [])``) or
    the multimodal array form::

        [{"type": "text",      "text": "..."},
         {"type": "image_url", "image_url": {"url": "data:image/png;base64,..." }}]

    Image URLs are extracted as raw base64 payloads for Ollama's
    ``messages[].images`` field. Only ``data:...;base64,...`` URLs are
    decoded here; ``https://`` URLs are skipped with a warning because
    fetching them would require outbound HTTP from the Kestrel compute
    node (usually blocked). Users should base64-encode client-side.
    """
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return "", []
    texts: list[str] = []
    images: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            texts.append(str(part.get("text", "")))
        elif ptype == "image_url":
            url_obj = part.get("image_url")
            url = url_obj.get("url", "") if isinstance(url_obj, dict) else (url_obj or "")
            if isinstance(url, str) and url.startswith("data:"):
                _, _, payload = url.partition("base64,")
                if payload:
                    images.append(payload)
            elif isinstance(url, str) and url.startswith(("http://", "https://")):
                print(
                    f"[ofa-serve] skipping image at {url[:60]}...: fetching "
                    "HTTP(S) URLs from the compute node is disabled; "
                    "base64-encode client-side instead",
                    file=sys.stderr,
                )
    return "\n".join(texts), images


# Regex used by _strip_fence_tool_protocol below. The markers live in
# prompts/common.txt and bracket the section of the system prompt that
# teaches the model ofa's ```bash / ```read / ```write / etc. fence
# tool convention. That teaching is *incompatible* with OpenAI-style
# tool_calls: given both, Gemma consistently emits fences instead of
# JSON tool_calls, silently breaking BYOK agent clients (langchain,
# langgraph, amrex-agent, ...). When a request supplies tools=[...] we
# elide the whole bracketed section (markers and all) so the model
# defaults to OpenAI's tool_calls convention.
_FENCE_TOOL_STRIP_RE = re.compile(
    r"\n===== BEGIN OFA FENCE-TOOL PROTOCOL[^\n]*\n"
    r".*?"
    r"===== END OFA FENCE-TOOL PROTOCOL =====\n",
    re.DOTALL,
)


def _strip_fence_tool_protocol(sys_prompt: str) -> str:
    """Remove the fence-tool teaching block from the system prompt.

    Safe no-op if the markers aren't present (older prompt files or
    mode-specific prompts that never included the block).
    """
    return _FENCE_TOOL_STRIP_RE.sub("\n", sys_prompt)


# When ``tools=[...]`` is present on an incoming request, we prepend this
# directive to the system prompt. Stripping the fence-tool block alone is
# not enough — the mode prompts (code.txt, cpp.txt, examples in common.txt)
# reference ```bash / ```write / etc. in passing, and Gemma follows those
# examples over the caller's OpenAI schema unless we override explicitly.
# This directive sits at the very top of the system prompt so it primes
# the model before any legacy fence teaching lower down.
_OPENAI_TOOL_MODE_DIRECTIVE = (
    "CRITICAL — OpenAI tool-call mode ACTIVE for this request.\n"
    "\n"
    "Your caller (a BYOK client) has supplied a list of tools in OpenAI's\n"
    "standard JSON-schema `tools` format. In this session you MUST invoke\n"
    "tools using OpenAI's native `tool_calls` mechanism, NOT ofa's\n"
    "fenced-code-block convention.\n"
    "\n"
    "Rules for this request only:\n"
    "- When you decide to invoke a tool, produce a native `tool_calls`\n"
    "  response with the function name and JSON arguments. Do NOT emit\n"
    "  ```bash / ```read / ```write / ```edit / ```search / ```fetch /\n"
    "  ```plan / ```validate_inputs / etc. fenced blocks — the caller\n"
    "  cannot execute them. Emitting a fenced block produces no tool\n"
    "  call and confuses the caller.\n"
    "- Ignore any fence-syntax examples that may appear elsewhere in this\n"
    "  system prompt — those are ofa's default CLI convention; the\n"
    "  OpenAI convention supersedes them for this request.\n"
    "- Continue to obey identity, thinking-channel (`<thought>`), retrieved-\n"
    "  context grounding, memory, and safety rules normally.\n"
    "\n"
    "---\n"
    "\n"
)


def _augment_messages(messages: list[dict], mode: str, has_tools: bool = False) -> list[dict]:
    """Return a new message list with RAG injected on the last user msg
    and an ofa system prompt prepended (replacing any inbound system msg).

    Handles both classic string ``content`` and the OpenAI multimodal
    array form. Extracted images are attached to the resulting message
    under an ``images`` key (Ollama's format), which ``chat_stream`` and
    ``_ollama_chat_raw`` forward untouched to ``/api/chat``.

    We replace inbound system messages on purpose: BYOK clients often
    ship their own system prompt that knows nothing about Kestrel /
    OpenFOAM, and our system prompt + memory injection is the whole
    point of routing through ofa.

    When ``has_tools`` is True (BYOK request supplied ``tools=[...]``),
    strip ofa's fence-tool protocol block from the system prompt. That
    block instructs the model to emit tool calls as ```bash / ```read /
    etc. fenced blocks, which collides with OpenAI's native
    ``tool_calls`` JSON convention; the model would pick the fence form
    (as observed with amrex-agent), leaving ``tool_calls=None`` in the
    response and breaking any external agent that expects the standard
    OpenAI format. The RAG-injection, identity, thinking-channel and
    behaviour rules all survive; only the fence syntax is elided.
    """
    sys_prompt = ofa_main.load_system_prompt(mode)
    if has_tools:
        sys_prompt = _OPENAI_TOOL_MODE_DIRECTIVE + _strip_fence_tool_protocol(sys_prompt)
    out = [{"role": "system", "content": sys_prompt}]
    last_user_idx = None
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            last_user_idx = i
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "system":
            # Drop inbound system messages — ours wins.
            continue
        content, images = _split_content(m.get("content", ""))
        if role == "user" and i == last_user_idx:
            content = _augment_user_message(content, mode)
        omsg: dict = {"role": role, "content": content}
        if images:
            omsg["images"] = images
        out.append(omsg)
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


# ---- tool-calling passthrough -------------------------------------------
#
# When ``--serve-enable-tools`` is set, the server forwards the OpenAI
# ``tools`` / ``tool_choice`` fields straight to Ollama's /api/chat and
# translates the model's ``tool_calls`` response back to the OpenAI SSE
# delta format VS Code's agent loop expects. Off by default because:
#
#   - VS Code Chat in Ask mode never sends ``tools``; this path is dead
#     code for the common case.
#   - Local 31B Gemma can be unreliable at emitting clean ``tool_calls``
#     JSON for VS Code's complex agent tool schemas. We'd rather the
#     opt-in user know they're trying experimental territory.

def _ollama_chat_raw(messages, tools, tool_choice, options):
    """Stream raw Ollama /api/chat response chunks (full dicts, not just
    text). Bypasses ``ofa_main.chat_stream`` because we need ``tool_calls``
    structure that ``chat_stream`` strips.

    Yields one dict per Ollama chunk; the last has ``done=True``.
    """
    import httpx  # noqa: PLC0415 — lazy import so test fixtures can fake
    payload = {
        "model": ofa_main.MODEL,
        "messages": messages,
        "stream": True,
        # See chat_stream() in ofa_main.py for the rationale. Ollama
        # returns empty `content` for reasoning-capable models unless
        # we opt out of its split-field thinking mode.
        "think": False,
        "options": options,
    }
    if tools:
        payload["tools"] = tools
    # `tool_choice` is OpenAI's name; Ollama doesn't currently support
    # forcing a specific tool, so we pass it through only for clients
    # that might use future Ollama versions. Ollama silently ignores
    # unknown fields.
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    with httpx.stream(
        "POST",
        f"{ofa_main.OLLAMA_HOST}/api/chat",
        json=payload,
        timeout=300.0,
    ) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if data.get("error"):
                raise RuntimeError(f"Ollama error: {data['error']}")
            yield data
            if data.get("done"):
                break


def _ollama_tool_call_to_openai(tc: dict, index: int) -> dict:
    """Translate one Ollama ``tool_calls`` entry to OpenAI SSE delta format.

    Ollama returns ``arguments`` as a parsed JSON object; OpenAI clients
    expect it as a JSON-encoded string. We also synthesise an ``id``
    because Ollama doesn't emit one and OpenAI clients usually need it
    to correlate the eventual ``tool`` reply message.
    """
    fn = tc.get("function") or {}
    args = fn.get("arguments")
    if isinstance(args, (dict, list)):
        args_str = json.dumps(args, ensure_ascii=False)
    elif args is None:
        args_str = ""
    else:
        args_str = str(args)
    return {
        "index": index,
        "id": tc.get("id") or f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": fn.get("name", ""),
            "arguments": args_str,
        },
    }


# ---- HTTP handler --------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    # Populated by ``serve()`` before the server starts accepting requests.
    api_key: str = ""
    expose_models: tuple[str, ...] = tuple(_MODEL_MODES)
    # When True, forward `tools` / `tool_choice` from incoming requests
    # to Ollama and translate Ollama's `tool_calls` responses back to
    # OpenAI SSE format. Off by default because local 31B models can be
    # unreliable at the tool-calling protocol VS Code expects.
    enable_tools: bool = False

    # Quiet down the default per-request stderr line; we log our own.
    def log_message(self, format, *args):  # noqa: A003 (stdlib name)
        return

    # ---- helpers ----
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected before we could finish writing the
            # response. This is normal (e.g. user Ctrl+C'd their sim,
            # or a slow model made the client time out on its end).
            # Log one line instead of the 30-line socketserver traceback.
            print(
                f"[ofa-serve] client {self.client_address[0]} disconnected "
                f"before response; nothing to do",
                file=sys.stderr,
            )

    def _send_error(self, status: int, message: str, type_: str = "invalid_request_error") -> None:
        self._send_json(status, {
            "error": {"message": message, "type": type_, "code": status}
        })

    def _auth_ok(self) -> bool:
        if not self.api_key:
            return True  # auth disabled (server started with --no-auth)
        # Accept multiple common auth header conventions: BYOK clients
        # vary in what they send.
        #   - OpenAI standard:  Authorization: Bearer <token>
        #   - Some clients omit "Bearer ":  Authorization: <token>
        #   - Azure / some custom endpoints:  api-key: <token>
        #   - Older OpenAI SDKs:  x-api-key: <token>, openai-api-key: <token>
        # We accept any of them as long as the value matches; constant-time
        # comparison prevents timing leaks.
        candidates = []
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            candidates.append(auth[len("Bearer "):].strip())
        elif auth:
            candidates.append(auth.strip())
        for hdr in ("api-key", "x-api-key", "openai-api-key"):
            v = self.headers.get(hdr, "")
            if v:
                candidates.append(v.strip())
        return any(secrets.compare_digest(c, self.api_key) for c in candidates if c)

    def _log_auth_failure(self) -> None:
        """Log a redacted summary of any auth-like headers we received.

        Helps diagnose why VS Code / curl / etc. got a 401 without leaking
        the token to logs.
        """
        seen = []
        for hdr in ("Authorization", "api-key", "x-api-key", "openai-api-key"):
            v = self.headers.get(hdr, "")
            if v:
                redacted = v[:8] + "..." if len(v) > 8 else v
                seen.append(f"{hdr}={redacted}")
        msg = ", ".join(seen) if seen else "no auth-like headers received"
        print(f"[ofa-serve] auth failed ({self.client_address[0]}): {msg}",
              file=sys.stderr)

    # ---- routes ----
    def do_GET(self):  # noqa: N802 (stdlib name)
        if self.path == "/healthz":
            return self._send_json(200, {"status": "ok"})
        if self.path == "/v1/models":
            if not self._auth_ok():
                self._log_auth_failure()
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
            self._log_auth_failure()
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

        # Tool-calling passthrough is opt-in (see --serve-enable-tools).
        # When disabled, we silently drop `tools` so the model treats the
        # request as plain chat — Ask-mode behaviour, regardless of what
        # the client sent.
        tools = req.get("tools") if self.enable_tools else None
        tool_choice = req.get("tool_choice") if self.enable_tools else None
        if not isinstance(tools, list) or not tools:
            tools = None
        use_tools = tools is not None

        # ---- build the message list ofa expects ----
        try:
            ofa_messages = _augment_messages(messages, mode, has_tools=use_tools)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            return self._send_error(500, f"failed to build prompt: {e}", "server_error")

        print(
            f"[ofa-serve] {model_id} ({mode}): {len(messages)} msg(s), "
            f"stream={stream}, tools={'on (fence-block elided)' if use_tools else 'off'}",
            file=sys.stderr,
        )

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if stream:
            return self._handle_stream(
                model_id, completion_id, created, ofa_messages, opt_overrides,
                tools, tool_choice,
            )
        return self._handle_blocking(
            model_id, completion_id, created, ofa_messages, opt_overrides,
            tools, tool_choice,
        )

    # ---- streaming branch ----
    def _handle_stream(self, model_id, completion_id, created, messages, opts,
                       tools=None, tool_choice=None):
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

        finish_reason = "stop"
        try:
            if tools:
                # Tools-on path: stream raw Ollama chunks so we can pull
                # out tool_calls structure.
                ollama_opts = dict(ofa_main.get_model_options())
                ollama_opts.update(opts)
                tool_index = 0
                for chunk in _ollama_chat_raw(messages, tools, tool_choice, ollama_opts):
                    msg = chunk.get("message") or {}
                    text = msg.get("content") or ""
                    if text:
                        self.wfile.write(_sse_chunk(model_id, completion_id, created,
                                                    {"content": text}))
                        self.wfile.flush()
                    tcs = msg.get("tool_calls") or []
                    for tc in tcs:
                        openai_tc = _ollama_tool_call_to_openai(tc, tool_index)
                        tool_index += 1
                        self.wfile.write(_sse_chunk(
                            model_id, completion_id, created,
                            {"tool_calls": [openai_tc]},
                        ))
                        self.wfile.flush()
                    if chunk.get("done"):
                        finish_reason = "tool_calls" if tool_index > 0 else "stop"
                        break
            else:
                # Existing text-only path.
                for chunk in ofa_main.chat_stream(messages, **opts):
                    if not chunk:
                        continue
                    self.wfile.write(_sse_chunk(model_id, completion_id, created,
                                                {"content": chunk}))
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
            self.wfile.write(_sse_chunk(model_id, completion_id, created, {},
                                        finish_reason=finish_reason))
            self.wfile.write(_sse_done())
            self.wfile.flush()
        except OSError:
            return

    # ---- non-streaming branch ----
    def _handle_blocking(self, model_id, completion_id, created, messages, opts,
                         tools=None, tool_choice=None):
        try:
            if tools:
                ollama_opts = dict(ofa_main.get_model_options())
                ollama_opts.update(opts)
                text_parts: list[str] = []
                tool_calls: list[dict] = []
                tool_index = 0
                for chunk in _ollama_chat_raw(messages, tools, tool_choice, ollama_opts):
                    msg = chunk.get("message") or {}
                    if msg.get("content"):
                        text_parts.append(msg["content"])
                    for tc in msg.get("tool_calls") or []:
                        tool_calls.append(_ollama_tool_call_to_openai(tc, tool_index))
                        tool_index += 1
                    if chunk.get("done"):
                        break
                text = "".join(text_parts)
                message_payload = {"role": "assistant", "content": text or None}
                if tool_calls:
                    message_payload["tool_calls"] = tool_calls
                finish_reason = "tool_calls" if tool_calls else "stop"
            else:
                text = "".join(ofa_main.chat_stream(messages, **opts))
                message_payload = {"role": "assistant", "content": text}
                finish_reason = "stop"
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
                "message": message_payload,
                "finish_reason": finish_reason,
            }],
            "usage": {
                # ofa doesn't surface token counts from Ollama in the
                # streaming path; report char counts as a coarse proxy.
                "prompt_tokens": sum(len(m.get("content") or "") for m in messages) // 4,
                "completion_tokens": len(text or "") // 4,
                "total_tokens": (sum(len(m.get("content") or "") for m in messages) + len(text or "")) // 4,
            },
        })


# ---- public entry point --------------------------------------------------

def _read_or_random_port(persist_path: str, port_min: int, port_span: int) -> int:
    """Return a persisted port, or generate+persist a fresh random one.

    Shared core for ``_default_serve_port`` and ``_default_local_port``.
    On read failure or out-of-range value we generate a new port. On
    write failure (e.g. read-only scratch) we still return a port; the
    caller just won't get persistence (no correctness issue, just a
    different port on the next run).

    Random pick uses ``secrets.randbelow`` so concurrent callers in the
    same scratch dir are statistically unlikely to collide.
    """
    if os.path.exists(persist_path):
        try:
            with open(persist_path) as fh:
                port = int(fh.read().strip())
            if 1024 <= port <= 65535:
                return port
        except (OSError, ValueError):
            pass
    port = port_min + secrets.randbelow(port_span)
    try:
        # 0o600 so a shared scratch can't leak this to other users.
        fd = os.open(persist_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, str(port).encode())
        finally:
            os.close(fd)
    except OSError:
        pass
    return port


def _default_serve_port(scratch_dir: str) -> int:
    """Return a stable Kestrel-side port for ``ofa --serve``.

    Persisted to ``$OFA_SCRATCH/.ofa_serve_port`` so the user's BYOK URL
    in VS Code stays valid across ``--serve`` restarts. Kestrel compute
    nodes can host up to 4 users (quarter-node GPU allocations), so we
    pick from a 10000-port range to keep the collision probability
    near-zero (~0.04% for 4 users on a node).
    """
    return _read_or_random_port(
        os.path.join(scratch_dir, ".ofa_serve_port"),
        port_min=40000,
        port_span=10000,
    )


def _default_local_port(scratch_dir: str) -> int:
    """Return a stable laptop-side port suggestion.

    First call generates a random port in the IANA dynamic/private range
    (well clear of 11434/11435/11436 which VS Code Remote-SSH likes to
    auto-forward) and persists it to ``$OFA_SCRATCH/.ofa_serve_local_port``.
    Subsequent calls return the same port so the user's BYOK config URL
    in VS Code does not have to change between ``ofa --serve`` runs.
    """
    return _read_or_random_port(
        os.path.join(scratch_dir, ".ofa_serve_local_port"),
        port_min=49200,
        port_span=15000,
    )


def serve(host: str = "0.0.0.0", port: int | None = None,
          api_key_file: str | None = None,
          no_auth: bool = False,
          local_port: int | None = None,
          enable_tools: bool = False) -> None:
    """Start the BYOK server. Blocks until Ctrl+C.

    Parameters
    ----------
    host:
        Address to bind. Default ``0.0.0.0`` so that an ``ssh -L`` through
        Kestrel's login node can reach the compute-node socket
        (``127.0.0.1`` is unreachable across that hop). The bearer token
        keeps this safe on Kestrel's internal network. Pass ``127.0.0.1``
        only when client + server run on the same machine.
    port:
        TCP port on the *server* side. ``None`` (default) uses a
        per-user-stable random port in 40000-49999 (persisted to
        ``$OFA_SCRATCH/.ofa_serve_port`` so the BYOK URL stays valid
        across restarts). Kestrel compute nodes can host up to 4 users
        (quarter-node GPU allocations), so picking from a 10000-port
        range keeps the collision probability near-zero. Pass ``0`` to
        let the OS pick (different port each restart) or a specific
        integer to pin.
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
    enable_tools:
        Forward OpenAI-format ``tools``/``tool_choice`` from incoming
        requests to Ollama and translate ``tool_calls`` responses back
        to OpenAI SSE format. Lets VS Code's Agent mode chain file
        edits / terminal commands through one approval gate instead of
        click-per-block. Off by default because local 31B models can
        emit malformed JSON for VS Code's complex tool schemas.
    """
    global ofa_main
    import ofa_main as _ofa_main  # noqa: PLC0415 — deliberate lazy import
    ofa_main = _ofa_main

    ofa_main.ensure_ollama_running()
    print("[ofa-serve] loading RAG index…", file=sys.stderr)
    ofa_main._init_rag()
    print("[ofa-serve] RAG ready.", file=sys.stderr)

    if port is None:
        port = _default_serve_port(ofa_main.OFA_SCRATCH)

    if no_auth:
        token = ""
        if host == "0.0.0.0":
            print(
                "[ofa-serve] WARNING: --serve-no-auth combined with host=0.0.0.0 means "
                "ANY USER on Kestrel's internal network can use your GPU allocation. "
                "Use only for short debugging sessions; prefer auth-on for normal use.",
                file=sys.stderr,
            )
        else:
            print("[ofa-serve] WARNING: started with --no-auth, all requests accepted",
                  file=sys.stderr)
    else:
        if api_key_file is None:
            api_key_file = os.path.join(ofa_main.OFA_SCRATCH, ".ofa_api_key")
        token = load_or_create_api_key(api_key_file)
        print(f"[ofa-serve] auth token in {api_key_file} (chmod 600)", file=sys.stderr)

    _Handler.api_key = token
    _Handler.enable_tools = enable_tools
    if enable_tools:
        print(
            "[ofa-serve] tool_calls passthrough is ENABLED (experimental). "
            "VS Code Agent mode requests will see real `tool_calls` from "
            "the model when it emits them.",
            file=sys.stderr,
        )
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
    print(_c(" 🌵 CONNECT FROM YOUR LAPTOP ", "bold", "cyan"), file=sys.stderr)
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
