"""ofa Python client — call `ofa --serve` from user code.

Zero external dependencies. Pure Python stdlib. Copy this file into any
Python environment on Kestrel (or symlink from ``$OFA_ROOT/src/ofa_client.py``)
and use::

    from ofa_client import ask, Session

    # One-shot, stateless
    text = ask("summarise this plot", image="output/step_0100.png")

    # Multi-turn, client-side history
    sess = Session(model="ofa-code")
    sess.ask("what turbulence model for cavity flow at Re=1e4?")
    sess.ask("show me a controlDict for that")   # sees the previous turn

Auto-detects the running ofa --serve via, in order:

  1. Explicit ``url=`` / ``token=`` kwargs.
  2. ``$OFA_BYOK_URL`` / ``$OFA_BYOK_TOKEN`` environment variables.
  3. ``$OFA_SCRATCH/.ofa_serve_port`` and ``$OFA_SCRATCH/.ofa_api_key``.
  4. ``/scratch/$USER/.ofa_serve_port`` and ``/scratch/$USER/.ofa_api_key``.

Raises RuntimeError with a clear message if no server can be located.
Import works in any Python 3.8+ interpreter regardless of what other
packages are installed — the client's only imports are stdlib.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Optional, Union

__all__ = ["ask", "Session", "MODEL_IDS"]

_DEFAULT_MODEL = "ofa-code"
_DEFAULT_TIMEOUT = 120.0
_FILE_TAIL_BYTES = 32 * 1024  # default cap when file= is passed

MODEL_IDS = (
    "ofa-openfoam", "ofa-hpc", "ofa-code", "ofa-amrex", "ofa-reframe",
)

PathLike = Union[str, os.PathLike]


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------

def _scratch_candidates() -> Iterable[Path]:
    """Yield scratch directories to search, most-preferred first."""
    if os.environ.get("OFA_SCRATCH"):
        yield Path(os.environ["OFA_SCRATCH"])
    user = os.environ.get("USER")
    if user:
        p = Path(f"/scratch/{user}")
        if p.is_dir():
            yield p


def _resolve_url(explicit: Optional[str]) -> str:
    if explicit:
        return explicit.rstrip("/")
    env = os.environ.get("OFA_BYOK_URL")
    if env:
        return env.rstrip("/")
    for scratch in _scratch_candidates():
        port_file = scratch / ".ofa_serve_port"
        if port_file.is_file():
            try:
                port = int(port_file.read_text().strip())
                return f"http://localhost:{port}"
            except (OSError, ValueError):
                continue
    raise RuntimeError(
        "no ofa server detected. Set OFA_BYOK_URL, or start `ofa --serve` "
        "on the same node so its port persists to $OFA_SCRATCH/.ofa_serve_port."
    )


def _resolve_token(explicit: Optional[str]) -> str:
    if explicit is not None:
        return explicit.strip()
    env = os.environ.get("OFA_BYOK_TOKEN")
    if env:
        return env.strip()
    for scratch in _scratch_candidates():
        key_file = scratch / ".ofa_api_key"
        if key_file.is_file():
            try:
                return key_file.read_text().strip()
            except OSError:
                continue
    # Empty token is valid: the server accepts it in --serve-no-auth mode.
    return ""


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------

def _encode_image(path: PathLike) -> str:
    """Read a local image file and return an OpenAI data-URL base64 encoding."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"image not found: {p}")
    mime, _ = mimetypes.guess_type(p.name)
    if mime is None:
        mime = "image/png"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _read_file_snippet(path: PathLike, full_file: bool = False,
                       max_bytes: int = _FILE_TAIL_BYTES) -> str:
    """Read a text file and return its content.

    By default only the *last* ``max_bytes`` bytes are returned — useful
    for huge solver logs where the tail is the interesting part. Pass
    ``full_file=True`` to read the whole file.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"file not found: {p}")
    if full_file:
        return p.read_text(errors="replace")
    size = p.stat().st_size
    if size <= max_bytes:
        return p.read_text(errors="replace")
    with p.open("rb") as f:
        f.seek(-max_bytes, os.SEEK_END)
        blob = f.read()
    text = blob.decode("utf-8", errors="replace")
    # Trim a possibly-broken first line at the seek boundary.
    if "\n" in text:
        text = text.split("\n", 1)[1]
    return f"[tail of {p.name}: last {len(blob)} bytes of {size}]\n{text}"


def _build_content(
    prompt: str,
    *,
    context: Optional[str],
    file: Optional[PathLike],
    files: Optional[list],
    image: Optional[PathLike],
    images: Optional[list],
    full_file: bool,
):
    """Assemble the OpenAI ``content`` field.

    Returns a plain string when there are no images (server accepts that
    shape), or a multimodal array of ``{type, ...}`` parts when at least
    one image is attached.
    """
    text_parts: list[str] = []
    if context:
        text_parts.append(f"Additional context:\n{context.strip()}")

    file_paths: list = []
    if file is not None:
        file_paths.append(file)
    if files:
        file_paths.extend(files)
    for fp in file_paths:
        body = _read_file_snippet(fp, full_file=full_file)
        text_parts.append(f"File `{Path(fp).name}`:\n```\n{body}\n```")

    text_parts.append(f"Question: {prompt}")
    combined_text = "\n\n".join(text_parts)

    image_paths: list = []
    if image is not None:
        image_paths.append(image)
    if images:
        image_paths.extend(images)

    if not image_paths:
        return combined_text

    parts = [{"type": "text", "text": combined_text}]
    for ip in image_paths:
        parts.append({
            "type": "image_url",
            "image_url": {"url": _encode_image(ip)},
        })
    return parts


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------

def _post(url: str, token: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}" if token else "",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            body_txt = e.read()[:500].decode(errors="replace")
        except Exception:
            body_txt = ""
        raise RuntimeError(
            f"ofa server returned HTTP {e.code}: {body_txt}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"ofa server unreachable at {url}: {e.reason}") from e


def _extract_text(payload: dict) -> str:
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"unexpected response shape: {payload!r}") from e


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask(
    prompt: str,
    *,
    image: Optional[PathLike] = None,
    images: Optional[list] = None,
    context: Optional[str] = None,
    file: Optional[PathLike] = None,
    files: Optional[list] = None,
    model: str = _DEFAULT_MODEL,
    url: Optional[str] = None,
    token: Optional[str] = None,
    timeout: float = _DEFAULT_TIMEOUT,
    full_file: bool = False,
) -> str:
    """Send a one-shot prompt (with optional images and file/text context)
    to ``ofa --serve`` and return the reply text. Stateless — each call
    is a fresh session.

    Parameters
    ----------
    prompt:
        The main question (required).
    image / images:
        Path (or list of paths) to local image files to attach.
    context:
        Inline text context to include verbatim in the user message.
    file / files:
        Path (or list of paths) to local text files whose contents get
        inlined (fenced with the filename). Only the last 32 KB of each
        file is included by default; pass ``full_file=True`` to override.
    model:
        One of ``MODEL_IDS``. Default ``"ofa-code"``.
    url / token / timeout:
        Overrides for auto-detection (see module docstring).
    full_file:
        If True, read entire ``file=`` / ``files=`` contents instead of
        only the last 32 KB.
    """
    if model not in MODEL_IDS:
        raise ValueError(f"unknown model {model!r}; expected one of {MODEL_IDS}")
    content = _build_content(
        prompt,
        context=context, file=file, files=files,
        image=image, images=images, full_file=full_file,
    )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
    }
    payload = _post(_resolve_url(url), _resolve_token(token), body, timeout)
    return _extract_text(payload)


class Session:
    """Multi-turn chat session with client-side history.

    Each ``.ask()`` call appends to the internal ``messages`` list and
    sends the whole thing on the next request, so the model sees prior
    context::

        sess = Session(model="ofa-code")
        sess.ask("what turbulence model for cavity flow?")
        sess.ask("show me a controlDict for that")   # sees turn 1

    Server-side state is deliberately NOT used: keeping the history
    client-side means it survives ``ofa --serve`` restarts, has no
    server memory footprint, and matches the OpenAI protocol shape.
    """

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        full_file: bool = False,
    ):
        if model not in MODEL_IDS:
            raise ValueError(
                f"unknown model {model!r}; expected one of {MODEL_IDS}"
            )
        self.model = model
        self._url = _resolve_url(url)
        self._token = _resolve_token(token)
        self.timeout = timeout
        self.full_file = full_file
        self.messages: list[dict] = []

    def ask(
        self,
        prompt: str,
        *,
        image: Optional[PathLike] = None,
        images: Optional[list] = None,
        context: Optional[str] = None,
        file: Optional[PathLike] = None,
        files: Optional[list] = None,
    ) -> str:
        """Same shape as ``ofa_client.ask``, but appends to and consults
        this session's message history."""
        content = _build_content(
            prompt,
            context=context, file=file, files=files,
            image=image, images=images, full_file=self.full_file,
        )
        self.messages.append({"role": "user", "content": content})
        body = {
            "model": self.model,
            "messages": self.messages,
            "stream": False,
        }
        payload = _post(self._url, self._token, body, self.timeout)
        reply = _extract_text(payload)
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def clear(self) -> None:
        """Forget all prior turns (start a fresh conversation)."""
        self.messages.clear()

    def __repr__(self) -> str:  # pragma: no cover — cosmetic
        return (
            f"Session(model={self.model!r}, url={self._url!r}, "
            f"turns={len(self.messages) // 2})"
        )
