#!/usr/bin/env python3
"""Demo: fit a noisy damped sinusoid + harmonic, ask ofa to critique
the fit and suggest better initial guesses, then refit.

Illustrates the two patterns you'll want in real sim code:

  1. Free-form ask (prose reply the human reads)
  2. Structured ask with a JSON schema (machine-parseable numbers back
     into the workflow)

And two `ofa_client` API entry points:

  - `Session(...)`: multi-turn, client-side history. Turn 2 can refer to
    "the plot above" and it resolves because the image + turn-1 reply
    are still in `sess.messages` on the client.
  - `extract_json(text)`: fenced-block-first fallback ladder for
    parsing structured LLM output.

Prereqs:
  - numpy, scipy, matplotlib in your Python environment.
  - `module load assistant` so `ofa_client` is on PYTHONPATH.
  - `ofa --serve` running on the same node so the client auto-detects
    the URL + token from $OFA_SCRATCH.

Run (from your own working directory, wherever you'd like the two PNG
outputs to land):

  cp $OFA_ROOT/examples/fit_and_ask.py .
  python3 fit_and_ask.py

Or in-place:

  cd $OFA_ROOT/examples && python3 fit_and_ask.py

Expected output on a successful run:

  RMS: 0.5370 -> 0.11xx  (IMPROVED, delta -0.42)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — no X needed
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from ofa_client import ask, Session


# ---------------------------------------------------------------------------
# 1. Ground truth + noisy data
# ---------------------------------------------------------------------------

def true_model(x, A, b, omega, phi):
    """Damped sinusoid + a harmonic at 2ω.

    The extra ``0.5 * cos(2*omega*x)`` term creates a spectrum with a
    strong second peak, which is exactly the kind of setup where
    Levenberg-Marquardt gets trapped in a local minimum — the optimizer
    happily locks onto the harmonic (ω ≈ 6) if you start it there, and
    the resulting fit looks vaguely right but is quantitatively wrong.
    """
    return A * np.exp(-b * x) * (np.cos(omega * x + phi) + 0.5 * np.cos(2 * omega * x))


TRUE_PARAMS = (2.0, 0.30, 3.0, 0.5)   # (A, b, omega, phi)
rng = np.random.default_rng(42)
x = np.linspace(0.0, 10.0, 120)
y = true_model(x, *TRUE_PARAMS) + rng.normal(scale=0.15, size=x.size)


# ---------------------------------------------------------------------------
# 2. Fit + plot helpers
# ---------------------------------------------------------------------------

def do_fit(p0, maxfev=1000):
    """Return a dict summarising the fit; never raises."""
    try:
        popt, _ = curve_fit(true_model, x, y, p0=p0, maxfev=maxfev)
        rms = float(np.sqrt(np.mean((y - true_model(x, *popt)) ** 2)))
        return {"popt": popt.tolist(), "rms": rms, "converged": True}
    except Exception as e:
        return {"popt": None, "rms": None, "converged": False, "error": str(e)}


def save_plot(popt, path, title):
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, "o", markersize=3, alpha=0.6, label="noisy data")
    plt.plot(x, true_model(x, *TRUE_PARAMS), "--", alpha=0.4, label="ground truth")
    if popt is not None:
        label = (f"fit: A={popt[0]:.2f}, b={popt[1]:.2f}, "
                 f"\u03c9={popt[2]:.2f}, \u03c6={popt[3]:.2f}")
        plt.plot(x, true_model(x, *popt), "-", linewidth=2, label=label)
    plt.xlabel("x"); plt.ylabel("y")
    plt.title(title); plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def fmt_summary(label, p0, result):
    return (
        f"{label}\n"
        f"  model:         y = A * exp(-b*x) * cos(omega*x + phi)\n"
        f"  initial guess: A={p0[0]}, b={p0[1]}, omega={p0[2]}, phi={p0[3]}\n"
        f"  fitted:        {result['popt']}\n"
        f"  residual RMS:  {result['rms']}\n"
        f"  converged:     {result['converged']}\n"
        f"  ground truth:  A={TRUE_PARAMS[0]}, b={TRUE_PARAMS[1]}, "
        f"omega={TRUE_PARAMS[2]}, phi={TRUE_PARAMS[3]}"
    )


# ---------------------------------------------------------------------------
# 3. First fit with a deliberately bad initial guess
#
# For the fundamental+harmonic model, starting at (or near) the harmonic
# frequency 2ω=6 tends to lock the optimizer onto the wrong peak. This
# is a *real* failure mode users see with multi-mode signals; the fit
# converges (converged=True) but the parameters are wrong and the
# residual is much worse than the noise floor.
# ---------------------------------------------------------------------------

p0_bad = [1.0, 0.01, 6.0, 0.0]
first = do_fit(p0_bad, maxfev=1000)
plot_before = "fit_before.png"
save_plot(first["popt"], plot_before, "Fit BEFORE (bad initial guess)")

summary_first = fmt_summary("=== initial fit (bad guess) ===", p0_bad, first)
print(summary_first)
print(f"plot: {plot_before}\n")


# ---------------------------------------------------------------------------
# 4. Open a Session and ask two follow-up questions in it.
#
# Using Session (instead of two independent ask() calls) means the
# second turn's user message no longer has to re-attach the plot or the
# fit summary — the model sees them in the accumulated history. That
# saves payload and, more importantly, lets the model's second reply
# build on its own first reply ("you said A should be around 2, so ...").
# ---------------------------------------------------------------------------

sess = Session(model="ofa-code", timeout=120)

# --- turn 1: free-form critique (natural language reply) ---
print("--- turn 1: asking ofa to critique the fit ---")
critique = sess.ask(
    "Look at the fit in this plot. I fit a damped sinusoid "
    "y = A * exp(-b*x) * cos(omega*x + phi) to noisy data. "
    "Is the fit good? If not, what's the main problem and roughly what "
    "should the parameters be?",
    image=plot_before,
    context=summary_first,
)
print(critique + "\n")

# --- turn 2: structured suggestion (JSON we can parse) ---
# No image/context re-attached — the Session's history carries turn 1's
# user message (with the image) AND turn 1's assistant critique. The
# second prompt can refer to "the plot above" and it will resolve.
print("--- turn 2: asking ofa for improved initial guesses (JSON) ---")
prompt = (
    "Based on the plot and your critique above, propose better initial "
    "guesses for scipy.optimize.curve_fit. The four parameters are "
    "(A, b, omega, phi) in that order.\n\n"
    "Reply with ONLY a fenced JSON code block matching this exact schema "
    "and NOTHING ELSE (no prose before or after):\n"
    "```json\n"
    "{\n"
    '  "p0":     [<A>, <b>, <omega>, <phi>],\n'
    '  "maxfev": <integer, e.g. 5000>,\n'
    '  "notes":  "<one sentence explaining the choice>"\n'
    "}\n"
    "```"
)
raw = sess.ask(prompt)
print("raw model reply:\n" + raw + "\n")
print(f"[session state] {len(sess.messages)} messages held client-side "
      f"({sess})\n")


# ---------------------------------------------------------------------------
# 6. Extract the JSON — tiered fallback so we're robust to the LLM adding
#    prose or dropping the fence
# ---------------------------------------------------------------------------

def extract_json(text: str):
    """Try increasingly permissive strategies to find a JSON object in *text*.

    Order:
      1. ```json ... ```  (the format we asked for)
      2. ``` ... ```      (fenced without language tag)
      3. balanced {...}   (last resort)
    Returns the parsed dict, or None if nothing looks like JSON.
    """
    patterns = [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    # Last resort: greediest {...} that json.loads accepts.
    for m in re.finditer(r"\{.*?\}", text, re.DOTALL):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return None


suggestion = extract_json(raw)
if suggestion is None:
    print("[!] Could not extract JSON from the reply. Refit skipped.")
    sys.exit(1)

print(f"parsed: {suggestion}\n")


# ---------------------------------------------------------------------------
# 7. Validate the suggestion, refit, compare
# ---------------------------------------------------------------------------

p0_new = suggestion.get("p0")
if not (isinstance(p0_new, list) and len(p0_new) == 4
        and all(isinstance(v, (int, float)) for v in p0_new)):
    print(f"[!] p0 must be a 4-element list of numbers, got {p0_new!r}.")
    sys.exit(1)
maxfev = int(suggestion.get("maxfev", 5000))

print(f"--- refitting with p0={p0_new}, maxfev={maxfev} ---")
second = do_fit(p0_new, maxfev=maxfev)
plot_after = "fit_after.png"
save_plot(second["popt"], plot_after, "Fit AFTER (ofa-suggested guess)")

print(fmt_summary("=== refit result ===", p0_new, second))
print(f"plot: {plot_after}\n")

# --- one-line before/after ---
if first["rms"] is not None and second["rms"] is not None:
    delta = first["rms"] - second["rms"]
    tol = 1e-6           # floating-point noise floor — treat as SAME
    verdict = (
        "IMPROVED" if delta >  tol
        else "WORSE"    if delta < -tol
        else "SAME"
    )
    print(f"RMS: {first['rms']:.4f} -> {second['rms']:.4f}  ({verdict}, "
          f"delta {delta:+.4f})")
else:
    print("RMS comparison unavailable (one of the fits failed)")
