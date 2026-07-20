import re

file_path = "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py"
with open(file_path, "r") as f:
    text = f.read()

# Ah! My patch for `fix_ollama_ctrlc.py` failed to apply correctly earlier because the Popen format was different than I assumed. It sends stdout to DEVNULL, not ollama.log.

old_popen = """    global _ollama_proc
    print("Starting Ollama server...", file=sys.stderr)
    _ollama_proc = subprocess.Popen(
        [OLLAMA_BIN, "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )"""

new_popen = """    global _ollama_proc
    print("Starting Ollama server...", file=sys.stderr)
    _ollama_proc = subprocess.Popen(
        [OLLAMA_BIN, "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setpgrp # <-- Critical: detach Ollama from Bash process group so Ctrl-C doesn't kill it
    )"""

if "preexec_fn=os.setpgrp" not in text:
    text = text.replace(old_popen, new_popen)
    with open(file_path, "w") as f:
        f.write(text)
    print("Applied setpgrp fix correctly.")
else:
    print("Already applied.")

