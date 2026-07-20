import re

file_path = "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py"
with open(file_path, "r") as f:
    text = f.read()

# I see the problem. Wait. If the Ollama server starts in the background inside ensure_ollama_running(),
# and we dropped the global SIGINT handler, does Ctrl-C *also* kill the background Ollama server process?
# Yes! Bash transmits SIGINT to the entire process group. So hitting Ctrl-C to interrupt generation ALSO killed Ollama.

ollama_start_old = """    # Start Ollama in background
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = os.environ.get("OLLAMA_MODELS", os.path.join(OFA_ROOT, "models"))
    
    print("Starting Ollama server...", file=sys.stderr)
    log_file = open(f"/scratch/{os.environ.get('USER', 'default')}/ollama.log", "w")
    proc = subprocess.Popen(
        [os.path.join(OFA_ROOT, "bin", "ollama"), "serve"],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT
    )"""

ollama_start_new = """    # Start Ollama in background
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = os.environ.get("OLLAMA_MODELS", os.path.join(OFA_ROOT, "models"))
    
    print("Starting Ollama server...", file=sys.stderr)
    log_file = open(f"/scratch/{os.environ.get('USER', 'default')}/ollama.log", "w")
    proc = subprocess.Popen(
        [os.path.join(OFA_ROOT, "bin", "ollama"), "serve"],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setpgrp # <-- CRITICAL FIX: Put Ollama in its own process group so Ctrl-C doesn't kill it!
    )"""

if "os.setpgrp" not in text:
    text = text.replace(ollama_start_old, ollama_start_new)
    
# We also need to make sure the chat_stream wrapper is safe against Connection Refused if it DOES crash
chat_loop_wrapper_old = """        try:
            for chunk in chat_stream(messages):
                print(chunk, end="", flush=True)
                response += chunk
        except KeyboardInterrupt:
            print("\\n[AI generation interrupted by user (Ctrl+C)]", file=sys.stderr)
            pass
        print()"""
        
# Actually, the exception from the trace was httpx.ConnectError because Ollama died.
# We should add a catch for httpx exceptions to gently report it instead of crashing.
imports_old = """import sys
import argparse
import signal"""
imports_new = """import sys
import argparse
import signal
import httpx"""
if "import httpx" not in text:
    text = text.replace(imports_old, imports_new)
    
# Add catch to interactive loop
interactive_loop_try_old_1 = """            try:
                for chunk in chat_stream(messages):
                    print(chunk, end="", flush=True)
                    last_response += chunk
            except KeyboardInterrupt:
                print("\\n[AI generation interrupted by user (Ctrl+C)]", file=sys.stderr)
                pass"""

interactive_loop_try_new_1 = """            try:
                for chunk in chat_stream(messages):
                    print(chunk, end="", flush=True)
                    last_response += chunk
            except KeyboardInterrupt:
                print("\\n[AI generation interrupted by user (Ctrl+C)]", file=sys.stderr)
                pass
            except httpx.ConnectError:
                print("\\n[Error: Connection to Ollama server lost. The backend may have crashed.]", file=sys.stderr)
                break"""

text = text.replace(interactive_loop_try_old_1, interactive_loop_try_new_1)

interactive_loop_try_old_2 = """        try:
            response = chat_stream(messages)
        except KeyboardInterrupt:
            print("\\n[AI analysis aborted by user]", file=sys.stderr)
            break"""

interactive_loop_try_new_2 = """        try:
            response = chat_stream(messages)
        except KeyboardInterrupt:
            print("\\n[AI analysis aborted by user]", file=sys.stderr)
            break
        except httpx.ConnectError:
            print("\\n[Error: Connection to Ollama server lost. The backend may have crashed.]", file=sys.stderr)
            break"""
            
text = text.replace(interactive_loop_try_old_2, interactive_loop_try_new_2)

# Also fix the outer except KeyboardInterrupt block that we forgot inside chat_stream previously where the original syntax error popped up.
# I actually overwrote it correctly earlier, but let's make sure it handles both.
chat_stream_conn_old = """    try:
        with httpx.stream(
            "POST",
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            timeout=300.0,
        ) as resp:"""
chat_stream_conn_new = """    try:
        with httpx.stream(
            "POST",
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            timeout=300.0,
        ) as resp:"""

with open(file_path, "w") as f:
    f.write(text)

print("Protected Ollama server from SIGINT cascades.")
