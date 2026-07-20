import re

file_path = "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py"
with open(file_path, "r") as f:
    text = f.read()

# Make the backend detect if the user sends Ctrl-C during AI generation to break the loop softly 
# currently ^C exits the whole python wrapper.
import_signal = """import json
import os"""

import_signal_new = """import json
import os
import signal"""

if "import signal" not in text:
    text = text.replace(import_signal, import_signal_new)

# Catch KeyboardInterrupt gracefully inside chat_stream where Ollama requests are generated
chat_stream_old = """    for chunk in ollama.chat(model="gemma2:27b", messages=messages, stream=True, options=opts):
        # We manually manage the space/newline logic for better format matching
        if "\\n" in chunk['message']['content']:
            sys.stdout.write(chunk['message']['content'])
        else:
            sys.stdout.write(chunk['message']['content'])
        sys.stdout.flush()
        response_text += chunk['message']['content']"""

chat_stream_new = """    try:
        for chunk in ollama.chat(model="gemma2:27b", messages=messages, stream=True, options=opts):
            # We manually manage the space/newline logic for better format matching
            if "\\n" in chunk['message']['content']:
                sys.stdout.write(chunk['message']['content'])
            else:
                sys.stdout.write(chunk['message']['content'])
            sys.stdout.flush()
            response_text += chunk['message']['content']
    except KeyboardInterrupt:
        print("\\n[AI generation interrupted by user (Ctrl+C)]", file=sys.stderr)
        # return what we managed to generate so far to allow execution/parsing to keep working
        pass"""

if "except KeyboardInterrupt:" not in text:
    text = text.replace(chat_stream_old, chat_stream_new)

# In the interactive_mode loop, also prevent Ctrl+C from killing the app
interactive_old = """        try:
            user_input = input("> ").strip()
            if not user_input:
                continue
        except (EOFError, KeyboardInterrupt):
            print("\\nGoodbye.")
            break"""

interactive_new = """        try:
            user_input = input("> ").strip()
            if not user_input:
                continue
        except KeyboardInterrupt:
            print("\\n(Ctrl+C pressed. Type 'quit' to exit safely.)", file=sys.stderr)
            continue
        except EOFError:
            print("\\nGoodbye.")
            break"""

text = text.replace(interactive_old, interactive_new)

# One more fix: The truncation limit is too loose for compilation logs! Reduce compilation output explicitly
truncation_old = """                lines = captured_text.split('\\n')
                if len(lines) > 200:
                    truncated = "\\n".join(lines[:100]) + "\\n... (output truncated, " + str(len(lines) - 200) + " lines omitted) ...\\n" + "\\n".join(lines[-100:])
                    out_str += truncated
                    print(truncated)
                else:"""
truncation_new = """                lines = captured_text.split('\\n')
                if len(lines) > 200:
                    truncated = "\\n".join(lines[:30]) + "\\n... (output truncated, " + str(len(lines) - 60) + " lines omitted) ...\\n" + "\\n".join(lines[-30:])
                    out_str += truncated
                    print(truncated)
                else:"""
                
text = text.replace(truncation_old, truncation_new)

# NEW FIX: The output string we append to all_outputs MUST be strictly bounded to prevent the AI from losing track.
bounds_old = """                all_outputs.append(out_str)"""
bounds_new = """                # Ensure we strictly limit what goes to the LLM to 3000 chars max
                if len(out_str) > 3000:
                     out_str = out_str[:1500] + "\\n...[OUTPUT TRUNCATED]...\\n" + out_str[-1500:]
                all_outputs.append(out_str)"""
if "OUTPUT TRUNCATED" not in text:
    text = text.replace(bounds_old, bounds_new)

with open(file_path, "w") as f:
    f.write(text)

print("Updated script gracefully handle Ctrl-C and clamped output limits.")
