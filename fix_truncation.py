import re

with open("/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py", "r") as f:
    text = f.read()

old_cmd_append = """            if cmd_out:
                messages.append({"role": "user", "content": f"Output from executed commands:\\n```text\\n{cmd_out}\\n```\\nPlease continue to assist the user using this information."})
                save_session(messages)
                print("\\n[AI is analyzing the output...]", flush=True)"""

# Add a limit to cmd_out length so the LLM doesn't crash on extremely large terminal outputs
new_cmd_append = """            if cmd_out:
                # If command output is extremely large, truncate it to prevent LLM context collapse
                if len(cmd_out) > 8000:
                    truncated = cmd_out[:4000] + "\\n...[OUTPUT TRUNCATED]...\\n" + cmd_out[-4000:]
                else:
                    truncated = cmd_out
                messages.append({"role": "user", "content": f"Output from executed commands:\\n```text\\n{truncated}\\n```\\nPlease continue to assist the user using this information."})
                save_session(messages)
                print("\\n[AI is analyzing the output...]", flush=True)"""

if old_cmd_append in text:
    text = text.replace(old_cmd_append, new_cmd_append)
    with open("/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py", "w") as f:
        f.write(text)
    print("Truncation applied.")
else:
    print("Failed to find cmd append logic.")
