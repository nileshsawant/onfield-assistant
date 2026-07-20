import re

file_path = "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py"
with open(file_path, "r") as f:
    text = f.read()

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
print("Fix successfully applied.")
