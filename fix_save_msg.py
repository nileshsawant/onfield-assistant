file_path = "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py"
with open(file_path, "r") as f:
    text = f.read()

old_str = "Type 'quit' to exit, 'save <dir>' to save last response (OpenFOAM mode only)."
new_str = "Type 'quit' to exit, 'save <dir>' to save last response."

text = text.replace(old_str, new_str)

with open(file_path, "w") as f:
    f.write(text)

print("Updated print string successfully.")
