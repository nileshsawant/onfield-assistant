with open("/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py", "r") as f:
    text = f.read()

old_block = """        # Skip executing file contents that look like scripts
        if cmd.startswith("#!/bin/bash") or cmd.startswith("#!/bin/sh") or "#SBATCH" in cmd:
        dangerous = False"""

new_block = """        # Skip executing file contents that look like scripts
        if cmd.startswith("#!/bin/bash") or cmd.startswith("#!/bin/sh") or "#SBATCH" in cmd:
            continue
        dangerous = False"""

if old_block in text:
    print("Replacing indentation...")
    text = text.replace(old_block, new_block)
    with open("/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py", "w") as f:
        f.write(text)
else:
    print("Could not find block.")
