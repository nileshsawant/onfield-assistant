import os
import re

files_to_fix = [
    "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ingest_amrex.py",
    "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py"
]

for fp in files_to_fix:
    with open(fp, "r") as f:
        content = f.read()

    # Change device="cpu" to device="cuda" if available
    if 'device="cpu"' in content:
        content = content.replace('device="cpu"', 'device="cuda"')
        with open(fp, "w") as f:
            f.write(content)
        print(f"Updated {fp} to use CUDA device.")
