import os
files = [
    "/nopt/nrel/apps/cpu_stack/modules/default/application/openfoam/assistant.lua",
    "/nopt/nrel/apps/gpu_stack/modules/default/application/openfoam/assistant.lua"
]

for file_path in files:
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            text = f.read()

        # Replace literal \n with actual newline
        text = text.replace("\\n    ofa --code", "\n    ofa --code")
        
        with open(file_path, "w") as f:
            f.write(text)
        print(f"Updated {file_path}")
