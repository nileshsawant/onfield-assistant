with open("/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/prompts/openfoam.txt", "r") as f:
    text = f.read()

# Add bash script command instructions + weather bypassing for openfoam.txt
if "If the user asks you to run or execute a system command" not in text:
    bash_instr = "\n\nIf the user asks you to run or execute a system command (like listing files, checking memory, navigating directories), output the exact bash command inside a ```bash ... ``` code block. The system will detect this block and run it on their behalf after asking for confirmation.\n\n"
    text = text.replace("If you are asked a question that you don't confidently know", bash_instr + "If you are asked a question that you don't confidently know")

if "wttr.in/CITY" not in text:
    weather_instr = """
If the user asks for the weather, do not use the search tool. Instead, execute this exact system command replacing CITY with their requested city (must not contain spaces, use +):
```bash
curl -s "wttr.in/CITY?T"
```
"""
    text = text.replace("Self-Knowledge:", weather_instr + "\nSelf-Knowledge:")

with open("/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/prompts/openfoam.txt", "w") as f:
    f.write(text)


with open("/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/prompts/cpp.txt", "r") as f:
    cpp_text = f.read()

# Make cpp.txt explicitly aware of Kestrel
old_intro = "You are an expert OpenFOAM C++ developer on the NREL Kestrel system."
new_intro = "You are an expert OpenFOAM C++ developer currently running natively on the NREL Kestrel supercomputer."
cpp_text = cpp_text.replace(old_intro, new_intro)

if "If the user asks you to run or execute a system command" not in cpp_text:
    cpp_text = cpp_text.replace("If you are asked a question that you don't confidently know", bash_instr + "If you are asked a question that you don't confidently know")

if "wttr.in/CITY" not in cpp_text:
    cpp_text = cpp_text.replace("Self-Knowledge:", weather_instr + "\nSelf-Knowledge:")

with open("/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/prompts/cpp.txt", "w") as f:
    f.write(cpp_text)

