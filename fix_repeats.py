with open("/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py", "r") as f:
    text = f.read()

text = text.replace('"repeat_penalty": 1.0', '"repeat_penalty": 1.15')
text = text.replace('"temperature": 0.6', '"temperature": 0.1')

with open("/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py", "w") as f:
    f.write(text)

