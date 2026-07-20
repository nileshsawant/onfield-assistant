with open("/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py", "r") as f:
    text = f.read()

# Make the web scraper provide more than just the first 2500 characters
text = text.replace('cleaned[:2500]', 'cleaned[:16000]')

with open("/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py", "w") as f:
    f.write(text)

