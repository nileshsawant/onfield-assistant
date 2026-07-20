import re

file_path = "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py"
with open(file_path, "r") as f:
    text = f.read()

# The original regex required strict newlines: ```read\npath\n```
# The LLM sometimes outputs inline blocks: ```read path``` or ```read path\n```
# Let's make search, fetch, and read blocks much more lenient by accepting spaces or newlines.

old_regexes = """    search_blocks = re.findall(r"```(?:search)\\n(.*?)\\n```", response_text, re.DOTALL)
    fetch_blocks = re.findall(r"```(?:fetch)\\n(.*?)\\n```", response_text, re.DOTALL)
    read_blocks = re.findall(r"```(?:read)\\n(.*?)\\n```", response_text, re.DOTALL)"""
    
new_regexes = """    search_blocks = re.findall(r"```(?:search)(?:\\s+|\\n)(.*?)\\s*```", response_text, re.DOTALL)
    fetch_blocks = re.findall(r"```(?:fetch)(?:\\s+|\\n)(.*?)\\s*```", response_text, re.DOTALL)
    read_blocks = re.findall(r"```(?:read)(?:\\s+|\\n)(.*?)\\s*```", response_text, re.DOTALL)"""

text = text.replace(old_regexes, new_regexes)

with open(file_path, "w") as f:
    f.write(text)

print("Updated parser logic to be more lenient on whitespaces.")
