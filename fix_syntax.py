import sys

file_path = "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py"
with open(file_path, "r") as f:
    text = f.read()

bad_str = '            prompt += "\\n\\n--- USER PREFERENCES ---\\n" + prefs'
bad_str_multiline = '            prompt += "\\n\\n--- USER PREFERENCES ---\\n\\n" + prefs'
bad_str_actual = '            prompt += "\\n\\n--- USER PREFERENCES ---\\n" + prefs'

# Notice in the sed output it was a raw newline inside a string.
# We will just rewrite the entire load_system_prompt logic safely.

import re
text = re.sub(
    r'prompt \+= "\n\n--- USER PREFERENCES ---\n" \+ prefs',
    r'prompt += "\\n\\n--- USER PREFERENCES ---\\n" + prefs',
    text
)

with open(file_path, "w") as f:
    f.write(text)
print("Fix applied")
