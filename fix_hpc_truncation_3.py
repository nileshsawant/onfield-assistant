import re

file_path = "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py"
with open(file_path, "r") as f:
    text = f.read()

# Fix the specific bounds loop in the bash processor 
old_block = """                else:
                    out_str += captured_text
                    print(captured_text, end="")
                    
                all_outputs.append(out_str)"""
new_block = """                else:
                    out_str += captured_text
                    print(captured_text, end="")
                
                # Hard limit character length to prevent context explosion on monolithic lines
                if len(out_str) > 3000:
                    out_str = out_str[:1500] + "\\n...[OUTPUT TRUNCATED]...\\n" + out_str[-1500:]
                all_outputs.append(out_str)"""

if old_block in text:
    text = text.replace(old_block, new_block)
    with open(file_path, "w") as f:
        f.write(text)
    print("Patch applied successfully.")
else:
    print("Could not find the target block.")

