import re

file_path = "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py"
with open(file_path, "r") as f:
    text = f.read()

# 1. Add edit_blocks regex
if "edit_blocks =" not in text:
    text = text.replace(
        'write_blocks = re.findall(r"```(?:write)\s+([^\\n]+)\\n(.*?)\\n```", response_text, re.DOTALL)',
        'write_blocks = re.findall(r"```(?:write)\s+([^\\n]+)\\n(.*?)\\n```", response_text, re.DOTALL)\n    edit_blocks = re.findall(r"```(?:edit)\s+([^\\n]+)\\n(.*?)\\n```", response_text, re.DOTALL)'
    )

    text = text.replace(
        'if not bash_blocks and not search_blocks and not fetch_blocks and not read_blocks and not write_blocks:',
        'if not bash_blocks and not search_blocks and not fetch_blocks and not read_blocks and not write_blocks and not edit_blocks:'
    )

# 2. Add the edit blocks execution logic right after write blocks
edit_logic = """
    # Process edit blocks
    for filepath, content in edit_blocks:
        filepath = filepath.strip()
        if not filepath: continue
        print(f"\\n[File Edit Suggested]")
        print(f"File: {filepath}")
        
        # Parse FIND and REPLACE sections
        if "<<FIND>>" in content and "<<REPLACE>>" in content:
            find_str = content.split("<<FIND>>")[1].split("<<REPLACE>>")[0].strip('\\n')
            replace_str = content.split("<<REPLACE>>")[1].strip('\\n')
            
            ans = input("Allow editing this file? [y/N]: ").strip().lower()
            if ans in ('y', 'yes'):
                print("-" * 60)
                try:
                    with open(filepath, 'r') as f:
                        file_data = f.read()
                    
                    if find_str in file_data:
                        file_data = file_data.replace(find_str, replace_str, 1)
                        with open(filepath, 'w') as f:
                            f.write(file_data)
                        out_str = f"\\n--- File Edit Success ---\\nSuccessfully edited {filepath}\\n----------------------------------\\n"
                        print(f"Edited target file successfully.")
                    else:
                        out_str = f"\\n--- File Edit Error ---\\nCould not find the exact <<FIND>> text in {filepath}. The file was not changed.\\n----------------------------------\\n"
                        print(out_str)
                except Exception as e:
                    out_str = f"\\n--- File Edit Error ---\\n{str(e)}\\n----------------------------------\\n"
                    print(out_str)
                all_outputs.append(out_str)
                print("-" * 60)
        else:
            out_str = f"\\n--- File Edit Error ---\\nEdit block missing <<FIND>> and <<REPLACE>> section markers.\\n----------------------------------\\n"
            print(out_str)
            all_outputs.append(out_str)
"""

if "# Process edit blocks" not in text:
    text = text.replace(
        '# Process bash blocks',
        edit_logic.strip() + '\n\n    # Process bash blocks'
    )

with open(file_path, "w") as f:
    f.write(text)

print("Added Edit blocks to ofa_main.py")
