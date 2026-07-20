import re

file_path = "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py"
with open(file_path, "r") as f:
    text = f.read()

new_logic = """def load_system_prompt(prompt_type="openfoam"):
    import os
    if prompt_type == "code":
        with open(CODE_PROMPT_PATH) as f: prompt = f.read().strip()
    elif prompt_type == "hpc":
        with open(HPC_PROMPT_PATH) as f: prompt = f.read().strip()
    elif prompt_type == "amrex":
        with open(os.path.join(OFA_ROOT, "prompts", "amrex.txt")) as f: prompt = f.read().strip()
    else:
        with open(OPENFOAM_PROMPT_PATH) as f: prompt = f.read().strip()

    prefs_file = f"/scratch/{os.environ.get('USER', 'default')}/.ofa_prefs.txt"
    if os.path.exists(prefs_file):
        with open(prefs_file) as f:
            prefs = f.read().strip()
        if prefs:
            prompt += "\\n\\n--- USER PREFERENCES ---\\n" + prefs
    return prompt
"""

if "def load_system_prompt(prompt_type" not in text:
    old_load = """def load_system_prompt():
    with open(OPENFOAM_PROMPT_PATH) as f:
        prompt = f.read().strip()
    prefs_file = f"/scratch/{os.environ.get('USER', 'default')}/.ofa_prefs.txt"
    if os.path.exists(prefs_file):
        with open(prefs_file) as f:
            prefs = f.read().strip()
        if prefs:
            prompt += "\\n\\nUser Preferences:\\n" + prefs
    return prompt"""
    
    if old_load in text:
        text = text.replace(old_load, new_logic)
        
        # Replace hardcoded prompt assignments in interactive_mode and hpc_single_query
        text = re.sub(
            r'system_prompt = AMREX_SYSTEM_PROMPT if amrex_mode else \(CODE_SYSTEM_PROMPT if code_mode else \(HPC_SYSTEM_PROMPT if hpc_mode else load_system_prompt\(\)\)\)',
            r'system_prompt = load_system_prompt("amrex") if amrex_mode else (load_system_prompt("code") if code_mode else (load_system_prompt("hpc") if hpc_mode else load_system_prompt("openfoam")))',
            text
        )
        
        text = re.sub(
            r'messages\[0\]\["content"\] = AMREX_SYSTEM_PROMPT if amrex_mode else \(CODE_SYSTEM_PROMPT if code_mode else HPC_SYSTEM_PROMPT\)',
            r'messages[0]["content"] = load_system_prompt("amrex") if amrex_mode else (load_system_prompt("code") if code_mode else load_system_prompt("hpc"))',
            text
        )
        
        text = re.sub(
            r'messages = \[\{"role": "system", "content": AMREX_SYSTEM_PROMPT if amrex_mode else \(CODE_SYSTEM_PROMPT if code_mode else HPC_SYSTEM_PROMPT\)\}\]',
            r'messages = [{"role": "system", "content": load_system_prompt("amrex") if amrex_mode else (load_system_prompt("code") if code_mode else load_system_prompt("hpc"))}]',
            text
        )
        
        with open(file_path, "w") as f:
            f.write(text)
        print("Updated preferences loading to apply across ALL modes.")
