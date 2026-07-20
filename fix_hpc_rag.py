file_path = "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py"
with open(file_path, "r") as f:
    text = f.read()

# In interactive_mode
old_ctx_logic = "            context = retrieve_hpc_context(user_input) if hpc_mode else retrieve_context(user_input)"
new_ctx_logic = "            context = retrieve_hpc_context(user_input) if (hpc_mode or code_mode) else retrieve_context(user_input)"

if old_ctx_logic in text:
    text = text.replace(old_ctx_logic, new_ctx_logic)

with open(file_path, "w") as f:
    f.write(text)

print("Updated RAG logic successfully.")
