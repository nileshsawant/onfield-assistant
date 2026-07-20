with open("/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py", "r") as f:
    text = f.read()

# Make sure we don't break existing strings
text = text.replace('"num_ctx": 16384', '"num_ctx": 65536')
text = text.replace('"num_predict": 8192', '"num_predict": 32768')
text = text.replace('def fetch_url_context(query: str, max_chars: int = 8000) -> str:', 'def fetch_url_context(query: str, max_chars: int = 64000) -> str:')

# Fix the truncation logic in interactive_mode and hpc_single_query
text = text.replace('len(cmd_out) > 8000:', 'len(cmd_out) > 96000:')
text = text.replace('cmd_out[:4000] + "\\n...[OUTPUT TRUNCATED]...\\n" + cmd_out[-4000:]', 'cmd_out[:48000] + "\\n...[OUTPUT TRUNCATED]...\\n" + cmd_out[-48000:]')

with open("/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py", "w") as f:
    f.write(text)

