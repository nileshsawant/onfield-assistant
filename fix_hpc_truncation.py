import re

with open("/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py", "r") as f:
    text = f.read()

# I missed updating hpc_single_query before, let's fix it completely to have the loop and the truncation.
old_hpc_single = """    print(f"\\n[HPC Documentation Assistant]\\nQuerying Kestrel docs...", file=sys.stderr)
    response = ""
    for chunk in chat_stream(messages):
        print(chunk, end="", flush=True)
        response += chunk
    print("\\n")
    messages.append({"role": "assistant", "content": response})
    save_session(messages)
    cmd_out = check_and_execute_bash(response)
    if cmd_out:
        messages.append({"role": "user", "content": f"Output from executed commands:\\n```text\\n{cmd_out}\\n```\\n(Please note the above command output for context)"})
        save_session(messages)
    return"""

new_hpc_single = """    print(f"\\n[HPC Documentation Assistant]\\nQuerying Kestrel docs...", file=sys.stderr)
    while True:
        response = ""
        for chunk in chat_stream(messages):
            print(chunk, end="", flush=True)
            response += chunk
        print("\\n")
        messages.append({"role": "assistant", "content": response})
        save_session(messages)
        
        cmd_out = check_and_execute_bash(response)
        if cmd_out:
            if len(cmd_out) > 8000:
                truncated = cmd_out[:4000] + "\\n...[OUTPUT TRUNCATED]...\\n" + cmd_out[-4000:]
            else:
                truncated = cmd_out
            messages.append({"role": "user", "content": f"Output from executed commands:\\n```text\\n{truncated}\\n```\\nPlease continue to assist the user using this information."})
            save_session(messages)
            print("\\n[AI is analyzing the output...]", flush=True)
        else:
            break
            
    return"""

text = text.replace(old_hpc_single, new_hpc_single)

with open("/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py", "w") as f:
    f.write(text)

