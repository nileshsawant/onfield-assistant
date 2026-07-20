import re

file_path = "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py"
with open(file_path, "r") as f:
    text = f.read()

# 1. Remove the global SIGINT handler so Python raises KeyboardInterrupt normally
text = re.sub(
    r'signal\.signal\(signal\.SIGINT, lambda \*_: \(print\("\\nGoodbye\."\), sys\.exit\(0\)\)\)',
    r'# Use default KeyboardInterrupt handling for SIGINT\n    signal.signal(signal.SIGINT, signal.default_int_handler)',
    text
)

# 2. Fix the interactive_mode input try/except logic
old_interactive_input = """        try:
            user_input = input("\\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\\nGoodbye.")
            break"""

new_interactive_input = """        try:
            user_input = input("\\n> ").strip()
        except KeyboardInterrupt:
            print("\\n(Ctrl+C pressed. Type 'quit' to exit safely.)", file=sys.stderr)
            continue
        except EOFError:
            print("\\nGoodbye.")
            break"""

if old_interactive_input in text:
    text = text.replace(old_interactive_input, new_interactive_input)

# 3. Just to be safe, wrap the chat_stream call in the main loop to ignore KeyboardInterrupt if it bubbles up
old_chat_stream_call = """        response = chat_stream(messages)"""
new_chat_stream_call = """        try:
            response = chat_stream(messages)
        except KeyboardInterrupt:
            print("\\n[AI analysis aborted by user]", file=sys.stderr)
            break"""
text = text.replace(old_chat_stream_call, new_chat_stream_call)

with open(file_path, "w") as f:
    f.write(text)

print("Patch applied for global Ctrl-C handling.")
