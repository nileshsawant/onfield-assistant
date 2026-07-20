import re

file_path = "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py"
with open(file_path, "r") as f:
    text = f.read()

# I see it now. The `chat_stream` function loops yielding chunks. 
# We need to catch the KeyboardInterrupt *around* the for loops that iterate over chat_stream, not just inside chat_stream itself because httpx bubbles the traceback to the outermost loop.
wrapper_old_1 = """            last_response = ""
            for chunk in chat_stream(messages):
                print(chunk, end="", flush=True)
                last_response += chunk
            print()"""

wrapper_new_1 = """            last_response = ""
            try:
                for chunk in chat_stream(messages):
                    print(chunk, end="", flush=True)
                    last_response += chunk
            except KeyboardInterrupt:
                print("\\n[AI generation interrupted by user (Ctrl+C)]", file=sys.stderr)
                pass
            print()"""

text = text.replace(wrapper_old_1, wrapper_new_1)

wrapper_old_2 = """        response = ""
        for chunk in chat_stream(messages):
            print(chunk, end="", flush=True)
            response += chunk
        print()"""
        
wrapper_new_2 = """        response = ""
        try:
            for chunk in chat_stream(messages):
                print(chunk, end="", flush=True)
                response += chunk
        except KeyboardInterrupt:
            print("\\n[AI generation interrupted by user (Ctrl+C)]", file=sys.stderr)
            pass
        print()"""

text = text.replace(wrapper_old_2, wrapper_new_2)

wrapper_old_3 = """        response = ""
        for chunk in chat_stream(messages):
            print(chunk, end="", flush=True)
            response += chunk
        print("\\n")"""
        
wrapper_new_3 = """        response = ""
        try:
            for chunk in chat_stream(messages):
                print(chunk, end="", flush=True)
                response += chunk
        except KeyboardInterrupt:
            print("\\n[AI generation interrupted by user (Ctrl+C)]", file=sys.stderr)
            pass
        print("\\n")"""

text = text.replace(wrapper_old_3, wrapper_new_3)

# And now fixing the exact traceback from the httpx backend inside chat_stream
chat_stream_backend = """                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
                if data.get("done"):
                    break"""

chat_stream_backend_new = """                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
                if data.get("done"):
                    break
        except KeyboardInterrupt:
            # Sub-catch inside the stream directly before it tears down httpx
            return"""
            
text = text.replace(chat_stream_backend, chat_stream_backend_new)

with open(file_path, "w") as f:
    f.write(text)

print("Applied strict Ctrl-C outer exception wrapping")
