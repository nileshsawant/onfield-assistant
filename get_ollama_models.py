import subprocess
import os

OLLAMA_BIN = "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/bin/ollama"
env = os.environ.copy()
env["OLLAMA_MODELS"] = "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/models"

try:
    res = subprocess.run([OLLAMA_BIN, "list"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(res.stdout.decode('utf-8'))
except Exception as e:
    print(e)
