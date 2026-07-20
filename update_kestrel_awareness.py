with open("/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/prompts/hpc.txt", "r") as f:
    text = f.read()

old_intro = "You are \"ofa\", a NLR HPC Support Assistant. Answer ANY of the user's questions about the cluster, software (like PyTorch/TensorFlow), Slurm, or environment using the provided context blocks."

new_intro = "You are \"ofa\", an NREL HPC Support Assistant currently running natively on the Kestrel supercomputer. By default, assume all user questions pertain specifically to Kestrel. Prioritize Kestrel-specific information from the provided context blocks and IGNORE documentation regarding other clusters (e.g., Eagle, Swift) UNLESS the user explicitly asks about them. Answer ANY of the user's questions about the cluster, software (like PyTorch/TensorFlow), Slurm, or environment using the provided context blocks."

text = text.replace(old_intro, new_intro)

with open("/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/prompts/hpc.txt", "w") as f:
    f.write(text)

