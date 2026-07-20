files = [
    "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/prompts/hpc.txt",
    "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/prompts/openfoam.txt",
    "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/prompts/cpp.txt"
]

old_prompt_chunk = """If you are asked a question that you don't confidently know the answer to or need more information, you can search the internet instead of guessing. Output your search query exactly inside a ```search ... ``` block. For example:
```search
how to submit openfoam job on slurm
```
The system will run the search and provide you the results.


If the user asks for the weather"""

new_prompt_chunk = """If you are asked a question that you don't confidently know the answer to or need more information, you can search the internet instead of guessing. Output your search query exactly inside a ```search ... ``` block. For example:
```search
how to submit openfoam job on slurm
```
The system will run the search and provide you the snippets/titles of the top results. You can then CHOOSE which url is the best one to actually read based on the results. To read a specific url from the results, output an exact ```fetch ... ``` block:
```fetch
https://docs.openfoam.com/example/url
```


If the user asks for the weather"""

for fn in files:
    with open(fn, "r") as f:
        content = f.read()
    content = content.replace(old_prompt_chunk, new_prompt_chunk)
    with open(fn, "w") as f:
        f.write(content)

print("Prompts updated successfully.")
