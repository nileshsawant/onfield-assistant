import glob

# Ensure all prompt files rigorously enforce the write rule and add edit rule
for fp in glob.glob("/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/prompts/*.txt"):
    if "amrex" in fp or "code" in fp:
        with open(fp, "r") as f:
            text = f.read()
        
        edit_instruction = """5. Editing Files: To edit an existing file without rewriting it completely, output an exact ```edit <filepath>``` block with <<FIND>> and <<REPLACE>> markers. The FIND text must match the existing file verbatim. For example:
```edit src/hello.py
<<FIND>>
print("Hello world!")
<<REPLACE>>
print("Hello Kestrel!")
```"""
        
        if "5. Editing Files:" not in text:
            # Add after writing
            text = text.replace(
                "4. Writing Files: To create or completely overwrite a file, output the file content inside a ```write <filepath> ... ``` block.\n   - CRITICAL: Never output a loose ```cpp``` or ```bash``` code block without the `write <file>` or explicit execution declaration. The system will IGNORE it.",
                "4. Writing Files: To create or completely overwrite a file, output the file content inside a ```write <filepath> ... ``` block.\n   - CRITICAL: Never output a loose ```cpp``` or ```bash``` code block without the `write <file>` or explicit execution declaration. The system will IGNORE it.\n" + edit_instruction
            )
        
        # Add Kestrel environment explicit instruction if AMReX
        if "amrex" in fp and "module load" not in text:
            text = text.replace(
                "Specialized Knowledge (AMReX & MARBLES):",
                "Specialized Knowledge (AMReX & MARBLES & Kestrel HPC):\n- You are operating natively on the NREL Kestrel supercomputer! When you need libraries like CUDA, Cray-MPICH, CMake, HDF5, or Python, DO NOT attempt to search standard Linux paths for them. You MUST load them via environment modules by generating a bash block (e.g., `module load cray-mpich cuda/12.4 cmake`)."
            )

        with open(fp, "w") as f:
            f.write(text)
        print(f"Updated {fp}")

