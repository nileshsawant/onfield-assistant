import glob

# Provide explicit instructions on HOW to save preferences to ensure the LLM triggers it
for fp in glob.glob("/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/prompts/*.txt"):
    with open(fp, "r") as f:
        text = f.read()

    # If the text has Self-Knowledge but lacks explicit PREFS instruction
    if "=== PREFS ===" not in text and "Self-Knowledge:" in text:
        text = text.replace(
            "Your conversation session state is stored in `/scratch/$USER/.ofa_session.json`.",
            "Your conversation session state is stored in `/scratch/$USER/.ofa_session.json`.\n- CRITICAL MEMORY RULE: If the user provides a permanent preference (e.g. \"always use tabs\", \"always add a certain flag\"), you MUST output it inside an exact `=== PREFS ===` and `=== END PREFS ===` block so it saves permanently offline."
        )
        with open(fp, "w") as f:
            f.write(text)
        print(f"Updated prefs rule in {fp}")

