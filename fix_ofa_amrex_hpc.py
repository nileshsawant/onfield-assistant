import re

file_path = "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py"
with open(file_path, "r") as f:
    text = f.read()

# 1. We need to append the HPC RAG context onto the AMReX context
amrex_func = """def retrieve_amrex_context(query: str, top_k: int = 5) -> str:
    _init_rag()
    query_embedding = _embed_model.encode([query])[0].tolist()
    context_parts = []
    
    # Try Marbles
    if _marbles_src_collection is not None:
        try:
            docs, metas = _hybrid_search(query=query, query_embedding=query_embedding, collection=_marbles_src_collection, coll_name="marbles_src", top_k=top_k)
            for s_doc, s_meta in zip(docs, metas):
                s_header = f"[MARBLES thermal C++ Source Code - src/{s_meta.get('filepath', '?')}]"
                context_parts.append(f"{s_header}\\n{s_doc}\\n")
        except Exception:
            pass

    # Try AMReX
    if _amrex_src_collection is not None:
        try:
            docs, metas = _hybrid_search(query=query, query_embedding=query_embedding, collection=_amrex_src_collection, coll_name="amrex_src", top_k=top_k)
            for s_doc, s_meta in zip(docs, metas):
                s_header = f"[AMReX Core Source Code - {s_meta.get('filepath', '?')}]"
                context_parts.append(f"{s_header}\\n{s_doc}\\n")
        except Exception:
            pass

    # Also grab generic Kestrel HPC docs so it knows about module paths / Slurm
    hpc_ctx = retrieve_hpc_context(query, top_k=2)
    if hpc_ctx:
        context_parts.append(hpc_ctx)

    return "\\n\\n---\\n\\n".join(context_parts)"""

if "hpc_ctx = retrieve_hpc_context" not in text:
    # replace the entire old function
    old_func = text.split("def retrieve_amrex_context")[1].split('return "\\n\\n---\\n\\n".join(context_parts)')[0] + 'return "\\n\\n---\\n\\n".join(context_parts)'
    text = text.replace("def retrieve_amrex_context" + old_func, amrex_func)

with open(file_path, "w") as f:
    f.write(text)

print("Appended Kestrel HPC context to the amrex retriever.")
