#!/usr/bin/env python3
import os
import sys
import subprocess
import glob
import tempfile
from pathlib import Path

# Try to import necessary libraries
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please run this script from an environment with chromadb and sentence_transformers installed.")
    sys.exit(1)

OFA_ROOT = os.environ.get("OFA_ROOT", str(Path(__file__).resolve().parent.parent))
VECTORDB_PATH = os.environ.get("OFA_VECTORDB", os.path.join(OFA_ROOT, "vectordb"))
TMP_CLONE_DIR = os.environ.get(
    "OFA_AMREX_CLONE_DIR",
    os.path.join(tempfile.gettempdir(), f"amrex_ingest_{os.environ.get('USER', 'default')}"),
)

# Ensure user directory setup
os.makedirs(TMP_CLONE_DIR, exist_ok=True)
os.chdir(TMP_CLONE_DIR)

print("Cloning repositories...")
if not os.path.exists("amrex"):
    subprocess.run(["git", "clone", "https://github.com/amrex-codes/amrex.git"])
if not os.path.exists("marblesThermal"):
    subprocess.run(["git", "clone", "https://github.com/nileshsawant/marblesThermal.git"])

# Load model
print("Loading embedding model...")
model_path = os.environ.get("OFA_EMBEDDING_MODEL", os.path.join(OFA_ROOT, "embedding_model"))
embed_model = SentenceTransformer(model_path, device="cpu")

# Initialize Master ChromaDB
print(f"Connecting to ChromaDB at {VECTORDB_PATH}...")
client = chromadb.PersistentClient(path=VECTORDB_PATH)

def process_repo(repo_name, coll_name, extensions):
    print(f"\nProcessing {repo_name} -> Collection: {coll_name}")
    try:
        collection = client.create_collection(coll_name)
    except ValueError:
        collection = client.get_collection(coll_name)
        print(f"Collection {coll_name} already exists. Appending...")

    files_to_process = []
    for ext in extensions:
        files_to_process.extend(glob.glob(f"{repo_name}/**/*{ext}", recursive=True))

    docs = []
    metadatas = []
    ids = []
    
    CHUNK_SIZE = 1500 # rough characters

    for i, fp in enumerate(files_to_process):
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            # Simple chunking for source code
            chunks = [content[i:i+CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE - 200)]
            
            for j, chunk in enumerate(chunks):
                doc_id = f"{repo_name}_{os.path.basename(fp)}_{i}_{j}"
                docs.append(chunk)
                metadatas.append({"filepath": fp, "repo": repo_name})
                ids.append(doc_id)
        except Exception as e:
             pass

    if docs:
        print(f"Embedding {len(docs)} chunks for {repo_name}...")
        # Embed in batches of 32
        BATCH = 32
        for b in range(0, len(docs), BATCH):
            batch_docs = docs[b:b+BATCH]
            batch_metas = metadatas[b:b+BATCH]
            batch_ids = ids[b:b+BATCH]
            embeddings = embed_model.encode(batch_docs).tolist()
            collection.add(
                documents=batch_docs,
                embeddings=embeddings,
                metadatas=batch_metas,
                ids=batch_ids
            )
            print(f"  Inserted {b+len(batch_docs)}/{len(docs)}", end='\r')
        print("\nDone.")
    else:
        print(f"No documents found for {repo_name}.")

process_repo("amrex", "amrex_src", [".cpp", ".H", ".h", ".f90"])
process_repo("marblesThermal", "marbles_src", [".cpp", ".H", ".h"])

print("\nIngestion complete. You can now use `ofa --amrex` safely!")
