import os
import sys
import glob

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please run this script from an environment with chromadb and sentence_transformers installed.")
    sys.exit(1)

VECTORDB_PATH = "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/vectordb"
REFRAME_REPO_DIR = "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/repos/reframe-universal"

print("Loading embedding model...")
model_path = "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/embedding_model"
embed_model = SentenceTransformer(model_path, device="cpu")

print(f"Connecting to ChromaDB at {VECTORDB_PATH}...")
client = chromadb.PersistentClient(path=VECTORDB_PATH)

def process_reframe(repo_dir, coll_name, extensions):
    print(f"\nProcessing {repo_dir} -> Collection: {coll_name}")
    try:
        collection = client.create_collection(coll_name)
    except ValueError:
        collection = client.get_collection(coll_name)

    files_to_process = []
    
    # Custom processing: ReFrame python scripts, bash scripts, and the PDF extraction
    for ext in extensions:
        files_to_process.extend(glob.glob(f"{repo_dir}/**/*{ext}", recursive=True))
        
    # Inject the PDF text extraction manually if available
    pdf_text = "/scratch/nsawant/rhel9_stack.txt"
    if os.path.exists(pdf_text):
        files_to_process.append(pdf_text)
        
    docs = []
    metadatas = []
    ids = []
    
    CHUNK_SIZE = 1500

    for i, fp in enumerate(files_to_process):
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            chunks = [content[j:j+CHUNK_SIZE] for j in range(0, len(content), CHUNK_SIZE - 200)]
            
            for j, chunk in enumerate(chunks):
                doc_id = f"reframe_{os.path.basename(fp)}_{i}_{j}"
                docs.append(chunk)
                metadatas.append({"filepath": fp})
                ids.append(doc_id)
        except Exception as e:
             pass

    if docs:
        print(f"Embedding {len(docs)} chunks...")
        BATCH = 32
        for b in range(0, len(docs), BATCH):
            batch_docs = docs[b:b+BATCH]
            batch_metas = metadatas[b:b+BATCH]
            batch_ids = ids[b:b+BATCH]
            embeddings = embed_model.encode(batch_docs).tolist()
            # Upsert will overwrite if the ID exists, updating our modified tests seamlessly
            collection.upsert(
                documents=batch_docs,
                embeddings=embeddings,
                metadatas=batch_metas,
                ids=batch_ids
            )
            print(f"  Inserted {b+len(batch_docs)}/{len(docs)}", end='\r')
        print("\nDone.")

process_reframe(REFRAME_REPO_DIR, "reframe_src", [".py", ".sh", ".md"])
