import os, json, hashlib, random
import numpy as np

MILVUS_URI = "http://192.168.86.132:19530"
COLLECTION_NAME = "enterprise_rag"
TEST_DATA_DIR = r"D:\Desktop\RAG\test_data"
EMBEDDING_DIM = 1024

def read_file(filepath):
    ext = filepath.rsplit(".", 1)[-1].lower()
    if ext in ("txt", "md"):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == "docx":
        try:
            from docx import Document
            doc = Document(filepath)
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        except:
            return ""
    elif ext == "xlsx":
        try:
            from openpyxl import load_workbook
            wb = load_workbook(filepath, read_only=True)
            texts = []
            for ws in wb.worksheets:
                for row in ws.iter_rows(values_only=True):
                    texts.append(" ".join([str(c) for c in row if c]))
            return "\n".join(texts)
        except:
            return ""
    elif ext == "pdf":
        try:
            import pdfplumber
            with pdfplumber.open(filepath) as pdf:
                return "\n".join([p.extract_text() or "" for p in pdf.pages])
        except:
            return ""
    return ""

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += "\n" + para if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk)
    if not chunks and text.strip():
        chunks = [text[:chunk_size]]
    return chunks

def generate_random_embedding():
    """Generate a random normalized embedding for testing."""
    vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()

def main():
    print("=" * 60)
    print("  Document Import (Quick - Random Embeddings)")
    print("=" * 60)
    
    print("\nStep 1: Reading documents...")
    all_chunks = []
    all_metadata = []
    for filename in sorted(os.listdir(TEST_DATA_DIR)):
        filepath = os.path.join(TEST_DATA_DIR, filename)
        if not os.path.isfile(filepath):
            continue
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext not in ("txt", "md", "docx", "xlsx", "pdf"):
            continue
        print(f"  Reading: {filename}")
        content = read_file(filepath)
        if not content.strip():
            print("    (empty)")
            continue
        chunks = chunk_text(content, chunk_size=500, overlap=100)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })
        print(f"    {len(chunks)} chunks")
    
    print(f"\nTotal chunks: {len(all_chunks)}")
    
    print("\nStep 2: Generating random embeddings (for testing)...")
    embeddings = [generate_random_embedding() for _ in all_chunks]
    print(f"Generated {len(embeddings)} embeddings")
    
    print("\nStep 3: Storing in Milvus...")
    from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
    
    client = MilvusClient(uri=MILVUS_URI)
    collections = client.list_collections()
    if COLLECTION_NAME in collections:
        print("Dropping existing collection...")
        client.drop_collection(COLLECTION_NAME)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=8192),
    ]
    schema = CollectionSchema(fields, description="Enterprise RAG embeddings")
    client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
    
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128},
    )
    client.create_index(COLLECTION_NAME, index_params)
    
    documents = []
    for i, (chunk, emb, meta) in enumerate(zip(all_chunks, embeddings, all_metadata)):
        documents.append({
            "document_id": hashlib.md5(meta["filename"].encode()).hexdigest()[:16],
            "chunk_id": f"{hashlib.md5(meta['filename'].encode()).hexdigest()[:8]}_{meta['chunk_index']}",
            "content": chunk[:65000],
            "embedding": emb,
            "metadata": json.dumps(meta, ensure_ascii=False),
        })
    
    result = client.insert(collection_name=COLLECTION_NAME, data=documents)
    print(f"Inserted {len(documents)} vectors")
    
    stats = client.get_collection_stats(COLLECTION_NAME)
    print(f"Collection stats: {stats}")
    
    print("\n" + "=" * 60)
    print("  Import complete!")
    print("=" * 60)
    print("\nNote: Using random embeddings for testing.")
    print("For production, use real embeddings from the Qwen API.")

if __name__ == "__main__":
    main()
