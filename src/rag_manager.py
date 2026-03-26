import os
import chromadb
from sentence_transformers import SentenceTransformer

class RAGManager:
    def __init__(self, db_path="data/vector_db", collection_name="bfsi_docs", model_name="all-MiniLM-L6-v2"):
        print(f"Initializing RAG Manager (ChromaDB at {db_path})...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # We share the same small SentenceTransformer model across Tier 1 and Tier 3
        # to save VRAM/RAM.
        print(f"Loading SentenceTransformer '{model_name}' for RAG...")
        self.model = SentenceTransformer(model_name)
        
    def chunk_document(self, text):
        """Simple chunking by double newlines (paragraphs/sections)"""
        chunks = text.split("\n\n")
        return [c.strip() for c in chunks if len(c.strip()) > 20]

    def ingest_directory(self, raw_docs_path="data/raw_documents"):
        print(f"Ingesting documents from {raw_docs_path}...")
        docs_to_add = []
        metadatas = []
        ids = []
        
        doc_idx = 0
        for filename in os.listdir(raw_docs_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(raw_docs_path, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                chunks = self.chunk_document(content)
                for i, chunk in enumerate(chunks):
                    docs_to_add.append(chunk)
                    metadatas.append({"source": filename, "chunk": i})
                    ids.append(f"{filename}_chunk_{i}")
                    
        if docs_to_add:
            print(f"Adding {len(docs_to_add)} chunks to ChromaDB...")
            embeddings = self.model.encode(docs_to_add, convert_to_tensor=False).tolist()
            
            # Upsert avoids duplicate errors on re-runs
            self.collection.upsert(
                documents=docs_to_add,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            print("Ingestion complete.")
        else:
            print("No new documents found.")

    def retrieve(self, query, top_k=2):
        """Retrieve most relevant document chunks for a query"""
        query_embedding = self.model.encode(query, convert_to_tensor=False).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        retrieved_texts = results["documents"][0]
        sources = [m["source"] for m in results["metadatas"][0]]
        
        # Combine retrieved chunks into a context string
        context = ""
        for i, (text, source) in enumerate(zip(retrieved_texts, sources)):
            context += f"[Source: {source}]\n{text}\n\n"
            
        return context.strip()

if __name__ == "__main__":
    # Test Ingestion
    rag = RAGManager()
    rag.ingest_directory()
    
    # Test Retrieval
    test_query = "What is the late payment fee for a premium credit card?"
    print(f"\nUser Query: {test_query}")
    print("\n--- Retrieved Context ---")
    context = rag.retrieve(test_query)
    print(context)
