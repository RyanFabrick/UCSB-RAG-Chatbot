import os
import json
import time
from typing import List, Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

# Load environment variables
load_dotenv()

class GeminiDocumentEmbedder:
    """Embed documents using Google Gemini and store in ChromaDB"""
    
    def __init__(self, collection_name: str = "ucsb_engineering"):
        # Configure Gemini API
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        
        self.embedding_model = "models/embedding-001"
        self.collection_name = collection_name
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path="./embeddings",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "UCSB College of Engineering documents"}
        )
        
        print(f"Initialized ChromaDB collection: {collection_name}")
    
    def get_embedding(self, text: str, retries: int = 3) -> List[float]:
        """Get embedding for text with retry logic"""
        for attempt in range(retries):
            try:
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_document"
                )
                return result['embedding']
            
            except Exception as e:
                print(f"Embedding attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    # Rate limiting - (longer waits for Gemini)
                    error_message = str(e).lower()
                    if "quota" in error_message:
                        wait_time = 60
                    else:
                        wait_time = 2 ** attempt
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    raise e
    
    def chunk_text(self, text: str, max_chars: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into chunks for better embedding"""
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chars
            
            # Try to break at sentence or word boundary
            if end < len(text):
                # Look for sentence boundary
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start:
                    end = sentence_end + 1
                else:
                    # Look for word boundary
                    word_end = text.rfind(' ', start, end)
                    if word_end > start:
                        end = word_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def embed_documents(self, documents_path: str, batch_size: int = 5):
        """Embed all documents and store in ChromaDB"""
        # Load processed documents
        with open(documents_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        print(f"Processing {len(documents)} documents...")
        
        # Clear existing collection
        self.collection.delete()
        
        batch_docs = []
        batch_embeddings = []
        batch_metadatas = []
        batch_ids = []
        
        for i, doc in enumerate(documents):
            try:
                content = doc['content']
                metadata = doc['metadata']
                doc_id = doc['doc_id']
                
                # Chunk long documents
                chunks = self.chunk_text(content)
                
                for chunk_idx, chunk in enumerate(chunks):
                    # Create embedding with rate limiting
                    embedding = self.get_embedding(chunk)
                    
                    # Create unique ID for chunk
                    if len(chunks) > 1:
                        chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                    else:
                        chunk_id = doc_id
                    
                    # Add to batch
                    batch_docs.append(chunk)
                    batch_embeddings.append(embedding)
                    batch_metadatas.append({
                        **metadata,
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks)
                    })
                    batch_ids.append(chunk_id)
                    
                    # Process batch when full
                    if len(batch_docs) >= batch_size:
                        self._add_batch_to_collection(batch_docs, batch_embeddings, batch_metadatas, batch_ids)
                        batch_docs, batch_embeddings, batch_metadatas, batch_ids = [], [], [], []
                    
                    # Rate limiting - be gentle with Gemini
                    time.sleep(1)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(documents)} documents...")
                
            except Exception as e:
                print(f"Error processing document {doc_id}: {e}")
                continue
        
        # Process remaining batch
        if batch_docs:
            self._add_batch_to_collection(batch_docs, batch_embeddings, batch_metadatas, batch_ids)
        
        print(f"Embedding complete! Total documents in collection: {self.collection.count()}")
    
    def _add_batch_to_collection(self, docs: List[str], embeddings: List[List[float]], 
                                metadatas: List[Dict], ids: List[str]):
        """Add batch to ChromaDB collection"""
        try:
            self.collection.add(
                documents=docs,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            print(f"Error adding batch to collection: {e}")
    
    def test_retrieval(self, query: str, n_results: int = 3):
        """Test retrieval with a sample query"""
        try:
            # Get query embedding
            query_result = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query"
            )
            query_embedding = query_result['embedding']
            
            # Search collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            print(f"\nQuery: {query}")
            print(f"Found {len(results['documents'][0])} results:")
            
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                print(f"\n{i+1}. {metadata['type'].title()}: {metadata.get('course_code', metadata.get('program', metadata.get('department', 'Unknown')))}")
                print(f"   Preview: {doc[:200]}...")
                
        except Exception as e:
            print(f"Error testing retrieval: {e}")

# Usage
if __name__ == "__main__":
    embedder = GeminiDocumentEmbedder()
    
    # Embed documents
    embedder.embed_documents('data/processed_documents.json')
    
    # Test retrieval
    embedder.test_retrieval("What computer science courses are available?")
    embedder.test_retrieval("Tell me about the mechanical engineering program")
    embedder.test_retrieval("What are the requirements for electrical engineering?")