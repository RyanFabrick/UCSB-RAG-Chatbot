import os
import json
import time
from typing import List, Dict, Any
import google.generativeai as genai
import chromadb
from chromadb.config import Settings

# Import from new configuration structure
from src.config.settings import Config

class GeminiDocumentEmbedder:
    """Embed documents using Google Gemini and store in ChromaDB"""
    
    def __init__(self, collection_name: str = None):
        # Use configuration for all settings
        self.config = Config()
        
        # Validate configuration
        is_valid, error_msg = self.config.validate_config()
        if not is_valid:
            raise ValueError(f"Configuration error: {error_msg}")
        
        # Configure Gemini API
        genai.configure(api_key=self.config.GOOGLE_API_KEY)
        
        # Use config for model and collection settings
        self.embedding_model = self.config.EMBEDDING_MODEL
        self.collection_name = collection_name or self.config.COLLECTION_NAME
        
        # Initialize ChromaDB with config path
        self.chroma_client = chromadb.PersistentClient(
            path=self.config.CHROMADB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "UCSB College of Engineering documents"}
        )
        
        print(f"🔧 Initialized ChromaDB collection: {self.collection_name}")
        print(f"📍 ChromaDB path: {self.config.CHROMADB_PATH}")
    
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
                print(f"⚠️ Embedding attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    # Rate limiting - be gentle with Gemini
                    error_message = str(e).lower()
                    if "quota" in error_message or "rate" in error_message:
                        wait_time = 60  # Wait 1 minute for quota issues
                    else:
                        wait_time = 2 ** attempt
                    print(f"⏳ Waiting {wait_time} seconds before retry...")
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
    
    def embed_documents(self, documents_file: str = None, batch_size: int = 5):
        """Embed all documents and store in ChromaDB"""
        # Use config for default path
        if documents_file is None:
            documents_file = os.path.join(self.config.DATA_PATH, 'processed_documents.json')
        
        # Check if file exists
        if not os.path.exists(documents_file):
            print(f"❌ Error: {documents_file} not found!")
            print("Please run the data processor first:")
            print("   python -m src.data.data_processor")
            return
        
        # Load processed documents
        with open(documents_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        print(f"📚 Processing {len(documents)} documents...")
        
        # Clear existing collection
        try:
            self.collection.delete()
            print("🧹 Cleared existing collection")
        except:
            pass  # Collection might be empty
        
        batch_docs = []
        batch_embeddings = []
        batch_metadatas = []
        batch_ids = []
        processed_count = 0
        
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
                    processed_count += 1
                    
                    # Process batch when full
                    if len(batch_docs) >= batch_size:
                        self._add_batch_to_collection(batch_docs, batch_embeddings, batch_metadatas, batch_ids)
                        print(f"📥 Added batch of {len(batch_docs)} chunks to collection")
                        batch_docs, batch_embeddings, batch_metadatas, batch_ids = [], [], [], []
                    
                    # Rate limiting - be gentle with Gemini
                    time.sleep(1)
                
                if (i + 1) % 10 == 0:
                    print(f"✅ Processed {i + 1}/{len(documents)} documents ({processed_count} chunks)...")
                
            except Exception as e:
                print(f"❌ Error processing document {doc_id}: {e}")
                continue
        
        # Process remaining batch
        if batch_docs:
            self._add_batch_to_collection(batch_docs, batch_embeddings, batch_metadatas, batch_ids)
            print(f"📥 Added final batch of {len(batch_docs)} chunks to collection")
        
        final_count = self.collection.count()
        print(f"🎉 Embedding complete! Total documents in collection: {final_count}")
        
        if final_count == 0:
            print("⚠️ Warning: No documents were added to the collection!")
            print("Check if your processed_documents.json file contains valid data.")
        
        return final_count
    
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
            print(f"❌ Error adding batch to collection: {e}")
    
    def test_retrieval(self, query: str, n_results: int = None):
        """Test retrieval with a sample query"""
        if n_results is None:
            n_results = self.config.DEFAULT_RETRIEVAL_COUNT
            
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
            
            print(f"\n🔍 Query: {query}")
            print(f"📊 Found {len(results['documents'][0])} results:")
            
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                doc_type = metadata.get('type', 'unknown')
                if doc_type == 'course':
                    identifier = metadata.get('course_name', 'Unknown Course')
                elif doc_type == 'program':
                    identifier = metadata.get('program', 'Unknown Program')
                elif doc_type == 'department':
                    identifier = metadata.get('department', 'Unknown Department')
                else:
                    identifier = 'Unknown'
                
                print(f"\n{i+1}. 📝 {doc_type.title()}: {identifier}")
                print(f"   🏛️ Department: {metadata.get('department', 'Unknown')}")
                print(f"   📄 Preview: {doc[:200]}...")
                
        except Exception as e:
            print(f"❌ Error testing retrieval: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "chromadb_path": self.config.CHROMADB_PATH,
                "embedding_model": self.embedding_model
            }
        except Exception as e:
            return {"error": str(e)}
    
    def query_documents(self, query: str, n_results: int = None, 
                       include_embeddings: bool = False) -> Dict[str, Any]:
        """Query documents and return structured results"""
        if n_results is None:
            n_results = self.config.DEFAULT_RETRIEVAL_COUNT
            
        try:
            # Get query embedding
            query_result = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query"
            )
            query_embedding = query_result['embedding']
            
            # Search collection
            include_list = ['documents', 'metadatas', 'distances']
            if include_embeddings:
                include_list.append('embeddings')
                
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=include_list
            )
            
            return {
                "query": query,
                "results": results,
                "count": len(results['documents'][0]) if results['documents'] else 0
            }
            
        except Exception as e:
            return {"error": str(e), "query": query}

# Usage and testing
if __name__ == "__main__":
    print("🚀 Starting UCSB Document Embedding Process")
    print("=" * 50)
    
    try:
        # Initialize embedder
        embedder = GeminiDocumentEmbedder()
        
        # Show current collection stats
        stats = embedder.get_collection_stats()
        print(f"📊 Current collection stats: {stats}")
        
        # Embed documents
        document_count = embedder.embed_documents()
        
        if document_count > 0:
            # Test retrieval
            print("\n" + "="*50)
            print("🧪 Testing retrieval with sample queries:")
            print("="*50)
            
            test_queries = [
                "What computer science courses are available?",
                "Tell me about the mechanical engineering program",
                "What are the requirements for electrical engineering?",
                "Show me courses about machine learning"
            ]
            
            for query in test_queries:
                embedder.test_retrieval(query)
                print("-" * 30)
        else:
            print("\n⚠️ No documents embedded. Please check your data processing.")
            
    except Exception as e:
        print(f"❌ Error during embedding process: {e}")
        print("\nMake sure you have:")
        print("1. ✅ Valid Google API key in .env file")
        print("2. ✅ Processed documents in data/processed_documents.json")
        print("3. ✅ ChromaDB directory created at ./embeddings")