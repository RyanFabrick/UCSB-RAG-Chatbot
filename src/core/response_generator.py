import json 
import time 
from typing import List, Dict, Any, Optional
import google.generativeai as genai
import chromadb
from chromadb.config import Settings

from ..config.settings import Config
from ..config.prompts import get_system_prompt

# ARCHITECTURE:
# 1. RAG PIPELINE: Retrieval-Augmented Generation combining vector search w/ LLM
# 2. ERROR HANDLING: Exception handling with retry logic
# 3. MODULAR DESIGN: Each method has single responsibility
# 4. USER INTERFACE: Multiple interaction modes (chat, batch, single query)
# 5. VECTOR SEARCH: ChromaDB for efficient document retrieval
# 6. API INTEGRATION: Google Gemini for embeddings and text generation
# 7. RATE LIMITING: Prevents API abuse with delays
# 8. CONTEXT MANAGEMENT: Structured document formatting for LLM input
# 9. SOURCE TRACKING: Maintains document provenance for citations
# 10. CONFIGURATION: Environment-based configuration management

class GeminiResponseGenerator:
    """
    Handles RAG (Retrieval-Augmented Generation) system
    - Connects to ChromaDB for document retrieval
    - Uses Gemini LLM for response generation
    - Implements retry logic for API calls
    - Provides interactive chat interface
    """
    
    def __init__(self, collection_name: str = None):
        """
        Initializes the RAG system using configuration
        Sets up all necessary components for the chatbot
        """
        
        # LOAD CONFIGURATION
        self.config = Config()
        
        # Use provided collection name or default from config
        self.collection_name = collection_name or self.config.COLLECTION_NAME
        
        # GEMINI API CONFIGURATION
        genai.configure(api_key=self.config.GOOGLE_API_KEY)
        
        # MODEL SELECTION from config
        self.embedding_model = self.config.EMBEDDING_MODEL
        self.chat_model = self.config.GEMINI_MODEL
        
        # CHROMADB INITIALIZATION using config path
        self.chroma_client = chromadb.PersistentClient(
            path=self.config.CHROMADB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # COLLECTION CONNECTION W/ ERROR HANDLING
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            print(f"Connected to ChromaDB collection: {self.collection_name}")
            print(f"Collection contains {self.collection.count()} documents")
        except Exception as e:
            print(f"Error: Could not connect to collection '{self.collection_name}': {e}")
            print("Please run embeddings.py first to create the collection")
            raise e
        
        # GENERATIVE MODEL INITIALIZATION
        self.model = genai.GenerativeModel(self.chat_model)
        
        # SYSTEM PROMPT from config
        self.system_prompt = get_system_prompt("ucsb_engineering")

    def get_query_embedding(self, query: str, retries: int = 3) -> List[float]:
        """
        EMBEDDING GENERATION: Converts text query to vector representation
        
        FUNCTIONALITY:
        - Implements exponential backoff retry logic
        - Uses Google's embedding model for vector conversion
        - Returns numerical vector representation of query
        """
        for attempt in range(retries):
            try:
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=query,
                    task_type="retrieval_query"
                )
                return result['embedding']
            
            except Exception as e:
                print(f"Query embedding attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    raise e

    def retrieve_relevant_docs(self, query: str, n_results: int = None) -> Dict[str, Any]:
        """
        Finds most relevant documents using vector similarity
        Uses configuration for default number of results
        """ 
        if n_results is None:
            n_results = self.config.DEFAULT_RETRIEVAL_COUNT
            
        try:
            # STEP 1: Convert query to vector
            query_embedding = self.get_query_embedding(query)
            
            # STEP 2: VECTOR SIMILARITY SEARCH
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # STEP 3: FORMAT RESULTS
            formatted_results = []
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                formatted_results.append({
                    'content': doc,
                    'metadata': metadata,
                    'rank': i + 1
                })
            
            return {
                'query': query,
                'results': formatted_results,
                'total_found': len(formatted_results)
            }
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return {'query': query, 'results': [], 'total_found': 0}

    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Converts retrieved docs into LLM-readable format
        Creates structured context with clear doc headers
        """
        if not retrieved_docs:
            return "No relevant documents found."
        
        context_parts = []
        for doc in retrieved_docs:
            metadata = doc['metadata']
            content = doc['content']
            
            # DYNAMIC HEADER GENERATION based on doc type
            if metadata['type'] == 'course':
                header = f"COURSE: {metadata.get('course_name', 'Unknown')} - {metadata.get('course_title', '')}"
                header += f" (Department: {metadata.get('department', 'Unknown')})"
            elif metadata['type'] == 'program':
                header = f"PROGRAM: {metadata.get('program', 'Unknown')}"
                header += f" (Department: {metadata.get('department', 'Unknown')})"
            elif metadata['type'] == 'department':
                header = f"DEPARTMENT: {metadata.get('department', 'Unknown')}"
            else:
                header = f"DOCUMENT: {metadata.get('type', 'Unknown').title()}"
            
            context_parts.append(f"--- {header} ---\n{content}\n")
        
        return "\n".join(context_parts)

    def generate_response(self, query: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        RESPONSE GENERATION: Main RAG pipeline combining retrieval and generation
        
        FUNCTIONALITY: 
        - Retrieves relevant documents
        - Formats context
        - Generates response using LLM
        - Handles errors gracefully
        """
        
        try:
            # STEP 1: DOCUMENT RETRIEVAL
            retrieval_results = self.retrieve_relevant_docs(query)
            
            if retrieval_results['total_found'] == 0:
                return {
                    'query': query,
                    'response': "I couldn't find relevant information about that topic in the UCSB College of Engineering database. Please try rephrasing your question or ask about specific courses, programs, or departments.",
                    'sources': [],
                    'success': True
                }
            
            # STEP 2: CONTEXT FORMATTING
            context = self.format_context(retrieval_results['results'])
            
            # STEP 3: PROMPT CONSTRUCTION
            prompt = f"""{self.system_prompt}

CONTEXT DOCUMENTS:
{context}

USER QUESTION: {query}

Please provide a helpful and accurate response based on the context documents above. If the context doesn't fully answer the question, mention what information is available and suggest how the user might find more details."""

            # STEP 4: LLM RESPONSE GENERATION w/ retry logic
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(prompt)
                    
                    # STEP 5: SOURCE EXTRACTION for citations
                    sources = []
                    for doc in retrieval_results['results']:
                        metadata = doc['metadata']
                        source_info = {
                            'type': metadata['type'],
                            'department': metadata.get('department', 'Unknown')
                        }
                        
                        if metadata['type'] == 'course':
                            source_info['title'] = f"{metadata.get('course_name', 'Unknown')} - {metadata.get('course_title', '')}"
                        elif metadata['type'] == 'program':
                            source_info['title'] = metadata.get('program', 'Unknown Program')
                        elif metadata['type'] == 'department':
                            source_info['title'] = f"{metadata.get('department', 'Unknown')} Department"
                        
                        sources.append(source_info)
                    
                    return {
                        'query': query,
                        'response': response.text,
                        'sources': sources,
                        'context_docs_used': len(retrieval_results['results']),
                        'success': True
                    }
                
                except Exception as e:
                    print(f"LLM generation attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        raise e
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return {
                'query': query,
                'response': f"I encountered an error while processing your question. Please try again later. Error: {str(e)}",
                'sources': [],
                'success': False
            }

    def chat_session(self):
        """
        Provides command-line interface for user interaction
        Continuous loop for user queries with graceful exit
        """
        
        print("UCSB Engineering Chatbot")
        print("Ask questions about UCSB College of Engineering courses, programs, and departments")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                query = input("\nYou: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not query:
                    continue
                
                print("\nThinking...")
                result = self.generate_response(query)
                
                print(f"\nBot: {result['response']}")
                
                if result['sources']:
                    print(f"\nSources consulted:")
                    for i, source in enumerate(result['sources'][:3], 1):
                        print(f"  {i}. {source['title']} ({source['department']} - {source['type']})")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    def batch_test_queries(self, test_queries: List[str]) -> List[Dict[str, Any]]:
        """
        Processes multiple queries for system evaluation
        Automated testing with rate limiting
        """
        
        results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nTesting query {i}/{len(test_queries)}: {query}")
            result = self.generate_response(query)
            results.append(result)
            
            # Rate limiting using config
            time.sleep(self.config.API_RATE_LIMIT_DELAY)
        
        return results

    def print_test_results(self, results: List[Dict[str, Any]]):
        """
        Formats and displays batch test results
        Structured output for test analysis
        """
        
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. QUERY: {result['query']}")
            print(f"   SUCCESS: {result['success']}")
            print(f"   SOURCES: {result.get('context_docs_used', 0)} documents")
            print(f"   RESPONSE: {result['response'][:200]}...")
            
            if result.get('sources'):
                print(f"   TOP SOURCES:")
                for source in result['sources'][:2]:
                    print(f"     - {source['title']} ({source['department']})")

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Returns statistics about the current collection
        Useful for debugging and monitoring
        """
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'status': 'connected'
            }
        except Exception as e:
            return {
                'collection_name': self.collection_name,
                'document_count': 0,
                'status': 'error',
                'error': str(e)
            }

# Main execution block
if __name__ == "__main__":
    
    try:
        generator = GeminiResponseGenerator()
        
        # Predefined test queries for system evaluation
        test_queries = [
            "What computer science courses are available?",
            "Tell me about the mechanical engineering program",
            "What are the prerequisites for CS 162?",
            "Show me electrical engineering courses about circuits",
            "What programs does the Materials department offer?",
            "How many units is ECE 10A?",
            "What courses can I take for machine learning?",
            "Tell me about the Computer Engineering program"
        ]
        
        print("Choose testing mode:")
        print("1. Interactive chat session")
        print("2. Batch test with sample queries")
        print("3. Single query test")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            generator.chat_session()
        
        elif choice == "2":
            print("Running batch tests...")
            results = generator.batch_test_queries(test_queries)
            generator.print_test_results(results)
        
        elif choice == "3":
            query = input("Enter your query: ").strip()
            if query:
                result = generator.generate_response(query)
                print(f"\nResponse: {result['response']}")
                if result['sources']:
                    print(f"\nSources:")
                    for source in result['sources']:
                        print(f"  - {source['title']} ({source['department']})")
        
        else:
            print("Invalid choice. Running interactive chat...")
            generator.chat_session()
            
    except Exception as e:
        print(f"Error initializing response generator: {e}")
        print("Make sure you have:")
        print("1. Run embeddings.py to create the ChromaDB collection")
        print("2. Set GOOGLE_API_KEY in your .env file")
        print("3. Installed required packages: google-generativeai, chromadb, python-dotenv")