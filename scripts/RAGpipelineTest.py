# RAG Pipeline Tester for Detailed Code Analysis
# This file tests a complete RAG (Retrieval-Augmented Generation) pipeline

import os
import json
import time
from typing import List, Dict, Any
import google.generativeai as genai  # Google Gemini API for embeddings and chat
from dotenv import load_dotenv           # Loads env variables from .env file
import chromadb                         # Vector database for storing embeddings
from chromadb.config import Settings

#ARCHITECTURE:
# 1. DESIGN PATTERN: Test-driven development with comprehensive validation
# 2. MODULARITY: Each test focuses on a specific component (embeddings, retrieval, generation)
# 3. ERROR HANDLING: Try/catch blocks w/  graceful degradation
# 4. RATE LIMITING: Built-in delays to respect API limits
# 5. SCORING SYSTEM: Quantitative evaluation of each component
# 6. INTEGRATION: Tests both individual components and end-to-end pipeline
# 7. REPORTING: Detailed results with JSON export for analysis
# 8. USER EXPERIENCE: Clear status indicators and actionable error messages

# Load environment variables from .env file
load_dotenv()

class RAGPipelineTester:
    """
    Tests complete RAG pipeline flow
    - Embeddings: Convert text to numerical vectors (semnatic meanings)
    - Retrieval: Find relevant documents using vector similarity (cosine similarity - text simialrity (vector direction))
    - Generation: Create responses using retrieved context (gemini 1.5 flash free tier)
    """
    
    def __init__(self, collection_name: str = "ucsb_engineering"):
        """
        Class constructor with default parameter
        Initialize all components needed for RAG testing
        """
        # Configure Gemini API using env vars
        # os.getenv() to safely access environment variables
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        
        # Sets up model names for different tasks
        self.embedding_model = "models/embedding-001"  # For converting text to vectors
        self.chat_model = "gemini-1.5-flash"           # For generating responses
        self.collection_name = collection_name          # ChromaDB collection name
        
        # Creates ChromaDB client with persistent storage
        # Vector database to store and search document embeddings (chromaDB)
        self.chroma_client = chromadb.PersistentClient(
            path="./embeddings",  # Local file storage path
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initializes generative model for chat responses
        self.model = genai.GenerativeModel(self.chat_model)
        
        # Empty list for storing test results
        self.test_results = []
    
    def check_prerequisites(self) -> bool:
        """
        Method returns boolean, uses -> type hint
        Verify all required components are available BEFORE testing
        """
        print("Checking RAG Pipeline Prerequisites...")
        
        # 1. CHECK API KEY
        # os.getenv() returns None if key doesn't exist
        # Ensures Google API key is available for embeddings/chat
        if not os.getenv('GOOGLE_API_KEY'):
            print("GOOGLE_API_KEY not found in environment variables")
            return False
        print("Google API key found")
        
        # 2. CHECK CHROMADB COLLECTION
        # try/except block for error handling
        # Verify vector database has documents stored
        try:
            collection = self.chroma_client.get_collection(name=self.collection_name)
            doc_count = collection.count()  # Get number of stored documents
            print(f"ChromaDB collection found with {doc_count} documents")
        except Exception as e:
            print(f"ChromaDB collection not found: {e}")
            print("   Run: python scripts/embeddings.py")  # Helpful error message
            return False
        
        # 3. TEST API CONNECTION
        # genai.embed_content() API call with parameters
        # Verify API is working with a simple test embedding
        try:
            test_result = genai.embed_content(
                model=self.embedding_model,
                content="test",                    # Simple test content
                task_type="retrieval_query"       # Optimize embedding for search
            )
            print("Gemini API connection working")
        except Exception as e:
            print(f"Gemini API connection failed: {e}")
            return False
        
        return True
    
    def test_embedding_quality(self) -> Dict[str, Any]:
        """
        Returns dict with Any type values
        Tests if similar queries produce similar embeddings
        """
        print("\nTesting Embedding Quality...")
        
        # List of dicts with test cases
        # FDefine pairs of queries to test similarity expectations
        test_cases = [
            {
                "query1": "computer science courses",
                "query2": "CS classes available",
                "should_be_similar": True    # Should have high similarity
            },
            {
                "query1": "mechanical engineering program",
                "query2": "ME department overview",
                "should_be_similar": True
            },
            {
                "query1": "computer science courses",
                "query2": "mechanical engineering program",
                "should_be_similar": False   # Should have low similarity
            }
        ]
        
        results = []
        # For loop iterating through test cases
        for case in test_cases:
            try:
                # Gets embeddings for both queries
                # genai.embed_content() returns dict with 'embedding' key
                emb1 = genai.embed_content(
                    model=self.embedding_model,
                    content=case["query1"],
                    task_type="retrieval_query"
                )['embedding']  # Extracts just the embedding vector
                
                emb2 = genai.embed_content(
                    model=self.embedding_model,
                    content=case["query2"],
                    task_type="retrieval_query"
                )['embedding']
                
                # Calculates how similar the embeddings are
                # Calls private method (prefixed with _)
                similarity = self._cosine_similarity(emb1, emb2)
                
                # Checks if similarity matches expectation
                # Conditional logic with different thresholds
                if case["should_be_similar"]:
                    passed = similarity > 0.7  # High similarity threshold
                    if passed:
                        status = "✅"
                    else:
                        status = "❌"
                else:
                    passed = similarity < 0.5  # Low similarity threshold
                    if passed:
                        status = "✅"
                    else:
                        status = "❌"
                
                # Appends dict to results list
                results.append({
                    "query1": case["query1"],
                    "query2": case["query2"],
                    "similarity": similarity,
                    "expected": "similar" if case["should_be_similar"] else "dissimilar",
                    "passed": passed
                })
                
                # Prints test result with formatting
                print(f"{status} {case['query1']} vs {case['query2']}: {similarity:.3f}")
                
                # Rate limiting to avoid API throttling hopefully
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error testing embeddings: {e}")
                results.append({
                    "query1": case["query1"],
                    "query2": case["query2"],
                    "error": str(e),
                    "passed": False
                })
        
        # Return dict with test metadata
        return {"test_type": "embedding_quality", "results": results}
    
    def test_retrieval_accuracy(self) -> Dict[str, Any]:
        """
        Tests if vector search returns relevant documents
        Uses ChromaDB collection for vector similarity search
        """
        print("\nTesting Retrieval Accuracy...")
        
        # Gets vector database collection
        collection = self.chroma_client.get_collection(name=self.collection_name)
        
        # Lists of test cases with expected document characteristics
        test_cases = [
            {
                "query": "CS 162 operating systems",
                "expected_type": "course",                    # Expect course document
                "expected_dept": "Computer Science"           # From CS department
            },
            {
                "query": "mechanical engineering program requirements",
                "expected_type": "program",
                "expected_dept": "Mechanical Engineering"
            },
            {
                "query": "electrical engineering department",
                "expected_type": "department",
                "expected_dept": "Electrical and Computer Engineering"
            }
        ]
        
        results = []
        for case in test_cases:
            try:
                # Cnverts query to embedding for vector search
                query_emb = genai.embed_content(
                    model=self.embedding_model,
                    content=case["query"],
                    task_type="retrieval_query"
                )['embedding']
                
                # Searches vector database for similar documents
                # ChromaDB query with embedding and result count
                search_results = collection.query(
                    query_embeddings=[query_emb],  # List of embeddings to search
                    n_results=3                    # Return top 3 matches
                )
                
                # Checks if top result matches expectations
                # Nested list access for search results
                if search_results['documents'][0]:
                    top_result = search_results['metadatas'][0][0]  # First result's metadata
                    
                    # Checks if doc type and department match
                    type_match = top_result.get('type') == case['expected_type']
                    dept_match = case['expected_dept'].lower() in top_result.get('department', '').lower()
                    
                    # Logical OR - either type or department should match
                    passed = type_match or dept_match
                    if passed:
                        status = "✅"
                    else:
                        status = "❌"
                    
                    results.append({
                        "query": case["query"],
                        "expected_type": case["expected_type"],
                        "expected_dept": case["expected_dept"],
                        "actual_type": top_result.get('type'),
                        "actual_dept": top_result.get('department'),
                        "passed": passed
                    })
                    
                    print(f"{status} {case['query']}: got {top_result.get('type')} from {top_result.get('department')}")
                else:
                    print(f"{case['query']}: No results found")
                    results.append({
                        "query": case["query"],
                        "error": "No results found",
                        "passed": False
                    })
                
                time.sleep(1)  # Rate limiting hopefully lol
                
            except Exception as e:
                print(f"❌ Error testing retrieval: {e}")
                results.append({
                    "query": case["query"],
                    "error": str(e),
                    "passed": False
                })
        
        return {"test_type": "retrieval_accuracy", "results": results}
    
    def test_response_generation(self) -> Dict[str, Any]:
        """
        Tests if LLM generates good responses using retrieved context
        Uses Gemini chat model with formatted prompts
        """
        print("\nTesting Response Generation...")
        
        collection = self.chroma_client.get_collection(name=self.collection_name)
        
        # List of test queries
        test_queries = [
            "What computer science courses are available?",
            "Tell me about the mechanical engineering program",
            "What are the prerequisites for ECE courses?",
            "How many units is a typical engineering course?"
        ]
        
        results = []
        for query in test_queries:
            try:
                # Gets relevant docs through vector search
                query_emb = genai.embed_content(
                    model=self.embedding_model,
                    content=query,
                    task_type="retrieval_query"
                )['embedding']
                
                search_results = collection.query(
                    query_embeddings=[query_emb],
                    n_results=3
                )
                
                # Formats retrieved documents for LLM
                context = self._format_context_for_llm(search_results)
                
                # fstring for prompt formatting
                # Creates structured prompt with context and question
                prompt = f"""You are a UCSB College of Engineering assistant. Answer the question based on the context provided.

CONTEXT:
{context}

QUESTION: {query}

Please provide a helpful answer based on the context above."""
                
                # Generates response using chat model
                response = self.model.generate_content(prompt)
                
                # Evaluates response quality
                response_text = response.text
                quality_score = self._evaluate_response_quality(query, response_text, search_results)
                
                results.append({
                    "query": query,
                    "response": response_text,
                    "quality_score": quality_score,
                    "passed": quality_score > 0.6  # 60% threshold
                })
                
                if quality_score > 0.6:
                    status = "✅"
                else:
                    status = "❌"
                print(f"{status} {query}: Quality score {quality_score:.2f}")
                
                time.sleep(2)  # Longer rate limiting for generation
                
            except Exception as e:
                print(f"❌ Error generating response: {e}")
                results.append({
                    "query": query,
                    "error": str(e),
                    "passed": False
                })
        
        return {"test_type": "response_generation", "results": results}
    
    def test_end_to_end_pipeline(self) -> Dict[str, Any]:
        """
        FUNCTION: Test complete pipeline using external response generator
        SYNTAX: Dynamic import and integration testing
        """
        print("\nTesting End-to-End Pipeline...")
        
        # Imports statement inside method (dynamic import)
        # Uses the actual response generator module
        from responseGenerator import GeminiResponseGenerator
        
        try:
            generator = GeminiResponseGenerator()
            
            # List of realistic user queries
            user_queries = [
                "What CS courses should I take for machine learning?",
                "Is there a computer engineering program at UCSB?",
                "What are the requirements for the Materials department?",
                "Show me electrical engineering courses about circuits"
            ]
            
            results = []
            for query in user_queries:
                try:
                    # Calls complete RAG pipeline
                    result = generator.generate_response(query)
                    
                    # Evaluates response quality with multiple criteria
                    has_response = len(result['response']) > 50      # Adequate length
                    has_sources = len(result.get('sources', [])) > 0  # Has source citations
                    no_errors = result.get('success', False)         # No errors occurred
                    
                    # Calculate avg score from boolean vals
                    quality_score = (has_response + has_sources + no_errors) / 3
                    
                    results.append({
                        "query": query,
                        "response_length": len(result['response']),
                        "sources_count": len(result.get('sources', [])),
                        "success": result.get('success', False),
                        "quality_score": quality_score,
                        "passed": quality_score > 0.6
                    })
                    
                    if quality_score > 0.6:
                        status = "✅"
                    else:
                        status = "❌"
                    print(f"{status} {query}: {quality_score:.2f} (response: {len(result['response'])} chars, sources: {len(result.get('sources', []))})")
                    
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"Error in end-to-end test: {e}")
                    results.append({
                        "query": query,
                        "error": str(e),
                        "passed": False
                    })
            
            return {"test_type": "end_to_end", "results": results}
            
        except Exception as e:
            print(f"Could not initialize response generator: {e}")
            return {"test_type": "end_to_end", "results": [], "error": str(e)}
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Calculates similarity between two embedding vectors
        List comprehension and math operations
        """
        # Generator expression w/ sum() for dot product
        dot_product = sum(x * y for x, y in zip(a, b))
        
        # List comprehension with sum() and exponentiation
        norm_a = sum(x * x for x in a) ** 0.5  # Square root of sum of squares
        norm_b = sum(x * x for x in b) ** 0.5
        
        # Cosine similarity formula (vector direction similarity)
        return dot_product / (norm_a * norm_b)
    
    def _format_context_for_llm(self, search_results: Dict[str, Any]) -> str:
        """
        Format retrieved docs for LLM consumption
        String formatting and list comprehension
        """
        context_parts = []
        # zip() to iterate over parallel lists
        for doc, metadata in zip(search_results['documents'][0], search_results['metadatas'][0]):
            doc_type = metadata.get('type', 'unknown')     # Safe dictionary access
            dept = metadata.get('department', 'Unknown')
            
            # fstring with newlines for formatting
            context_parts.append(f"[{doc_type.upper()} - {dept}]\n{doc}\n")
        
        # Joins list elements with newlines
        return "\n".join(context_parts)
    
    def _evaluate_response_quality(self, query: str, response: str, search_results: Dict[str, Any]) -> float:
        """
        Basic evaluation of response quality
        Score accumulation w/ multiple criteria
        """
        score = 0.0
        
        # CRITERION 1: Response length (not too short, not too long)
        if 50 <= len(response) <= 1000:
            score += 0.3
        
        # CRITERION 2: Relevance (contains query terms)
        query_terms = query.lower().split()
        response_lower = response.lower()
        # Generator expression w/ conditional
        relevant_terms = sum(1 for term in query_terms if term in response_lower)
        if relevant_terms > 0:
            score += 0.3
        
        # CRITERION 3: Context usage (uses retrieved info)
        if search_results['documents'][0]:
            context_terms = []
            for doc in search_results['documents'][0]:
                context_terms.extend(doc.lower().split())
            
            # List slicing [:20] to limit terms checked
            context_usage = sum(1 for term in context_terms[:20] if term in response_lower)
            if context_usage > 0:
                score += 0.4
        
        return score
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Execute complete test suite and generate summary
        Dict comprehension and string formatting
        """
        print("="*60)
        print("UCSB RAG PIPELINE COMPREHENSIVE TEST")
        print("="*60)
        
        # Checsk prerequisites before running tests
        if not self.check_prerequisites():
            return {"status": "failed", "error": "Prerequisites not met"}
        
        # Runs all test methods and collect results
        test_results = {
            "prerequisite_check": True,
            "embedding_quality": self.test_embedding_quality(),
            "retrieval_accuracy": self.test_retrieval_accuracy(),
            "response_generation": self.test_response_generation(),
            "end_to_end": self.test_end_to_end_pipeline()
        }
        
        # Generates summary statistics
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        total_passed = 0
        total_tests = 0
        
        # Dict items() iteration
        for test_name, test_result in test_results.items():
            if test_name == "prerequisite_check":
                continue
            
            # isinstance() for type checking
            if isinstance(test_result, dict) and 'results' in test_result:
                # Generator expression w/ sum() for counting
                passed = sum(1 for r in test_result['results'] if r.get('passed', False))
                total = len(test_result['results'])
                total_passed += passed
                total_tests += total
                
                if passed == total:
                    status = "✅"
                else:
                    status = "❌"
                # String method chaining for formatting
                print(f"{status} {test_name.replace('_', ' ').title()}: {passed}/{total} passed")
        
        if total_passed == total_tests:
            overall_status = "PASSED"
        else:
            overall_status = "FAILED"
        print(f"\nOVERALL: {overall_status} ({total_passed}/{total_tests} tests)")
        
        return {
            "status": "passed" if total_passed == total_tests else "failed",
            "total_passed": total_passed,
            "total_tests": total_tests,
            "detailed_results": test_results
        }

# For script execution
if __name__ == "__main__":
    # Creates tester instance and run all tests
    tester = RAGPipelineTester()
    results = tester.run_all_tests()
    
    # Saves results to JSON file for analysis
    # With statement for file handling
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to test_results.json")
    
    # Provides next steps based on test results
    if results['status'] == 'passed':
        print("\nAll tests passed! Your RAG pipeline is ready.")
        print("Next steps:")
        print("1. Run: python scripts/responseGenerator.py")
        print("2. Test with interactive chat")
        print("3. Integrate with Streamlit frontend")
    else:
        print("\nSome tests failed. Check the details above.")
        print("Fix issues before proceeding to frontend integration.")