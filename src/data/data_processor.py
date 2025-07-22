import json
import os
from typing import List, Dict, Any
from src.config.settings import Config

class UCSBDataProcessor:
    """Process UCSB Engineering data into documents for embedding"""
    
    def __init__(self, input_file: str = None):
        self.config = Config()
        # Use config for file paths
        self.input_file = input_file or os.path.join(self.config.DATA_PATH, "complete_UCSB_data.json")
        self.output_file = os.path.join(self.config.DATA_PATH, "processed_documents.json")
    
    def process_data(self) -> List[Dict[str, Any]]:
        """Transform hierarchical UCSB data into flat document list"""
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        # Load data
        with open(self.input_file, 'r', encoding='utf-8') as f:
            departments = json.load(f)
        
        documents = []
        
        for dept in departments:
            dept_name = dept['name']
            dept_code = dept['code']
            dept_overview = dept['overview']
            
            # 1. Create department overview document
            dept_doc = {
                "doc_id": f"dept_{dept_code}",
                "content": f"Department: {dept_name} ({dept_code})\n\nOverview: {dept_overview}",
                "metadata": {
                    "type": "department",
                    "department": dept_name,
                    "department_code": dept_code,
                    "source": "UCSB College of Engineering"
                }
            }
            documents.append(dept_doc)
            
            # 2. Create program documents
            for i, program in enumerate(dept.get('programs', [])):
                prog_doc = {
                    "doc_id": f"prog_{dept_code}_{i}",
                    "content": f"Program: {program['name']} ({program['code']})\n"
                              f"Department: {dept_name}\n"
                              f"URL: {program['url']}\n\n"
                              f"Description: {program['description']}",
                    "metadata": {
                        "type": "program",
                        "department": dept_name,
                        "department_code": dept_code,
                        "program": program['name'],
                        "program_code": program['code'],
                        "url": program['url'],
                        "source": "UCSB College of Engineering"
                    }
                }
                documents.append(prog_doc)
            
            # 3. Create course documents
            for i, course in enumerate(dept.get('courses', [])):
                course_content = f"Course: {course['name']} - {course['title']}\n"
                course_content += f"Department: {dept_name}\n"
                course_content += f"Units: {course['units']}\n"
                course_content += f"Grading: {course['grading']}\n\n"
                course_content += f"Description: {course['description']}\n"
                
                if course.get('prerequisites'):
                    course_content += f"\nPrerequisites: {course['prerequisites']}"
                
                course_doc = {
                    "doc_id": f"course_{dept_code}_{i}",
                    "content": course_content,
                    "metadata": {
                        "type": "course",
                        "department": dept_name,
                        "department_code": dept_code,
                        "course_name": course['name'],
                        "course_title": course['title'],
                        "units": course['units'],
                        "grading": course['grading'],
                        "source": "UCSB College of Engineering"
                    }
                }
                documents.append(course_doc)
        
        print(f"Processed {len(documents)} documents:")
        dept_count = len([d for d in documents if d['metadata']['type'] == 'department'])
        prog_count = len([d for d in documents if d['metadata']['type'] == 'program'])
        course_count = len([d for d in documents if d['metadata']['type'] == 'course'])
        print(f"- {dept_count} departments")
        print(f"- {prog_count} programs")
        print(f"- {course_count} courses")
        
        return documents
    
    def save_processed_data(self, documents: List[Dict[str, Any]]):
        """Save processed documents to JSON file"""
        # Ensure data directory exists using config
        os.makedirs(self.config.DATA_PATH, exist_ok=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        
        print(f"Saved processed documents to {self.output_file}")
    
    def preview_documents(self, documents: List[Dict[str, Any]], n: int = 3):
        """Preview first n documents"""
        print(f"\nPreview of first {n} documents:")
        for i, doc in enumerate(documents[:n]):
            print(f"\n--- Document {i+1} ---")
            print(f"ID: {doc['doc_id']}")
            print(f"Type: {doc['metadata']['type']}")
            print(f"Content (first 200 chars): {doc['content'][:200]}...")
            print(f"Metadata: {doc['metadata']}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        if not os.path.exists(self.output_file):
            return {"status": "not_processed", "documents": 0}
        
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            stats = {
                "status": "processed",
                "total_documents": len(documents),
                "types": {},
                "departments": {}
            }
            
            for doc in documents:
                doc_type = doc['metadata']['type']
                dept = doc['metadata'].get('department', 'Unknown')
                
                stats["types"][doc_type] = stats["types"].get(doc_type, 0) + 1
                stats["departments"][dept] = stats["departments"].get(dept, 0) + 1
            
            return stats
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

def main():
    """Main function for direct execution"""
    processor = UCSBDataProcessor()
    
    try:
        # Process the data
        print(f"Processing data from: {processor.input_file}")
        documents = processor.process_data()
        
        # Preview some documents
        processor.preview_documents(documents)
        
        # Save processed data
        processor.save_processed_data(documents)
        
        print(f"\n✅ Processing complete!")
        print(f"Ready for embedding! Next run:")
        print(f"  python -m src.core.embeddings")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure your data file exists at: {processor.input_file}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Usage
if __name__ == "__main__":
    main()