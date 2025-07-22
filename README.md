# UCSB College of Engineering RAG Chatbot

An intelligent conversational AI assistant specifically designed for UCSB College of Engineering students, faculty, and prospective students. This RAG (Retrieval-Augmented Generation) chatbot provides accurate, source-backed information about courses, programs, prerequisites, and departmental details.

## Table of Contents

## Overview

## Why Did I Build This?

### Key Capabilties 

## Features

### Core Functionality
- **Intelligent Q&A**: Natural language queries about UCSB College of Engineering departments, programs, and courses
- **Source Attribution**: All responses backed by most recent (2024 - 2025) official UCSB documents and catalog
- **Real-time Chat**: Interactive Streamlit interface with full conversation history
- **Semantic Search**: Advanced document retrieval using vector embeddings
- **Multi-Department Support**: Covers all engineering departments (CS, ECE, ME, etc.)

### User Experience
- **Responsive Design**: UCSB branded user interface with school colors and styling
- **Full Chat History**: Persistent conversation storage within sessions
- **Export Functionality**: Download chat transcripts for later reference
- **Sample Questions**: Quick start prompts for common queries for user
- **System Diagnostics**: Real-time status monitoring and error handling

### Technical Features
- **RAG Pipeline**: Combines retrieval and generation for accurate responses
- **Vector Database**: ChromaDB for efficient similarity search
- **Modern LLM**: Google Gemini 1.5 Flash for high-quality responses with little to no relative cost
- **Modular Architecture**: Clean separation of concerns with organized codebase
- **Error Handling**: Graceful failures and thorough error management

## System Architecture

```
ARCHTIECRURE DIAGRAM HERE
```

## Demo GIFs

DEMO GIFS HERE

### Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python & LangChain  
- **LLM**: Google Gemini 1.5 Flash
- **Vector DB**: ChromaDB
- **Embeddings**: Google Embedding-001

## Project Structure

```
UCSB-RAG-CHATBOT/
├── src/
│   ├── config/
│   │   ├── prompts.py
│   │   └── settings.py
│   ├── core/
│   │   ├── embeddings.py
│   │   ├── rag_pipeline.py
│   │   └── response_generator.py
│   ├── data/
│   │   ├── data_cleaner.py
│   │   ├── data_processor.py
│   │   ├── data_scraper.js
│   │   └── data_validator.py
│   └── utils/
│       └── api_tester.py
├── styles/
│   └── app.css
├── tests/
│   ├── test_embeddings.py
│   └── test_results.json
├── .env
├── .gitignore
├── app.py
├── LICENSE
├── package-lock.json
├── package.json
├── README.md
└── requirements.txt
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- JavaScript (Web Scraper)
- Google API key (Gemini)
- ChromaDB
- LangChain

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/ucsb-rag-chatbot.git
   cd ucsb-rag-chatbot
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

5. **Process Documents**
   ```bash
   python src/core/embeddings.py
   ```

6. **Launch Application**
   ```bash
   streamlit run app.py
   ```

The application will be available at `http://localhost:8501`

## Requirements

### Dependencies
```
streamlit>=1.28.0
python-dotenv>=1.0.0
chromadb>=0.4.15
langchain>=0.0.350
langchain-community>=0.0.10
google-generativeai>=0.3.0
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
pathlib2>=2.3.0
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional
COLLECTION_NAME=ucsb_engineering_docs
PERSIST_DIRECTORY=./chroma_db
MODEL_NAME=gemini-1.5-flash
EMBEDDING_MODEL=models/embedding-001
```

## System Requirements

**Hardware Requirements:**
- **CPU**: Sufficient enough for embedding data
- **RAM**: Sufficient enough for embedding data
- **Storage**: Sufficient amount for documents and embeddings 

## Usage

### Basic Queries
- "What CS courses are available?"
- "Tell me about the ME program"
- "Prerequisites for ECE 10A?"
- "Computer Engineering requirements"

### Advanced Features

#### Chat History Management
- View complete conversation history
- Export chat transcripts
- Clear history for new sessions

#### System Controls
- Monitor system status
- Restart RAG pipeline
- View document metrics

#### Source Verification
- All responses include specific source documents
- Expandable source details
- Department and document type attribution

## Testing

### Run Test Suite
```bash
# All tests
pytest

# Specific test categories
pytest tests/test_embeddings.py
pytest src//core/rag_pipeline.py

```

### Manual Testing
```bash
# Test embeddings generation
python src/core/embeddings.py --test

# Test RAG pipeline
python src/core/rag_pipeline.py --query "test query"

# Test API connection
python src/utils/api_tester.py
```

## Performance

### Benchmarks
- **Response Time**: Approximately 2-5 seconds per query

### Optimization Tips
- Pre-generate embeddings for faster startup
- Use SSD storage for vector database
- Monitor API rate limits
- Cache frequent queries

### Production Considerations
- Integration with UCSB GOLD system
- Personal academic planning assistance
- Include academic and recreational club information
- Expand to UCSB College of Letters & Science
- Expand to UCSB College of Creative Studies

## Troubleshooting Configurations

### Common Issues

#### "Failed to initialize the system"
- **Solution**: Verify Google API key in `.env`
- **Check**: Run embeddings script first
- **Verify**: All dependencies installed correctly

#### "No documents found"
- **Solution**: Run `python src/core/embeddings.py`
- **Check**: Data folder contains UCSB documents
- **Verify**: ChromaDB permissions and storage

#### "Import errors"
- **Solution**: Activate virtual environment
- **Check**: Install all requirements
- **Verify**: Python version compatibility

## Contributing

This project was developed as a personal learning project. For future questions and/or suggestions:

1. Open an issue describing the enhancement or bug
2. Fork the repository and create a feature branch
3. Follow coding standards
4. Write tests for new functionality
5. Update documentation as needed
6. Submit a pull request with detailed description of changes

## License

This project is open source and available under the MIT License.

## Author

**Ryan Fabrick**
- Statistics and Data Science (B.S) Student, University of California Santa Barbara
- GitHub: [https://github.com/RyanFabrick](https://github.com/RyanFabrick)
- LinkedIn: [www.linkedin.com/in/ryan-fabrick](https://www.linkedin.com/in/ryan-fabrick)
- Email: ryanfabrick@gmail.com

## Acknowledgments & References

- **[UCSB General Catalog](https://catalog.ucsb.edu/)** - info here
- **[Google AI Studio](https://aistudio.google.com/)** - info here
- **[Google Gemini](https://gemini.google.com/)** - info here
- **[ChromaDB](https://www.trychroma.com/)** - info here
- **[LangChain](https://www.langchain.com/)** - info here
- **[Puppeteer Community](https://pptr.dev/)** - info here
- **[Streamlit Community](https://flask.palletsprojects.com/)** - info here

________________________________________
Built with ❤️ for the UCSB community

