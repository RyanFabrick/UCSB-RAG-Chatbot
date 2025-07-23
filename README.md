# UCSB College of Engineering RAG Chatbot

A conversational Artifical Intelligence assistant specifically designed for UCSB College of Engineering students, faculty, and prospective students. This RAG (Retrieval-Augmented Generation) chatbot provides up to date, accurate, source-backed detailed and thorough information about departments, programs, and course information.

## Table of Contents

- [Overview](#overview)
- [Why Did I Build This?](#why-did-i-build-this)
- [Key Capabilities](#key-capabilties)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Demo GIFs](#demo-gifs)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [System Requirements](#system-requirements)
- [Usage](#usage)
- [Testing](#testing)
- [Performance](#performance)
- [Troubleshooting Configurations](#troubleshooting-configurations)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Acknowledgments & References](#acknowledgments--references)

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


## System Architecture

```
ARCHTIECRURE DIAGRAM HERE
```

## Demo GIFs

DEMO GIFS HERE

## Technology Stack

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

- Python 3.8 or higher (Backend and Streamlit)
- Node.js 16 or higher (Web Scraper)
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

### System Metrics
- **Document Collection**: 314 engineering documents indexed
- **Response Time**: Approximately 2-5 seconds per query
- **Response Length**: 800-1,800 characters average
- **Sources Retrieved**: 5 relevant documents per query

### Test Results
**Overall System Performance**: 85.7% success rate

**Component Performance:**
- **Retrieval Accuracy**: 3/3 passed (100%) - Correctly identifies relevant documents
- **Response Generation**: 4/4 passed (100%) - High-quality responses with proper formatting
- **End-to-End Pipeline**: 4/4 passed (100%) - Complete user query processing
- **Embedding Quality**: 1/3 passed (33%) - Some cross-domain similarity challenges

**Cosine Similarity Analysis:**
- **High Similarity** (>0.80): "computer science courses" vs "CS classes available" (0.819)
- **Medium Similarity** (0.65-0.80): "mechanical engineering program" vs "ME department overview" (0.698)
- **Cross-Domain** (<0.70): "computer science courses" vs "mechanical engineering program" (0.683) - Expected low similarity
- **Similarity Threshold**: 0.80 for reliable semantic matching

**Quality Scores:**
- Machine Learning course recommendations: 1.00
- Computer Engineering program queries: 1.00  
- Materials department requirements: 1.00
- Circuit-related course searches: 1.00

### Known Limitations
- Cross-departmental query similarity requires optimization (e.g., CS vs ME topics)
- Embedding model may need fine-tuning for domain-specific engineering terminology

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

- **[UCSB General Catalog](https://catalog.ucsb.edu/)** - Official academic catalog published by UCSB's Registrar's Office, containing comprehensive course descriptions, academic requirements, and program information for all colleges and majors at UCSB
- **[Google AI Studio](https://aistudio.google.com/)** - Google's platform for experimenting with Generative AI models including the Gemini family, providing direct API access, prototyping capabilities, and billing & usage information
- **[Google Gemini](https://gemini.google.com/)** - Google's generative AI model family offering powerful text generation capabilities, integrated with LangChain for building GenAI applications with function calling
- **[ChromaDB](https://www.trychroma.com/)** - An open-source vector database designed for storing and querying embeddings, enabling efficient similarity search and retrieval-augmented generation workflows
- **[LangChain](https://www.langchain.com/)** - A framework for developing LLM-powered applications by connecting with external data sources, providing chains and agents for complex reasoning and information processing
- **[Puppeteer Community](https://pptr.dev/)** - A Node.js library providing an API for controlling Chrome/Chromium browsers, essential for web scraping and automated data collection from dynamic web pages
- **[Streamlit Community](https://flask.palletsprojects.com/)** - An open-source Python framework for building and deploying interactive web applications with seamless integration for AI and machine learning projects

________________________________________
Built with ❤️ for the UCSB community

