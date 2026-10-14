# AgentAssist — AI-Powered Document Research and Task Automation Assistant

AgentAssist is an AI-powered assistant that combines **Retrieval-Augmented Generation (RAG), agentic workflows, and task automation** to help users interact with documents and perform useful tasks through a simple web interface.

Built using Python, Streamlit, LangGraph, ChromaDB, Sentence Transformers, and the Groq API, AgentAssist can answer document-related questions, perform calculations, generate structured reports, and respond to general queries.

## Features

- **Document Question Answering:** Upload PDF and TXT files and ask questions about their content.
- **Retrieval-Augmented Generation (RAG):** Retrieves relevant document chunks to provide context-aware answers.
- **Intelligent Query Routing:** Uses LangGraph to direct queries to the appropriate processing workflow.
- **Calculator Tool:** Evaluates supported mathematical expressions using a controlled calculation tool.
- **Report Generation:** Generates structured reports using relevant retrieved information.
- **General AI Assistant:** Handles general-purpose questions beyond document-based queries.
- **Source References:** Displays relevant document sources for retrieved information.
- **Execution Trace:** Provides visibility into the selected route and processing steps.
- **Web-Based Interface:** Interact with the assistant through a Streamlit application.
- **Cloud Deployment:** Hosted using Streamlit Community Cloud with Groq API integration.

## Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Agent Workflow | LangGraph |
| Large Language Model API | Groq API |
| Language Model | `openai/gpt-oss-20b` |
| Document Processing | PyPDF |
| Text Embeddings | Sentence Transformers |
| Embedding Model | `all-MiniLM-L6-v2` |
| Vector Database | ChromaDB |
| Testing | Pytest |
| Deployment | Streamlit Community Cloud |
| Version Control | Git and GitHub |

## System Architecture

```text
             User
               |
               v
       Streamlit Interface
               |
               v
        LangGraph Agent
               |
               v
        Query Classification
               |
       +-------+--------+----------------+
       |       |        |                |
       v       v        v                v
   Document  Calculator Report        General
    Search     Tool    Generation      Chat
       |       |        |                |
       v       v        v                v
   ChromaDB  Result   Retrieved       Groq API
   Retrieval          Context
       |                |
       +-------+--------+
               |
               v
       Generated Response
               |
               v
     Answer, Sources and Trace
```

The agent routes incoming questions to the appropriate workflow. Document-related queries use retrieval to obtain relevant context, calculator queries use the calculation tool, report requests generate structured output, and general queries use the language model directly.

## How It Works

### 1. Document Upload and Processing

Users upload PDF or TXT files through the Streamlit interface. The application extracts the text and divides it into smaller chunks suitable for retrieval.

### 2. Text Embedding and Indexing

The extracted chunks are converted into numerical embeddings using the `all-MiniLM-L6-v2` Sentence Transformer model. ChromaDB stores these embeddings for similarity-based retrieval.

### 3. Agent Routing

LangGraph manages the workflow and directs each query to the appropriate node based on its intent.

The supported workflows include:

- **Document Search:** Retrieves relevant chunks and uses them as context for answering questions.
- **Calculator:** Processes supported mathematical expressions through a dedicated tool.
- **Report Generation:** Creates structured reports using relevant information.
- **General Chat:** Handles general questions using the language model.

### 4. Response Generation

The Groq API provides language-model responses. For document queries, retrieved context helps ground the answer in the uploaded material.

### 5. Source and Execution Visibility

The interface displays the generated response along with relevant sources and an execution trace to help users understand how the query was processed.

## Project Structure

```text
AgentAssist/
├── app.py                  # Streamlit application and UI
├── agent.py                # LangGraph workflow and routing
├── rag.py                  # Document processing and retrieval
├── tools.py                # Calculator and report tools
├── llm.py                  # Groq API integration
├── requirements.txt        # Python dependencies
├── tests/
│   ├── test_tools.py       # Tool and calculator tests
│   ├── test_retrieval.py   # Retrieval tests
│   └── test_routing.py     # Agent routing tests
├── .gitignore              # Excluded files and secrets
└── README.md               # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.12 recommended
- Git
- A Groq API key

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/AgentAssist.git
cd AgentAssist
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Obtain an API key from the [Groq Console](https://console.groq.com/).

**Security:** Never commit `.env` or expose your API key in source code or public repositories.

### 5. Run the Application

```bash
streamlit run app.py
```

Streamlit will provide a local URL, usually `http://localhost:8501`, where you can access AgentAssist.

## Running Tests

Run the test suite from the project root:

```bash
pytest -v
```

The tests cover agent routing, document retrieval, and tool execution.

## Key Concepts Demonstrated

AgentAssist demonstrates concepts from Generative AI, Natural Language Processing, RAG, and Agentic AI:

- **Prompt Engineering:** Designing instructions that guide language-model responses.
- **Embeddings and Semantic Search:** Representing text numerically and retrieving semantically relevant content.
- **Retrieval-Augmented Generation:** Combining retrieved document context with language-model generation.
- **Agentic Workflows:** Using LangGraph to organize query processing into task-specific nodes.
- **Tool Calling:** Connecting AI workflows to calculation and report-generation functions.
- **Workflow Routing:** Directing different query types to appropriate processing paths.
- **Testing and Evaluation:** Testing routing, retrieval, and tool behavior.

## Future Scope

Potential future improvements include:

- More advanced intent classification and routing.
- Support for additional document formats.
- Improved retrieval evaluation and answer-quality measurement.
- Expanded tool integrations and workflow automation.
- Enhanced source attribution and response validation.

## License

This project is intended for educational and demonstration purposes. Add a license file if you wish to specify permissions for reuse and redistribution.
