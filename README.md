# HR Policy RAG Assistant

A Streamlit based HR policy assistant built with LangChain. It answers employee questions by retrieving relevant chunks from an HR policy document and passing them to an LLM agent.

Live app: https://v1-ops-hr-rag-assistant-app-4jyezi.streamlit.app/

The project includes:

- RAG pipeline over `data/hr_policy.txt`
- Qdrant vector store for document retrieval
- Groq hosted LLM for answer generation
- Jina embeddings for document embeddings
- Input and output guardrails
- LangSmith observability
- LangSmith evaluation with correctness and groundedness checks

## Project Flow

1. Load the HR policy document from `data/hr_policy.txt`.
2. Split the document into chunks.
3. Create embeddings using Jina.
4. Store and reuse vectors in Qdrant.
5. Create a retriever tool for HR policy search.
6. Use a LangChain agent with Groq to answer questions.
7. Run guardrails before and after the agent response.
8. Log and trace the run for observability.

## Features

### RAG Assistant

The assistant searches the HR policy document before answering. If the Qdrant collection already exists, the app connects to it directly instead of rebuilding embeddings.

Main files:

- `hr_assistant/pipeline.py`
- `hr_assistant/vector_store.py`
- `hr_assistant/tools.py`
- `hr_assistant/agent.py`

### Guardrails

Guardrails are implemented in `hr_assistant/Guardrails.py`.

The app checks:

- User input before sending it to the agent
- Agent output before showing it in the UI

The guardrails block unsafe requests such as prompt injection, requests for other employee private data, unsafe output, suspicious links, and unauthorized promises.

### Observability

Observability is implemented through logging and LangSmith tracing.

Logging:

- Logs are written to the `logs/` folder.
- Each run creates a timestamped log file.

LangSmith:

- Tracing can be enabled using environment variables.
- Traces help inspect agent calls, retrieval, tool usage, and evaluation runs.

Main files:

- `hr_assistant/logger.py`
- `hr_assistant/tracing.py`

### Evaluation

Evaluation is implemented in `hr_assistant/evaluation.py`.

The evaluation uses LangSmith datasets and checks:

- Correctness against expected answers
- Groundedness against retrieved context

Run it with:

```bash
python evaluate.py
```

## Setup

Create and activate the virtual environment:

```bash
pip install uv
uv venv basicragenv
basicragenv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root.

Required:

```env
GROQ_API_KEY=your_groq_api_key
JINA_API_KEY=your_jina_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=hr_policy
```

Optional for LangSmith observability:

```env
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=hr_rag
```

## Run the Streamlit App

```bash
basicragenv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL shown in the terminal.

## Run the CLI Demo

```bash
python main.py
```

## Run Evaluation

```bash
python evaluate.py
```

The evaluation results will be available in your LangSmith project.

## Folder Structure

```text
HR_RAG/
  app.py
  main.py
  evaluate.py
  requirements.txt
  data/
    hr_policy.txt
  hr_assistant/
    agent.py
    config.py
    document_loader.py
    embeddings.py
    evaluation.py
    Guardrails.py
    llm.py
    logger.py
    pipeline.py
    splitter.py
    tools.py
    tracing.py
    vector_store.py
```
