# DocStudio Quick Start Guide

## Start the Application with Docker

### One-Command Launch
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your OpenAI/Azure API credentials
nano .env  # or vim, code, etc

# Start both API and UI
docker compose up
```

Access the application:
- API: http://localhost:8000
- UI: http://localhost:8501
- API Docs: http://localhost:8000/docs

The UI automatically connects to the backend

## Requirements

1. Docker and Docker Compose ([Install here](https://docs.docker.com/get-docker/))
2. OpenAI or Azure OpenAI API Key
   - Azure: Create resource at [portal.azure.com](https://portal.azure.com)
   - OpenAI: Get key at [platform.openai.com](https://platform.openai.com)

## Configuration

Minimum required in .env:
```bash
OPENAI_API_KEY=your_key_here
OPENAI_API_BASE=https://your-resource.openai.azure.com
OPENAI_API_TYPE=azure
OPENAI_API_DEPLOYMENT_NAME=gpt-4o
```

Full configuration options in .env.example

## System Architecture

```
orchestrate.py (Core Logic)
    |
    v
trace.py (Logging) + policy.py (RL Bandit)
    |
    v
api/app.py (REST API)
    |
    v
ui/app.py (Streamlit UI)
```

## Key Files

| File | Purpose | Key Function |
|------|---------|--------------|
| orchestrate.py | Core RAG orchestration | Orchestrator.run() |
| trace.py | Event logging | TraceLogger.save() |
| policy.py | Bandit policy | EpsilonGreedyPolicy.choose() |
| api/app.py | REST backend | POST /orchestrate |
| ui/app.py | Web interface | Streamlit app |

## Using the Application

### 1. Upload Documents
Via UI: Go to "Upload Docs" tab and select .txt or .md files

Via API:
```bash
curl -X POST http://localhost:8000/upload_docs \
  -H "Content-Type: application/json" \
  -d '{
    "docs": [{
      "doc_id": "paper1",
      "text": "Your document text here",
      "meta": {"title": "Paper Title"}
    }]
  }'
```

### 2. Run a Query
Via UI: Go to "Studio" tab, enter question, click "Run Orchestrator"

Via API:
```bash
curl -X POST http://localhost:8000/orchestrate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is reinforcement learning?",
    "arm_name": "default",
    "arm_params": {"k": 5, "mmr_lambda": 0.6}
  }'
```

### 3. Run RL Experiments
Go to "RL Tuning Lab" tab in UI:
- Configure arms (different retrieval settings)
- Set number of iterations
- Watch learning curves in real-time

## Common Commands

### Docker Management
```bash
# Start in background
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down

# Rebuild after code changes
docker compose up --build

# Reset everything
docker compose down -v
rm -rf data/ mlruns/
```

### Health Checks
```bash
# API health
curl http://localhost:8000/health

# Check document count
curl http://localhost:8000/health | jq '.docs_indexed'
```

## Important Data Files

| File | Contains | Location |
|------|----------|----------|
| index.faiss | Vector embeddings | data/ |
| index.meta.jsonl | Document metadata | data/ |
| traces.jsonl | Detailed run logs | data/ |
| rewards.csv | Quality/reward per run | data/ |
| rl_bandit_rewards.csv | Bandit policy stats | data/ |
| docstudio.db | Query history (SQLite) | project root |
| mlruns/ | MLflow experiments | project root |

Note: These are auto-created on first run. All are gitignored

## Troubleshooting

| Problem | Solution |
|---------|----------|
| UI shows "Cannot connect" | 1. Check API is running: docker compose ps<br>2. Wait 30s for startup<br>3. Check API logs: docker compose logs api |
| "Module not found" | Run docker compose up --build to rebuild |
| FAISS errors | Delete data/index.faiss and restart |
| Port already in use | Change ports in docker-compose.yml |
| Out of memory | Reduce batch size or use smaller embedding model |

### Debug Mode
```bash
# Enable verbose logging
echo "LOG_LEVEL=DEBUG" >> .env
docker compose up
```

## Local Development (without Docker)

If you prefer to run locally:

```bash
# 1. Install dependencies
pip install -e .

# 2. Set environment variables
export OPENAI_API_KEY=your_key
export OPENAI_API_BASE=https://your-resource.openai.azure.com
# ... other vars from .env.example

# 3. Start API (Terminal 1)
cd src
uvicorn doc_studio.api.app:app --reload --port 8000

# 4. Start UI (Terminal 2)
streamlit run src/doc_studio/ui/app.py --server.port 8501
```

## Learn More

- Architecture Details: See README.md
- API Reference: http://localhost:8000/docs (when running)
- Component Demos: examples/component_demos/
- Evaluation Scripts: examples/evaluation/

## Quick Test

1. Start both services
2. UI sidebar shows "Connected" status
3. Upload a document or use seed corpus
4. Go to "Studio" tab
5. Enter: "What is reinforcement learning?"
6. Click "Run Orchestrator"
7. View results

## Docker Architecture

```
docker-compose.yml
    |
    +-------+-------+
    |               |
   API             UI
  :8000          :8501
    |               |
    +-------+-------+
            |
         data/
        mlruns/
```

Networking:
- UI connects to API via http://api:8000 (internal Docker network)
- You connect via localhost:8000 and localhost:8501

Need help? Check README.md for more details
