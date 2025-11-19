# DocStudio: RL-Tuned RAG Orchestrator

**Multi-agent retrieval-augmented generation system with reinforcement learning-based hyperparameter optimization**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DocStudio demonstrates rigorous evaluation and efficiency optimization in RAG systems: systematic ablation studies across 5 configurations, cost-aware reward functions balancing quality vs latency/tokens, and reproducible benchmarks with comprehensive statistical analysis (mean, std, median, p95)

## Key Features

- Rigorous Evaluation: 5 ablation configs, fixed-seed reproducibility, comprehensive metrics (mean/std/median/p95) for quality and efficiency
- Efficiency Optimization: Cost-aware reward function penalizing tokens and latency, achieving 21.3% quality improvement with 15% token reduction
- RL-Based Tuning: Epsilon-greedy (1/sqrt(t) decay) and UCB1 policies with O(sqrt(KT log T)) regret bounds for hyperparameter optimization
- Multi-Agent Architecture: Coordinated Summarizer, Analyst (Critic), and Orchestrator with retry logic and quality thresholds
- Advanced Retrieval: FAISS vector search with MMR diversity reranking for coverage-relevance tradeoffs
- Production-Ready: Docker Compose deployment, structured event tracing, REST API with interactive dashboard

## Performance Results

Evaluated on 15-query research paper corpus with fixed seeds (seed=7) for reproducibility

| Configuration | Overall Quality | Tokens | Latency | Reward | vs Baseline |
|--------------|----------------|---------|---------|---------|-------------|
| Embeddings + MMR | 0.822 | 1,106 | 11.9s | 0.699 | +31.4% |
| Embeddings Only | 0.782 | 1,123 | 11.9s | 0.664 | +24.8% |
| Baseline (TF-IDF) | 0.678 | 1,301 | 15.8s | 0.532 | baseline |

Key Insights (Quality vs Efficiency Tradeoffs):
- **Best quality-efficiency tradeoff**: Embeddings + MMR achieves 21.3% quality improvement while reducing tokens by 15.0% and latency by 24.7%
- **Stability**: Lowest variance (±0.167) across all metrics, indicating robust performance
- **Reliability**: Reduced retry rate from 53.3% to 26.7% (50% reduction in failed generations)
- **Cost awareness**: Reward function successfully balances quality gains against token/latency costs

Statistical Rigor:
- All configs evaluated on identical query set with fixed random seed
- Metrics tracked: mean, standard deviation, median, 95th percentile
- Results reproducible via `python examples/evaluation/offline_eval.py`

## Architecture

```
User Query
    |
    v
Orchestrator (orchestrate.py)
  - Configures retrieval params
  - Manages retry logic with quality thresholds
  - Computes efficiency-aware rewards
    |
    +------------------+
    |                  |
    v                  v
SummarizerAgent    AnalystAgent
- FAISS retrieval  - Scores output (faithfulness, coverage, clarity)
- MMR reranking    - Suggests fixes via JSON-strict prompting
- GPT-4o LLM       - LLM-as-judge pattern
    |                  |
    +--------+---------+
             |
             v
    Bandit Policy (policy.py)
    - Epsilon-greedy or UCB1 selection
    - Online learning of optimal params
```

Core Components:
- Orchestrator: Coordinates agent workflow, implements retry logic with quality thresholds
- Summarizer Agent: Retrieves top-k docs via FAISS, applies MMR diversity reranking, generates citations
- Analyst Agent: LLM-as-judge pattern scoring faithfulness, coverage, clarity
- Bandit Policy: Multi-armed bandit for online learning of optimal retrieval configurations
- Trace Logger: Structured event logging for debugging and analysis

## Quick Start

Prerequisites:
- Docker and Docker Compose
- OpenAI or Azure OpenAI API key

1. Clone and Configure

```bash
git clone https://github.com/Krupa-pv/DocStudio.git
cd SimplifyResearch

cp .env.example .env
# Edit .env with your API credentials
```

2. Launch with Docker Compose

```bash
docker compose up
```

Services:
- API: http://localhost:8000 (FastAPI backend)
- UI: http://localhost:8501 (Streamlit dashboard)

3. Use the Application

- Upload Documents (tab 1): Add txt or md files to build your knowledge base
- Studio (tab 2): Run single queries with customizable parameters
- RL Tuning Lab (tab 3): Train bandit policy over multiple configurations
- Evaluation (tab 4): View quality-vs-efficiency metrics and RL learning curves
- History (tab 5): Browse query logs and aggregate statistics

## Technical Stack

| Layer | Technologies |
|-------|-------------|
| LLM | Azure OpenAI (GPT-4o), OpenAI API |
| Embeddings | Hugging Face Transformers (BGE-small-en-v1.5) |
| Vector Search | FAISS (FlatIP index), MMR reranking |
| Backend API | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit, Altair charts |
| Logging | SQLite, MLflow, structured JSON traces |
| Deployment | Docker, Docker Compose |
| Language | Python 3.11+ |

## Project Structure

```
SimplifyResearch/
├── src/doc_studio/
│   ├── agents/               # AI agent implementations
│   │   ├── summarizer.py     # RAG with MMR
│   │   └── analyst.py        # LLM-as-judge critic
│   ├── orchestrator/         # Coordination logic
│   │   ├── orchestrate.py    # Main workflow controller
│   │   └── policy.py         # Epsilon-greedy and UCB1 bandits
│   ├── retrieval/            # Vector search components
│   │   ├── faiss_retriever.py
│   │   ├── hf_embedder.py
│   │   └── mmr.py            # Diversity reranking
│   ├── core/                 # Utilities (config, LLM client, types)
│   ├── api/                  # FastAPI application
│   └── ui/                   # Streamlit dashboard
├── examples/                 # Demo scripts
│   ├── component_demos/      # Individual component tests
│   ├── evaluation/           # Offline benchmarking
│   └── data_preparation/     # Corpus building
├── docker-compose.yml        # Production deployment
├── pyproject.toml            # Python packaging config
└── .env.example              # Configuration template
```

## Evaluation and Optimization Methodology

### Cost-Aware Reward Function
```python
reward = overall_quality - alpha * (tokens/1000) - beta * (latency_sec)
# alpha = 0.02 (token cost penalty)
# beta = 0.01 (latency penalty)
```
Explicitly penalizes efficiency costs to drive hyperparameter optimization toward quality-efficiency Pareto frontier. The reward function balances three objectives: maximizing answer quality while minimizing both token usage (API costs) and latency (user experience)

### Rigorous Evaluation Protocol
**Ablation Study Design**:
- 5 configurations isolating effects: baseline (TF-IDF), embeddings only, embeddings+MMR, efficient MMR, aggressive efficiency
- Each config varies k (retrieval count), context_budget (chars), mmr_lambda (diversity vs relevance)
- Fixed random seed (7) ensures reproducible comparisons

**Statistical Analysis**:
- Per-query metrics: overall quality, faithfulness, coverage, clarity, tokens, latency, attempts
- Aggregate statistics: mean, standard deviation, median, 95th percentile for all metrics
- Delta calculations vs baseline for resume-ready improvement percentages
- Output: `data/eval_results.csv` (per-run), `data/eval_stats.csv` (aggregates)

**Reproducibility**:
```bash
python examples/evaluation/offline_eval.py --seed 7 --eval-file data/eval.jsonl
```

### Multi-Armed Bandit for Online Learning
- Arms represent retrieval configs (k, MMR lambda, context budget)
- Epsilon-greedy: Decaying exploration (ε = ε₀/√t) for convergence guarantees
- UCB1: Theoretical regret bound O(√(KT log T)) for optimal exploration-exploitation
- Logged to `data/rl_bandit_rewards.csv` with running statistics

## API Endpoints

```bash
# Health check
GET /health

# Upload documents
POST /upload_docs
{
  "docs": [{"doc_id": "paper1", "text": "...", "meta": {...}}]
}

# Run orchestrator
POST /orchestrate
{
  "query": "Explain MMR in RAG",
  "arm_name": "default",
  "arm_params": {"k": 5, "mmr_lambda": 0.6}
}

# RL batch experiment
POST /rl/run
{
  "queries": ["Query 1", "Query 2"],
  "epsilon": 0.1,
  "iterations": 200,
  "arms": {"A": {...}, "B": {...}}
}
```

## Development

Local Development (without Docker)

```bash
# Install dependencies
pip install -e .

# Terminal 1: Start API
cd src
uvicorn doc_studio.api.app:app --reload --port 8000

# Terminal 2: Start UI
streamlit run src/doc_studio/ui/app.py --server.port 8501
```

Run Component Demos

```bash
# Test retrieval pipeline
python examples/component_demos/smoke_orchestrate.py

# Test bandit policy
python examples/component_demos/demo_bandit.py

# Run offline evaluation
python examples/evaluation/offline_eval.py
```

Configuration

Edit .env or set environment variables:
- OPENAI_API_KEY: Your API key
- OPENAI_API_BASE: API endpoint
- DOCSTUDIO_MLFLOW=1: Enable MLflow tracking
- LOG_LEVEL=DEBUG: Verbose logging

## Technical Highlights

This project uses several concepts in applied AI:

**System Design**
Multi-agent coordination with specialized roles (retrieval, generation, critique), modular architecture enabling independent component testing and replacement

**Machine Learning Engineering**
Dense embeddings for semantic search, FAISS vector indexing for scalable retrieval, MMR algorithm for diversity-aware reranking

**Reinforcement Learning**
Multi-armed bandit algorithms (epsilon-greedy with decay, UCB1) for hyperparameter optimization, achieving theoretical regret bounds O(sqrt(KT log T))

**Production Engineering**
Docker containerization for reproducible deployments, FastAPI REST services, structured event tracing for debugging, environment-based configuration

**Evaluation Rigor**
Systematic ablation studies across 5 configurations isolating effects of embeddings vs MMR vs efficiency tradeoffs. Fixed random seeds for reproducibility. Comprehensive statistical analysis tracking mean, std, median, and p95 for all metrics (quality scores, token usage, latency, retry rates). Multi-objective evaluation combining quality, cost, and stability. Results logged to CSV with delta percentages vs baseline for resume-ready metrics

## Use Cases

This system is designed for:
- Research paper analysis and question answering
- Technical documentation search with quality guarantees
- Cost-aware production RAG where token usage matters
- Experimentation with retrieval strategies and hyperparameters
- Learning multi-agent system design and RL applications




