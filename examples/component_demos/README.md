# Component Demos

Quick demonstrations of individual DocStudio components.

## Multi-Armed Bandit RL Demo

**File**: `demo_bandit.py`

Demonstrates epsilon-greedy and UCB1 reinforcement learning for RAG hyperparameter optimization.

### Quick Run

```bash
# From project root
python examples/component_demos/demo_bandit.py
```

### What It Shows

1. **Epsilon-Greedy Learning**: Policy learns to identify best RAG config through exploration-exploitation with decaying ε
2. **UCB1 Learning**: Confidence bound-based selection with theoretical regret guarantees
3. **Comparison**: Side-by-side performance comparison

### Expected Output

You'll see:
- Real-time learning progress (iteration by iteration)
- Running mean rewards converging to true values
- Final rankings showing discovered best configuration
- Comparison of both algorithms

### Output Files

- `data/demo_bandit_epsilon.csv` - Epsilon-greedy learning curve
- `data/demo_bandit_ucb.csv` - UCB1 learning curve

Format: `ts_utc, arm, reward, count, mean_reward, epsilon/ucb_score`

### Learning Curve

The demo simulates 4 RAG configurations with different "true" rewards:
- `baseline` (TF-IDF): 0.55 reward
- `embeddings`: 0.68 reward
- `mmr_balanced`: 0.75 reward (best)
- `efficient`: 0.62 reward

Watch as the policy discovers `mmr_balanced` is best!

### What This Demonstrates

**For Resume/Interviews**:
- Understanding of exploration-exploitation tradeoffs
- Implementation of two bandit algorithms (epsilon-greedy, UCB1)
- Knowledge of theoretical guarantees (regret bounds)
- Ability to apply RL to practical hyperparameter optimization

**Key Concepts Shown**:
- Incremental mean updates
- Decaying exploration (ε = ε₀/√t)
- Upper confidence bounds
- Regret analysis

### Understanding the Output

The demo simulates 4 RAG configurations with different true rewards:
- baseline (TF-IDF): 0.55 reward
- embeddings: 0.68 reward
- mmr_balanced: 0.75 reward (best)
- efficient: 0.62 reward

**Epsilon-Greedy Output Columns**:
- Iter: Iteration number
- Arm: Which configuration was chosen
- Reward: Observed reward (true reward + noise)
- Count: How many times this arm has been tried
- Mean: Running average reward for this arm
- Epsilon: Current exploration rate (decays over time)
- Action: Whether this was exploration or exploitation

**UCB1 Output Columns**:
- Iter: Iteration number
- Arm: Which configuration was chosen
- Reward: Observed reward
- Count: How many times this arm has been tried
- Mean: Running average reward for this arm
- UCB Score: Upper confidence bound = mean + c*sqrt(log(T)/n)

**What to Look For**:
- Both algorithms should discover mmr_balanced as best (highest mean reward)
- Epsilon-greedy: Mean rewards converge to true values as count increases
- UCB1: Arms with fewer samples get higher UCB scores (exploration bonus)
- Final results: mmr_balanced should be ranked #1 with most trials

---

## Other Demos

### Orchestrator Demo
**File**: `smoke_orchestrate.py`

Tests the full multi-agent workflow: orchestrator → retriever → summarizer → analyst.

```bash
python examples/component_demos/smoke_orchestrate.py
```

### MLflow Demo
**File**: `smoke_mlflow.py`

Tests optional MLflow experiment tracking integration.

```bash
# Set environment variable first
export DOCSTUDIO_MLFLOW=1
python examples/component_demos/smoke_mlflow.py
```

---

## Troubleshooting

**Import errors**: Make sure you're running from project root and have installed dependencies:
```bash
pip install -e .
```

**No output files**: Check that `data/` directory exists (created automatically on first run).

**Module not found**: Ensure you're in the project root directory when running.
