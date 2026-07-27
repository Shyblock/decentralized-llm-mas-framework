# Decentralized LLM Multi-Agent System with Blockchain-Enforced Collaboration

A decentralized Large Language Model (LLM) multi-agent system implementing the methodology proposed in:

> **Toward Transparent and Incentive-Compatible Collaboration in Decentralized LLM Multi-Agent Systems: A Blockchain-Driven Approach**

This project extends the paper by replacing probabilistic simulation with **real LLM-generated code execution**, **persistent agent memory**, **teacher-guided reflection**, and **on-chain enforcement** using Solidity smart contracts deployed on a local Ganache blockchain.

---

## Overview

This system simulates a decentralized ecosystem of autonomous AI agents that compete and collaborate to solve programming tasks.

Each agent independently evaluates incoming tasks, estimates its expected utility, participates in an auction, generates executable Python code using an LLM, and continuously improves through experience.

Unlike conventional simulations, success is determined through deterministic execution against hidden test cases rather than random Bernoulli sampling.

The blockchain acts as a decentralized coordination layer, ensuring transparent task assignment, immutable reputation updates, capability evolution, and auditability.

---

# Architecture

```
                    Programming Dataset
                           │
                           ▼
                Task Broadcast System
                           │
         ┌─────────────────┴─────────────────┐
         │                                   │
         ▼                                   ▼
   Autonomous Agents (20)             Blockchain Layer
         │                                   │
         │ Utility Estimation                │
         │ Auction Bidding                   │
         │ Capability Tracking               │
         │ Reputation Updates                │
         ▼                                   ▼
   Winning Agent Selected  ─────────────► Smart Contracts
         │
         ▼
  LLM Code Generation
         │
         ▼
 Deterministic Evaluation
         │
   ┌─────┴──────┐
   │            │
 Pass         Fail
   │            │
   ▼            ▼
Capability    Teacher LLM
 Update       Reflection
   │            │
   └────► Persistent Memory
```

---

# Features

## Autonomous Multi-Agent System

- 20 independent autonomous agents
- Capability vectors initialized using Beta(2,5) distribution
- Reputation-based decentralized task allocation
- Utility-driven auction mechanism
- Dynamic workload balancing
- Emergent specialization over time

---

## Real LLM Code Generation

Supports multiple providers:

- Groq
- Gemini

Each winning agent generates executable Python code directly from natural language programming tasks.

---

## Deterministic Evaluation

Instead of simulated success probabilities:

- Generated code executes inside a secure sandbox
- Hidden unit tests evaluate correctness
- Quality score is computed deterministically
- Deadline compliance contributes to performance
- Capability updates depend on actual execution

---

## Teacher-Guided Reflection

Failed attempts trigger an independent teacher LLM.

The teacher:

- analyzes failures
- classifies failure type
- generates reusable lessons
- produces task-specific repair guidance

Reflection is completely separated from task execution.

---

## Persistent Agent Memory

Each agent maintains long-term memory.

Stored lessons include:

- reusable programming knowledge
- associated capability tags
- failure categories
- originating task

Relevant lessons are retrieved before future generations using capability-tag indexing.

Duplicate lessons are automatically removed.

---

## Blockchain Enforcement Layer

Three Solidity smart contracts implement the paper's decentralized protocol.

### AgentRegistry.sol

Stores:

- agent identity
- reputation
- capability vector
- workload

---

### TaskManager.sol

Maintains:

- task submission
- assignment
- completion
- workload synchronization
- immutable event logs

---

### IncentiveContract.sol

Implements on-chain exponential moving average updates for:

- Reputation
- Capability

Implements the reputation and capability update equations described in the reference paper.

---

## Visualization

Automatic dashboard generation including:

- Task Success Rate
- Capability Match
- Utility
- Reputation
- Capability Entropy
- Bid Rate
- Allocation Delay
- Agent Capability Heatmaps
- Trend Lines
- Paper Comparison

---

# Project Structure

```
.
├── blockchain/
│   ├── contracts/
│   │   ├── AgentRegistry.sol
│   │   ├── TaskManager.sol
│   │   └── IncentiveContract.sol
│   │
│   ├── deployed/
│   ├── scripts/
│   ├── hardhat.config.js
│   └── package.json
│
├── data/
│   └── dataset.json
│
├── memory/
│   └── agent_x.json
│
├── runs/
│   └── run_timestamp/
│
├── blockchain_bridge.py
├── config.py
├── llm_provider.py
├── memory_manager.py
├── reflection.py
├── simulation_engine.py
├── run_simulation.py
├── summary_table.py
├── visualize_results.py
├── requirements.txt
└── README.md
```

---

# Dataset

The benchmark consists of programming tasks spanning:

- Dynamic Programming
- Graphs
- Trees
- Strings
- Searching
- Sorting
- Hash Maps
- Lists
- Mathematics
- Recursion

Each task contains:

- natural language description
- paraphrased prompt
- capability tags
- reference solution
- reward
- deadline
- deterministic test cases

---

# Simulation Pipeline

1. Initialize agents
2. Broadcast task
3. Agents compute expected utility
4. Auction selects winner
5. Task recorded on blockchain
6. Winning agent generates Python solution
7. Sandbox executes solution
8. Hidden test cases evaluate correctness
9. Reputation updated
10. Capability updated
11. Reflection generated if failure occurs
12. Validated lesson stored in persistent memory
13. Metrics logged
14. Dashboard generated

---

# Installation

## Clone Repository

```bash
git clone https://github.com/JayaJagriti/decentralized-llm-mas-framework.git

cd decentralized-llm-mas-framework
```

---

## Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## Install Blockchain Dependencies

```bash
cd blockchain

npm install
```

---

## Start Ganache

Generate at least 25 accounts.

GUI:

```
Accounts & Keys
Total Accounts = 25
```

or

```bash
ganache --wallet.totalAccounts 25 --server.port 7545
```

---

## Deploy Contracts

```bash
cd blockchain

npx hardhat compile

npx hardhat run scripts/deploy.js --network ganache
```

---

## Configure Environment

Create a `.env`

```env
ENABLE_BLOCKCHAIN=true

GANACHE_URL=http://127.0.0.1:7545

GROQ_API_KEY_1=...

GROQ_API_KEY_2=...

GEMINI_API_KEY=...
```

---


# Evaluation Metrics

The implementation tracks the metrics reported in the original paper:

- Task Success Rate
- Mean Task Quality
- Capability Match Score
- Agent Utility
- Allocation Delay
- Bid Rate
- Mean Reputation
- Mean Capability
- Capability Entropy
- Average Workload
- Number of Bidders

Additional metrics introduced in this implementation include:

- Reflection Success
- Memory Growth
- Retry Success Rate
- Persistent Learning Statistics

---

# Blockchain Workflow

```
Agent Registration
        │
        ▼
Task Submission
        │
        ▼
Auction Winner
        │
        ▼
On-chain Assignment
        │
        ▼
Code Execution
        │
        ▼
Performance Evaluation
        │
        ▼
Reputation Update
        │
        ▼
Capability Update
        │
        ▼
Task Completion
```

---

# Research Contributions

Compared with the original paper, this implementation introduces:

- Real LLM-generated executable code
- Deterministic correctness evaluation
- Persistent agent memory
- Teacher-guided reflection
- Retrieval-augmented prompting
- Multi-provider LLM support
- Automatic API key rotation
- Interactive visualization dashboard
- Fully operational blockchain enforcement layer
- Reproducible benchmark dataset with hidden test cases

---

# Technologies Used

### AI

- Groq
- Gemini

### Blockchain

- Solidity
- Hardhat
- Ganache
- Web3.py

### Python

- NumPy
- Pandas
- Plotly

---

# Future Work

Potential extensions include:

- Multi-chain deployment
- Peer-to-peer agent networking
- Decentralized memory sharing
- Byzantine agent detection
- Multi-agent negotiation protocols
- Federated capability learning
- Economic incentive optimization


---

---

# How to Run

## 1. Start Ganache

Launch Ganache (GUI or CLI) with **at least 25 accounts**.

```bash
ganache --wallet.totalAccounts 25 --server.port 7545
```

---

## 2. Deploy the Smart Contracts

From the `blockchain` directory:

```bash
cd blockchain

npx hardhat compile

npx hardhat run scripts/deploy.js --network ganache

cd ..
```

---

## 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
ENABLE_BLOCKCHAIN=true
GANACHE_URL=http://127.0.0.1:7545

GROQ_API_KEY_1=YOUR_API_KEY
GROQ_API_KEY_2=YOUR_API_KEY   # Optional
GEMINI_API_KEY=YOUR_API_KEY   # Optional
```

---

## 4. Select an LLM Provider

Open `config.py` and choose the provider:

```python
PROVIDER = "groq"
```

Supported providers:

- `groq`
- `gemini`

---

## 5. Run the Simulation

From the project root:

```bash
python run_simulation.py
```

The simulation will:

- Initialize 20 autonomous agents
- Register agents on the blockchain
- Broadcast programming tasks
- Run decentralized auctions
- Generate solutions using the selected LLM
- Execute solutions against hidden test cases
- Update reputation and capability on-chain
- Generate reflections and persistent memories (if needed)
- Save all metrics, logs, and visualizations

---

## 6. View Results

Each run creates a timestamped directory inside `runs/`:

```text
runs/
└── run_YYYYMMDD_HHMMSS_provider/
```

Typical contents:

```text
dashboard.html          # Interactive Plotly dashboard
metrics.json            # Per-round metrics
results.json            # Final simulation results
failure_logs/           # Failed task logs
memory/                 # Updated agent memories
```

Optionally regenerate the reports:

```bash
python visualize_results.py
```

```bash
python summary_table.py
```

# License

This project is intended for academic and research purposes.

```

---