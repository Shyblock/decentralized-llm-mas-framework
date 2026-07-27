# Blockchain Enforcement Layer — Setup

Implements methodology Steps 2, 4, 10, 14–19 (paper Section IV) as three
Solidity contracts on a local Ganache chain, driven from Python via
`blockchain_bridge.py`.

## What each file is

- `blockchain/contracts/AgentRegistry.sol` — agent identity, reputation, capability ← UPDATED
  vector, workload (Step 2 / Section IV-A)
- `blockchain/contracts/TaskManager.sol` — task metadata, assignment, action logging, ← UPDATED
  completion, workload sync (Steps 4, 10, 16, 18 / Section IV-C)
- `blockchain/contracts/IncentiveContract.sol` — computes the reputation EMA (Eq. 4) ← UPDATED
  and capability EMA (Eq. 7) **on-chain** from an off-chain performance
  score (Steps 14–15)
- `blockchain_bridge.py` — Python/web3.py wrapper the simulation calls into
- `blockchain/hardhat.config.js`, `blockchain/scripts/deploy.js`, ← UPDATED
  `blockchain/package.json` — Hardhat project that compiles/deploys the
  contracts to Ganache

`run_simulation.py` and `config.py` are your originals, patched to call the
bridge at each step; `simulation_engine.py` is unchanged.

## One-time setup

```bash
# 1. Install Node deps (Hardhat)
cd blockchain                      # ← UPDATED
npm install

# Return to the project root when finished.
cd ..                              # ← ADDED

# 2. Start Ganache with at least NUM_AGENTS + 1 = 21 unlocked accounts
#    (GUI: Settings -> Accounts & Keys -> "Total Accounts to Generate" -> 25)
#    (CLI: )
ganache --wallet.totalAccounts 25 --server.port 7545

# 3. In a second terminal, compile + deploy the contracts
cd blockchain                      # ← UPDATED
npx hardhat compile
npx hardhat run scripts/deploy.js --network ganache
#    -> writes blockchain/deployed/addresses.json and                 ← UPDATED
#       blockchain/deployed/abi/*.json

# Return to the project root before running Python.
cd ..                              # ← ADDED

# 4. Install the Python side (adds web3.py to your existing venv)
pip install -r requirements.txt

# 5. Add to your .env (see .env.example)
ENABLE_BLOCKCHAIN=true
GANACHE_URL=http://127.0.0.1:7545
```

## Run

Run the simulation **from the project root**:

```bash
python run_simulation.py
```

Each round now:
1. registers agents on-chain once, at startup (`AgentRegistry.registerAgent`)
2. submits task metadata + hash on-chain per task (`TaskManager.submitTask`)
3. commits the auction winner on-chain (`TaskManager.assignTask`)
4. drives the on-chain reputation/capability EMA (`IncentiveContract.updateReputation` / `updateCapability`)
5. marks task completion (`TaskManager.completeTask`)
6. reconciles workload once per round (`TaskManager.syncWorkload`)

If Ganache isn't running or the contracts aren't deployed, the simulation
logs a warning and **continues off-chain only** — it won't crash your run.
Set `ENABLE_BLOCKCHAIN=false` to skip the chain entirely.

At the end of a run, `runs/<run_dir>/chain_state.json` snapshots each
agent's final reputation/capabilities/workload as read directly back from
`AgentRegistry`, so you can diff it against `agent_stats` in
`results_*.json` as a sanity check that on-chain and off-chain state agree.

## Design notes / trade-offs

- **Identity**: agent 1..20 map to `Ganache.accounts[1..20]`, coordinator =
  `accounts[0]`. This uses Ganache's unlocked dev accounts instead of
  generating+funding separate keypairs, per your fresh-setup answer — swap
  in `eth_account.Account` + manual signing if you want agents to hold
  real, independently-generated keys.
- **Fixed-point**: all `[0,1]` values (reputation, capability weights,
  performance scores) are scaled by `SCALE=10000`; rewards by
  `REWARD_SCALE=100`. This is the "scaling by 10000" step from your
  methodology doc.
- **On-chain EMA**: `IncentiveContract` recomputes Eq. 4 / Eq. 7 itself
  from on-chain prior state + the passed-in score, rather than just
  storing a Python-computed number — this is what makes the reputation
  update actually *enforced* on-chain rather than merely logged.
- **Workload**: `assignTask` increments on-chain workload immediately
  (Step 10); the round-based release-one-unit-per-agent behavior in
  `run_simulation.py` is reconciled once per round via `syncWorkload`
  (Step 18), rather than decrementing per-task in `completeTask`, to stay
  faithful to your existing off-chain semantics.