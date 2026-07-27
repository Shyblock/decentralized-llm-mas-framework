"""
blockchain_bridge.py

Thin web3.py wrapper around the three deployed contracts (AgentRegistry,
TaskManager, IncentiveContract). Solidity has no floating point, so every
[0,1]-valued quantity from the paper (reputation, capability weights,
performance scores) is scaled by SCALE=10000 on the way in/out; task
rewards are scaled by REWARD_SCALE=100 (2 decimal places).

Identity model for this simulation: Ganache's own unlocked dev accounts are
used as agent identities (accounts[0] = coordinator/relay, accounts[1..N] =
agents 1..N), since Ganache exposes those accounts unlocked and eth_sign-able
without extra key management. Make sure Ganache is configured with at least
NUM_AGENTS + 1 accounts (Settings -> "Total Accounts to Generate", or
`ganache --wallet.totalAccounts 25` on the CLI).

For a real (non-simulation) deployment, each agent would instead hold its
own private key locally and sign registerAgent itself - swap
`self.accounts[agent_id]` for a locally-managed `eth_account.Account`
and use `w3.eth.account.sign_transaction` + `send_raw_transaction`.
"""

import json
import logging
import time
from pathlib import Path

from web3 import Web3

logger = logging.getLogger(__name__)

SCALE = 10000
REWARD_SCALE = 100

DEPLOYED_DIR = Path(__file__).parent / "blockchain" / "deployed"


class BlockchainBridge:
    def __init__(self, ganache_url: str = "http://127.0.0.1:7545", min_accounts: int = 21):
        self.w3 = Web3(Web3.HTTPProvider(ganache_url))

        if not self.w3.is_connected():
            raise ConnectionError(
                f"Could not connect to Ganache at {ganache_url}. "
                f"Start Ganache first (GUI or `ganache --wallet.totalAccounts 25`)."
            )

        addresses_path = DEPLOYED_DIR / "addresses.json"
        if not addresses_path.exists():
            raise FileNotFoundError(
                f"{addresses_path} not found. Run `npx hardhat run scripts/deploy.js "
                f"--network ganache` first."
            )

        addresses = json.loads(addresses_path.read_text())
        self.registry = self._load_contract("AgentRegistry", addresses["AgentRegistry"])
        self.task_manager = self._load_contract("TaskManager", addresses["TaskManager"])
        self.incentive = self._load_contract("IncentiveContract", addresses["IncentiveContract"])

        self.accounts = self.w3.eth.accounts
        if len(self.accounts) < min_accounts:
            raise RuntimeError(
                f"Ganache exposes only {len(self.accounts)} unlocked accounts; "
                f"need at least {min_accounts} (1 coordinator + N agents). "
                f"Increase 'Total Accounts to Generate' in Ganache."
            )
        self.coordinator = self.accounts[0]

        # ----------------------------------------
        # Blockchain Metrics
        # ----------------------------------------
        self.metrics = {
            "transactions": 0,
            "failed_transactions": 0,
            "gas_used": 0,
            "events": 0,
            "confirmation_times": [],
        }

    def _load_contract(self, name: str, address: str):
        abi = json.loads((DEPLOYED_DIR / "abi" / f"{name}.json").read_text())
        return self.w3.eth.contract(address=Web3.to_checksum_address(address), abi=abi)

    def agent_account(self, agent_id: int) -> str:
        return self.accounts[agent_id]

    # ---------------------------------------------------------------
    # Fixed-point helpers
    # ---------------------------------------------------------------
    @staticmethod
    def scale_unit(x: float) -> int:
        """Scale a [0,1]-valued float to a uint256, clamped to [0,1]."""
        return int(round(max(0.0, min(1.0, x)) * SCALE))

    @staticmethod
    def unscale_unit(x: int) -> float:
        return x / SCALE

    @staticmethod
    def scale_reward(x: float) -> int:
        return int(round(x * REWARD_SCALE))

    def _send(self, fn, from_addr: str):

        start = time.perf_counter()

        tx_hash = fn.transact({"from": from_addr})

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

        elapsed = time.perf_counter() - start

        self.metrics["transactions"] += 1
        self.metrics["gas_used"] += receipt.gasUsed
        self.metrics["events"] += len(receipt.logs)
        self.metrics["confirmation_times"].append(elapsed)

        if receipt.status == 0:
            self.metrics["failed_transactions"] += 1

        return receipt

    # ---------------------------------------------------------------
    # Step 2: Agent Registration
    # ---------------------------------------------------------------
    def register_agent(self, agent_id: int, capabilities: dict, reputation: float, capability_tags: list):
        caps_scaled = [self.scale_unit(capabilities.get(tag, 0.0)) for tag in capability_tags]
        fn = self.registry.functions.registerAgent(agent_id, caps_scaled, self.scale_unit(reputation))
        return self._send(fn, self.agent_account(agent_id))

    # ---------------------------------------------------------------
    # Step 4: Task Submission
    # ---------------------------------------------------------------
    def submit_task(self, task: dict, task_id: int, deadline: int):
        payload = json.dumps(
            {k: v for k, v in sorted(task.items()) if k != "reference_solution"},
            sort_keys=True,
        )
        task_hash = Web3.keccak(text=payload)
        reward_scaled = self.scale_reward(task.get("reward", 0.0))
        fn = self.task_manager.functions.submitTask(task_id, reward_scaled, task_hash, int(deadline))
        receipt = self._send(fn, self.coordinator)
        return task_hash, receipt

    # ---------------------------------------------------------------
    # Step 10: Task Assignment
    # ---------------------------------------------------------------
    def assign_task(self, task_id: int, agent_id: int, score: float):
        fn = self.task_manager.functions.assignTask(task_id, agent_id, self.scale_unit(score))
        return self._send(fn, self.coordinator)

    # ---------------------------------------------------------------
    # Behavioral logging
    # ---------------------------------------------------------------
    def log_action(self, agent_id: int, task_id: int, action: str):
        fn = self.task_manager.functions.logAction(agent_id, task_id, action)
        return self._send(fn, self.coordinator)

    # ---------------------------------------------------------------
    # Step 16: Task Completion
    # ---------------------------------------------------------------
    def complete_task(self, task_id: int, success: bool, quality: float):
        fn = self.task_manager.functions.completeTask(task_id, success, self.scale_unit(quality))
        return self._send(fn, self.coordinator)

    # ---------------------------------------------------------------
    # Step 18: Workload sync (once per round, per agent)
    # ---------------------------------------------------------------
    def sync_workload(self, agent_id: int, workload: int):
        fn = self.task_manager.functions.syncWorkload(agent_id, int(workload))
        return self._send(fn, self.coordinator)

    # ---------------------------------------------------------------
    # Step 14: Reputation EMA - computed ON-CHAIN by IncentiveContract
    # ---------------------------------------------------------------
    def update_reputation(self, agent_id: int, performance_score: float) -> float:
        fn = self.incentive.functions.updateReputation(agent_id, self.scale_unit(performance_score))
        self._send(fn, self.coordinator)
        return self.unscale_unit(self.registry.functions.getReputation(agent_id).call())

    # ---------------------------------------------------------------
    # Step 15: Capability EMA - computed ON-CHAIN by IncentiveContract
    # ---------------------------------------------------------------
    def update_capability(self, agent_id: int, tag_index: int, performance_score: float) -> float:
        fn = self.incentive.functions.updateCapability(agent_id, tag_index, self.scale_unit(performance_score))
        self._send(fn, self.coordinator)
        return self.unscale_unit(self.registry.functions.getCapability(agent_id, tag_index).call())

    # ---------------------------------------------------------------
    # Read helpers (for auditing / final snapshot)
    # ---------------------------------------------------------------
    def get_agent_state(self, agent_id: int, capability_tags: list) -> dict:
        reputation = self.unscale_unit(self.registry.functions.getReputation(agent_id).call())
        caps_raw = self.registry.functions.getCapabilities(agent_id).call()
        workload = self.registry.functions.getWorkload(agent_id).call()
        return {
            "reputation": reputation,
            "capabilities": {tag: self.unscale_unit(caps_raw[i]) for i, tag in enumerate(capability_tags)},
            "workload": workload,
        }

    def get_blockchain_metrics(self, num_rounds):

        tx = self.metrics["transactions"]

        return {
            "total_transactions": tx,
            "failed_transactions": self.metrics["failed_transactions"],
            "total_gas_used": self.metrics["gas_used"],
            "average_gas_per_transaction":
                self.metrics["gas_used"] / tx if tx else 0,

            "total_events": self.metrics["events"],

            "average_confirmation_time":
                (
                    sum(self.metrics["confirmation_times"])
                    / len(self.metrics["confirmation_times"])
                )
                if self.metrics["confirmation_times"]
                else 0,

            "transactions_per_round":
                tx / num_rounds if num_rounds else 0,

            "events_per_round":
                self.metrics["events"] / num_rounds if num_rounds else 0,
        }