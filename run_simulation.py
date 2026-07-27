import os
import json
import time
import math
import re
import ast
import random
import logging
import difflib
import shutil
import datetime
import numpy as np
from dotenv import load_dotenv
from collections import defaultdict
from llm_provider import generate
from reflection import generate_lesson
from config import PROVIDER, MODELS, ENABLE_BLOCKCHAIN, GANACHE_URL
from simulation_engine import (
    Agent,
    run_auction,
    apply_capability_recovery,
    evaluate_agent_code,
)
from memory_manager import (
    load_memory,
    add_lesson,
    retrieve_lessons,
    MEMORY_DIR,
)
from blockchain_bridge import BlockchainBridge

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ==================================================
# System Initialization Parameters (Step 1)
# ==================================================
NUM_AGENTS = 20
DATASET_PATH = "data/dataset.json"

# Performance Weight Constants (Step 13, 14, 15)
ALPHA = 0.7     # Importance of quality
DELTA = 0.3     # Importance of timeliness
LAMBDA_REP = 0.7  # Reputation memory factor
MU_CAP = 0.6      # Capability update factor

# --------------------------------------------------
# Reward normalization.
#
# The utility formula (paper Eq. 1/3) is:
#     U = score * reward - beta*workload - gamma*mismatch
# with beta=0.1, gamma=0.4 (paper Appendix B). For beta/gamma to actually
# discourage overloaded/mismatched agents, `reward` needs to be on a
# comparable order of magnitude to those cost terms (max realistic cost is
# roughly 0.1*3 + 0.4*4 ~= 1.9), not tens or hundreds.
# --------------------------------------------------
REWARD_MIN = 2.0
REWARD_MAX = 20.0

# --------------------------------------------------
# Agent capability initialization.
# See Agent.__init__ docstring in simulation_engine.py - the paper (Sec.
# V-B) samples purely from Beta(2,5) with no forced specialization boost.
# --------------------------------------------------
FORCE_INIT_SPECIALIZATION = False

# --------------------------------------------------
# Modeled task allocation delay (NOT wall-clock time). See prior discussion
# - measuring time.time() around the in-memory auction call captures
# interpreter noise, not anything comparable to the paper's Fig. 3(f).
# --------------------------------------------------
ALLOCATION_DELAY_BASE = 0.26
ALLOCATION_DELAY_PER_BIDDER = 0.26
ALLOCATION_DELAY_JITTER = 0.26

# Retry behavior for transient LLM provider errors (502/503/504)
MAX_GENERATION_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2

# Near-duplicate retry detection threshold (Recommendation 2).
DUPLICATE_SIMILARITY_THRESHOLD = 0.95

# --------------------------------------------------
# Run configuration.
# --------------------------------------------------
DEBUG_MODE = False
DEBUG_ROUND_SIZE = 1
DEBUG_MAX_TEST_TASKS = 5

PROD_ROUND_SIZE = 5

# Paper (Sec. V-C): 50 rounds x 5 tasks/round, drawn from a 100-task pool.
NUM_ROUNDS = 50

#DIFFICULTY_FILTER = {"easy", "medium"}  # set to None for full dataset
DIFFICULTY_FILTER = None

# --------------------------------------------------
# Per-run output isolation.
# Every call to main() writes into its own timestamped folder under
# RUNS_BASE_DIR instead of the working directory, so consecutive runs
# never overwrite each other's metrics/results/failure_logs/dashboard.
# --------------------------------------------------
RUNS_BASE_DIR = "runs"


def extract_function_signature(reference_solution: str) -> str:
    """
    Extract only the function signature from the reference solution.

    Example:
        def solve(nums, k):
            ...
    ->
        def solve(nums, k):
    """
    match = re.search(r"def\s+\w+\s*\([^)]*\)\s*:", reference_solution)

    if not match:
        raise ValueError("Could not extract function signature.")

    return match.group(0)


def _build_lesson_block(lessons: list | None) -> str:
    if not lessons:
        return ""

    lesson_text = "\n".join(f"• {item['lesson']}" for item in lessons)

    return f"""
====================================================
PAST VALIDATED LESSONS
====================================================

These lessons represent validated programming knowledge
obtained from previous successful retries.

Use them only when they are relevant.

Apply the underlying programming principles rather than
copying the lessons verbatim.

Lessons:

{lesson_text}

====================================================
"""


def _build_task_specific_block(task_specific_lesson: str | None) -> str:
    if not task_specific_lesson:
        return ""

    return f"""
====================================================
TASK-SPECIFIC REPAIR
====================================================

The previous implementation failed.

This guidance is ONLY for repairing the current solution.

Preserve the existing algorithm whenever possible.

Preserve all correct parts of the previous implementation.

Apply this repair before considering broader changes.

Repair:

• {task_specific_lesson}

====================================================
"""


def _build_previous_code_block(previous_code: str | None) -> str:
    if not previous_code:
        return ""

    return f"""
====================================================
PREVIOUS FAILED IMPLEMENTATION
====================================================

Repair the following implementation.

Preserve all correct logic.

Rewrite the algorithm only if it cannot be repaired.

{previous_code}

====================================================
"""


def _validate_generated_code(code: str, signature: str) -> str | None:
    """
    Parses and validates generated code against the expected signature.
    Returns the code if valid, otherwise None (after logging why).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        logger.info("\n========== RAW MODEL OUTPUT ==========")
        logger.info(code)
        logger.info("======================================\n")
        logger.info("        [!] Invalid Python generated")
        return None

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

    if not functions:
        logger.info("        [!] No function generated")
        return None

    if functions[0].name != "solve":
        logger.info("        [!] Function must be named solve")
        return None

    expected_args_str = signature[signature.find("(") + 1: signature.find(")")].strip()
    expected_args = (
        [arg.strip() for arg in expected_args_str.split(",")] if expected_args_str else []
    )
    generated_args = [arg.arg for arg in functions[0].args.args]

    if len(expected_args) != len(generated_args):
        logger.info(
            "        [!] Wrong parameter count (Expected %d, Got %d)",
            len(expected_args), len(generated_args),
        )
        return None

    return code


def _code_similarity(a: str | None, b: str | None) -> float:
    """
    Ratio of textual similarity, 0.0-1.0, between two code strings.
    """
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def generate_agent_solution(
    agent: Agent,
    task_dict: dict,
    lessons=None,
    task_specific_lesson=None,
    previous_code=None,
    use_paraphrase=False,
    _retry_count: int = 0,
) -> str | None:
    """
    Constructs the prompt and calls the LLM to generate code.
    Retries transient provider errors (502/503/504) up to MAX_GENERATION_RETRIES times.
    """
    signature = extract_function_signature(task_dict["reference_solution"])

    task_description = (
        task_dict["paraphrase"] if use_paraphrase else task_dict["description"]
    )

    lesson_block = _build_lesson_block(lessons)
    task_specific_block = _build_task_specific_block(task_specific_lesson)
    previous_code_block = _build_previous_code_block(previous_code)

    if previous_code:
        objective = """
You are repairing a previously failed Python solution.

Repair the existing implementation.

Make the smallest possible changes needed for the implementation to pass all test cases.

Preserve all correct logic.

Apply the task-specific repair.

Rewrite the algorithm only if it cannot be repaired.
"""
    else:
        objective = """
You are an expert algorithmic Python developer.

Solve the following programming problem.
"""

    full_prompt = f"""
{objective}

Problem:
{task_description}

Required Tags:
{task_dict["required_tags"]}

{lesson_block}

{task_specific_block}

{previous_code_block}

The solution MUST begin EXACTLY with this signature (same name, same parameters):

{signature}
"""

    try:
        messages = [
            {
                "role": "system",
                "content": """
        You are an expert competitive programmer.

        Return EXACTLY ONE complete Python function named solve, and nothing else.

        Rules:
        - Output ONLY executable Python code - no markdown, no explanations, no comments, no reasoning, no multiple solutions.
        - The first line of your response must be the function definition, using the exact name and parameters given in the prompt's signature. Do not change the function name or parameters.
        - Never use input() or print().
        - Define every helper function you use; the solution must be completely self-contained.
        - If you use recursion, call the correct recursive function name exactly as defined.
        - When repairing a previous implementation, return the complete corrected solution, not only the changed lines.
        - Ensure the code is syntactically valid Python. Finish the entire implementation before stopping, and end your response immediately after the final return statement.
        """,
            },
            {"role": "user", "content": full_prompt},
        ]

        code = generate(
            messages=messages,
            temperature=0 if previous_code is None else 0.3,
            max_tokens=1500,
        )

        code = code.replace("```python", "").replace("```", "").strip()

        match = re.search(r"def\s+solve\s*\(", code)
        if match:
            code = code[match.start():]
        else:
            logger.info("\n========== RAW MODEL OUTPUT ==========")
            logger.info(code)
            logger.info("======================================")
            logger.info("[!] No solve() function found")
            return None

        return _validate_generated_code(code, signature)

    except Exception as e:
        error = str(e)

        if any(x in error for x in ("502", "503", "504")):
            if _retry_count >= MAX_GENERATION_RETRIES:
                logger.info(
                    "        [!] Groq still overloaded after %d retries. Giving up.",
                    MAX_GENERATION_RETRIES,
                )
                return None

            logger.info(
                "        [!] Groq overloaded. Retrying (%d/%d)...",
                _retry_count + 1, MAX_GENERATION_RETRIES,
            )
            time.sleep(RETRY_BACKOFF_SECONDS * (_retry_count + 1))

            return generate_agent_solution(
                agent,
                task_dict,
                lessons=lessons,
                task_specific_lesson=task_specific_lesson,
                previous_code=previous_code,
                use_paraphrase=use_paraphrase,
                _retry_count=_retry_count + 1,
            )

        if "429" in error or "rate limit" in error.lower():
            logger.info("[!] %s API limit reached: %s", PROVIDER.upper(), e)
            return None

        logger.info("        [!] LLM API Error: %s", error)
        return None


def _normalize_rewards(dataset: list, reward_min: float, reward_max: float) -> list:
    """
    Rescale task['reward'] into [reward_min, reward_max] (in place).
    """
    raw_rewards = [t.get("reward") for t in dataset]
    numeric = [r for r in raw_rewards if isinstance(r, (int, float))]

    if not numeric or max(numeric) == min(numeric):
        default_reward = (reward_min + reward_max) / 2
        for t in dataset:
            t["reward"] = default_reward
        logger.info(
            "[*] No usable reward variation in dataset; assigned flat reward=%.2f to all tasks.",
            default_reward,
        )
        return dataset

    r_min, r_max = min(numeric), max(numeric)
    for t in dataset:
        raw = t.get("reward")
        if not isinstance(raw, (int, float)):
            raw = r_min
        scaled = reward_min + (raw - r_min) * (reward_max - reward_min) / (r_max - r_min)
        t["reward"] = scaled

    logger.info(
        "[*] Rewards rescaled from raw range [%.2f, %.2f] to [%.2f, %.2f].",
        r_min, r_max, reward_min, reward_max,
    )
    return dataset


def _load_dataset(path: str, difficulty_filter: set | None) -> list:
    try:
        with open(path, "r") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"[!] Dataset not found at '{path}'. Check DATASET_PATH.")
    except json.JSONDecodeError as e:
        raise SystemExit(f"[!] Dataset at '{path}' is not valid JSON: {e}")

    # Task IDs excluded due to malformed reference solutions / degenerate or
    # ambiguous test cases identified during dataset curation. Verify
    # against your own dataset revision before reusing this list as-is.
    EXCLUDED_TASKS = {134, 142, 187, 248}
    dataset = [task for task in dataset if task["id"] not in EXCLUDED_TASKS]

    if difficulty_filter is not None:
        dataset = [task for task in dataset if task.get("difficulty") in difficulty_filter]
        logger.info("[*] Difficulty filter: %s", difficulty_filter)

    dataset = _normalize_rewards(dataset, REWARD_MIN, REWARD_MAX)

    return dataset


def _sample_round_tasks(dataset: list, round_size: int) -> list:
    """
    Draws `round_size` tasks from the full pool for one round, reusing the
    pool across rounds (paper Sec. V-C: 50 rounds x 5 tasks/round from a
    100-task pool implies reuse, since 250 > 100).
    """
    if len(dataset) >= round_size:
        return random.sample(dataset, round_size)
    return [random.choice(dataset) for _ in range(round_size)]


def agent_capability_entropy(agent: Agent) -> float:
    """
    Shannon entropy (bits) of an agent's capability vector, matching the
    paper's Section VI-B framing (higher entropy = more generalist, lower
    = more specialized). Values are treated as an unnormalized weight
    distribution over skill tags.
    """
    values = np.array(list(agent.capabilities.values()), dtype=float)
    values = values[values > 0]
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    probs = values / values.sum()
    return float(-(probs * np.log2(probs)).sum())


def _make_run_dir(base: str = RUNS_BASE_DIR) -> str:
    """
    Creates and returns a fresh, timestamped folder for this run's outputs
    (metrics/results/failure_logs/dashboard), so consecutive runs never
    clobber each other. Format: runs/run_YYYYMMDD_HHMMSS_<provider>/
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, f"run_{timestamp}_{PROVIDER}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _early_late_split(values: list, frac: float = 0.2) -> tuple[float | None, float | None]:
    """Average of the first/last `frac` of rounds, matching the paper's
    'early rounds' / 'later rounds' framing (Section VI-A)."""
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    n = max(1, int(len(values) * frac))
    return float(np.mean(values[:n])), float(np.mean(values[-n:]))


def main():

    random.seed(42)
    np.random.seed(42)

    run_dir = _make_run_dir()
    logger.info(os.path.abspath(__file__))
    logger.info("[*] Run outputs will be saved to: %s", os.path.abspath(run_dir))
    logger.info("[*] Initializing Multi-Agent Simulation...")

    agents = [Agent(i, force_specialization=FORCE_INIT_SPECIALIZATION) for i in range(1, NUM_AGENTS + 1)]
    logger.info("[*] Created %d autonomous agents.", NUM_AGENTS)

    # -----------------------------
    # Blockchain enforcement layer (Step 2 / paper Section IV-A).
    # Any failure here (Ganache not running, contracts not deployed) is
    # caught and logged - the simulation still runs off-chain-only rather
    # than crashing, since the incentive mechanism itself is fully defined
    # in Python regardless of on-chain enforcement.
    # -----------------------------
    bridge = None
    if ENABLE_BLOCKCHAIN:
        try:
            bridge = BlockchainBridge(ganache_url=GANACHE_URL, min_accounts=NUM_AGENTS + 1)
            logger.info("[*] Connected to blockchain enforcement layer at %s", GANACHE_URL)

            for agent in agents:
                bridge.register_agent(
                    agent.agent_id, agent.capabilities, agent.reputation, Agent.CAPABILITY_TAGS
                )
            logger.info("[*] Registered %d agents on-chain (AgentRegistry.registerAgent).", NUM_AGENTS)
        except Exception as e:
            logger.warning(
                "[!] Blockchain layer unavailable (%s). Continuing off-chain only.", e
            )
            bridge = None

    dataset = _load_dataset(DATASET_PATH, DIFFICULTY_FILTER)

    random.shuffle(dataset)

    logger.info("[*] Loaded %d tasks.", len(dataset))

    if DEBUG_MODE:
        dataset = dataset[:DEBUG_MAX_TEST_TASKS]
        round_size = DEBUG_ROUND_SIZE
        num_rounds = math.ceil(len(dataset) / round_size)
        use_pool_resampling = False
        logger.info(
            "[*] DEBUG_MODE on: using %d tasks, round_size=%d, num_rounds=%d",
            len(dataset), round_size, num_rounds,
        )
    else:
        round_size = PROD_ROUND_SIZE
        num_rounds = NUM_ROUNDS
        use_pool_resampling = True
        logger.info(
            "[*] Production mode: %d rounds x %d tasks/round, sampled from a %d-task pool.",
            num_rounds, round_size, len(dataset),
        )

    failure_logs = []
    metrics_history = []

    # -----------------------------
    # Metric Histories
    # -----------------------------
    success_history = []
    effective_success_history = []
    quality_history = []
    capmatch_history = []
    utility_history = []
    load_history = []
    delay_history = []
    reputation_history = []
    capability_history = []
    entropy_history = []
    bid_history = []
    bid_count_per_task = []

    capability_stats = defaultdict(lambda: {"attempted": 0, "passed": 0, "failed": 0})
    agent_performance = defaultdict(lambda: {"attempted": 0, "passed": 0, "failed": 0})

    # -----------------------------
    # Reflection-arc counters (simulation-wide, novel contribution - not
    # part of the base paper's mechanism)
    # -----------------------------
    reflection_total_failures = 0
    reflection_total_retries_attempted = 0
    reflection_total_retries_succeeded = 0
    reflection_total_lessons_stored = 0
    cumulative_lessons_stored = 0

    # -----------------------------
    # Round Loop
    # -----------------------------
    for round_num in range(num_rounds):

        logger.info("\n========== ROUND %d ==========", round_num + 1)

        if use_pool_resampling:
            round_tasks = _sample_round_tasks(dataset, round_size)
        else:
            round_tasks = dataset[round_num * round_size: (round_num + 1) * round_size]

        round_successes = 0
        round_effective_successes = 0  # counts first-attempt AND retry-recovered successes
        round_quality = []
        round_capmatch = []
        round_utilities = []
        round_loads = []
        round_delays = []
        round_bidders = []
        last_winner = None

        # Reflection-arc counters (this round only)
        round_failures = 0
        round_retries_attempted = 0
        round_retries_succeeded = 0
        round_lessons_stored = 0

        for idx, task in enumerate(round_tasks):

            task_id = task.get("id", round_num * round_size + idx + 1)

            logger.info("\n--- Task %s Broadcast ---", task_id)

            # -----------------------------
            # Step 4: Task metadata registered on-chain before the auction
            # runs, so assignment/logging below can reference it.
            # -----------------------------
            if bridge:
                try:
                    bridge.submit_task(task, task_id, int(task.get("deadline", 5)))
                except Exception as e:
                    logger.warning("      [!] On-chain submitTask failed: %s", e)

            # -----------------------------
            # Auction
            # -----------------------------
            winner, num_bidders, winner_score, winner_utility = run_auction(task, agents, temperature=0.05)

            round_bidders.append(num_bidders)

            # Modeled allocation delay (see constants above) - NOT
            # wall-clock time. A function of bidder contention plus jitter.
            allocation_delay = (
                ALLOCATION_DELAY_BASE
                + ALLOCATION_DELAY_PER_BIDDER * num_bidders
                + random.uniform(0, ALLOCATION_DELAY_JITTER)
            )
            round_delays.append(allocation_delay)

            if not winner:
                logger.info(
                    "      [-] No agents bid on this task (Utility <= 0). Skipping..."
                )
                continue
            last_winner = winner

            logger.info(
                "      [+] Auction Winner: Agent %d (Workload is now %s)",
                winner.agent_id, winner.workload,
            )
            agent_performance[winner.agent_id]["attempted"] += 1

            # -----------------------------
            # Step 10: commit the auction winner and log the acceptance
            # action on-chain (TaskAssigned event, ActionLogged event).
            # -----------------------------
            if bridge:
                try:
                    bridge.assign_task(task_id, winner.agent_id, winner_score or 0.0)
                    bridge.log_action(winner.agent_id, task_id, "task_accepted")
                except Exception as e:
                    logger.warning("      [!] On-chain assignTask failed: %s", e)

            required_tags = task.get("required_tags", [])
            bid_count_per_task.append({
                "task_id": task_id,
                "num_bidders": num_bidders,
                "required_tags": required_tags,
            })

            # -----------------------------
            # Capability Match (for logging only - independent of workload,
            # so no staleness risk from recomputing it here)
            # -----------------------------
            if required_tags:
                cap_match = sum(
                    winner.capabilities.get(tag, 0.0) for tag in required_tags
                ) / len(required_tags)
            else:
                cap_match = 0.0

            round_capmatch.append(cap_match)

            # -----------------------------
            # Utility - taken directly from run_auction's own scoring
            # decision, rather than recomputed here.
            # -----------------------------
            round_utilities.append(winner_utility)

            # -----------------------------
            # Generate Code
            # -----------------------------
            lessons = retrieve_lessons(winner.agent_id, required_tags)

            logger.info("      [*] Retrieved %d lesson(s).", len(lessons))
            for i, item in enumerate(lessons, 1):
                logger.info("          %d. %s", i, item["lesson"])

            logger.info("      [*] Agent is generating code...")

            code = generate_agent_solution(winner, task, lessons)

            if not code:
                logger.info("      [-] Agent failed to generate code.")

                eval_result = {
                    "quality_score": 0.0,
                    "success": False,
                    "delay_ratio": 1.0,
                    "reason": "Failed to generate syntactically valid Python",
                    "failure_type": "syntax_error",
                    "failed_test_case": None,
                }

                failure_logs.append({
                    "task_id": task_id,
                    "description": task.get("description"),
                    "required_tags": required_tags,
                    "generated_code": None,
                    "error": eval_result["reason"],
                    "failure_type": eval_result["failure_type"],
                    "failed_test_case": None,
                })

            else:
                eval_result = evaluate_agent_code(task, code)

                logger.info(
                    "      [*] Sandbox Result: Success=%s, Quality=%.2f, Delay Ratio=%.2f (%s)",
                    eval_result["success"], eval_result["quality_score"],
                    eval_result["delay_ratio"], eval_result["reason"],
                )

                if not eval_result["success"]:
                    logger.info(code)
                    failure_logs.append({
                        "task_id": task_id,
                        "description": task.get("description"),
                        "required_tags": required_tags,
                        "generated_code": code,
                        "error": eval_result["reason"],
                        "failure_type": eval_result.get("failure_type", "unknown"),
                        "failed_test_case": eval_result.get("failed_test_case"),
                    })

            q_i_t = eval_result["quality_score"]       # continuous, in [0,1]
            success_i_t = eval_result["success"]        # strict all-tests-passed
            d_i_t = eval_result["delay_ratio"]

            # -----------------------------
            # Success Metrics
            # -----------------------------
            if success_i_t:
                round_successes += 1
            round_quality.append(q_i_t)

            if success_i_t:
                agent_performance[winner.agent_id]["passed"] += 1
            else:
                agent_performance[winner.agent_id]["failed"] += 1

            for tag in required_tags:
                capability_stats[tag]["attempted"] += 1
                if success_i_t:
                    capability_stats[tag]["passed"] += 1
                else:
                    capability_stats[tag]["failed"] += 1

            # -----------------------------
            # Performance / Reputation / Capability Updates
            #
            # NOTE: this block intentionally uses q_i_t (first-attempt
            # quality only), per the paper's definition of the per-round
            # performance signal. It must NOT be changed to reflect retry
            # outcomes - only the success-rate reporting below does that.
            # -----------------------------
            performance_score = ALPHA * q_i_t + DELTA * (1 - d_i_t)

            old_rep = winner.reputation
            winner.reputation = LAMBDA_REP * old_rep + (1 - LAMBDA_REP) * performance_score

            for tag in required_tags:
                old_cap = winner.capabilities.get(tag, 0.0)
                winner.capabilities[tag] = MU_CAP * old_cap + (1 - MU_CAP) * performance_score

            logger.info(
                "      [*] Agent %d Reputation updated: %.2f -> %.2f",
                winner.agent_id, old_rep, winner.reputation,
            )

            # -----------------------------
            # Steps 14-16: mirror the same update on-chain. IncentiveContract
            # recomputes the EMA itself (Eq. 4 / Eq. 7) from the on-chain
            # prior state and this performance_score, so the on-chain result
            # is independently verifiable rather than just a copy of the
            # Python value.
            # -----------------------------
            if bridge:
                try:
                    bridge.update_reputation(winner.agent_id, performance_score)
                    for tag in required_tags:
                        tag_index = Agent.CAPABILITY_TAGS.index(tag)
                        bridge.update_capability(winner.agent_id, tag_index, performance_score)
                    bridge.complete_task(task_id, success_i_t, q_i_t)
                except Exception as e:
                    logger.warning("      [!] On-chain reputation/capability sync failed: %s", e)

            # -----------------------------
            # Reflection / Retry
            # -----------------------------
            retry_code = None
            retry_result = None
            generic_lesson = None
            task_specific_lesson = None

            if not success_i_t:
                round_failures += 1
                reflection_total_failures += 1

                logger.info("      [*] Teacher is generating lesson...")

                generic_lesson, task_specific_lesson = generate_lesson(
                    task,
                    code if code is not None else "No valid Python code was generated.",
                    eval_result["reason"],
                    eval_result["failure_type"],
                    eval_result.get("failed_test_case"),
                )

                if generic_lesson is None:
                    logger.info("      [-] Reflection generation failed.")

                logger.info("      [*] Generic Lesson: %s", generic_lesson)
                logger.info("      [*] Task-specific Lesson: %s", task_specific_lesson)
                logger.info("      [*] Retrying on paraphrased task...")

                retry_code = generate_agent_solution(
                    winner,
                    task,
                    lessons=lessons,
                    task_specific_lesson=task_specific_lesson,
                    previous_code=code,
                    use_paraphrase=True,
                )

                # Recommendation 2: near-duplicate detection.
                if retry_code is not None and code is not None:
                    similarity = _code_similarity(retry_code, code)

                    if retry_code == code:
                        logger.info("      [!] Retry produced IDENTICAL code — no repair happened.")

                    if similarity >= DUPLICATE_SIMILARITY_THRESHOLD:
                        logger.info(
                            "      [!] Retry is %.1f%% identical to original — forcing a distinct rewrite.",
                            similarity * 100,
                        )

                        forced_task_specific_lesson = (
                            (task_specific_lesson or "")
                            + " IMPORTANT: your previous repair attempt was nearly "
                              "identical to the failing code and did not fix the bug. "
                              "You must change the actual logic, not just formatting, "
                              "variable names, or whitespace."
                        )

                        retry_code = generate_agent_solution(
                            winner,
                            task,
                            lessons=lessons,
                            task_specific_lesson=forced_task_specific_lesson,
                            previous_code=code,
                            use_paraphrase=True,
                        )

                if retry_code is None:
                    logger.info("      [-] Retry generation failed.")
                else:
                    round_retries_attempted += 1
                    reflection_total_retries_attempted += 1

                    retry_result = evaluate_agent_code(task, retry_code)
                    logger.info(
                        "      [*] Retry Result: Success=%s, Quality=%.2f, Reason=%s",
                        retry_result["success"], retry_result["quality_score"], retry_result["reason"],
                    )

                    if retry_result["success"]:
                        round_retries_succeeded += 1
                        reflection_total_retries_succeeded += 1

                        logger.info("      [+] Lesson validated!")

                        stored = add_lesson(
                            winner.agent_id,
                            {
                                "task_id": task_id,
                                "required_tags": required_tags,
                                "failure_type": eval_result["failure_type"],
                                "lesson": generic_lesson,
                            },
                        )

                        if stored:
                            round_lessons_stored += 1
                            reflection_total_lessons_stored += 1

                        logger.info(
                            "      [+] Lesson stored." if stored else "      [*] Lesson already existed."
                        )

                        # Driven by the retry's own graded performance,
                        # matching Eq. (5)/(7)'s form - see
                        # apply_capability_recovery's docstring.
                        retry_performance_score = (
                            ALPHA * retry_result["quality_score"]
                            + DELTA * (1 - retry_result["delay_ratio"])
                        )
                        apply_capability_recovery(
                            winner, required_tags, MU_CAP, retry_performance_score
                        )
                        logger.info("      [+] Capability recovery applied.")

                        if bridge:
                            try:
                                for tag in required_tags:
                                    tag_index = Agent.CAPABILITY_TAGS.index(tag)
                                    bridge.update_capability(
                                        winner.agent_id, tag_index, retry_performance_score
                                    )
                            except Exception as e:
                                logger.warning(
                                    "      [!] On-chain capability recovery sync failed: %s", e
                                )

                        retrieved = retrieve_lessons(winner.agent_id, required_tags)
                        logger.info(
                            "      [*] Retrieved %d lesson(s) after storing:", len(retrieved)
                        )
                        for i, item in enumerate(retrieved, 1):
                            logger.info("          %d. %s", i, item["lesson"])
                    else:
                        logger.info(
                            "      [-] Lesson rejected. Reason: %s", retry_result["reason"]
                        )

                for log in reversed(failure_logs):
                    if log["task_id"] == task_id:
                        log["generic_lesson"] = generic_lesson
                        log["task_specific_lesson"] = task_specific_lesson
                        log["retry_code"] = retry_code
                        log["retry_result"] = retry_result
                        break

            # -----------------------------
            # Effective success (first attempt OR retry recovery).
            # -----------------------------
            effective_success = (
                success_i_t
                or (retry_result is not None and retry_result.get("success"))
            )
            if effective_success:
                round_effective_successes += 1

            round_loads.append(np.mean([a.workload for a in agents]))

        # ==================================
        # ROUND SUMMARY
        # ==================================
        success_rate = round_successes / len(round_tasks) if round_tasks else 0
        effective_success_rate = round_effective_successes / len(round_tasks) if round_tasks else 0
        mean_quality = np.mean(round_quality) if round_quality else 0
        mean_capmatch = np.mean(round_capmatch) if round_capmatch else 0
        mean_utility = np.mean(round_utilities) if round_utilities else 0
        mean_load = np.mean(round_loads) if round_loads else 0
        mean_delay = np.mean(round_delays) if round_delays else 0
        mean_bidders = np.mean(round_bidders) if round_bidders else 0

        avg_rep = np.mean([a.reputation for a in agents])
        avg_cap = np.mean([np.mean(list(a.capabilities.values())) for a in agents])
        avg_entropy = np.mean([agent_capability_entropy(a) for a in agents])

        cumulative_lessons_stored += round_lessons_stored
        round_retry_recovery_rate = (
            (round_retries_succeeded / round_failures) if round_failures > 0 else None
        )

        success_history.append(success_rate)
        effective_success_history.append(effective_success_rate)
        quality_history.append(mean_quality)
        capmatch_history.append(mean_capmatch)
        utility_history.append(mean_utility)
        load_history.append(mean_load)
        delay_history.append(mean_delay)
        reputation_history.append(avg_rep)
        capability_history.append(avg_cap)
        entropy_history.append(float(avg_entropy))
        bid_history.append(mean_bidders)

        # Release one unit of workload for every agent
        for agent in agents:
            agent.workload = max(0, agent.workload - 1)

        # Step 18: reconcile on-chain workload with the just-updated
        # off-chain values (TaskManager.syncWorkload).
        if bridge:
            try:
                for agent in agents:
                    bridge.sync_workload(agent.agent_id, agent.workload)
            except Exception as e:
                logger.warning("[!] On-chain workload sync failed: %s", e)

        metrics_history.append({
            "round": round_num + 1,
            "difficulty_filter": sorted(DIFFICULTY_FILTER) if DIFFICULTY_FILTER else "all",
            "tasks_in_round": len(round_tasks),
            "task_success_rate": success_rate,
            "effective_task_success_rate": effective_success_rate,
            "reflection_lift": effective_success_rate - success_rate,
            "mean_task_quality": mean_quality,
            "capability_match_score": mean_capmatch,
            "agent_utility": mean_utility,
            "load_distribution": mean_load,
            "task_allocation_delay": mean_delay,
            "task_bid_rate": mean_bidders / NUM_AGENTS,
            "mean_bidders": mean_bidders,
            "average_reputation": avg_rep,
            "average_capability": avg_cap,
            "capability_entropy": float(avg_entropy),
            "round_failures": round_failures,
            "round_retries_attempted": round_retries_attempted,
            "round_retries_succeeded": round_retries_succeeded,
            "round_retry_recovery_rate": round_retry_recovery_rate,
            "round_lessons_stored": round_lessons_stored,
            "cumulative_lessons_stored": cumulative_lessons_stored,
        })

        logger.info("\n----- ROUND %d SUMMARY -----", round_num + 1)
        logger.info("Task Success Rate           : %.3f", success_rate)
        logger.info("Effective Success Rate (retry): %.3f", effective_success_rate)
        logger.info("Mean Quality      : %.3f", mean_quality)
        logger.info("Capability Match  : %.3f", mean_capmatch)
        logger.info("Agent Utility     : %.3f", mean_utility)
        logger.info("Average Load      : %.3f", mean_load)
        logger.info("Allocation Delay  : %.3f", mean_delay)
        logger.info("Average Reputation: %.3f", avg_rep)
        logger.info("Average Capability: %.3f", avg_cap)
        logger.info("Capability Entropy: %.3f", avg_entropy)
        logger.info("Average Bidders   : %.2f", mean_bidders)
        logger.info(
            "Reflection: %d failures, %d retries attempted, %d succeeded, %d lessons stored (cumulative: %d)",
            round_failures, round_retries_attempted, round_retries_succeeded,
            round_lessons_stored, cumulative_lessons_stored,
        )

        if last_winner:
            logger.info("      Capabilities: %s", last_winner.capabilities)
            logger.info("      Reputation: %.3f", last_winner.reputation)

    # ==================================
    # SIMULATION COMPLETE
    # ==================================
    agent_stats = []
    for a in agents:
        stats = agent_performance[a.agent_id]
        agent_stats.append({
            "agent_id": a.agent_id,
            "reputation": a.reputation,
            "workload": a.workload,
            "capabilities": a.capabilities,
            "capability_entropy": agent_capability_entropy(a),
            "tasks_attempted": stats["attempted"],
            "tasks_passed": stats["passed"],
            "tasks_failed": stats["failed"],
            "success_rate": stats["passed"] / stats["attempted"] if stats["attempted"] > 0 else 0,
        })

    capability_success = {
        tag: {
            "attempted": stats["attempted"],
            "passed": stats["passed"],
            "failed": stats["failed"],
            "success_rate": stats["passed"] / stats["attempted"] if stats["attempted"] > 0 else 0,
        }
        for tag, stats in capability_stats.items()
    }

    # -----------------------------
    # Final overall summary - all the "at a glance" numbers in one place:
    # simulation-wide averages, the literal final round's values, an
    # early-vs-late split (mirroring the paper's Section VI-A framing), and
    # a dedicated reflection-arc summary quantifying what the reflection
    # module actually bought you on top of the paper's base mechanism.
    # -----------------------------
    reputations_final = [a.reputation for a in agents]
    entropies_final = [agent_capability_entropy(a) for a in agents]

    success_early, success_late = _early_late_split(success_history)
    effective_early, effective_late = _early_late_split(effective_success_history)
    quality_early, quality_late = _early_late_split(quality_history)
    capmatch_early, capmatch_late = _early_late_split(capmatch_history)
    utility_early, utility_late = _early_late_split(utility_history)

    final_summary = {
        "rounds_completed": num_rounds,
        "tasks_per_round": round_size,
        "overall": {
            "task_success_rate": _mean(success_history),
            "effective_task_success_rate": _mean(effective_success_history),
            "mean_task_quality": _mean(quality_history),
            "capability_match_score": _mean(capmatch_history),
            "agent_utility": _mean(utility_history),
            "load_distribution": _mean(load_history),
            "task_allocation_delay": _mean(delay_history),
            "task_bid_rate": (_mean(bid_history) / NUM_AGENTS) if bid_history else None,
            "average_reputation": _mean(reputation_history),
            "average_capability": _mean(capability_history),
            "average_capability_entropy": _mean(entropy_history),
        },
        "final_round": metrics_history[-1] if metrics_history else None,
        "early_vs_late": {
            "task_success_rate": {"early": success_early, "late": success_late},
            "effective_task_success_rate": {"early": effective_early, "late": effective_late},
            "mean_task_quality": {"early": quality_early, "late": quality_late},
            "capability_match_score": {"early": capmatch_early, "late": capmatch_late},
            "agent_utility": {"early": utility_early, "late": utility_late},
        },
        "final_agent_reputation": {
            "mean": _mean(reputations_final),
            "std": float(np.std(reputations_final)) if reputations_final else None,
            "min": float(np.min(reputations_final)) if reputations_final else None,
            "max": float(np.max(reputations_final)) if reputations_final else None,
        },
        "final_agent_capability_entropy": {
            "mean": _mean(entropies_final),
            "std": float(np.std(entropies_final)) if entropies_final else None,
            "min": float(np.min(entropies_final)) if entropies_final else None,
            "max": float(np.max(entropies_final)) if entropies_final else None,
        },
        # Reflection module (novel contribution, not part of the base
        # paper's mechanism) - quantifies retry/lesson effectiveness.
        "reflection": {
            "total_failures": reflection_total_failures,
            "total_retries_attempted": reflection_total_retries_attempted,
            "total_retries_succeeded": reflection_total_retries_succeeded,
            "retry_recovery_rate": (
                reflection_total_retries_succeeded / reflection_total_retries_attempted
                if reflection_total_retries_attempted > 0 else None
            ),
            "failure_recovery_rate": (
                reflection_total_retries_succeeded / reflection_total_failures
                if reflection_total_failures > 0 else None
            ),
            "total_lessons_stored": reflection_total_lessons_stored,
            "raw_vs_effective_success_lift": (
                _mean(effective_success_history) - _mean(success_history)
                if success_history and effective_success_history else None
            ),
        },
    }

    results = {
        "num_agents": NUM_AGENTS,
        "round_size": round_size,
        "num_rounds": num_rounds,
        "success_history": success_history,
        "effective_success_history": effective_success_history,
        "quality_history": quality_history,
        "capmatch_history": capmatch_history,
        "utility_history": utility_history,
        "load_history": load_history,
        "delay_history": delay_history,
        "reputation_history": reputation_history,
        "capability_success": capability_success,
        "capability_history": capability_history,
        "entropy_history": entropy_history,
        "agent_stats": agent_stats,
        "final_agent_entropy": {a.agent_id: agent_capability_entropy(a) for a in agents},
        "bid_history": bid_history,
        "bid_count_per_task": bid_count_per_task,
        "bid_participation_history": [b / NUM_AGENTS for b in bid_history],
        "final_summary": final_summary,
    }

    results_path = os.path.join(run_dir, f"results_{PROVIDER}.json")
    metrics_path = os.path.join(run_dir, f"metrics_{PROVIDER}.json")
    failure_logs_path = os.path.join(run_dir, f"failure_logs_{PROVIDER}.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=4)

    with open(failure_logs_path, "w") as f:
        json.dump(failure_logs, f, indent=4)

    logger.info("[*] Results saved to %s", results_path)
    logger.info("[*] Round metrics saved to %s", metrics_path)
    logger.info("[*] Failure logs saved to %s", failure_logs_path)

    # -----------------------------
    # Snapshot the (cumulative, cross-run) agent memory into this run's
    # folder for reference/reproducibility. This does NOT reset or isolate
    # memory between runs - agents keep learning across separate `main()`
    # invocations by design (memory/ is a persistent store) - it just
    # copies the state as it stood at the end of THIS run alongside the
    # rest of this run's artifacts.
    # -----------------------------
    try:
        if os.path.isdir(MEMORY_DIR):
            shutil.copytree(
                MEMORY_DIR,
                os.path.join(run_dir, "memory_snapshot"),
                dirs_exist_ok=True,
            )
            logger.info("[*] Memory snapshot saved to %s", os.path.join(run_dir, "memory_snapshot"))
    except Exception as e:
        logger.warning("[!] Could not snapshot memory/: %s", e)

    # -----------------------------
    # Snapshot final on-chain agent state (read back directly from
    # AgentRegistry, not from the Python objects) so it can be diffed
    # against `agent_stats` in results_*.json as an audit check.
    # -----------------------------
    if bridge:
        try:
            chain_state = {
                str(a.agent_id): bridge.get_agent_state(a.agent_id, Agent.CAPABILITY_TAGS)
                for a in agents
            }
            chain_state_path = os.path.join(run_dir, "chain_state.json")
            with open(chain_state_path, "w") as f:
                json.dump(chain_state, f, indent=4)
            logger.info("[*] On-chain agent state snapshot saved to %s", chain_state_path)
        except Exception as e:
            logger.warning("[!] Could not snapshot on-chain state: %s", e)


    # -----------------------------
    # Blockchain metrics
    # -----------------------------
    if bridge:
        try:
            blockchain_metrics = bridge.get_blockchain_metrics(num_rounds)

            blockchain_metrics_path = os.path.join(
                run_dir,
                "blockchain_metrics.json"
            )

            with open(blockchain_metrics_path, "w") as f:
                json.dump(blockchain_metrics, f, indent=4)

            logger.info(
                "[*] Blockchain metrics saved to %s",
                blockchain_metrics_path,
            )

        except Exception as e:
            logger.warning(
                "[!] Could not save blockchain metrics: %s",
                e,
            )

    # -----------------------------
    # Auto-build the dashboard into the same run folder. Soft-fails (with a
    # clear message) if plotly isn't installed, rather than crashing a
    # simulation run that otherwise completed successfully.
    # -----------------------------
    try:
        from visualize_results import build_dashboard
        dashboard_path = os.path.join(run_dir, f"dashboard_{PROVIDER}.html")
        build_dashboard(metrics_path, results_path, dashboard_path)
        logger.info("[*] Dashboard saved to %s", dashboard_path)
    except ImportError:
        logger.warning(
            "[!] plotly not installed - skipping dashboard generation. "
            "Install it (`pip install plotly`) and run: "
            "python visualize_results.py --run-dir %s",
            run_dir,
        )
    except Exception as e:
        logger.warning("[!] Dashboard generation failed: %s", e)

    logger.info("[*] All artifacts for this run are in: %s", os.path.abspath(run_dir))


if __name__ == "__main__":
    main()