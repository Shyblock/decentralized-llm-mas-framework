import os
import json
import random
import time
import math
import heapq
import itertools
import collections
import functools
import bisect as _bisect_module
import types
import inspect
import traceback
import builtins as _builtins_module
import numpy as np
from typing import Tuple
from multiprocessing import Process, Manager
from collections import deque, Counter, defaultdict
from typing import Dict, List, Any, Optional
import ast as _ast_module

# ==========================================
# MODULE 1: AGENT INITIALIZATION (Step 1)
# ==========================================
class Agent:
    """
    Autonomous entity within the simulation initialized with Beta and Uniform distributions
    matching the structural specifications of the methodology.
    """
    CAPABILITY_TAGS = [
        "dynamic_programming",
        "graphs",
        "hashmaps",
        "lists",
        "math",
        "recursion",
        "searching",
        "sorting",
        "strings",
        "trees"
    ]

    def __init__(self, agent_id: int, force_specialization: bool = False):
        """
        force_specialization: if True, deterministically boosts one random
        capability tag to Uniform(0.75, 0.95) at init time. The paper
        (Sec. V-B) samples w0_i purely from Beta(2,5) with no such forced
        boost - specialization there is meant to *emerge* from the
        incentive dynamics over 50 rounds. Defaulting this to False keeps
        initialization paper-faithful; flip it on only if you deliberately
        want a more differentiated starting population and are prepared to
        document that as a methodological deviation (it will also make
        your initial/round-0 capability entropy lower than a pure Beta(2,5)
        draw, which matters if you're comparing against the paper's
        Fig. 4(b) entropy trend).
        """
        self.agent_id: int = agent_id

        # System Initialization Parameters (Step 1)
        self.reputation: float = 0.5
        self.workload: int = random.choice([0, 1, 2])  # l_i^0 ~ Uniform{0,1,2}

        # Initialize Capability Vector (Step 1)
        self.capabilities: Dict[str, float] = self._generate_capabilities(force_specialization)

    def _generate_capabilities(self, force_specialization: bool) -> Dict[str, float]:
        """
        Generates capability scores using Beta(2,5), matching paper Sec. V-B.
        Optionally (off by default) sets one random tag to a higher
        Uniform(0.75, 0.95) draw - see docstring on __init__.
        """
        num_tags = len(self.CAPABILITY_TAGS)
        base_caps = np.random.beta(2, 5, num_tags)

        if force_specialization:
            specialized_idx = random.randint(0, num_tags - 1)
            base_caps[specialized_idx] = np.random.uniform(0.75, 0.95)

        return {tag: float(base_caps[i]) for i, tag in enumerate(self.CAPABILITY_TAGS)}

# ==========================================
# FAILURE CLASSIFICATION
# ==========================================
def classify_failure(error):
    error = str(error)

    if "SyntaxError" in error:
        return "syntax_error"

    if "Timeout" in error:
        return "timeout"

    # Must be checked before the generic fallback and is unambiguous:
    # this sandbox deliberately excludes __import__ from SAFE_BUILTINS, so
    # any `import` statement in generated code raises this specific error.
    # Classifying it separately lets reflection give a mechanically correct
    # lesson ("don't use import statements at all") instead of a generic
    # runtime-error lesson that sends the teacher model down the wrong path.
    if "ImportError" in error:
        return "import_error"

    if "IndexError" in error:
        return "index_error"

    if "ZeroDivisionError" in error:
        return "division_by_zero"

    if "KeyError" in error:
        return "key_error"

    if "TypeError" in error:
        return "type_error"

    if "NameError" in error:
        return "name_error"

    if "RecursionError" in error:
        return "recursion_error"

    if "AttributeError" in error:
        return "attribute_error"

    if "Wrong Answer" in error:
        return "wrong_answer"

    return "runtime_error"


def normalize_nested_lists(obj):
    """
    Recursively sorts nested lists so logically equivalent outputs compare equal.
    Useful for anagrams, subsets, permutations, etc.
    """
    if isinstance(obj, list):
        normalized = [normalize_nested_lists(x) for x in obj]
        try:
            return sorted(normalized)
        except TypeError:
            return sorted(normalized, key=str)
    return obj


def _normalize_containers(obj):
    """
    Recursively converts tuples to lists so a solution that builds its
    result out of tuples instead of lists (a cosmetic, not logical,
    difference) isn't marked wrong purely on container type.
    """
    if isinstance(obj, (list, tuple)):
        return [_normalize_containers(x) for x in obj]
    return obj


def compare_outputs(task_dict, result, expected):
    """
    Compare outputs while allowing multiple valid representations
    for certain task categories.
    """

    tags = set(task_dict.get("required_tags", []))
    description = task_dict.get("description", "").lower()

    # -----------------------------
    # Group Anagrams
    # -----------------------------
    if "anagram" in description:
        return (
            normalize_nested_lists(result)
            == normalize_nested_lists(expected)
        )

    # -----------------------------
    # Permutations / Combinations / Subsets / Groupings
    # -----------------------------
    # These problems have many valid enumeration orders. A correct solution
    # that emits the same set of results in a different order should not be
    # marked wrong - only the multiset of results matters here. Widened to
    # include grouping/shifted-string tasks (e.g. task 234/148/25 - group
    # strings by shift sequence), which have the identical "many valid
    # group orders" property as anagram grouping but weren't originally
    # covered because "anagram" doesn't appear in their description.
    ORDER_INDEPENDENT_KEYWORDS = (
        "permutation", "combination", "subset", "arrangement",
        "group", "grouped", "shifted",
    )
    if any(kw in description for kw in ORDER_INDEPENDENT_KEYWORDS):
        if isinstance(result, list) and isinstance(expected, list):
            return (
                normalize_nested_lists(result)
                == normalize_nested_lists(expected)
            )

    # -----------------------------
    # "Find/return all X" style tasks
    # -----------------------------
    # Problems phrased as "find all words/elements that satisfy condition Y"
    # describe a set, not a sequence - there's no implied order in the
    # problem statement itself (e.g. task 220: "Find all words ... that can
    # be created by joining shorter dictionary words"). Scoped narrowly to
    # flat lists of primitives so it doesn't accidentally cover tasks where
    # order IS meaningful (e.g. "return the elements in sorted order").
    FIND_ALL_PHRASES = ("find all", "return all")
    if any(p in description for p in FIND_ALL_PHRASES) and "sorted" not in description:
        if (
            isinstance(result, list) and isinstance(expected, list)
            and all(not isinstance(x, (list, tuple, dict)) for x in result)
            and all(not isinstance(x, (list, tuple, dict)) for x in expected)
        ):
            try:
                return sorted(result, key=str) == sorted(expected, key=str)
            except TypeError:
                pass

    # -----------------------------
    # Topological Sort
    # -----------------------------
    if "topological" in description:

        test_cases = task_dict.get("test_cases", [])
        if not test_cases:
            return False

        edges = test_cases[0]["input"][0]

        if not isinstance(result, list):
            return False

        if len(result) != len(expected):
            return False

        pos = {node: i for i, node in enumerate(result)}

        for u, v in edges:
            if u not in pos or v not in pos:
                return False
            if pos[u] > pos[v]:
                return False

        return True

    # -----------------------------
    # Default comparison
    # -----------------------------
    # Normalize tuple/list distinction before comparing. The reference
    # answers are plain Python lists; a correct algorithm that happens to
    # build its result with tuples (e.g. task 232: reconstructing
    # [height, count] pairs as (height, count) tuples) should not fail
    # purely on container type - the values and structure are identical.
    return _normalize_containers(result) == _normalize_containers(expected)


# ==========================================
# SANDBOX BUILTINS
# ==========================================
# Only expose what algorithmic solutions plausibly need. Deliberately excludes
# open, eval, exec, compile, __import__, input, exit/quit, and similar
# filesystem/process/interpreter-escape hatches, since this executes
# untrusted, LLM-generated code. The subprocess boundary gives crash
# isolation but is not itself a security sandbox, so we also restrict what
# the executed code's global namespace can reach.
_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
    "callable", "chr", "complex", "dict", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "hash", "hex", "id", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min", "next", "oct",
    "ord", "pow", "print", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "type", "zip",
    "True", "False", "None", "NotImplemented",
    # Exception types generated solutions may legitimately raise/catch.
    "Exception", "ArithmeticError", "AssertionError", "AttributeError",
    "IndexError", "KeyError", "NameError", "OverflowError",
    "RecursionError", "RuntimeError", "StopIteration", "TypeError",
    "ValueError", "ZeroDivisionError",
]

SAFE_BUILTINS = {
    name: getattr(_builtins_module, name)
    for name in _SAFE_BUILTIN_NAMES
    if hasattr(_builtins_module, name)
}


# ==========================================
# RECOMMENDATION 1: IMPORT STATEMENT STRIPPING
# ==========================================
def _strip_import_statements(code: str) -> str:
    """
    Removes all `import X` / `from X import Y` statements from generated
    code, wherever they appear (top-level or inside function bodies).

    __import__ is intentionally excluded from SAFE_BUILTINS (see comment
    above), so any import statement - regardless of where it sits - raises
    `ImportError: __import__ not found` the moment it executes, since
    Python's `import` statement itself calls the __import__ builtin under
    the hood. Moving an import from inside a function to the top of the
    file does NOT fix this; the statement has to not exist at all. Needed
    names are instead supplied directly as globals in exec_env below.

    If the code fails to parse, it's returned unmodified - downstream
    syntax validation / execution will report the real error.
    """
    try:
        tree = _ast_module.parse(code)
    except SyntaxError:
        return code

    class ImportStripper(_ast_module.NodeTransformer):
        def visit_Import(self, node):
            return None

        def visit_ImportFrom(self, node):
            return None

    tree = ImportStripper().visit(tree)
    _ast_module.fix_missing_locations(tree)

    try:
        return _ast_module.unparse(tree)
    except Exception:
        # Extremely defensive fallback: if unparse ever fails for some
        # exotic AST shape, run the original code rather than crashing the
        # whole evaluation - it will simply fail with its original error.
        return code


# ==========================================
# MODULE 2: EXECUTOR SANDBOX (Steps 12 & 13)
# ==========================================
def _sandbox_worker(task_dict: Dict[str, Any], generated_code_str: str, return_dict: Dict[str, Any]) -> None:
    """
    Isolated worker execution thread injecting context utilities and evaluating
    against precise ast literal mappings.

    Evaluates ALL test cases (rather than stopping at the first failure) so
    the caller can compute partial-credit quality (pass_count/total_count)
    instead of a strict all-or-nothing pass/fail. Debug output and the
    detailed error message shown to the reflection module are still only
    captured from the FIRST failing test case, to keep repair guidance
    focused rather than dumping every failure at once.
    """
    import ast

    exec_env = {
        '__builtins__': SAFE_BUILTINS,
        'math': math,
        'collections': collections,
        'itertools': itertools,
        'heapq': heapq,
        'deque': deque,
        'Counter': Counter,
        'defaultdict': defaultdict,
        'List': List,
        'Dict': Dict,
        'Optional': Optional,
        'Any': Any,
        'Tuple': Tuple,
        'Set': set,
        # Recommendation 1: names commonly reached for via `import`, now
        # pre-supplied directly so stripped-import code still resolves.
        'functools': functools,
        'cmp_to_key': functools.cmp_to_key,
        'bisect': _bisect_module,
        'bisect_left': _bisect_module.bisect_left,
        'bisect_right': _bisect_module.bisect_right,
        'gcd': math.gcd,
    }

    # Assigned up front so the except-block below can always reference them,
    # even if exec() itself fails before the test-case loop runs.
    input_data = None
    expected_output = None

    try:
        # Recommendation 1: strip any import statements (top-level or
        # in-function) before execution, since __import__ is disabled.
        generated_code_str = _strip_import_statements(generated_code_str)

        print("=" * 80)
        print(generated_code_str)
        print("=" * 80)
        exec(generated_code_str, exec_env, exec_env)

        target_func = exec_env.get("solve")

        if not callable(target_func):
            return_dict['success'] = False
            return_dict['error'] = "Function 'solve' not found."
            return_dict['pass_count'] = 0
            return_dict['total_count'] = len(task_dict.get('test_cases', []))
            return

        all_passed = True
        pass_count = 0
        total_count = 0
        first_failure_captured = False

        for tc in task_dict.get('test_cases', []):
            total_count += 1

            # Safe literal evaluation parsing
            if isinstance(tc['input'], str):
                try:
                    input_data = ast.literal_eval(tc['input'])
                except (ValueError, SyntaxError):
                    input_data = tc['input']
            else:
                input_data = tc['input']

            if isinstance(tc['expected'], str):
                try:
                    expected_output = ast.literal_eval(tc['expected'])
                except (ValueError, SyntaxError):
                    expected_output = tc['expected']
            else:
                expected_output = tc['expected']

            if not isinstance(input_data, (list, tuple)):
                input_data = [input_data]

            result = target_func(*input_data)

            passed = compare_outputs(task_dict, result, expected_output)

            if passed:
                pass_count += 1
                continue

            all_passed = False

            # Only record debug output / detailed error for the FIRST
            # failure - keep counting pass/fail for every remaining test
            # case, but don't re-capture or overwrite the reflection-facing
            # error message with later failures.
            if first_failure_captured:
                continue

            first_failure_captured = True

            # Recommendation 3: detect order-only mismatches (same
            # elements/groups, wrong order) so the failure message
            # points the teacher model at the actual bug instead of
            # letting it guess and go down a sorting-key rabbit hole.
            order_only_mismatch = False
            try:
                if (
                    isinstance(result, list)
                    and isinstance(expected_output, list)
                    and normalize_nested_lists(result) == normalize_nested_lists(expected_output)
                ):
                    order_only_mismatch = True
            except Exception:
                pass

            print("=" * 70)
            print("DEBUG OUTPUT COMPARISON")
            print("EXPECTED :", repr(expected_output))
            print("TYPE     :", type(expected_output))
            print("RESULT   :", repr(result))
            print("TYPE     :", type(result))
            print("ORDER ONLY MISMATCH:", order_only_mismatch)
            print("=" * 70)

            # --------------------------------------------------
            # Provide explicit type mismatch information to the
            # reflection module (e.g. list vs tuple, int vs str).
            # --------------------------------------------------
            type_hint = ""

            if type(result) != type(expected_output):
                type_hint = (
                    f" TYPE MISMATCH: Expected "
                    f"{type(expected_output).__name__}, "
                    f"got {type(result).__name__}."
                )

            if order_only_mismatch:
                return_dict["error"] = (
                    f"Wrong Answer (ORDERING ONLY). Output contains the correct "
                    f"elements/groups, but their order does not match. "
                    f"Expected={expected_output}, Got={result}. Ensure output "
                    f"order follows the original input order of first "
                    f"appearance - do not sort or reorder groups/elements "
                    f"unless the problem explicitly requires it."
                    f"{type_hint}"
                )
            else:
                return_dict["error"] = (
                    f"Wrong Answer. Expected={expected_output}, Got={result}."
                    f"{type_hint}"
                )

            return_dict["failed_test_case"] = {
                "input": input_data,
                "expected": expected_output,
                "actual": result
            }

        return_dict['success'] = all_passed
        return_dict['pass_count'] = pass_count
        return_dict['total_count'] = total_count

    except Exception:
        return_dict["success"] = False
        return_dict["error"] = traceback.format_exc()
        return_dict["failed_test_case"] = {
            "input": input_data,
            "expected": expected_output,
            "actual": None
        }
        return_dict['pass_count'] = 0
        return_dict['total_count'] = len(task_dict.get('test_cases', []))


def evaluate_agent_code(task_dict: Dict[str, Any], generated_code_str: str) -> Dict[str, Any]:
    """
    Wrapper tracking time metrics and isolating execution parameters from main loop.

    Returns both:
      - 'success': strict all-tests-passed boolean (drives task_success_rate)
      - 'quality_score': continuous pass_count/total_count in [0,1] (drives
        the paper's Eq. (5) performance_score, and the "Mean Task Quality"
        history - previously this field WAS the binary success flag, which
        is why round-averaged quality/success could only land on multiples
        of 1/round_size instead of varying smoothly.
    """
    # --------------------------------------------------
    # Fast-fail on invalid Python before spawning a worker
    # --------------------------------------------------
    try:
        _ast_module.parse(generated_code_str)
    except SyntaxError as e:
        return {
            "quality_score": 0.0,
            "success": False,
            "delay_ratio": 0.0,
            "reason": f"SYNTAX ERROR: {e.msg} at line {e.lineno}.",
            "failure_type": "syntax_error",
            "failed_test_case": None,
            "pass_count": 0,
            "total_count": len(task_dict.get("test_cases", [])),
        }

    deadline = float(task_dict.get('deadline', 5))
    manager = Manager()

    try:
        return_dict = manager.dict()
        return_dict['success'] = False
        return_dict['error'] = None

        process = Process(target=_sandbox_worker, args=(task_dict, generated_code_str, return_dict))

        start_time = time.time()
        process.start()
        process.join(deadline)

        if process.is_alive():
            process.terminate()
            process.join()
            return {
                'quality_score': 0.0,
                'success': False,
                'delay_ratio': 1.0,
                'reason': 'Timeout threshold reached.',
                'failure_type': 'timeout',
                "failed_test_case": None,
                "pass_count": 0,
                "total_count": len(task_dict.get("test_cases", [])),
            }

        actual_time = time.time() - start_time
        delay_ratio = min(1.0, actual_time / deadline)

        success = bool(return_dict.get('success', False))
        pass_count = return_dict.get('pass_count', 0)
        total_count = return_dict.get('total_count', 0)

        if success:
            quality_score = 1.0
        else:
            quality_score = (pass_count / total_count) if total_count > 0 else 0.0

        if success:
            return {
                'quality_score': quality_score,
                'success': True,
                'delay_ratio': delay_ratio,
                'reason': 'Passed all deterministic checks.',
                'failure_type': None,
                "failed_test_case": None,
                "pass_count": pass_count,
                "total_count": total_count,
            }
        else:
            error = return_dict.get('error', 'Assertion failed.')

            return {
                'quality_score': quality_score,
                'success': False,
                'delay_ratio': delay_ratio,
                'reason': error,
                'failure_type': classify_failure(error),
                "failed_test_case": return_dict.get("failed_test_case", None),
                "pass_count": pass_count,
                "total_count": total_count,
            }
    finally:
        # Manager() spawns its own background server process; without an
        # explicit shutdown it leaks one process per evaluation call.
        manager.shutdown()


# ==========================================
# MODULE 3: THE AUCTIONEER (Steps 5 - 10)
# ==========================================
#for min-max normalization
def normalized_binarized_capabilities(agent: Agent, threshold: float = 0.4):
    """
    Per-agent min-max normalization + binarization (Paper Sec. V-B).
    """

    values = agent.capabilities

    v_min = min(values.values())
    v_max = max(values.values())

    if v_max == v_min:
        return {tag: 0 for tag in values}

    return {
        tag: 1 if ((value - v_min) / (v_max - v_min)) >= threshold else 0
        for tag, value in values.items()
    }


def run_auction(
    task_dict: Dict[str, Any], agents: List[Agent], temperature: float = 1.0
) -> Tuple[Optional[Agent], int, Optional[float], Optional[float]]:
    """
    Calculates operational capability indexing matches, score values, and bidding
    utility weights, then samples a winner via a temperature-scaled softmax
    over bidder scores (rather than deterministic argmax), so the highest-
    scoring agent isn't guaranteed to win every eligible task.

    Returns (winner_agent, num_bidders, winner_score, winner_utility).
    winner_score/winner_utility are returned so the caller doesn't have to
    recompute them (previously the caller recomputed both AFTER
    winner.workload had already been incremented here, so the logged
    utility used a stale, post-increment workload rather than the value
    that actually governed the auction decision).
    """
    required_tags = task_dict.get("required_tags", [])
    reward = task_dict.get("reward", 0)

    if not required_tags:
        return None, 0, None, None

    bidders = []

    for agent in agents:
        # Step 5: Compute Capability Match
        cap_match_sum = sum(agent.capabilities.get(tag, 0.0) for tag in required_tags)
        cap_match = cap_match_sum / len(required_tags)

        # Step 6: Compute Agent Score, per Eq. (8). NOT clipped to [0,1].
        # The paper softmaxes this raw score directly in Eq. (9); clipping
        # it first would compress differentiation between agents whose true
        # scores differ (e.g. a heavily-loaded agent scoring -0.3 vs. a
        # well-matched agent scoring 0.7 would otherwise be pushed toward
        # the same post-clip value before the softmax ever sees them).
        # Note the weights (lambda1=0.5, lambda2=0.3) mean the unclipped
        # score can reach at most 0.8 when workload=0, so an upper clip at
        # 1.0 was never actually binding in practice - only the lower
        # clip at 0.0 was, which is what's removed here.
        score = (0.5 * cap_match) + (0.3 * agent.reputation) - (0.2 * agent.workload)

        # Step 7: Compute Capability Mismatch
        
        w_hat = normalized_binarized_capabilities(agent)

        mismatch = sum(
            1 - w_hat.get(tag, 0)
            for tag in required_tags
        )

        # Step 8: Compute Utility
        utility = (score * reward) - (0.1 * agent.workload) - (0.4 * mismatch)

        # Step 9: Collect Positive Utility Bidders
        if utility > 0:
            bidders.append({
                'agent': agent,
                'score': score,
                'utility': utility
            })

    if not bidders:
        return None, 0, None, None

    # Step 10: Assignment Selection via softmax sampling. Score -> probability
    # through a temperature-scaled softmax, then sample (rather than always
    # taking argmax) so agents with lower-but-positive utility still
    # occasionally win, matching the "exponential transformation -> highest
    # score matches highest assignment probability" intent literally.
    scores = np.array([b['score'] for b in bidders], dtype=float)
    scaled = scores / max(temperature, 1e-8)
    exp_scores = np.exp(scaled - scaled.max())  # numerically stable softmax
    probs = exp_scores / exp_scores.sum()

    winner_idx = np.random.choice(len(bidders), p=probs)
    winner_package = bidders[winner_idx]
    winner_agent = winner_package['agent']

    winner_agent.workload += 1

    return winner_agent, len(bidders), winner_package['score'], winner_package['utility']


# ==========================================
# MODULE 4: CAPABILITY RECOVERY (Step 17)
# ==========================================
def apply_capability_recovery(
    agent: Agent,
    lesson_tags: list,
    mu: float,
    performance_score: float,
):
    """
    Step 17:
    Recover capability using a validated lesson.

    Uses the SAME exponential-smoothing form as the paper's Eq. (7) update,
    driven by the retry's own performance_score (ALPHA*quality + DELTA*(1
    - delay_ratio), computed by the caller from the retry's own evaluation)
    rather than a hardcoded full-credit value of 1.0. Previously this
    always pushed capability toward a fixed LESSON_VALUE=1.0 regardless of
    how well the retry actually performed - a stronger, uncapped boost than
    the graded first-attempt update path used for the exact same equation.
    Passing in the real performance_score keeps both write paths to
    agent.capabilities structurally consistent with Eq. (7): they differ
    only in *when* they fire (first attempt vs. a validated
    reflection-guided retry), not in *how* they compute the update target.
    """

    for tag in lesson_tags:

        old_cap = agent.capabilities.get(tag, 0.0)

        agent.capabilities[tag] = (
            mu * old_cap
            + (1 - mu) * performance_score
        )
        print(
            f"          {tag}: "
            f"{old_cap:.3f} -> "
            f"{agent.capabilities[tag]:.3f}"
        )
