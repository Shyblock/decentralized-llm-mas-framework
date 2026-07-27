import os
import json
import logging
from groq import APIError, APIConnectionError, RateLimitError
from dotenv import load_dotenv

from llm_provider import generate_groq

load_dotenv()

logger = logging.getLogger(__name__)

TEACHER_MODEL = "llama-3.3-70b-versatile"
MAX_COMPLETION_TOKENS = 350
TEMPERATURE = 0.1
MAX_RETRIES = 3

GENERIC_LESSON_WORD_LIMIT = 40
TASK_SPECIFIC_LESSON_WORD_LIMIT = 80

# --------------------------------------------------------------------
# The teacher model always runs on Groq (llama-3.3-70b-versatile),
# independent of config.PROVIDER. This is now an intentional, explicit
# dependency: previously this module built its own single, non-rotating
# Groq client here (`teacher_client = Groq(api_key=...)`), which silently
# ignored config.PROVIDER entirely and had no failover if that one key got
# rate-limited mid-run - it would just retry the same exhausted key 3x and
# give up. generate_groq() (from llm_provider.py) reuses the shared Groq
# key pool and rotation logic that the main agent generation path already
# has, so a rate-limited key here fails over the same way.
# --------------------------------------------------------------------


def _format_failed_case(failed_test_case: dict | None) -> str:
    if failed_test_case is None:
        return (
            "No failed test case was available. The failure occurred before a "
            "valid test case result could be produced, or execution terminated "
            "before completion."
        )
    return (
        f"Input:\n{failed_test_case.get('input')}\n\n"
        f"Expected Output:\n{failed_test_case.get('expected')}\n\n"
        f"Actual Output:\n{failed_test_case.get('actual')}"
    )


def _failure_guidance(failure_type: str) -> str:
    if failure_type == "syntax_error":
        return (
            "The agent failed to generate syntactically valid Python code. Since no "
            "executable solution is available, do NOT infer an algorithmic lesson from "
            "the reference solution. Instead, identify ONE reusable principle that "
            "improves code-generation quality (e.g. preserving the required function "
            "signature, producing syntactically valid Python, matching parentheses/"
            "indentation/control flow, correct recursive calls, returning only "
            "executable Python). Focus on generation correctness, not algorithm choice."
        )

    if failure_type == "import_error":
        return (
            "The agent's code used an `import` statement. This sandbox has "
            "`__import__` disabled for security, so ANY import statement - anywhere "
            "in the code, top-level or inside a function - fails immediately, "
            "regardless of where it is placed. Do NOT recommend moving the import; "
            "recommend removing it entirely. The names Counter, defaultdict, "
            "cmp_to_key, bisect, bisect_left, bisect_right, gcd, math, collections, "
            "itertools, heapq, deque, and functools are already available as globals "
            "without importing them, so the fix is almost always to delete the "
            "import line and use the name directly, or implement the needed logic "
            "inline if the name is not one of those."
        )

    return (
        "Compare the generated solution against the reference solution across: "
        "algorithm selection, correctness, edge-case handling, recursion/iteration "
        "strategy, data structures, state management, index handling, complexity, "
        "and robustness. Determine whether the failure is primarily an incorrect "
        "algorithm choice, a flawed implementation of a correct algorithm, missing "
        "edge-case handling, incorrect state/data-structure use, or incorrect "
        "recursion/iteration logic. Use the stated failure type as one signal, but "
        "verify it against the code, execution result, and reference solution rather "
        "than trusting it alone."
    )


def _build_prompt(
    task: dict,
    generated_code: str,
    error: str,
    failure_type: str,
    failed_test_case: dict | None,
) -> str:
    is_syntax_error = failure_type == "syntax_error"

    reference_solution_block = (
        "Not provided. No executable solution was generated, so focus on producing "
        "valid Python rather than algorithmic correctness."
        if is_syntax_error
        else task.get("reference_solution", "Not provided.")
    )

    return f"""You are an expert software engineer, competitive programmer, and code reviewer.
Analyze this failed programming attempt and produce two lessons.

=== TASK ===
Description: {task.get("description", "Not provided.")}
Required Capability Tags: {", ".join(task.get("required_tags", []))}

=== REFERENCE SOLUTION ===
{reference_solution_block}

=== FAILED AGENT SOLUTION ===
{generated_code}

=== EXECUTION RESULT ===
{error}
Failure Type: {failure_type}

=== FAILED TEST CASE ===
{_format_failed_case(failed_test_case)}

=== ANALYSIS GUIDANCE ===
{_failure_guidance(failure_type)}

=== OBJECTIVES ===

LESSON 1 — Generic Lesson (permanently stored in memory, must generalize to future
unrelated tasks with similar capability tags):
- Reusable, transferable programming/algorithmic principle that would have prevented
  this class of failure.
- Implementation-oriented, not a description of this specific bug.
- No variable names, function names, literal values, or references to this task.
- Max {GENERIC_LESSON_WORD_LIMIT} words.

LESSON 2 — Task-specific Lesson (used only for the immediate retry, not stored):
- Identify the exact mistake in THIS implementation.
- Recommend the smallest concrete change likely to make it pass; preserve all
  correct parts of the existing code and modify only the incorrect logic.
- Only recommend a different algorithm if the current one is fundamentally
  unrepairable.
- Concrete enough for another programmer to implement directly.
- Max {TASK_SPECIFIC_LESSON_WORD_LIMIT} words.

The two lessons serve different purposes and must NOT be paraphrases of each other:
Lesson 1 optimizes for durable, transferable insight; Lesson 2 optimizes for making
this specific retry succeed. If improving one would weaken the other, prioritize
each lesson's own purpose over making them similar.

Internally: first decide whether the current algorithm is fundamentally correct: if
so, isolate the implementation bug; if not, identify the minimal algorithmic fix. Do
not output this reasoning.

=== OUTPUT FORMAT ===
Return EXACTLY one JSON object, no markdown, no code fences, no explanations, no
extra keys, nothing before or after it. Both values must be non-empty strings:

{{
    "generic_lesson": "<string>",
    "task_specific_lesson": "<string>"
}}"""


def _truncate_to_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def _call_teacher(prompt: str) -> dict:
    """Single API call + parse. Raises on any failure so the caller can retry."""
    content = generate_groq(
        messages=[{"role": "user", "content": prompt}],
        model=TEACHER_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_COMPLETION_TOKENS,
        response_format={"type": "json_object"},
    ).strip()

    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("```", 1)[0].strip()

    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError("Teacher did not return a JSON object.")

    generic_lesson = result.get("generic_lesson")
    task_specific_lesson = result.get("task_specific_lesson")

    if not isinstance(generic_lesson, str) or not generic_lesson.strip():
        raise ValueError("generic_lesson missing or not a non-empty string.")
    if not isinstance(task_specific_lesson, str) or not task_specific_lesson.strip():
        raise ValueError("task_specific_lesson missing or not a non-empty string.")

    return {
        "generic_lesson": generic_lesson.strip(),
        "task_specific_lesson": task_specific_lesson.strip(),
    }


def generate_lesson(
    task: dict,
    generated_code: str,
    error: str,
    failure_type: str,
    failed_test_case: dict | None = None,
) -> tuple[str | None, str | None]:
    """
    Generate a reusable programming lesson from a failed attempt.
    The generic lesson should generalize to future tasks with similar capability tags;
    the task-specific lesson is only for the immediate retry.

    Returns (generic_lesson, task_specific_lesson), or (None, None) if generation
    fails after retries.
    """
    prompt = _build_prompt(task, generated_code, error, failure_type, failed_test_case)

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = _call_teacher(prompt)

            generic_lesson = _truncate_to_words(
                result["generic_lesson"], GENERIC_LESSON_WORD_LIMIT
            )
            task_specific_lesson = _truncate_to_words(
                result["task_specific_lesson"], TASK_SPECIFIC_LESSON_WORD_LIMIT
            )

            return generic_lesson, task_specific_lesson

        except (RateLimitError, APIConnectionError, APIError) as e:
            last_error = e
            logger.warning(
                "[Reflection] Teacher API error on attempt %d/%d: %s",
                attempt, MAX_RETRIES, e,
            )
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            logger.warning(
                "[Reflection] Invalid teacher response on attempt %d/%d: %s",
                attempt, MAX_RETRIES, e,
            )

    logger.error("[Reflection] Giving up after %d attempts: %s", MAX_RETRIES, last_error)
    return None, None
