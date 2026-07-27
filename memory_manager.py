import os
import json
import tempfile

MEMORY_DIR = "memory"

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


def get_memory_path(agent_id):
    return os.path.join(
        MEMORY_DIR,
        f"agent_{agent_id}.json"
    )


def load_memory(agent_id):
    """
    Memory format:

    {
        "next_lesson_id": 1,

        "lessons": {
            "L1": {
                "lesson": "...",
                "required_tags": [...],
                "failure_type": "...",
                "task_id": ...
            }
        },

        "memory_index": {
            "sorting": ["L1"],
            "graphs": [],
            ...
        }
    }
    """

    path = get_memory_path(agent_id)

    if not os.path.exists(path):

        return {
            "next_lesson_id": 1,
            "lessons": {},
            "memory_index": {
                tag: []
                for tag in CAPABILITY_TAGS
            }
        }

    with open(path, "r") as f:
        return json.load(f)


def save_memory(agent_id, memory):
    """
    Writes memory to disk atomically: creates MEMORY_DIR if missing, then
    writes to a temp file and renames it over the target so a crash or
    concurrent write mid-save can't leave a truncated/corrupt JSON file
    that load_memory() would fail to parse on the next run.
    """

    os.makedirs(MEMORY_DIR, exist_ok=True)

    path = get_memory_path(agent_id)

    fd, tmp_path = tempfile.mkstemp(
        dir=MEMORY_DIR,
        prefix=f"agent_{agent_id}_",
        suffix=".json.tmp",
    )

    try:
        with os.fdopen(fd, "w") as f:
            json.dump(memory, f, indent=4)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def add_lesson(agent_id, lesson):
    """
    Store a validated lesson only if it is not already present
    among lessons sharing at least one capability tag.
    """

    memory = load_memory(agent_id)

    lessons = memory["lessons"]
    memory_index = memory["memory_index"]

    # ---------------------------------------
    # Retrieve candidate lesson ids
    # ---------------------------------------
    candidate_ids = set()

    for tag in lesson["required_tags"]:
        candidate_ids.update(
            memory_index.get(tag, [])
        )

    # ---------------------------------------
    # Duplicate detection
    # ---------------------------------------
    for lesson_id in candidate_ids:

        existing = lessons[lesson_id]

        if existing["lesson"] == lesson["lesson"]:
            return False

    # ---------------------------------------
    # Generate new lesson id
    # ---------------------------------------
    lesson_id = f"L{memory['next_lesson_id']}"
    memory["next_lesson_id"] += 1

    lesson["required_tags"] = sorted(
        lesson["required_tags"]
    )

    # ---------------------------------------
    # Store lesson
    # ---------------------------------------
    lessons[lesson_id] = lesson

    # ---------------------------------------
    # Update capability index
    # ---------------------------------------
    for tag in lesson["required_tags"]:

        if tag not in memory_index:
            memory_index[tag] = []

        if lesson_id not in memory_index[tag]:
            memory_index[tag].append(lesson_id)

    save_memory(agent_id, memory)

    return True


def retrieve_lessons(agent_id, required_tags, k=3):
    """
    Retrieve Top-K relevant lessons using the capability index.
    """

    memory = load_memory(agent_id)

    lessons = memory["lessons"]
    memory_index = memory["memory_index"]

    # ---------------------------------------
    # Candidate lesson ids
    # ---------------------------------------
    candidate_ids = set()

    for tag in required_tags:
        candidate_ids.update(
            memory_index.get(tag, [])
        )

    # ---------------------------------------
    # Score candidates
    # ---------------------------------------
    scored = []

    required_tags = set(required_tags)

    for lesson_id in candidate_ids:

        lesson = lessons[lesson_id]

        overlap = len(
            required_tags &
            set(lesson["required_tags"])
        )

        if overlap > 0:

            scored.append(
                (
                    overlap,
                    lesson
                )
            )

    # ---------------------------------------
    # Top-K retrieval
    # ---------------------------------------
    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        lesson
        for _, lesson in scored[:k]
    ]
