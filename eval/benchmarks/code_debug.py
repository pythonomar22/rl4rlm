#!/usr/bin/env python3
"""
Code Debug Benchmark — Find bugs planted in large codebases.

The model receives a large codebase as `context` with planted bugs.
It must systematically search through functions to find and describe the bugs.

This tests the RLM's ability to:
1. Navigate large codebases by chunking
2. Use llm_query() to analyze individual functions
3. Cross-reference functions (bugs in function A that affect function B)
4. Aggregate findings across the codebase

Task types:
- Single bug: Find one planted bug in a large codebase
- Multi bug: Find N bugs scattered throughout
- Logic bug: Incorrect algorithm logic (not syntax)
- Off-by-one: Classic off-by-one errors

Context format:
```
QUESTION: Find the bug(s) in this codebase. Describe each bug: which function,
what's wrong, and what the fix should be.

CODE:
# ===== module: utils.py =====
def calculate_mean(values):
    ...
# ===== module: data.py =====
...
```
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


# Templates for generating plausible Python functions
CLEAN_FUNCTIONS = [
    {
        "name": "calculate_mean",
        "code": '''def calculate_mean(values):
    """Calculate arithmetic mean of a list."""
    if not values:
        return 0.0
    return sum(values) / len(values)''',
        "module": "stats",
    },
    {
        "name": "calculate_median",
        "code": '''def calculate_median(values):
    """Calculate median of a list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 0:
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    return sorted_vals[n // 2]''',
        "module": "stats",
    },
    {
        "name": "calculate_std",
        "code": '''def calculate_std(values):
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5''',
        "module": "stats",
    },
    {
        "name": "binary_search",
        "code": '''def binary_search(arr, target):
    """Binary search in sorted array. Returns index or -1."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1''',
        "module": "search",
    },
    {
        "name": "merge_sort",
        "code": '''def merge_sort(arr):
    """Merge sort implementation."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result''',
        "module": "sort",
    },
    {
        "name": "validate_email",
        "code": '''def validate_email(email):
    """Basic email validation."""
    if not email or "@" not in email:
        return False
    parts = email.split("@")
    if len(parts) != 2:
        return False
    local, domain = parts
    if not local or not domain:
        return False
    if "." not in domain:
        return False
    return True''',
        "module": "validation",
    },
    {
        "name": "flatten_list",
        "code": '''def flatten_list(nested):
    """Flatten a nested list."""
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result''',
        "module": "utils",
    },
    {
        "name": "count_words",
        "code": '''def count_words(text):
    """Count word frequencies in text."""
    words = text.lower().split()
    counts = {}
    for word in words:
        word = word.strip(".,!?;:")
        if word:
            counts[word] = counts.get(word, 0) + 1
    return counts''',
        "module": "text",
    },
    {
        "name": "matrix_multiply",
        "code": '''def matrix_multiply(a, b):
    """Multiply two matrices."""
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    if cols_a != rows_b:
        raise ValueError("Incompatible dimensions")
    result = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result''',
        "module": "math_ops",
    },
    {
        "name": "parse_csv_line",
        "code": '''def parse_csv_line(line):
    """Parse a CSV line respecting quoted fields."""
    fields = []
    current = ""
    in_quotes = False
    for char in line:
        if char == '"':
            in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            fields.append(current.strip())
            current = ""
        else:
            current += char
    fields.append(current.strip())
    return fields''',
        "module": "parser",
    },
    {
        "name": "lru_cache",
        "code": '''class LRUCache:
    """Least Recently Used cache."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = []

    def get(self, key):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return -1

    def put(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)''',
        "module": "cache",
    },
    {
        "name": "dijkstra",
        "code": '''def dijkstra(graph, start):
    """Dijkstra's shortest path algorithm."""
    import heapq
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    visited = set()
    while pq:
        dist, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    return distances''',
        "module": "graph",
    },
]

# Bug templates: (original_line, buggy_line, bug_description, bug_type)
BUG_TEMPLATES = {
    "calculate_mean": [
        (
            "return sum(values) / len(values)",
            "return sum(values) / len(values) - 1",
            "Off-by-one error in calculate_mean: divides by (len-1) instead of len, giving incorrect mean for lists of length > 1",
            "off_by_one",
        ),
    ],
    "calculate_median": [
        (
            "return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2",
            "return (sorted_vals[n // 2] + sorted_vals[n // 2 + 1]) / 2",
            "Off-by-one in calculate_median: uses wrong indices for even-length lists, accessing n//2 and n//2+1 instead of n//2-1 and n//2",
            "off_by_one",
        ),
    ],
    "calculate_std": [
        (
            "variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)",
            "variance = sum((x - mean) ** 2 for x in values) / len(values)",
            "Logic bug in calculate_std: uses population variance (divides by N) instead of sample variance (divides by N-1)",
            "logic",
        ),
    ],
    "binary_search": [
        (
            "left = mid + 1",
            "left = mid",
            "Infinite loop bug in binary_search: left = mid instead of mid + 1, causing infinite loop when target > arr[mid]",
            "logic",
        ),
    ],
    "validate_email": [
        (
            'if len(parts) != 2:',
            'if len(parts) > 2:',
            "Logic bug in validate_email: allows empty local part by checking > 2 instead of != 2, accepts emails like '@domain.com'",
            "logic",
        ),
    ],
    "flatten_list": [
        (
            "if isinstance(item, list):",
            "if isinstance(item, (list, tuple)):",
            "Over-flattening bug in flatten_list: also flattens tuples, which may be intentional data structures that should be preserved",
            "logic",
        ),
    ],
    "count_words": [
        (
            'counts[word] = counts.get(word, 0) + 1',
            'counts[word] = counts.get(word, 1)',
            "Logic bug in count_words: counts.get(word, 1) starts count at 1 and never increments, so every word appears exactly once",
            "logic",
        ),
    ],
    "matrix_multiply": [
        (
            "result[i][j] += a[i][k] * b[k][j]",
            "result[i][j] += a[i][k] * b[j][k]",
            "Index swap bug in matrix_multiply: uses b[j][k] instead of b[k][j], computing incorrect multiplication",
            "logic",
        ),
    ],
    "lru_cache": [
        (
            "oldest = self.order.pop(0)",
            "oldest = self.order.pop()",
            "Logic bug in LRUCache.put: pops most recently used (last) instead of least recently used (first), evicting wrong entry",
            "logic",
        ),
    ],
    "dijkstra": [
        (
            "if new_dist < distances[neighbor]:",
            "if new_dist <= distances[neighbor]:",
            "Performance bug in dijkstra: uses <= instead of <, causing unnecessary re-exploration of equal-distance paths",
            "logic",
        ),
    ],
}


def _generate_filler_code(seed: int, n_lines: int = 50) -> str:
    """Generate realistic-looking filler code to pad the codebase."""
    rng = random.Random(seed)
    templates = [
        "    # Process {thing} for {module}\n    result = []\n    for item in data:\n        if item.get('{field}'):\n            result.append(item['{field}'])",
        "    # Initialize {thing}\n    config = {{\n        'max_retries': {n1},\n        'timeout': {n2},\n        'batch_size': {n3},\n    }}",
        "    # Log {thing} metrics\n    logger.info(f'Processed {{len(items)}} {thing}')\n    stats = {{'count': len(items), 'time': elapsed}}",
        "    # Validate {thing}\n    if not isinstance({thing}_data, dict):\n        raise TypeError(f'Expected dict, got {{type({thing}_data)}}')\n    required_fields = ['{field}', 'timestamp', 'status']",
        "    # Cache {thing} results\n    cache_key = f'{thing}_{{hash(str(params))}}'\n    if cache_key in self._cache:\n        return self._cache[cache_key]",
    ]
    things = ["user", "order", "payment", "session", "request", "response", "config", "metric", "event", "record"]
    fields = ["name", "id", "value", "status", "type", "timestamp", "amount", "count", "score", "label"]
    modules = ["auth", "billing", "analytics", "notifications", "scheduler", "api", "db", "cache"]

    lines = []
    for _ in range(n_lines // 5):
        tmpl = rng.choice(templates)
        code = tmpl.format(
            thing=rng.choice(things),
            field=rng.choice(fields),
            module=rng.choice(modules),
            n1=rng.randint(1, 10),
            n2=rng.randint(5, 120),
            n3=rng.randint(16, 256),
        )
        func_name = f"process_{rng.choice(things)}_{rng.choice(modules)}"
        lines.append(f"def {func_name}(data, params=None):\n{code}\n")

    return "\n\n".join(lines)


@dataclass
class CodeDebugTask:
    """A code debugging task."""
    task_id: str
    prompt: str
    question: str
    bugs: list[dict]  # [{function, description, bug_type}, ...]
    n_functions: int
    n_bugs: int
    total_lines: int


def generate_code_debug_task(
    task_idx: int,
    n_bugs: int = 1,
    n_filler_functions: int = 20,
    seed: int | None = None,
) -> CodeDebugTask:
    """Generate a code debugging task with planted bugs."""
    seed = seed if seed is not None else task_idx + 95000
    rng = random.Random(seed)

    # Select functions and bugs
    available_funcs = list(CLEAN_FUNCTIONS)
    rng.shuffle(available_funcs)

    # Select n_bugs functions to have bugs
    buggable = [f for f in available_funcs if f["name"] in BUG_TEMPLATES]
    n_bugs = min(n_bugs, len(buggable))
    buggy_funcs = buggable[:n_bugs]
    clean_funcs = [f for f in available_funcs if f not in buggy_funcs][:8]

    # Create bugs
    bugs = []
    code_sections = []

    for func in buggy_funcs:
        bug_options = BUG_TEMPLATES[func["name"]]
        original, buggy, description, bug_type = rng.choice(bug_options)
        buggy_code = func["code"].replace(original, buggy)
        code_sections.append({
            "module": func["module"],
            "code": buggy_code,
            "has_bug": True,
        })
        bugs.append({
            "function": func["name"],
            "description": description,
            "bug_type": bug_type,
        })

    for func in clean_funcs:
        code_sections.append({
            "module": func["module"],
            "code": func["code"],
            "has_bug": False,
        })

    # Add filler code
    for i in range(n_filler_functions):
        filler = _generate_filler_code(seed * 100 + i, n_lines=rng.randint(20, 40))
        module = rng.choice(["utils", "helpers", "services", "handlers", "middleware", "core"])
        code_sections.append({
            "module": f"{module}_{i}",
            "code": filler,
            "has_bug": False,
        })

    # Shuffle
    rng.shuffle(code_sections)

    # Build codebase text
    codebase_parts = []
    for section in code_sections:
        codebase_parts.append(f"# ===== module: {section['module']}.py =====\n{section['code']}")

    codebase = "\n\n".join(codebase_parts)
    total_lines = codebase.count("\n") + 1

    if n_bugs == 1:
        question = (
            f"This codebase contains exactly 1 bug. Find it and describe:\n"
            f"1. Which function contains the bug\n"
            f"2. What the bug is\n"
            f"3. What the correct fix should be\n"
            f"Return your answer as: FUNCTION: <name> | BUG: <description> | FIX: <fix>"
        )
    else:
        question = (
            f"This codebase contains exactly {n_bugs} bugs. Find all of them.\n"
            f"For each bug, describe:\n"
            f"1. Which function contains the bug\n"
            f"2. What the bug is\n"
            f"3. What the correct fix should be\n"
            f"Return each bug on a separate line as: FUNCTION: <name> | BUG: <description> | FIX: <fix>"
        )

    prompt = f"QUESTION: {question}\n\nCODE:\n{codebase}"

    task_id = f"debug_{task_idx:03d}_{n_bugs}bugs_{total_lines}lines"

    return CodeDebugTask(
        task_id=task_id,
        prompt=prompt,
        question=question,
        bugs=bugs,
        n_functions=len(code_sections),
        n_bugs=n_bugs,
        total_lines=total_lines,
    )


def score_code_debug(answer: str | None, bugs: list[dict]) -> dict:
    """Score a code debugging answer."""
    if answer is None:
        return {"score": 0.0, "found": 0, "total": len(bugs)}

    answer_lower = answer.lower()
    found = 0
    details = []

    for bug in bugs:
        func_name = bug["function"].lower()
        # Check if the function name is mentioned
        if func_name in answer_lower:
            found += 1
            details.append({"function": bug["function"], "found": True})
        else:
            details.append({"function": bug["function"], "found": False})

    score = found / len(bugs) if bugs else 0
    return {"score": score, "found": found, "total": len(bugs), "details": details}


def generate_code_debug_suite(
    n_tasks: int = 15,
    seed_offset: int = 95000,
) -> list[CodeDebugTask]:
    """Generate code debug benchmark suite.

    Configurations:
    - 1 bug in small codebase (~200 lines): easy
    - 1 bug in large codebase (~500 lines): medium
    - 2 bugs in large codebase: hard
    - 3 bugs in very large codebase (~1000 lines): very hard
    """
    configs = [
        # (n_bugs, n_filler, count)
        (1, 10, 4),     # Easy: ~200 lines
        (1, 30, 4),     # Medium: ~500 lines
        (2, 30, 4),     # Hard: ~500 lines, 2 bugs
        (3, 50, 3),     # Very hard: ~1000 lines, 3 bugs
    ]

    tasks = []
    idx = 0
    for n_bugs, n_filler, count in configs:
        for _ in range(count):
            if len(tasks) >= n_tasks:
                return tasks
            tasks.append(generate_code_debug_task(
                task_idx=idx,
                n_bugs=n_bugs,
                n_filler_functions=n_filler,
                seed=idx + seed_offset,
            ))
            idx += 1

    return tasks


if __name__ == "__main__":
    tasks = generate_code_debug_suite(n_tasks=4)
    for t in tasks:
        print(f"{t.task_id}: {t.n_bugs} bugs in {t.n_functions} functions ({t.total_lines} lines, {len(t.prompt):,} chars)")
        for b in t.bugs:
            print(f"  BUG: {b['function']} — {b['description'][:80]}...")
        print()
