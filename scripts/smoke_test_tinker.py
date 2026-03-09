"""
Smoke test: Verify Qwen3.5 models can write RLM code via Tinker API.

Tests basic RLM capabilities:
1. Can the model write valid Python?
2. Does it access the `context` variable?
3. Does it use `llm_query()` for sub-calls?
4. Does it terminate with FINAL/FINAL_VAR?
5. Does it get correct answers?

Usage:
    uv run python scripts/smoke_test_tinker.py --model Qwen/Qwen3.5-4B
    uv run python scripts/smoke_test_tinker.py --model Qwen/Qwen3.5-35B-A3B
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scaffold.llm_query import TinkerModel
from scaffold.rlm import rlm, trajectory_to_dict
from scaffold.prompts.qwen35_35b import QWEN35_35B_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


SMOKE_TASKS = [
    {
        "name": "needle_in_haystack",
        "prompt": (
            "QUESTION: What is the secret password?\n\n"
            "The weather forecast for tomorrow indicates partly cloudy skies with a high "
            "of 72 degrees Fahrenheit. Morning commuters should expect light traffic on "
            "the interstate. LOCAL NEWS: The city council met yesterday to discuss the "
            "proposed budget for the upcoming fiscal year. Several council members raised "
            "concerns about the allocation for road maintenance. In other news, the local "
            "library will be hosting a book fair next weekend.\n"
            "SECRET PASSWORD: PHOENIX-42\n"
            "Sports update: The regional basketball tournament concluded with an exciting "
            "final game. The home team won by a margin of 12 points. Weather experts "
            "predict rainfall later this week."
        ),
        "expected": "PHOENIX-42",
        "description": "Find a password in ~500 char buffer",
    },
    {
        "name": "count_lines",
        "prompt": (
            "QUESTION: How many lines are in this document?\n\n"
            "Line one of the document.\n"
            "Line two of the document.\n"
            "Line three of the document.\n"
            "Line four of the document.\n"
            "Line five of the document.\n"
            "Line six of the document.\n"
            "Line seven of the document."
        ),
        "expected": "7",
        "description": "Count lines (test basic Python)",
    },
    {
        "name": "first_last_word",
        "prompt": (
            "QUESTION: What are the first and last words in this document?\n\n"
            "Artificial intelligence has transformed many industries over the past "
            "decade, from healthcare to transportation to entertainment."
        ),
        "expected": "Artificial entertainment",
        "description": "Extract first and last word (test string ops)",
    },
    {
        "name": "multi_fact_extraction",
        "prompt": (
            "QUESTION: What is the name and city mentioned?\n\n"
            "The annual technology conference was held in Seattle this year. "
            "Dr. Sarah Mitchell presented her groundbreaking research on quantum "
            "computing applications. The event attracted over 500 participants "
            "from around the world."
        ),
        "expected": "Sarah Mitchell Seattle",
        "description": "Extract name and city (test multi-fact)",
    },
    {
        "name": "character_count",
        "prompt": (
            "QUESTION: How many characters (including spaces) are in the document below?\n\n"
            "Hello World"
        ),
        "expected": "11",
        "description": "Count characters (test len())",
    },
]


def run_smoke_test(model_name: str, system_prompt: str, verbose: bool = False):
    """Run smoke tests and return results."""
    logger.info(f"=== Smoke Test: {model_name} ===")

    model = TinkerModel(
        model_name=model_name,
        max_new_tokens=2048,
        temperature=0.7,
    )

    results = []
    for task in SMOKE_TASKS:
        logger.info(f"\n--- Task: {task['name']} ({task['description']}) ---")
        t0 = time.time()

        try:
            trajectory = rlm(
                prompt=task["prompt"],
                model=model,
                system_prompt=system_prompt,
                max_iterations=5,
                verbose=verbose,
            )
            elapsed = time.time() - t0

            # Check quality
            answer = trajectory.answer or ""
            correct = task["expected"].lower() in answer.lower() if answer else False

            # Analyze code quality
            code_blocks = []
            for turn in trajectory.turns:
                if turn.get("parsed_code"):
                    code_blocks.append(turn["parsed_code"])

            uses_context = any("context" in c for c in code_blocks)
            uses_llm_query = any("llm_query" in c for c in code_blocks)
            uses_final = any("FINAL" in c for c in code_blocks)
            valid_python = len(code_blocks) > 0  # At least one parseable code block

            result = {
                "task": task["name"],
                "correct": correct,
                "answer": answer[:200] if answer else None,
                "expected": task["expected"],
                "valid_python": valid_python,
                "uses_context": uses_context,
                "uses_llm_query": uses_llm_query,
                "uses_final": uses_final,
                "turns": len(trajectory.turns),
                "terminated": trajectory.terminated,
                "time": elapsed,
            }
            results.append(result)

            status = "CORRECT" if correct else "WRONG"
            logger.info(
                f"  {status} | answer='{answer[:100]}' | "
                f"turns={len(trajectory.turns)} | time={elapsed:.1f}s"
            )

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"  ERROR: {e}")
            results.append({
                "task": task["name"],
                "correct": False,
                "error": str(e),
                "time": elapsed,
                "valid_python": False,
                "uses_context": False,
                "uses_llm_query": False,
                "uses_final": False,
            })

    return results, model.total_stats()


def print_summary(results: list[dict], model_stats: dict, model_name: str):
    """Print formatted summary."""
    n = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    valid = sum(1 for r in results if r.get("valid_python", False))
    context = sum(1 for r in results if r.get("uses_context", False))
    llm_q = sum(1 for r in results if r.get("uses_llm_query", False))
    final = sum(1 for r in results if r.get("uses_final", False))

    print(f"\n{'='*60}")
    print(f"SMOKE TEST SUMMARY: {model_name}")
    print(f"{'='*60}")
    print(f"  Correct answers:    {correct}/{n} ({100*correct/n:.0f}%)")
    print(f"  Valid Python:       {valid}/{n} ({100*valid/n:.0f}%)")
    print(f"  Uses context:       {context}/{n} ({100*context/n:.0f}%)")
    print(f"  Uses llm_query:     {llm_q}/{n} ({100*llm_q/n:.0f}%)")
    print(f"  Uses FINAL:         {final}/{n} ({100*final/n:.0f}%)")
    print(f"  Total time:         {sum(r.get('time', 0) for r in results):.1f}s")
    print(f"  Model stats:        {model_stats}")
    print()

    # Decision guidance
    if correct >= 4:
        print("  VERDICT: Strong base capability. Proceed with SFT + RL pipeline.")
    elif correct >= 2:
        print("  VERDICT: Moderate capability. SFT should help significantly.")
    else:
        print("  VERDICT: Weak capability. May need intensive SFT or larger model.")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Smoke test RLM on Tinker")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B",
                        help="Model name on Tinker")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="System prompt to use (default: qwen35_35b)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Save results to directory")
    args = parser.parse_args()

    system_prompt = args.system_prompt or QWEN35_35B_SYSTEM_PROMPT

    results, model_stats = run_smoke_test(args.model, system_prompt, args.verbose)
    print_summary(results, model_stats, args.model)

    # Save results
    save_dir = args.save_dir or f"results/smoke_test_tinker/{args.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "smoke_test_results.json"), "w") as f:
        json.dump({
            "model": args.model,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "model_stats": model_stats,
            "summary": {
                "correct": sum(1 for r in results if r.get("correct")),
                "total": len(results),
                "accuracy": sum(1 for r in results if r.get("correct")) / len(results),
            },
        }, f, indent=2)

    logger.info(f"Results saved to {save_dir}/")


if __name__ == "__main__":
    main()
