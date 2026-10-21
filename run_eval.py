"""
STEP 4: Evaluation Harness
-----------------------------
Runs the agent over a curated set of real bug-fixing tasks and reports
a RESOLVE RATE -- not just "does it work on one example I've been
staring at", but "how often does it actually work, and on what kinds
of bugs does it fail".

Why not real SWE-bench-lite? Genuine SWE-bench tasks are full GitHub
issues against real historical commits of large projects (Django,
requests, etc.), each needing its own dependency/environment setup --
that's exactly why the OFFICIAL SWE-bench harness runs every task
inside its own Docker container. That's real infrastructure work, not
a weekend addition on top of everything else here. Swapping in real
SWE-bench-lite later is a legitimate "next step" for this project, and
worth saying so plainly in interviews rather than overclaiming.

Instead, this is a small hand-built benchmark of 6 genuinely DIFFERENT
bug patterns (null handling, off-by-one, wrong comparison operator,
missing branch, mutable default argument, falsy-vs-None) -- enough to
measure real generalization without the environment-provisioning
problem.
"""

import io
import os
import sys
import time
from contextlib import redirect_stdout

import agent
import sandbox

TASKS = [
    {
        "name": "none_check",
        "repo": "benchmark/task_01_none_check",
        "task": (
            "The login method in app.py has a bug: it crashes instead of "
            "returning False when the username doesn't exist. There's a "
            "test suite at test_app.py -- find the bug, propose a fix, "
            "and make sure all tests pass."
        ),
    },
    {
        "name": "off_by_one",
        "repo": "benchmark/task_02_off_by_one",
        "task": (
            "sum_first_n() in app.py is supposed to sum the first n "
            "numbers of a list but returns the wrong total for some "
            "inputs. There's a test suite at test_app.py -- find the "
            "bug, propose a fix, and make sure all tests pass."
        ),
    },
    {
        "name": "wrong_operator",
        "repo": "benchmark/task_03_wrong_operator",
        "task": (
            "is_senior_discount_eligible() in app.py should return True "
            "at exactly age 65 and above, but doesn't for one edge case. "
            "There's a test suite at test_app.py -- find the bug, "
            "propose a fix, and make sure all tests pass."
        ),
    },
    {
        "name": "missing_return",
        "repo": "benchmark/task_04_missing_return",
        "task": (
            "classify_number() in app.py should return 'positive', "
            "'negative', or 'zero', but one of those cases doesn't work "
            "correctly. There's a test suite at test_app.py -- find the "
            "bug, propose a fix, and make sure all tests pass."
        ),
    },
    {
        "name": "mutable_default",
        "repo": "benchmark/task_05_mutable_default",
        "task": (
            "add_to_cart() in app.py has a subtle bug where separate "
            "calls unexpectedly share state with each other. There's a "
            "test suite at test_app.py -- find the bug, propose a fix, "
            "and make sure all tests pass."
        ),
    },
    {
        "name": "falsy_vs_none",
        "repo": "benchmark/task_06_falsy_vs_none",
        "task": (
            "get_or_default() in app.py should only fall back to the "
            "default when the value is missing (None), but it incorrectly "
            "falls back for some other values too. There's a test suite "
            "at test_app.py -- find the bug, propose a fix, and make "
            "sure all tests pass."
        ),
    },
]


class _Tee(io.StringIO):
    """Capture the agent's output for the transcript while still showing
    it live, so a long run never looks frozen."""

    def __init__(self, console):
        super().__init__()
        self.console = console

    def write(self, text):
        self.console.write(text)
        self.console.flush()
        return super().write(text)


def _tests_pass(repo_dir):
    """Reuse sandbox.run_tests directly against the real repo dir (not
    a sandbox copy) -- this is the ground truth check of whether the
    task is ACTUALLY solved, independent of what the agent claims."""
    passed, _ = sandbox.run_tests(repo_dir)
    return passed


def run_evaluation(tasks=TASKS, transcript_dir="eval_transcripts"):
    os.makedirs(transcript_dir, exist_ok=True)
    results = []

    for task in tasks:
        print(f"\n{'=' * 60}\nTASK: {task['name']}\n{'=' * 60}")

        app_path = os.path.join(task["repo"], "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            original_code = f.read()  # so we can restore it after this run

        transcript = _Tee(sys.stdout)
        start = time.time()
        crashed = None
        try:
            with redirect_stdout(transcript):
                agent.run_agent(task["task"], repo_path=task["repo"])
        except Exception as e:  # noqa: BLE001 -- eval harness must not die on one bad task
            crashed = str(e)
        duration = round(time.time() - start, 1)

        solved = crashed is None and _tests_pass(task["repo"])
        transcript_text = transcript.getvalue()
        attempts = transcript_text.count("[Action] propose_edit")

        # Restore the ORIGINAL buggy code so re-running this evaluation
        # later starts from the same clean baseline every time.
        with open(app_path, "w", encoding="utf-8") as f:
            f.write(original_code)

        with open(os.path.join(transcript_dir, f"{task['name']}.txt"), "w", encoding="utf-8") as f:
            f.write(transcript_text if crashed is None else f"{transcript_text}\n\nCRASHED: {crashed}")

        status = "PASS" if solved else "FAIL"
        print(f"[Eval] {task['name']}: {status} ({attempts} attempt(s), {duration}s)")
        if crashed:
            print(f"[Eval] Crashed: {crashed}")

        results.append(
            {"name": task["name"], "solved": solved, "attempts": attempts, "seconds": duration, "crashed": crashed}
        )

    print_summary(results)
    return results


def print_summary(results):
    solved_count = sum(1 for r in results if r["solved"])
    total = len(results)

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {solved_count}/{total} solved ({100 * solved_count / total:.0f}%)")
    print("=" * 60)
    for r in results:
        mark = "PASS" if r["solved"] else "FAIL"
        note = f" (crashed: {r['crashed']})" if r["crashed"] else ""
        print(f"  [{mark}] {r['name']:<20} {r['attempts']} attempt(s), {r['seconds']}s{note}")


if __name__ == "__main__":
    run_evaluation()
