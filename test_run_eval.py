"""
Tests for run_eval.py's ORCHESTRATION logic (backup/restore, resolve-
rate calculation, crash resilience, transcript writing) -- with
agent.run_agent mocked, since we're testing the harness itself here,
not the agent's actual bug-fixing ability (that needs the real API,
by design -- this just proves the harness reports correctly whatever
the agent does).

Run with:  python -m pytest test_run_eval.py -v
"""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(__file__))
import run_eval  # noqa: E402


def make_task(tmp_path, name, buggy_code, test_code):
    repo = tmp_path / name
    repo.mkdir()
    (repo / "app.py").write_text(buggy_code)
    (repo / "test_app.py").write_text(test_code)
    return {"name": name, "repo": str(repo), "task": "fix it"}


def test_resolve_rate_and_restore_after_solved_and_unsolved_tasks(tmp_path):
    task_a = make_task(
        tmp_path, "task_a",
        buggy_code="def add(a, b):\n    return a - b\n",
        test_code=(
            "import sys, os\nsys.path.insert(0, os.path.dirname(__file__))\n"
            "from app import add\ndef test_add():\n    assert add(2, 3) == 5\n"
        ),
    )
    task_b = make_task(
        tmp_path, "task_b",
        buggy_code="def sub(a, b):\n    return a + b\n",
        test_code=(
            "import sys, os\nsys.path.insert(0, os.path.dirname(__file__))\n"
            "from app import sub\ndef test_sub():\n    assert sub(5, 3) == 2\n"
        ),
    )
    original_buggy_b = (tmp_path / "task_b" / "app.py").read_text()

    def fake_run_agent(task_text, repo_path=None):
        # Simulate: the agent successfully fixes task_a, but never
        # manages to fix task_b (leaves it untouched, still broken).
        if repo_path.endswith("task_a"):
            with open(os.path.join(repo_path, "app.py"), "w") as f:
                f.write("def add(a, b):\n    return a + b\n")
        return None

    with patch("agent.run_agent", side_effect=fake_run_agent):
        results = run_eval.run_evaluation(
            tasks=[task_a, task_b], transcript_dir=str(tmp_path / "transcripts")
        )

    solved = {r["name"]: r["solved"] for r in results}
    assert solved == {"task_a": True, "task_b": False}

    # Both tasks should be back to their ORIGINAL buggy state after the
    # run -- a solved task gets reset too, so re-running the eval later
    # starts from the same baseline every time.
    assert "a - b" in (tmp_path / "task_a" / "app.py").read_text()
    assert (tmp_path / "task_b" / "app.py").read_text() == original_buggy_b

    # Transcripts should exist for both tasks.
    assert os.path.exists(tmp_path / "transcripts" / "task_a.txt")
    assert os.path.exists(tmp_path / "transcripts" / "task_b.txt")


def test_one_crashed_task_does_not_stop_the_rest(tmp_path):
    task_a = make_task(
        tmp_path, "task_a",
        buggy_code="def add(a, b):\n    return a - b\n",
        test_code=(
            "import sys, os\nsys.path.insert(0, os.path.dirname(__file__))\n"
            "from app import add\ndef test_add():\n    assert add(2, 3) == 5\n"
        ),
    )
    task_b = make_task(
        tmp_path, "task_b",
        buggy_code="def sub(a, b):\n    return a + b\n",
        test_code=(
            "import sys, os\nsys.path.insert(0, os.path.dirname(__file__))\n"
            "from app import sub\ndef test_sub():\n    assert sub(5, 3) == 2\n"
        ),
    )

    def fake_run_agent(task_text, repo_path=None):
        if repo_path.endswith("task_a"):
            raise RuntimeError("simulated API meltdown")
        # task_b succeeds normally
        with open(os.path.join(repo_path, "app.py"), "w") as f:
            f.write("def sub(a, b):\n    return a - b\n")
        return None

    with patch("agent.run_agent", side_effect=fake_run_agent):
        results = run_eval.run_evaluation(
            tasks=[task_a, task_b], transcript_dir=str(tmp_path / "transcripts")
        )

    by_name = {r["name"]: r for r in results}
    assert by_name["task_a"]["solved"] is False
    assert "meltdown" in by_name["task_a"]["crashed"]
    # task_b should be UNAFFECTED by task_a's crash.
    assert by_name["task_b"]["solved"] is True


def test_attempt_count_reflects_number_of_propose_edit_calls(tmp_path, capsys):
    task_a = make_task(
        tmp_path, "task_a",
        buggy_code="def add(a, b):\n    return a - b\n",
        test_code=(
            "import sys, os\nsys.path.insert(0, os.path.dirname(__file__))\n"
            "from app import add\ndef test_add():\n    assert add(2, 3) == 5\n"
        ),
    )

    def fake_run_agent(task_text, repo_path=None):
        # Simulate the printed transcript of two propose_edit attempts,
        # the way the real agent.py would print them.
        print("[Action] propose_edit({'target_name': 'add', ...})")
        print("[Verify] Tests FAILED.")
        print("[Action] propose_edit({'target_name': 'add', ...})")
        with open(os.path.join(repo_path, "app.py"), "w") as f:
            f.write("def add(a, b):\n    return a + b\n")
        return None

    with patch("agent.run_agent", side_effect=fake_run_agent):
        results = run_eval.run_evaluation(tasks=[task_a], transcript_dir=str(tmp_path / "transcripts"))

    assert results[0]["attempts"] == 2
    # Output is shown live AND saved to the transcript.
    assert "[Verify] Tests FAILED." in capsys.readouterr().out
    saved = (tmp_path / "transcripts" / "task_a.txt").read_text()
    assert "[Verify] Tests FAILED." in saved


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
