"""
STEP 3: The Sandbox
---------------------
This is where the agent stops just TALKING about a fix and starts
proving it actually works.

The idea in plain words: imagine a padded practice room. We copy the
target repo into a temporary folder (the sandbox) before the agent's
edit ever gets applied. If the edit is wrong, nothing about your real
project is touched -- we just throw the sandbox away and try again
from a clean copy.

NOTE on scope: a production coding agent would usually isolate this
with git worktrees or a Docker container for stronger guarantees
(e.g. the edited code can't accidentally read your real filesystem).
For this project's MVP, a plain temp-directory copy is enough
isolation to prove the concept safely, and it needs zero extra
installs. That trade-off (simplicity vs. stronger isolation) is worth
mentioning if you're asked about it in an interview.
"""

import ast
import os
import shutil
import subprocess
import sys
import tempfile


def create_sandbox(source_repo):
    """Copy the target repo into an isolated temp folder and return its
    path. The ORIGINAL repo is never written to."""
    sandbox_dir = tempfile.mkdtemp(prefix="coding_agent_sandbox_")
    shutil.copytree(source_repo, sandbox_dir, dirs_exist_ok=True)
    return sandbox_dir


def apply_edit(sandbox_dir, file_relative_path, target_name, fixed_code):
    """
    Replace ONE top-level function/class inside a file with new code.

    We reuse the same `ast` trick from step 1 to find exactly which
    lines belong to `target_name` (e.g. "login" or "UserManager"), so
    we only touch those lines -- not the whole file. This is closer
    to how a real diff/patch works than overwriting everything.

    Returns (success: bool, message: str).
    """
    file_path = os.path.join(sandbox_dir, file_relative_path)
    if not os.path.exists(file_path):
        return False, f"File not found in sandbox: {file_relative_path}"

    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, f"Could not parse {file_relative_path}: {e}"

    target_node = None
    for node in ast.iter_child_nodes(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == target_name
        ):
            target_node = node
            break

    if target_node is None:
        return False, (
            f"Could not find a top-level function/class named '{target_name}' "
            f"in {file_relative_path}. Check the exact name and try again."
        )

    source_lines = source.splitlines(keepends=True)
    start = target_node.lineno - 1          # ast lines are 1-indexed
    end = target_node.end_lineno            # end_lineno is inclusive

    new_code = fixed_code if fixed_code.endswith("\n") else fixed_code + "\n"
    new_lines = source_lines[:start] + [new_code] + source_lines[end:]

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    return True, "Edit applied in sandbox."


def run_tests(sandbox_dir, timeout_seconds=30):
    """
    Run pytest inside the sandbox and report pass/fail + output.

    A timeout is included on purpose: a bad edit could introduce an
    infinite loop in the code under test, and we don't want that to
    hang our whole agent run.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", sandbox_dir, "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        passed = result.returncode == 0
        output = (result.stdout + result.stderr).strip()
        return passed, output
    except subprocess.TimeoutExpired:
        return False, f"Tests did not finish within {timeout_seconds}s (possible infinite loop in the edit)."


def cleanup_sandbox(sandbox_dir):
    shutil.rmtree(sandbox_dir, ignore_errors=True)
