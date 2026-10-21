"""
Tests for the agent loop, WITHOUT calling the real Gemini API.

Two kinds of test in here:
  1. Loop-wiring tests -- mock verify_edit() entirely, so we're only
     checking the LOOP's decisions (stop on success? retry on failure?
     feed the failure back correctly?) in isolation.
  2. A real integration test -- lets verify_edit() actually run against
     real temp files (a real sandbox, a real pytest run), so we know
     the whole pipeline genuinely works together, not just in theory.

Run with:  python -m pytest test_agent.py -v
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")

sys.path.insert(0, os.path.dirname(__file__))
import agent  # noqa: E402
import sandbox  # noqa: E402


def make_text_part(text):
    return SimpleNamespace(text=text, function_call=None)


def make_function_call_part(name, args, call_id="call_1"):
    fc = SimpleNamespace(name=name, args=args, id=call_id)
    return SimpleNamespace(text=None, function_call=fc)


def make_response(parts):
    content = SimpleNamespace(parts=parts, role="model")
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)])


# ---------------------------------------------------------------------
# 1. Loop-wiring tests (verify_edit mocked)
# ---------------------------------------------------------------------

def test_stops_and_copies_fix_back_when_tests_pass(tmp_path):
    """When verify_edit reports success, the loop should stop AND copy
    the fixed file from the sandbox back onto the real repo."""
    real_repo = tmp_path / "repo"
    real_repo.mkdir()
    (real_repo / "app.py").write_text("original buggy content")

    fake_sandbox = tmp_path / "fake_sandbox"
    fake_sandbox.mkdir()
    (fake_sandbox / "app.py").write_text("fixed content")

    round_1 = make_response(
        [make_function_call_part(
            "propose_edit",
            {"file": "app.py", "target_name": "login", "explanation": "fixed it", "fixed_code": "..."},
            call_id="c1",
        )]
    )

    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [round_1]

    with patch("agent.genai.Client", return_value=fake_client), \
         patch("agent.verify_edit", return_value=(True, "4 passed", str(fake_sandbox))):
        result = agent.run_agent("fix it", repo_path=str(real_repo))

    assert result is round_1
    assert fake_client.models.generate_content.call_count == 1
    assert (real_repo / "app.py").read_text() == "fixed content"


def test_retries_with_failure_feedback_then_succeeds(tmp_path):
    """
    Round 1: proposed fix FAILS verification.
    Round 2: proposed fix PASSES.
    The loop must retry (not give up after round 1) AND the failure
    message from round 1 must actually reach Gemini in round 2's request.
    """
    real_repo = tmp_path / "repo"
    real_repo.mkdir()
    (real_repo / "app.py").write_text("original")

    fake_sandbox = tmp_path / "fake_sandbox"
    fake_sandbox.mkdir()
    (fake_sandbox / "app.py").write_text("correct fix")

    round_1 = make_response(
        [make_function_call_part(
            "propose_edit",
            {"file": "app.py", "target_name": "login", "explanation": "attempt 1", "fixed_code": "wrong"},
            call_id="c1",
        )]
    )
    round_2 = make_response(
        [make_function_call_part(
            "propose_edit",
            {"file": "app.py", "target_name": "login", "explanation": "attempt 2", "fixed_code": "right"},
            call_id="c2",
        )]
    )

    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [round_1, round_2]

    verify_results = [
        (False, "1 failed, 3 passed -- AttributeError: 'NoneType' object has no attribute 'x'", None),
        (True, "4 passed", str(fake_sandbox)),
    ]

    with patch("agent.genai.Client", return_value=fake_client), \
         patch("agent.verify_edit", side_effect=verify_results):
        result = agent.run_agent("fix it", repo_path=str(real_repo))

    assert fake_client.models.generate_content.call_count == 2
    assert result is round_2
    assert (real_repo / "app.py").read_text() == "correct fix"

    # The failure text from round 1 must show up in round 2's request.
    second_call_kwargs = fake_client.models.generate_content.call_args_list[1].kwargs
    sent_contents = second_call_kwargs["contents"]
    fed_back_results = [
        part.function_response.response["result"]
        for turn in sent_contents
        for part in getattr(turn, "parts", [])
        if getattr(part, "function_response", None) is not None
    ]
    assert any("AttributeError" in text for text in fed_back_results)


def test_respects_iteration_cap_if_never_calling_propose_edit():
    infinite = make_response(
        [make_function_call_part("read_file", {"path": "nonexistent.py"}, call_id="loop")]
    )
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = infinite

    with patch("agent.genai.Client", return_value=fake_client), \
         patch.dict(agent.TOOL_FUNCTIONS, {"read_file": lambda a: "nothing here"}):
        result = agent.run_agent("this will never finish")

    assert fake_client.models.generate_content.call_count == agent.MAX_ITERATIONS
    assert result is None


# ---------------------------------------------------------------------
# 2. Real integration test -- verify_edit runs for real, nothing mocked
#    except the Gemini API call itself.
# ---------------------------------------------------------------------

def test_full_verify_loop_against_a_real_tiny_repo(tmp_path):
    """
    Builds an actual tiny buggy repo + real pytest test, then runs the
    REAL sandbox + apply_edit + run_tests pipeline (no mocking of
    verify_edit this time) through two propose_edit attempts: a wrong
    one, then a correct one. This is the closest thing to a true
    end-to-end check we can do without spending real API calls.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text(
        "def add(a, b):\n"
        "    return a - b  # BUG: should be +\n"
    )
    (repo / "test_app.py").write_text(
        "import sys, os\n"
        "sys.path.insert(0, os.path.dirname(__file__))\n"
        "from app import add\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n"
    )

    wrong_fix = "def add(a, b):\n    return a - b\n"          # still broken
    correct_fix = "def add(a, b):\n    return a + b\n"        # actually fixed

    round_1 = make_response(
        [make_function_call_part(
            "propose_edit",
            {"file": "app.py", "target_name": "add", "explanation": "attempt 1", "fixed_code": wrong_fix},
            call_id="c1",
        )]
    )
    round_2 = make_response(
        [make_function_call_part(
            "propose_edit",
            {"file": "app.py", "target_name": "add", "explanation": "attempt 2", "fixed_code": correct_fix},
            call_id="c2",
        )]
    )

    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [round_1, round_2]

    with patch("agent.genai.Client", return_value=fake_client):
        result = agent.run_agent("fix add()", repo_path=str(repo))

    assert result is round_2
    assert fake_client.models.generate_content.call_count == 2
    # The REAL file on "disk" should now contain the correct fix.
    assert "a + b" in (repo / "app.py").read_text()


# ---------------------------------------------------------------------
# 3. Retry logic tests (carried over from step 2 -- call_with_retry is
#    unaffected by the verify-loop changes, still needs covering)
# ---------------------------------------------------------------------

def test_retries_on_503_then_succeeds():
    from google.genai import errors

    success_response = make_response([make_text_part("All good.")])
    server_error = errors.ServerError(503, {"error": {"message": "high demand"}}, None)

    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [server_error, server_error, success_response]

    with patch("agent.time.sleep"):
        result = agent.call_with_retry(fake_client, model="x", contents=[], config=None)

    assert result is success_response
    assert fake_client.models.generate_content.call_count == 3


def test_gives_up_after_max_retries():
    from google.genai import errors

    server_error = errors.ServerError(503, {"error": {"message": "still down"}}, None)

    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = server_error

    with patch("agent.time.sleep"), pytest.raises(errors.ServerError):
        agent.call_with_retry(fake_client, model="x", contents=[], config=None)

    assert fake_client.models.generate_content.call_count == agent.MAX_RETRIES


def test_does_not_retry_non_rate_limit_client_errors():
    from google.genai import errors

    bad_request = errors.ClientError(400, {"error": {"message": "bad request"}}, None)

    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = bad_request

    with patch("agent.time.sleep"), pytest.raises(errors.ClientError):
        agent.call_with_retry(fake_client, model="x", contents=[], config=None)

    assert fake_client.models.generate_content.call_count == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))