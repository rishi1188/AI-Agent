"""
STEP 2: The Agent Loop (Thought -> Action -> Observation) -- Gemini version
-----------------------------------------------------------------------------
Same idea as before, now built on Google's Gemini API instead of Anthropic's.
The LOOP is what changes between providers -- your tools (search_codebase,
read_file) stay exactly the same.

How tool calling works here, in plain words:
  1. We send Gemini the task, plus a MENU of tools it's allowed to use.
  2. Gemini doesn't run anything itself. It replies with a `function_call`
     part: "I want to call search_codebase with query='login bug'."
  3. OUR code is the one that actually runs the tool -- Gemini never
     touches your filesystem directly.
  4. We send the tool's result back as a `function_response`, matched to
     the original call by its `id` (Gemini 3 requires this id to be
     echoed back correctly, or it can't tell which result answers which
     call -- important when it calls more than one tool at once).
  5. Gemini reads that and decides what to do next.
This repeats until Gemini calls propose_edit, or we hit MAX_ITERATIONS.
"""

import os
import shutil
import time

import chromadb
from google import genai
from google.genai import types
from google.genai import errors

import sandbox

MAX_ITERATIONS = 10  # bumped from 8 -- a full run now needs room for
                      # investigation AND one or more verify attempts

# Retry settings for TRANSIENT failures -- a 503 ("server's overloaded
# right now") or a 429 ("you're calling too fast") aren't bugs in your
# code. They're normal, expected noise when talking to any third-party
# API, and preview-tier models like gemini-3-flash-preview hit 503s
# more often than stable ones because they get less server capacity.
# A production-grade agent has to expect this and recover on its own
# instead of crashing on the first hiccup.
MAX_RETRIES = 4
INITIAL_BACKOFF_SECONDS = 2

# gemini-3-flash-preview is Google's newest model, but as a preview
# model it gets less server capacity -- which is exactly the 503s
# you've been hitting. Switching to gemini-2.5-flash (fully stable,
# generally available) trades a bit of raw capability for much more
# consistent availability while you're actively developing. Just
# remember this generation is scheduled to retire mid-October 2026,
# so revisit this choice before then.
MODEL = "gemini-3.8-flash"


# ---------- Tool implementations (identical to the Anthropic version) ----

def search_codebase(query, db_path="./chroma_db", top_k=3):
    """Find code chunks related to `query` by MEANING (same as step 1)."""
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection("codebase")
    results = collection.query(query_texts=[query], n_results=top_k)

    if not results["ids"][0]:
        return "No matches found."

    pieces = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        code = results["documents"][0][i]
        pieces.append(f"File: {meta['file']}\n{meta['type']} {meta['name']}:\n{code}")
    return "\n\n---\n\n".join(pieces)


def read_file(path):
    """Read a file's full contents WITH line numbers."""
    if not os.path.exists(path):
        return f"Error: file not found: {path}"
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    return "".join(f"{i + 1}: {line}" for i, line in enumerate(lines))


# NOTE: Gemini hands back `args` as a plain dict already (unlike some
# APIs that give you a JSON *string* you'd need to parse yourself).
TOOL_FUNCTIONS = {
    "search_codebase": lambda args: search_codebase(args["query"]),
    "read_file": lambda args: read_file(args["path"]),
}


# ---------- Tool menu Gemini sees ----------
# Gemini wants plain dict schemas wrapped in a types.Tool, rather than
# Anthropic's list-of-dicts-with-input_schema shape.

FUNCTION_DECLARATIONS = [
    {
        "name": "search_codebase",
        "description": (
            "Search the indexed codebase for code related to a natural-"
            "language description. Use this FIRST to find where relevant "
            "code lives before reading anything."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What you're looking for, in plain English."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read the FULL contents of one specific file, with line "
            "numbers, so you can see exact context before proposing an edit."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file, e.g. dummy_repo/app.py"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "propose_edit",
        "description": (
            "Propose a fix and have it AUTOMATICALLY applied and tested. "
            "You'll be told whether the tests passed -- if they didn't, "
            "you'll see the failure and can try again with a corrected fix. "
            "IMPORTANT: target_name must be a TOP-LEVEL function or class "
            "name in the file. If the bug is inside a METHOD, set "
            "target_name to the name of its ENCLOSING CLASS and give the "
            "full corrected class (all its methods) as fixed_code."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "File to change, e.g. dummy_repo/app.py"},
                "target_name": {
                    "type": "string",
                    "description": "Name of the top-level function or class being replaced.",
                },
                "explanation": {"type": "string", "description": "Why this fixes the bug."},
                "fixed_code": {
                    "type": "string",
                    "description": "The full replacement source, starting from 'def' or 'class'.",
                },
            },
            "required": ["file", "target_name", "explanation", "fixed_code"],
        },
    },
]

TOOLS = types.Tool(function_declarations=FUNCTION_DECLARATIONS)


def call_with_retry(client, **kwargs):
    """
    Wraps the API call with automatic retries + exponential backoff for
    transient failures. "Exponential backoff" just means: wait a bit,
    and if it fails again, wait TWICE as long next time (2s, 4s, 8s...)
    instead of hammering an already-overloaded server at a fixed rate.
    """
    backoff = INITIAL_BACKOFF_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.models.generate_content(**kwargs)
        except errors.ServerError as e:
            # 5xx -- the server is having a bad day, not us. Always worth retrying.
            if attempt == MAX_RETRIES:
                raise
            print(f"[Retry] Server error ({e.code}): {e.message}. "
                  f"Retrying in {backoff}s (attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(backoff)
            backoff *= 2
        except errors.ClientError as e:
            # Only 429 (rate limited) is worth retrying. A 400 (bad
            # request) or 403 (bad API key) will fail the exact same
            # way every time -- retrying those just wastes time, so we
            # re-raise immediately instead.
            if e.code == 429 and attempt < MAX_RETRIES:
                print(f"[Retry] Rate limited (429). "
                      f"Retrying in {backoff}s (attempt {attempt}/{MAX_RETRIES})...")
                time.sleep(backoff)
                backoff *= 2
            else:
                raise


def verify_edit(repo_path, file_relative_path, target_name, fixed_code):
    """
    Apply the agent's proposed edit inside a FRESH sandbox copy of the
    repo, then run the test suite there. Returns (passed, message,
    sandbox_dir). The sandbox is the caller's responsibility to clean
    up (or keep, if it passed and we want to copy the fix back).
    """
    sb = sandbox.create_sandbox(repo_path)

    applied, apply_msg = sandbox.apply_edit(sb, file_relative_path, target_name, fixed_code)
    if not applied:
        sandbox.cleanup_sandbox(sb)
        return False, f"Could not apply the edit: {apply_msg}", None

    passed, test_output = sandbox.run_tests(sb)
    if passed:
        return True, test_output, sb

    sandbox.cleanup_sandbox(sb)
    return False, test_output, None


def run_agent(task, repo_path="dummy_repo", system_prompt=None):
    """The core Thought -> Action -> Observation -> Verify loop."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing GEMINI_API_KEY environment variable.\n"
            "Get a free key from https://aistudio.google.com/app/apikey and set it, e.g. (Windows):\n"
            '  setx GEMINI_API_KEY "your-key-here"\n'
            "Then open a NEW terminal window and try again."
        )
    client = genai.Client(api_key=api_key)

    system_prompt = system_prompt or (
        "You are a careful coding assistant. You have tools to search and "
        "read a codebase. ALWAYS investigate with search_codebase and "
        "read_file before proposing a fix -- never guess at code you "
        "haven't read. When you call propose_edit, your fix is AUTOMATICALLY "
        "tested. If the tests fail, you'll see exactly why and should call "
        "propose_edit again with a corrected fix -- don't repeat the same "
        "mistake twice."
    )
    config = types.GenerateContentConfig(tools=[TOOLS], system_instruction=system_prompt)

    contents = [types.Content(role="user", parts=[types.Part(text=task)])]

    for round_num in range(1, MAX_ITERATIONS + 1):
        print(f"\n=== Round {round_num} ===")
        response = call_with_retry(client, model=MODEL, contents=contents, config=config)

        candidate_content = response.candidates[0].content
        contents.append(candidate_content)  # Gemini's "model" turn goes back into history

        function_calls = [p.function_call for p in candidate_content.parts if p.function_call]
        for p in candidate_content.parts:
            if p.text and p.text.strip():
                print(f"[Thought] {p.text.strip()}")

        if not function_calls:
            # Plain text answer, no tool call -- Gemini is done (or stuck).
            return response

        response_parts = []
        for call in function_calls:
            print(f"[Action] {call.name}({call.args})")

            if call.name == "propose_edit":
                args = call.args
                passed, message, sb = verify_edit(
                    repo_path, args["file"], args["target_name"], args["fixed_code"]
                )

                if passed:
                    print("\n[Verify] Tests PASSED.")
                    # Copy the working fix from the sandbox back onto the
                    # real repo -- this is the only point where the
                    # agent's edit touches your actual files.
                    real_file = os.path.join(repo_path, args["file"])
                    sandboxed_file = os.path.join(sb, args["file"])
                    shutil.copyfile(sandboxed_file, real_file)
                    sandbox.cleanup_sandbox(sb)
                    print(f"[Done] Fix applied to {real_file}")
                    print(f"Explanation: {args['explanation']}")
                    return response  # success -- stop the loop

                print(f"\n[Verify] Tests FAILED. Feeding failure back to the agent.")
                print(message[:400])
                # This is the key reliability moment: instead of crashing
                # or giving up, the failure becomes an OBSERVATION the
                # agent gets to read and react to, just like any other
                # tool result.
                response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=call.name,
                            response={"result": f"Tests FAILED after applying this edit:\n{message}"},
                            id=call.id,
                        )
                    )
                )
                continue

            tool_fn = TOOL_FUNCTIONS.get(call.name)
            result_text = tool_fn(call.args) if tool_fn else f"Unknown tool: {call.name}"
            print(f"[Observation] {result_text[:200]}...")

            # Built directly (not via Part.from_function_response) so we
            # can set `id` explicitly -- required for Gemini 3 to match
            # this result back to the right call, especially when
            # multiple tools are called in the same round.
            response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=call.name, response={"result": result_text}, id=call.id
                    )
                )
            )

        contents.append(types.Content(role="user", parts=response_parts))

    print(f"\nHit the {MAX_ITERATIONS}-round cap without a passing fix.")
    return None


if __name__ == "__main__":
    import sys

    default_task = (
        "The login method in dummy_repo/app.py has a bug: it crashes "
        "instead of returning False when the username doesn't exist. "
        "There's a test suite at dummy_repo/test_app.py -- find the bug, "
        "propose a fix, and make sure all tests pass."
    )
    task = " ".join(sys.argv[1:]) or default_task
    try:
        run_agent(task)
    except errors.ServerError as e:
        print(f"\nGemini is still unavailable after {MAX_RETRIES} retries ({e.code}: {e.message}).")
        print("This is Google's server, not your code -- wait a few minutes and try again.")