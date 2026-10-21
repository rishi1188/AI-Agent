"""
Bug Fix Studio -- a web UI for the agent
-----------------------------------------
Pick a buggy repo (a benchmark task, or paste your own code + tests),
describe the bug, and watch the agent work round by round. A fix is
only shown once it has passed the tests in the sandbox, as a diff you
can download as a .patch.

Run locally:  streamlit run streamlit_app.py
"""

import json
import os
import shutil
import tempfile
import time

import streamlit as st

import agent
import run_eval

st.set_page_config(page_title="Bug Fix Studio", page_icon="🛠️", layout="wide")

DEMO_TASK = {
    "name": "demo_login",
    "repo": "dummy_repo",
    "task": (
        "The login method in app.py has a bug: it crashes instead of returning "
        "False when the username doesn't exist. There's a test suite at "
        "test_app.py -- find the bug, propose a fix, and make sure all tests pass."
    ),
}
SAMPLES = {t["name"]: t for t in [DEMO_TASK, *run_eval.TASKS]}

CUSTOM_CODE = '''def average(numbers):
    """Return the mean of a list of numbers."""
    return sum(numbers) / len(numbers) - 1
'''
CUSTOM_TESTS = '''import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from app import average


def test_average():
    assert average([2, 4, 6]) == 4


def test_single_value():
    assert average([5]) == 5
'''

TOOL_ICONS = {"search_codebase": "🔎", "read_file": "📄", "propose_edit": "🧪"}


def server_setting(name, default=""):
    """Read a value from Streamlit secrets or the environment. These stay
    on the server: never pass them into a widget, or they're sent to
    every visitor's browser."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # noqa: BLE001 -- no secrets file is fine
        pass
    return os.environ.get(name, default)


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def prepare_workdir(source):
    """Copy the code into a throwaway folder, so a run never edits the
    benchmark files in this repo."""
    workdir = tempfile.mkdtemp(prefix="bugfix_studio_")
    if source["kind"] == "sample":
        shutil.copytree(source["repo"], workdir, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
    else:
        with open(os.path.join(workdir, "app.py"), "w", encoding="utf-8") as f:
            f.write(source["code"])
        with open(os.path.join(workdir, "test_app.py"), "w", encoding="utf-8") as f:
            f.write(source["tests"])
    return workdir


def render_event(event, box):
    """Write one agent step into the live timeline."""
    kind = event["kind"]
    if kind == "round":
        box.markdown(f"**Round {event['round']}**")
    elif kind == "thought":
        box.markdown(f"💭 {event['text']}")
    elif kind == "action":
        icon = TOOL_ICONS.get(event["tool"], "🔧")
        args = event["args"]
        if event["tool"] == "propose_edit":
            label = f"propose_edit → `{args.get('target_name', '?')}` in `{args.get('file', '?')}`"
        else:
            label = f"{event['tool']}(" + ", ".join(f"{k}={v!r}" for k, v in args.items()) + ")"
        box.markdown(f"{icon} {label}")
        if event["tool"] == "propose_edit" and args.get("fixed_code"):
            with box.expander("Proposed code"):
                st.code(args["fixed_code"], language="python")
    elif kind == "observation":
        with box.expander(f"Result of {event['tool']}"):
            st.code(event["text"][:4000])
    elif kind == "refused":
        box.warning(f"Refused: {event['reason']}")
    elif kind == "verify_failed":
        box.error("Tests failed in the sandbox. The failure goes back to the agent.")
        with box.expander("Test output"):
            st.code(event["test_output"][-4000:])
    elif kind == "fixed":
        box.success(f"Tests passed. Fix applied to {event['file']}.")
    elif kind == "gave_up":
        box.warning(event["reason"])


def run(source, task_text, api_key, model, max_rounds):
    workdir = prepare_workdir(source)
    events = []
    start = time.time()
    error = None
    with st.status("Agent is working...", expanded=True) as status:
        def on_event(event):
            events.append(event)
            render_event(event, st)
            if event["kind"] == "round":
                status.update(label=f"Agent is working... round {event['round']} of {max_rounds}")

        try:
            agent.run_agent(task_text, repo_path=workdir, on_event=on_event,
                            api_key=api_key, model=model, max_rounds=max_rounds)
        except Exception as e:  # noqa: BLE001 -- show any API/network failure in the UI
            error = f"{type(e).__name__}: {e}"
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        fixed = next((e for e in events if e["kind"] == "fixed"), None)
        if fixed:
            status.update(label=f"Fixed in {fixed['rounds']} round(s)", state="complete", expanded=False)
        else:
            status.update(label="No verified fix", state="error", expanded=True)

    st.session_state["last_run"] = {
        "events": events,
        "error": error,
        "seconds": round(time.time() - start, 1),
        "name": source.get("name", "custom"),
    }


def show_result(last_run):
    events = last_run["events"]
    fixed = next((e for e in events if e["kind"] == "fixed"), None)
    attempts = sum(1 for e in events if e["kind"] == "action" and e["tool"] == "propose_edit")
    rounds = max((e["round"] for e in events if e["kind"] == "round"), default=0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Result", "Fixed" if fixed else "Not fixed")
    c2.metric("Rounds / fix attempts", f"{rounds} / {attempts}")
    c3.metric("Time", f"{last_run['seconds']} s")

    if last_run["error"]:
        st.error(f"The run stopped with an error: {last_run['error']}")
    if not fixed:
        return

    st.subheader("Verified fix")
    if fixed["explanation"]:
        st.markdown(f"**Why this works:** {fixed['explanation']}")
    st.code(fixed["diff"], language="diff")
    st.download_button("Download .patch", fixed["diff"], file_name=f"{last_run['name']}.patch",
                       mime="text/x-diff")
    with st.expander("Test results"):
        st.code(fixed["test_output"][-6000:])


def fix_tab(api_key, model, max_rounds):
    left, right = st.columns([2, 3], gap="large")

    with left:
        st.subheader("1. Choose the code")
        mode = st.radio("Code source", ["Sample bug", "Paste your own"], horizontal=True,
                        label_visibility="collapsed")
        if mode == "Sample bug":
            name = st.selectbox("Sample", list(SAMPLES), format_func=lambda n: n.replace("_", " "))
            sample = SAMPLES[name]
            source = {"kind": "sample", "repo": sample["repo"], "name": name}
            default_task = sample["task"]
            with st.expander("View the buggy code"):
                st.code(read(os.path.join(sample["repo"], "app.py")), language="python")
            with st.expander("View the tests"):
                st.code(read(os.path.join(sample["repo"], "test_app.py")), language="python")
        else:
            code = st.text_area("app.py", CUSTOM_CODE, height=180)
            tests = st.text_area("test_app.py (pytest)", CUSTOM_TESTS, height=220)
            source = {"kind": "custom", "code": code, "tests": tests, "name": "custom"}
            default_task = ("average() in app.py returns the wrong value. Tests are in "
                            "test_app.py -- find the bug, fix it, and make sure all tests pass.")

        st.subheader("2. Describe the bug")
        task_text = st.text_area("Bug description", default_task, height=120,
                                 key=f"task_{source['name']}", label_visibility="collapsed")
        go = st.button("Run agent", type="primary", width="stretch",
                       disabled=not api_key)
        if not api_key:
            st.info("Add a Gemini API key in the sidebar to run the agent.")

    with right:
        st.subheader("3. Watch it work")
        if go:
            run(source, task_text, api_key, model, max_rounds)
        last_run = st.session_state.get("last_run")
        if last_run:
            if not go:
                with st.expander("Steps from the last run"):
                    for event in last_run["events"]:
                        render_event(event, st)
            show_result(last_run)
        elif not go:
            st.caption("Each round streams in here: what the agent searches, which files it "
                       "reads, and every fix it tries. A fix is shown only after it passes "
                       "the tests in an isolated sandbox.")


def benchmark_tab():
    path = "eval_results.json"
    if not os.path.exists(path):
        st.info("No benchmark results yet. Run `python run_eval.py` to create `eval_results.json`, "
                "then refresh this page.")
        st.dataframe(
            [{"Task": t["name"], "Bug pattern": t["repo"].split("_", 2)[-1].replace("_", " ")}
             for t in run_eval.TASKS],
            hide_index=True, width="stretch",
        )
        return

    data = json.loads(read(path))
    tasks = data["tasks"]
    solved = [t for t in tasks if t["solved"]]
    c1, c2, c3 = st.columns(3)
    c1.metric("Resolve rate", f"{data['solved']}/{data['total']}",
              f"{100 * data['solved'] / max(data['total'], 1):.0f}%", delta_color="off")
    c2.metric("Avg fix attempts (solved)",
              f"{sum(t['attempts'] for t in solved) / len(solved):.1f}" if solved else "–")
    c3.metric("Avg time per task", f"{sum(t['seconds'] for t in tasks) / max(len(tasks), 1):.0f} s")
    st.caption(f"Model: {data['model']} · run at {data['run_at']}")
    st.dataframe(
        [{"Task": t["name"], "Solved": "✅" if t["solved"] else "❌", "Fix attempts": t["attempts"],
          "Seconds": t["seconds"], "Error": t["crashed"] or ""} for t in tasks],
        hide_index=True, width="stretch",
    )


def about_tab():
    st.markdown(
        "The agent fixes Python bugs and proves each fix with the project's own tests "
        "before it touches a file."
    )
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR; node [shape=box, style=rounded, fontname="Helvetica"];
          task [label="Bug description"];
          loop [label="Agent loop\\nGemini + tools", shape=diamond];
          index [label="Vector index\\nAST chunks in Chroma"];
          repo [label="Target repo"];
          sandbox [label="Sandbox copy\\napply edit + pytest"];
          done [label="Verified fix"];
          task -> loop;
          loop -> index [label="search_codebase"];
          loop -> repo [label="read_file"];
          loop -> sandbox [label="propose_edit"];
          sandbox -> loop [label="tests fail"];
          sandbox -> done [label="tests pass"];
        }
        """
    )
    st.markdown(
        "- **Verify before apply:** edits run in a sandbox copy first, and only a green test run is kept.\n"
        "- **Self-correction:** a failing test run goes back to the model as feedback.\n"
        "- **AST-aware:** code is searched and replaced per function or class.\n"
        "- **Guard rails:** tools are scoped to the repo, and test files can't be edited.\n\n"
        "Source code: [github.com/rishi1188/AI-Agent](https://github.com/rishi1188/AI-Agent)"
    )


with st.sidebar:
    st.title("🛠️ Bug Fix Studio")
    st.caption("An AI agent that fixes Python bugs and proves each fix with tests.")
    server_key = server_setting("GEMINI_API_KEY")
    # The field always starts empty: a visitor's own key wins, otherwise
    # the server's key is used without ever being shown in the page.
    own_key = st.text_input("Gemini API key", type="password",
                            placeholder="Using the app's key" if server_key else "Paste your key",
                            help="Free key: https://aistudio.google.com/app/apikey. "
                                 "It is only used for this session.")
    api_key = own_key or server_key
    model = st.text_input("Model", value=server_setting("GEMINI_MODEL", agent.MODEL),
                          help="Try gemini-flash-latest if the default is overloaded.")
    max_rounds = st.slider("Max rounds", 3, 15, agent.MAX_ITERATIONS)

st.title("Bug Fix Studio")
tab_fix, tab_bench, tab_about = st.tabs(["Fix a bug", "Benchmark", "How it works"])
with tab_fix:
    fix_tab(api_key, model, max_rounds)
with tab_bench:
    benchmark_tab()
with tab_about:
    about_tab()
