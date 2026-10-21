"""
STEP 1: Codebase Indexer
-------------------------
What this file does, in plain words:

  1. Walk through a folder of Python code and find every .py file.
  2. Open each file and pull out every function and class as its own
     small "chunk" (instead of treating the whole file as one blob).
  3. Turn each chunk into an embedding -- a list of numbers that
     represents what the code MEANS, not just what it says.
  4. Store those chunks + embeddings in Chroma, a local vector
     database, so we can search them later by meaning.

Think of step 3 & 4 like this: imagine turning every function into a
dot on a giant map, where similar-meaning functions land near each
other. Later, when the agent asks "where's the login code?", we just
find the dots closest to that question -- even if the code doesn't
contain the word "login" anywhere.
"""

import ast
import os

import chromadb

# Folders we never want to look inside -- virtual envs, git internals,
# caches, etc. Indexing these would just add noise.
IGNORED_DIRS = {".git", "venv", ".venv", "__pycache__", "node_modules", ".mypy_cache"}


def find_python_files(repo_path):
    """Return the full path of every .py file inside repo_path."""
    py_files = []
    for root, dirs, files in os.walk(repo_path):
        # Modifying dirs[:] in place tells os.walk to skip these folders
        # entirely, instead of just filtering them out after the fact.
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        for filename in files:
            if filename.endswith(".py"):
                py_files.append(os.path.join(root, filename))
    return py_files


def chunk_file(file_path):
    """
    Parse ONE Python file and pull out each top-level function and
    class as its own chunk of source code.

    We use Python's built-in `ast` (Abstract Syntax Tree) module here.
    Instead of treating the file as plain text, `ast` understands
    Python's actual structure -- it knows exactly where a function
    starts and ends, even across multiple lines. That's what lets us
    cut the file into clean, meaningful pieces instead of arbitrary
    chunks of text.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        # Some files won't parse (old Python 2 code, weird encodings).
        # We just skip those rather than crashing the whole indexing run.
        return []

    source_lines = source.splitlines()
    chunks = []

    # NOTE: we use ast.iter_child_nodes() here, NOT ast.walk().
    # walk() would recurse into every nested level and return a
    # class's methods as separate chunks IN ADDITION TO the class
    # itself -- duplicate, overlapping data in the vector store.
    # iter_child_nodes() only looks at the top level of the file, so
    # a class's methods stay bundled inside that one class chunk.
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_line = node.lineno - 1  # ast line numbers start at 1
            end_line = getattr(node, "end_lineno", start_line + 1)
            code_snippet = "\n".join(source_lines[start_line:end_line])

            chunks.append(
                {
                    "id": f"{file_path}::{node.name}::{start_line}",
                    "file": file_path,
                    "name": node.name,
                    "type": type(node).__name__,  # "FunctionDef" or "ClassDef"
                    "code": code_snippet,
                }
            )
    return chunks


def build_index(repo_path, db_path="./chroma_db"):
    """The full pipeline: scan -> chunk -> embed -> store."""
    print(f"Scanning {repo_path} for Python files...")
    files = find_python_files(repo_path)
    print(f"Found {len(files)} Python file(s).")

    all_chunks = []
    for file_path in files:
        all_chunks.extend(chunk_file(file_path))
    print(f"Extracted {len(all_chunks)} function/class chunk(s).")

    if not all_chunks:
        print("Nothing to index -- check the repo path.")
        return None

    # We deliberately do NOT install sentence-transformers/torch here.
    # Chroma ships its own lightweight built-in embedding function
    # (a small ONNX model) -- if you just don't pass `embeddings=`
    # yourself, Chroma embeds `documents` for you automatically the
    # first time you call .add(). Same idea (meaning -> numbers),
    # much lighter to install.
    print("Connecting to local Chroma database (this downloads a small")
    print("embedding model the very first time you run it)...")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection("codebase")

    print("Embedding and storing chunks...")
    texts = [c["code"] for c in all_chunks]

    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=texts,
        metadatas=[
            {"file": c["file"], "name": c["name"], "type": c["type"]} for c in all_chunks
        ],
    )

    print(f"Done. Indexed {len(all_chunks)} chunks into '{db_path}'.")
    return collection


if __name__ == "__main__":
    import sys

    target_repo = sys.argv[1] if len(sys.argv) > 1 else "."
    build_index(target_repo)
