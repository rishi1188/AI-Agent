"""
Sanity check for Step 1: search the index by MEANING, not exact text.

Run this AFTER indexer.py has built the database, to prove retrieval
actually works. Try a query that doesn't share any words with the
code itself -- e.g. "code that checks a password" should still find
the `login` method, because embeddings compare meaning, not spelling.

Usage:
    python indexer.py dummy_repo          # build the index first
    python query_test.py "checks a password"
"""

import chromadb


def search_codebase(query, db_path="./chroma_db", top_k=3):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection("codebase")

    results = collection.query(query_texts=[query], n_results=top_k)

    if not results["ids"][0]:
        print("No results -- did you run indexer.py first?")
        return

    print(f'Top {len(results["ids"][0])} matches for: "{query}"\n')
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]  # lower = more similar
        print(f"--- Match {i + 1}: {meta['name']} ({meta['type']}) in {meta['file']} ---")
        print(f"(similarity distance: {distance:.3f})")
        print(results["documents"][0][i][:200])
        print()


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "function that handles user login"
    search_codebase(query)
