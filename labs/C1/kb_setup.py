"""
ISDO - Knowledge Base indexer
-----------------------------
1. Reads every .md file in data/kb/
2. Splits each article at level-2 (##) headings into chunks
3. Stores all chunks in the ChromaDB collection 'isdo_kb'
4. Runs 4 sample queries and prints the best matching article + confidence

Requires only:  pip install chromadb
Run from anywhere:  python build_kb_index.py
"""

import re
from pathlib import Path

import chromadb

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
def find_project_root(start: Path) -> Path:
    """Walk up from the script's folder until a folder containing data/kb is found."""
    for folder in [start, *start.parents]:
        if (folder / "data" / "kb").is_dir():
            return folder
    raise FileNotFoundError(
        f"Could not find a 'data/kb' folder in {start} or any parent folder"
    )


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
KB_DIR = PROJECT_ROOT / "data" / "kb"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "isdo_kb"

SAMPLE_QUERIES = [
    "VPN not connecting after I changed my password",
    "My account is locked and I can't log in",
    "SAP shows DBCON_FAIL error for the whole finance team",
    "Outlook on my phone stopped syncing emails",
]


# ----------------------------------------------------------------------------
# 1 + 2. Read markdown and split at ## headings
# ----------------------------------------------------------------------------
def split_at_h2(text: str):
    """Return (title, [(section_heading, section_body), ...]).

    Content before the first '## ' (title + metadata) becomes an 'Overview'
    chunk. '###' sub-headings stay inside their parent '##' section.
    """
    title_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Untitled"

    parts = re.split(r"^##\s+", text, flags=re.MULTILINE)
    sections = []

    preamble = parts[0].strip()
    if preamble:
        sections.append(("Overview", preamble))

    for part in parts[1:]:
        heading, _, body = part.partition("\n")
        body = body.strip()
        if body:
            sections.append((heading.strip(), body))

    return title, sections


def load_chunks(kb_dir: Path):
    ids, documents, metadatas = [], [], []

    md_files = sorted(kb_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {kb_dir}")

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        title, sections = split_at_h2(text)

        for i, (heading, body) in enumerate(sections):
            # Prefix title + heading so each chunk carries its own context
            chunk_text = f"{title}\n## {heading}\n{body}"
            ids.append(f"{md_file.stem}::{i:02d}")
            documents.append(chunk_text)
            metadatas.append(
                {
                    "article": md_file.name,
                    "title": title,
                    "section": heading,
                    "chunk_index": i,
                }
            )

        print(f"  {md_file.name:<28} -> {len(sections)} chunks")

    return ids, documents, metadatas


# ----------------------------------------------------------------------------
# 3. Store in ChromaDB
# ----------------------------------------------------------------------------
def build_collection(ids, documents, metadatas):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Rebuild from scratch so re-runs don't duplicate or leave stale chunks
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # Cosine distance -> confidence = 1 - distance is easy to read
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection


# ----------------------------------------------------------------------------
# 4. Test queries
# ----------------------------------------------------------------------------
def run_test_queries(collection, queries):
    print("\n" + "=" * 78)
    print("TEST QUERIES")
    print("=" * 78)

    for q in queries:
        res = collection.query(query_texts=[q], n_results=1)
        meta = res["metadatas"][0][0]
        distance = res["distances"][0][0]
        confidence = max(0.0, 1.0 - distance)

        print(f"\nQuery      : {q}")
        print(f"Best match : {meta['article']}  ({meta['title']})")
        print(f"Section    : {meta['section']}")
        print(f"Confidence : {confidence:.3f}")


def main():
    print(f"Reading KB articles from: {KB_DIR}")
    ids, documents, metadatas = load_chunks(KB_DIR)

    print(f"\nStoring {len(ids)} chunks in ChromaDB collection '{COLLECTION_NAME}'")
    print(f"Persist directory: {CHROMA_DIR}")
    collection = build_collection(ids, documents, metadatas)
    print(f"Collection count: {collection.count()}")

    run_test_queries(collection, SAMPLE_QUERIES)


if __name__ == "__main__":
    main()