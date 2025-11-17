from __future__ import annotations
import os, json, re, uuid
from pathlib import Path

# Choose ONE of these:
# pip install pypdf
from pypdf import PdfReader

# If pypdf text isn't good on your PDFs, try PyMuPDF:
# pip install pymupdf
# import fitz

DATA_DIR = Path("data")
PAPERS_DIR = DATA_DIR / "papers"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
EVAL_PATH = DATA_DIR / "eval.jsonl"

MAX_CHARS = 1000
OVERLAP = 150

def clean_text(txt: str) -> str:
    # normalize whitespace
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = txt.strip()
    # drop everything after a typical "References" heading (heuristic)
    refs = re.search(r"\n\s*(references|bibliography)\s*\n", txt, re.IGNORECASE)
    if refs:
        txt = txt[:refs.start()]
    return txt

def chunk(text: str, max_chars=MAX_CHARS, overlap=OVERLAP) -> list[str]:
    text = " ".join(text.split())
    out, i = [], 0
    step = max_chars - overlap
    while i < len(text):
        out.append(text[i:i+max_chars])
        i += step
    return [c for c in out if c.strip()]

def extract_pdf_text_and_meta(pdf_path: Path) -> tuple[str, dict]:
    # Option A: PyPDF
    reader = PdfReader(str(pdf_path))
    meta = reader.metadata or {}
    # fallbacks
    title = str(meta.title or pdf_path.stem)
    author = str(meta.author or "")
    # Concatenate page text
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    txt = "\n".join(pages)

    # If PyPDF extraction is poor, switch to PyMuPDF:
    # doc = fitz.open(pdf_path)
    # title = doc.metadata.get("title") or pdf_path.stem
    # author = doc.metadata.get("author") or ""
    # txt = "\n".join(page.get_text("text") for page in doc)

    txt = clean_text(txt)
    meta_out = {
        "title": title.strip(),
        "author": author.strip(),
        "filename": pdf_path.name,
    }
    return txt, meta_out

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not PAPERS_DIR.exists():
        raise SystemExit(f"Put your PDFs under {PAPERS_DIR}/ then rerun.")

    n_docs = 0
    with open(CORPUS_PATH, "w") as fout:
        for pdf in PAPERS_DIR.rglob("*.pdf"):
            text, meta = extract_pdf_text_and_meta(pdf)
            if not text or len(text) < 300:
                continue
            # Optional: try to grab year/doi from the first 1k chars
            head = text[:1000]
            year = re.search(r"\b(20\d{2}|19\d{2})\b", head)
            doi  = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", head)
            meta["year"] = year.group(1) if year else ""
            meta["doi"]  = doi.group(0) if doi else ""

            for idx, ch in enumerate(chunk(text)):
                rec = {
                    "id": f"{pdf.stem}-{idx}-{uuid.uuid4().hex[:6]}",
                    "text": ch,
                    "meta": meta,
                }
                fout.write(json.dumps(rec) + "\n")
                n_docs += 1

    # If you don't have an eval set yet, make a simple one from titles.
    if not EVAL_PATH.exists():
        with open(EVAL_PATH, "w") as fe:
            # Use each paper title as a query seed; edit later as you like.
            titles = set()
            with open(CORPUS_PATH) as f:
                for line in f:
                    m = json.loads(line)
                    titles.add(m["meta"]["title"])
            for t in list(titles)[:50]:
                fe.write(json.dumps({"query": f"Summarize the key contributions of '{t}' with citations."}) + "\n")

    print(f"Wrote {n_docs} chunks to {CORPUS_PATH}")
    print(f"Eval set at {EVAL_PATH} (edit freely).")

if __name__ == "__main__":
    main()
