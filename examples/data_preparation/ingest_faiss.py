from __future__ import annotations
import os, csv, glob, argparse, json
from typing import Iterable
from doc_studio.retrieval import Document
from doc_studio.retrieval.hf_embedder import HFEmbedder
from doc_studio.retrieval.faiss_retriever import FaissRetriever

# quick read helpers
def _iter_txt_md(path:str)->Iterable[Document]:
    i=1
    for p in glob.glob(os.path.join(path, "**/*.txt"), recursive=True)+glob.glob(os.path.join(path, "**/*.md"), recursive=True):
        with open(p,"r",encoding="utf-8",errors="ignore") as f:
            t=f.read().strip()
            if not t: continue
            yield Document(f"d{i}", t, meta={"path":p}); i+=1

def _iter_csv(path:str, text_col:str="text")->Iterable[Document]:
    i=1
    for p in glob.glob(os.path.join(path, "**/*.csv"), recursive=True):
        with open(p,"r",encoding="utf-8",errors="ignore") as f:
            r=csv.DictReader(f)
            for row in r:
                t=(row.get(text_col) or "").strip()
                if not t: continue
                yield Document(f"d{i}", t, meta={"path":p}); i+=1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="folder with .txt/.md/.csv files")
    ap.add_argument("--csv_text_col", default="text", help="csv column to use as doc text")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--out_index", default="data/index.faiss")
    ap.add_argument("--out_meta", default="data/index.meta.jsonl")
    args=ap.parse_args()

    os.makedirs(os.path.dirname(args.out_index), exist_ok=True)

    # gather docs
    docs: list[Document] = []
    docs.extend(list(_iter_txt_md(args.input_dir)))
    docs.extend(list(_iter_csv(args.input_dir, text_col=args.csv_text_col)))
    if not docs:
        print("no docs found"); return

    # embed + build faiss
    emb=HFEmbedder(args.model)
    r=FaissRetriever(emb)
    r.add(docs)
    print("indexed:", r.size())

    # save
    r.save(args.out_index, args.out_meta)
    print("saved:", args.out_index, args.out_meta)

if __name__=="__main__":
    main()
