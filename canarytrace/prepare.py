"""Stage 1b -- Prepare canary files from the synthesis output.

Convert the stage-1 ``synthetic_data_*.json`` (records with ``synthetic_text`` and
``question``) into the BEIR-style JSONL files the retrieval / evaluation stages read:

    trap_corpus_path   {"_id": "SYN-0000",   "title": " ", "text": <synthetic_text>}
    trap_queries_path  {"_id": "PLSYN-0000", "text": <question>}

Canary document ``SYN-i`` and its probing question ``PLSYN-i`` share the index ``i``.
This was previously done by hand in tools/dataset_process.ipynb.
"""

import os
import json

from . import config
from .cli import parse_and_load


def main(cfg):
    src = config.resolve_path(cfg["synthesize"]["output"])
    with open(src, "r") as f:
        data = json.load(f)

    corpus_path = config.resolve_path(cfg["data"]["trap_corpus_path"])
    queries_path = config.resolve_path(cfg["data"]["trap_queries_path"])
    os.makedirs(corpus_path.parent, exist_ok=True)
    os.makedirs(queries_path.parent, exist_ok=True)

    with open(corpus_path, "w") as fc, open(queries_path, "w") as fq:
        for i, d in enumerate(data):
            fc.write(json.dumps({"_id": f"SYN-{i:04d}", "title": " ", "text": d["synthetic_text"]}) + "\n")
            fq.write(json.dumps({"_id": f"PLSYN-{i:04d}", "text": d["question"]}) + "\n")

    print(f"[prepare] {len(data)} canaries: {corpus_path}  |  {queries_path}")


if __name__ == "__main__":
    main(parse_and_load("CanaryTrace stage 1b: build canary corpus/query files from synthesis output"))
