"""Stage 2 -- Retrieval.

Retrieve the top-K documents for each probing (trap) query against a knowledge
base of ``base_dataset`` + IP dataset + inserted canaries, and save the retrieved
document ids per query. Consolidates the former ``rag.py`` and its variants
(``rag_ance``, ``rag_b2m``, ``rag_ablation``, ...); the differences are all
expressed as config values:

    retriever:   contriever | ance
    ip_loader:   beir_split | corpus_file      (Mathematica uses corpus_file)
    task:        trap | original               (which query set to retrieve)
    base_dataset_num, k_values, and the various corpus/query paths.
"""

import os
import json
import pathlib
from itertools import islice

import torch
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from transformers import AutoTokenizer

from . import config
from .cli import parse_and_load

config.setup_import_paths()


def load_queries(path):
    queries = {}
    with open(path, encoding="utf8") as fin:
        for line in fin:
            line = json.loads(line)
            queries[line.get("_id")] = line.get("text")
    return queries


def beir_data_path(dataset):
    """Return a local BEIR dataset folder, downloading it on first use."""
    out_dir = str(config.DATASETS_DIR)
    data_path = os.path.join(out_dir, dataset)
    if not os.path.exists(data_path):
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        data_path = util.download_and_unzip(url, out_dir)
    return data_path


def build_retriever(name, k_values, device):
    if name == "ance":
        from beir.retrieval import models   # pulls in sentence_transformers; only needed for ANCE
        model = DRES(models.SentenceBERT("msmarco-roberta-base-ance-firstp"))
        return EvaluateRetrieval(model, score_function="cos_sim", k_values=[k_values])
    if name == "contriever":
        from contriever import Contriever          # third_party/contriever
        from beir_utils import DenseEncoderModel    # third_party/contriever/src
        encoder = Contriever.from_pretrained("facebook/contriever-msmarco").to(device)
        tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
        model = DRES(DenseEncoderModel(encoder, doc_encoder=encoder, tokenizer=tokenizer), batch_size=256)
        return EvaluateRetrieval(model, score_function="cos_sim", k_values=[k_values])
    raise ValueError(f"Unknown retriever: {name!r} (expected 'contriever' or 'ance')")


def main(cfg):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    data = cfg["data"]
    rag = cfg["rag"]

    # --- base (knowledge) corpus ------------------------------------------- #
    ms_path = beir_data_path(data["base_dataset"])
    ms_corpus, _, _ = GenericDataLoader(data_folder=ms_path).load(split="test")
    ms_corpus = dict(islice(ms_corpus.items(), data["base_dataset_num"]))

    # --- IP corpus + queries ----------------------------------------------- #
    if data.get("ip_loader", "beir_split") in ("corpus_file", "local"):
        ip_corpus = GenericDataLoader(corpus_file=str(config.resolve_path(data["ip_corpus_path"]))).load_corpus()
    else:
        ip_path = beir_data_path(data["ip_dataset"])
        ip_corpus, _, _ = GenericDataLoader(data_folder=ip_path).load(split="test")
    ip_queries = load_queries(config.resolve_path(data["ip_queries_path"]))

    # --- inserted canaries (trap) ------------------------------------------ #
    trap_corpus = GenericDataLoader(corpus_file=str(config.resolve_path(data["trap_corpus_path"]))).load_corpus()
    trap_queries = load_queries(config.resolve_path(data["trap_queries_path"]))

    corpus = {**ms_corpus, **ip_corpus, **trap_corpus}

    retriever = build_retriever(rag.get("retriever", "contriever"), rag.get("k_values", 3), device)

    task = rag.get("task", "trap")
    query_set = trap_queries if task == "trap" else ip_queries
    results = retriever.retrieve(corpus, query_set)

    save_path = config.resolve_path(rag["save"])
    os.makedirs(save_path.parent, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f)
    print(f"[rag] saved retrieval results for {len(query_set)} queries -> {save_path}")


if __name__ == "__main__":
    main(parse_and_load("CanaryTrace stage 2: retrieval"))
