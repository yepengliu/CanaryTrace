"""Stage 3 -- Response generation and watermark scoring.

Feed the retrieved documents + probing question to the (suspicious) RA-LLM,
collect its response, and score the green-list watermark z-statistic of that
response. Consolidates ``evaluation.py`` and all of its variants via a single
``task`` switch:

    rag_watermark      base pipeline: generate a response, score its watermark
                       (covers _ance/_b2m/_k1/_para/_bias/_orig -- pure config diffs)
    score_docs         score canary/IP documents directly, no RAG    (was _un)
    answerability      classify Answerable/Unanswerable               (was _fact/_fact2)
    watermarked_rallm  RA-LLM itself watermarks its output w/ a different key (_wm_rallm)

All paths, K, base_dataset_num, system prompt, and slice ranges come from config.
"""

import os
from itertools import islice

import torch
import pandas as pd
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer
from beir.datasets.data_loader import GenericDataLoader

from . import config
from .cli import parse_and_load
from .utils import vocab_segmentation, load_model

from .detection import detect

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
SYS_PROMPT_FILES = {"easy": "easy.txt", "eval": "eval.txt", "hard": "hard.txt"}


def load_queries(path):
    queries = {}
    with open(path, encoding="utf8") as fin:
        import json
        for line in fin:
            line = json.loads(line)
            queries[line.get("_id")] = line.get("text")
    return queries


def load_rag_data(path):
    import json
    with open(path, "r") as f:
        trap = json.load(f)
    return [{"id": key, "doc": value} for key, value in trap.items()]


def get_sys_prompt(name):
    fname = SYS_PROMPT_FILES.get(name, SYS_PROMPT_FILES["easy"])
    with open(os.path.join(PROMPTS_DIR, fname), "r") as f:
        return f.read()


def load_ip_corpus(cfg):
    """Just the IP corpus -- no base corpus, no canaries."""
    from .rag import beir_data_path
    data = cfg["data"]
    if data.get("ip_loader", "beir_split") in ("corpus_file", "local"):
        return GenericDataLoader(corpus_file=str(config.resolve_path(data["ip_corpus_path"]))).load_corpus()
    ip_path = beir_data_path(data["ip_dataset"])
    ip_corpus, _, _ = GenericDataLoader(data_folder=ip_path).load(split="test")
    return ip_corpus


def build_corpus(cfg):
    """Union corpus (base + IP + canaries) plus IP-corpus and combined query lookup."""
    from .rag import beir_data_path
    data = cfg["data"]
    ms_path = beir_data_path(data["base_dataset"])
    ms_corpus, _, _ = GenericDataLoader(data_folder=ms_path).load(split="test")
    ms_corpus = dict(islice(ms_corpus.items(), data["base_dataset_num"]))

    if data.get("ip_loader", "beir_split") in ("corpus_file", "local"):
        ip_corpus = GenericDataLoader(corpus_file=str(config.resolve_path(data["ip_corpus_path"]))).load_corpus()
    else:
        ip_path = beir_data_path(data["ip_dataset"])
        ip_corpus, _, _ = GenericDataLoader(data_folder=ip_path).load(split="test")

    ip_queries = load_queries(config.resolve_path(data["ip_queries_path"]))
    trap_corpus = GenericDataLoader(corpus_file=str(config.resolve_path(data["trap_corpus_path"]))).load_corpus()
    trap_queries = load_queries(config.resolve_path(data["trap_queries_path"]))

    corpus = {**ms_corpus, **ip_corpus, **trap_corpus}
    queries = {**ip_queries, **trap_queries}
    return corpus, ip_corpus, queries


def build_context(context_list, k):
    docs = context_list[:k]
    if k == 1:
        return f"Document: {docs[0]}"
    return "\n ".join(f"Document {i + 1}: {c}" for i, c in enumerate(docs))


# Responses are flushed every FLUSH_EVERY rows so a killed job keeps its progress
# without rewriting the whole CSV per row.
FLUSH_EVERY = 50


def save_rows(rows, save_path, force=False):
    if not force and len(rows) % FLUSH_EVERY:
        return
    os.makedirs(save_path.parent, exist_ok=True)
    pd.DataFrame(rows).to_csv(save_path, index=False)


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
def _make_pipeline(model_id, dtype="float16"):
    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    pipe = transformers.pipeline(
        "text-generation", model=model_id, device_map="auto", torch_dtype=torch_dtype,
    )
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    return pipe


def task_rag_watermark(cfg, ev, device):
    pipe = _make_pipeline(cfg["models"]["rag_llm"], cfg["models"].get("dtype", "float16"))
    green_list_ids = vocab_segmentation(pipe.tokenizer, device, cfg["watermark"]["gamma"], cfg["watermark"]["key"])
    sys_prompt = get_sys_prompt(ev.get("sys_prompt", "easy"))
    corpus, _, queries = build_corpus(cfg)

    rag_data = load_rag_data(config.resolve_path(ev["rag_data"]))
    sl = ev.get("slice")
    if sl:
        rag_data = rag_data[sl[0]:sl[1]]

    k = ev.get("k", 3)
    save_path = config.resolve_path(ev["save"])
    rows = []
    for data in tqdm(rag_data):
        query = queries[data["id"]]
        context_list = [corpus[doc_id]["text"] for doc_id in data["doc"]]
        context = build_context(context_list, k)
        template = (
            f"Here is the set of retrieved documents you will use to answer the question:\n "
            f"<documents>\n{context}\n</documents> \n Now, here is the question you need to answer:\n "
            f"<question>\n{query}\n</question>"
        )
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": template}]
        outputs = pipe(messages, max_new_tokens=ev.get("max_new_tokens", 600))
        outputs = outputs[0]["generated_text"][-1]["content"]
        score = detect(outputs, pipe.tokenizer, green_list_ids, cfg["watermark"]["gamma"])
        rows.append({"query_id": data["id"], "output": outputs, "score": score})
        save_rows(rows, save_path)
    save_rows(rows, save_path, force=True)
    print(f"[evaluation:rag_watermark] {len(rows)} responses -> {save_path}")


def task_score_docs(cfg, ev, device):
    """Score the watermark of canary/IP documents directly (no retrieval or LLM)."""
    tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["rag_llm"])
    green_list_ids = vocab_segmentation(tokenizer, device, cfg["watermark"]["gamma"], cfg["watermark"]["key"])
    ip_corpus = load_ip_corpus(cfg)      # no base corpus needed: nothing is retrieved here
    save_path = config.resolve_path(ev["save"])
    rows = []
    for query_id, doc in tqdm(ip_corpus.items()):
        text = doc["text"]
        score = detect(text, tokenizer, green_list_ids, cfg["watermark"]["gamma"])
        rows.append({"query_id": query_id, "output": text, "score": score})
        save_rows(rows, save_path)
    save_rows(rows, save_path, force=True)
    print(f"[evaluation:score_docs] {len(rows)} documents -> {save_path}")


def task_answerability(cfg, ev, device):
    """Ask the RA-LLM whether the question is answerable from the retrieved docs."""
    pipe = _make_pipeline(cfg["models"]["rag_llm"], cfg["models"].get("dtype", "float16"))
    sys_prompt = get_sys_prompt(ev.get("sys_prompt", "easy"))
    corpus, _, queries = build_corpus(cfg)
    rag_data = load_rag_data(config.resolve_path(ev["rag_data"]))
    k = ev.get("k", 3)
    mode = ev.get("answerability_mode", "with_context")
    save_path = config.resolve_path(ev["save"])
    rows = []
    for data in tqdm(rag_data):
        query = queries[data["id"]]
        if mode == "question_only":
            template = (
                f"\n        Here is the question you need to answer:\n <question>\n{query}\n</question>\n\n"
                "        Do not hallucinate the answer. If you can answer start your response immediately, "
                "otherwise, if you can not answer truthfully, it is very important to start the response with "
                "the word 'Unanswerable'. Feel free to explain why it is unanswerable after, but always start "
                "such responses with the word Unanswerable.\n        "
            )
            messages = [{"role": "user", "content": template}]
        else:
            context_list = [corpus[doc_id]["text"] for doc_id in data["doc"]]
            context = build_context(context_list, k)
            template = (
                f"\n        Here is the set of retrieved documents you will use to answer the question:\n "
                f"<documents>\n{context}\n</documents> \n \n        \n        Now, here is the question you need "
                f"to answer:\n <question>\n{query}\n</question>\n\n        Do not hallucinate the answer. If you "
                "can answer, please response 'Answerable'. Otherwize, if you can not answer truthfully, please "
                "response 'Unanswerable'. Please not that you should only response 'Answerable' or 'Unanswerable'. "
                "Do not response anything else.\n        "
            )
            messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": template}]
        outputs = pipe(messages, max_new_tokens=ev.get("max_new_tokens", 10))
        outputs = outputs[0]["generated_text"][-1]["content"]
        if mode == "question_only":
            score = 0 if "Unanswerable" in outputs else -1
        else:
            score = 1 if "Answerable" in outputs else (0 if "Unanswerable" in outputs else -1)
        rows.append({"query_id": data["id"], "output": outputs, "score": score})
        save_rows(rows, save_path)
    save_rows(rows, save_path, force=True)
    print(f"[evaluation:answerability] {len(rows)} responses -> {save_path}")


def task_watermarked_rallm(cfg, ev, device):
    """Adversarial RA-LLM that re-watermarks its own output with a different key."""
    from .watermark_unigram import Watermark
    wm = cfg["watermark"]
    watermark_model, watermark_tokenizer = load_model(cfg["models"]["rag_llm"], cfg["models"].get("dtype", "float16"))
    watermark = Watermark(
        device=device, watermark_tokenizer=watermark_tokenizer, watermark_model=watermark_model,
        gamma=wm["gamma"], bias=ev.get("bias", 2.0), top_k=50, top_p=0.9, repetition_penalty=1.0,
        no_repeat_ngram_size=0, max_new_tokens=ev.get("max_new_tokens", 600), key=ev["wm_key"],
    )
    green_list_ids = vocab_segmentation(watermark_tokenizer, device, wm["gamma"], ev["detect_key"])
    sys_prompt = get_sys_prompt(ev.get("sys_prompt", "easy"))
    corpus, _, queries = build_corpus(cfg)
    rag_data = load_rag_data(config.resolve_path(ev["rag_data"]))
    k = ev.get("k", 3)
    save_path = config.resolve_path(ev["save"])
    rows = []
    for data in tqdm(rag_data):
        query = queries[data["id"]]
        context_list = [corpus[doc_id]["text"] for doc_id in data["doc"]]
        context = build_context(context_list, k)
        template = (
            f"Here is the set of retrieved documents you will use to answer the question:\n "
            f"<documents>\n{context}\n</documents> \n Now, here is the question you need to answer:\n "
            f"<question>\n{query}\n</question>"
        )
        message = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>{template}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>"
        )
        outputs = watermark.generate_watermarked(message)
        score = detect(outputs, watermark_tokenizer, green_list_ids, wm["gamma"])
        rows.append({"query_id": data["id"], "output": outputs, "score": score})
        save_rows(rows, save_path)
    save_rows(rows, save_path, force=True)
    print(f"[evaluation:watermarked_rallm] {len(rows)} responses -> {save_path}")


TASKS = {
    "rag_watermark": task_rag_watermark,
    "score_docs": task_score_docs,
    "answerability": task_answerability,
    "watermarked_rallm": task_watermarked_rallm,
}


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ev = cfg["evaluation"]
    task = ev.get("task", "rag_watermark")
    if task not in TASKS:
        raise ValueError(f"Unknown evaluation task {task!r}; expected one of {list(TASKS)}")
    TASKS[task](cfg, ev, device)


if __name__ == "__main__":
    main(parse_and_load("CanaryTrace stage 3: response generation + watermark scoring"))
