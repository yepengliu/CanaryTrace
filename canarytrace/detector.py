"""Stage 4 -- Watermark detection across query quotas.

Given the per-query response CSV from stage 3, form ``num_combinations`` groups of
``step`` responses (the query quota), concatenate each group, and score its
green-list watermark z-statistic. Repeating this over a range of ``steps`` yields
the detection-vs-query-quota curves in the paper.

Consolidates ``watermark_detector.py`` and its variants through a ``pairing`` mode:

    random        random groups of ``step`` distinct responses (default; most variants)
    stride        repeat a single response ``step`` times          (was _single)
    mixed_stolen  mix watermarked + unwatermarked responses         (was _substolen)

``input`` may be one CSV or a list of CSVs (concatenated first, e.g. the _bias runs).
"""

import os
import random
from math import comb

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from . import config
from .cli import parse_and_load
from .utils import vocab_segmentation

from .detection import detect


# Rows are flushed every FLUSH_EVERY scores so that a killed job keeps its progress
# without rewriting the whole CSV on every single row (the file holds the full
# concatenated text, so rewriting per row costs GBs of I/O over a run).
FLUSH_EVERY = 50


def _write(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)


def _maybe_write(rows, path, force=False):
    if force or len(rows) % FLUSH_EVERY == 0:
        _write(rows, path)


def get_unique_random_pairs(n, k, step):
    """Up to k unique sorted tuples of ``step`` distinct indices drawn from range(n).

    There are only C(n, step) such tuples; asking for more than that would loop
    forever, so k is capped at the number that actually exists.
    """
    available = comb(n, step)
    if k > available:
        print(f"[detector] only {available} distinct {step}-subsets of {n} responses "
              f"exist; using {available} instead of num_combinations={k}")
        k = available
    pairs = set()
    while len(pairs) < k:
        pairs.add(tuple(sorted(random.sample(range(n), step))))
    return list(pairs)


def load_input(spec):
    """Load one CSV path or a list of CSV paths (concatenated)."""
    paths = spec if isinstance(spec, list) else [spec]
    frames = [pd.read_csv(config.resolve_path(p)) for p in paths]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _score(gamma, text, tokenizer, green_list_ids):
    return detect(text, tokenizer, green_list_ids, gamma)


def run_random(df, gamma, tokenizer, green_list_ids, steps, k, population, name, out_dir):
    n = population or len(df)
    for step in steps:
        print(f"[detector] pairing=random step={step}")
        rows = []
        path = out_dir / f"{name}_combine_{step}.csv"
        for cid, group in enumerate(tqdm(get_unique_random_pairs(n, k, step))):
            text = " ".join(str(df.iloc[i]["output"]) for i in group)
            rows.append({"c_id": cid, "c_output": text, "score": _score(gamma, text, tokenizer, green_list_ids)})
            _maybe_write(rows, path)
        _write(rows, path)


def run_stride(df, gamma, tokenizer, green_list_ids, steps, name, out_dir):
    for step in steps:
        print(f"[detector] pairing=stride step={step}")
        rows = []
        path = out_dir / f"{name}_combine_{step}.csv"
        for i in range(0, len(df), step):
            text = " ".join(str(df.iloc[i]["output"]) for _ in range(step))
            rows.append({"c_id": i, "c_output": text, "score": _score(gamma, text, tokenizer, green_list_ids)})
            _maybe_write(rows, path)
        _write(rows, path)


def run_mixed_stolen(cfg, det, gamma, tokenizer, green_list_ids, k, name, out_dir):
    df_wm = pd.read_csv(config.resolve_path(det["input_wm"]))
    df_unwm = pd.read_csv(config.resolve_path(det["input_unwm"]))
    canary_num_in_list = det["canary_num_in_list"]
    total_query = det["total_query"]
    for c in tqdm(canary_num_in_list):
        combos_wm = get_unique_random_pairs(len(df_wm), k, c)
        combos_unwm = get_unique_random_pairs(len(df_unwm), k, total_query - c)
        rows = []
        path = out_dir / f"{name}_combine_wm{c}.csv"
        for cid, (cw, cu) in enumerate(zip(combos_wm, combos_unwm)):
            text = " ".join(str(df_wm.iloc[i]["output"]) for i in cw) + " " + \
                   " ".join(str(df_unwm.iloc[i]["output"]) for i in cu)
            rows.append({"c_id": cid, "c_output": text, "score": _score(gamma, text, tokenizer, green_list_ids)})
            _maybe_write(rows, path)
        _write(rows, path)


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det = cfg["detector"]
    wm = cfg["watermark"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["rag_llm"])
    green_list_ids = vocab_segmentation(tokenizer, device, wm["gamma"], wm["key"])

    out_dir = config.resolve_path(det["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    name = det.get("name", "trap_response_results")
    steps = det.get("steps", [2, 4, 6, 8, 10, 12, 14])
    k = det.get("num_combinations", 500)
    pairing = det.get("pairing", "random")

    if pairing == "mixed_stolen":
        run_mixed_stolen(cfg, det, wm["gamma"], tokenizer, green_list_ids, k, name, out_dir)
    else:
        df = load_input(det["input"])
        if pairing == "stride":
            run_stride(df, wm["gamma"], tokenizer, green_list_ids, steps, name, out_dir)
        else:
            run_random(df, wm["gamma"], tokenizer, green_list_ids, steps, k, det.get("population"), name, out_dir)
    print(f"[detector] done -> {out_dir}")


if __name__ == "__main__":
    main(parse_and_load("CanaryTrace stage 4: watermark detection across query quotas"))
