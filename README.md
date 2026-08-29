# CanaryTrace: Dataset Protection via Watermarked Canaries in Retrieval-Augmented LLMs

 Official implementation of the paper:

["Dataset Protection via Watermarked Canaries in Retrieval-Augmented LLMs"](https://arxiv.org/abs/2502.10673) by Yepeng Liu, Xuandong Zhao, Dawn Song, Yuheng Bu.

Retrieval-Augmented Generation (RAG) has become an effective method for enhancing large language models (LLMs) with up-to-date knowledge. However, it may pose a risk of copyright infringement, as IP datasets may be incorporated into the knowledge database by malicious Retrieval-Augmented LLMs (RA-LLMs) without authorization. To protect the rights of the dataset owner, an effective dataset membership inference algorithm for RA-LLMs is needed. In this work, we introduce a novel approach, \textit{CanaryTrace}, to safeguard the ownership of text datasets and effectively detect unauthorized use by the RA-LLMs. Our approach preserves the original data completely unchanged while protecting it by inserting specifically designed canary documents into the IP dataset. These canary documents are created with synthetic content and embedded watermarks to ensure uniqueness, consistency, and statistical provability. During the detection process, unauthorized usage is identified by querying the canary documents and analyzing the responses of RA-LLMs for statistical evidence of the embedded watermark. Our experimental results demonstrate high query efficiency, detectability, and consistency, along with minimal perturbation to the original dataset, all without compromising the performance of the RAG system.


<img width="2227" height="1113" alt="overview" src="https://github.com/user-attachments/assets/26ac6727-9cb7-4bb9-a55b-1fc7496d486f" />




## Repository layout

```
canarytrace/          The pipeline; one module per stage
  config.py             Path resolution + YAML config loading (_base_ inheritance)
  synthesize.py         Stage 1  synthesize watermarked canary documents + questions
  prepare.py            Stage 1b synthesis JSON -> trap_corpus / trap_queries JSONL
  rag.py                Stage 2  retrieve top-K documents (Contriever / ANCE)
  evaluation.py         Stage 3  RA-LLM responses + watermark z-score
  detector.py           Stage 4  detection across query quotas
  watermark_unigram.py  Green-list (unigram) watermarked generation
  detection.py          Green-list watermark detector (z-test)
  prompts/              RA-LLM system prompts (easy / eval / hard)
configs/              One YAML per experiment (see the table below)
scripts/slurm/        SLURM launchers for each stage
log/                  SLURM job logs land here (submit from the repository root)
```

`datasets/` and `results/` are created on first run and are not tracked.

## Installation

```bash
conda create -n canarytrace python=3.9 -y && conda activate canarytrace

# 1. PyTorch matching your GPU (see the note at the top of requirements.txt)
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# 2. the rest of the stack
pip install -r requirements.txt

# 3. Contriever (retrieval stage only; not vendored — CC BY-NC 4.0, see Licensing)
git clone https://github.com/facebookresearch/contriever third_party/contriever
```

You also need:

- **`OPENAI_API_KEY`** — used only by the canary-synthesis stage. Put it in the
  environment or in a `.env` file at the repository root (`.env` is gitignored):
  ```bash
  echo 'OPENAI_API_KEY=sk-...' > .env
  ```
- **Hugging Face access to `meta-llama/Llama-3.1-70B-Instruct`** (`huggingface-cli login`).
  It is both the canary generator and the RA-LLM. `configs/smoke.yaml` substitutes
  `Llama-3.2-1B-Instruct` so you can exercise the code without the 70B download.
- Retriever checkpoints `facebook/contriever-msmarco` and, for the ANCE ablation,
  `msmarco-roberta-base-ance-firstp`. Both download automatically.


## Configuration

Each experiment is one YAML file under `configs/`, which may inherit from another via
`_base_`. All paths resolve relative to the repository root, so there are no
machine-specific absolute paths; override the root with `CANARYTRACE_ROOT` if needed.

```bash
python -m canarytrace.<stage> --config <name>
python -m canarytrace.<stage> --config <name> --set section.key=value   # ad-hoc override
```

`main_nfcorpus` is the base experiment; every other config overrides only what differs.

| Config | Paper result | What it changes |
| --- | --- | --- |
| `main_nfcorpus` | Table 1, Fig. 4 | Main run: NFCorpus IP set, MS MARCO base, Contriever |
| `ance` | Table 13 | ANCE retriever instead of Contriever |
| `b2m` | Table 14 | 2M-document base dataset |
| `k1` | Table 15 | Retrieve top-1 (K = 1) |
| `bias1`, `bias3` | Table 11 | Watermark strength δ = 1, 3 |
| `hard_prompt` | Table 6 | Restrictive RA-LLM system prompt |
| `wm_rallm` | Table 10 | Adversarial RA-LLM re-watermarks its output with another key |
| `single` | Table 8 | Query one canary repeatedly instead of distinct canaries |
| `unwatermarked` | — | Baseline: score documents directly, no RAG |
| `substolen` | Appendix B.1 | Only a subset of the IP dataset is stolen |
| `smoke` | — | Environment check: 5 canaries on a 1B model; numbers are meaningless |
| `real_mini` | — | Same, but on the real 70B model: 8 canaries, reduced base corpus |

Order matters for a few of them: `single` and `substolen` read the response CSV that
`main_nfcorpus` produces, and `substolen` additionally needs `unwatermarked`.


## Running the pipeline

```bash
python -m canarytrace.synthesize --config main_nfcorpus   # 1.  watermarked canaries (JSON)
python -m canarytrace.prepare    --config main_nfcorpus   # 1b. -> trap_corpus / trap_queries
python -m canarytrace.rag        --config main_nfcorpus   # 2.  retrieve top-K
python -m canarytrace.evaluation --config main_nfcorpus   # 3.  RA-LLM responses + z-score
python -m canarytrace.detector   --config main_nfcorpus   # 4.  detection vs. query quota
```

On SLURM (pass your cluster's partition; the scripts set no `--partition` directive):

```bash
sbatch --partition=<gpu-partition> --export=ALL,CONFIG=main_nfcorpus \
       scripts/slurm/pipeline.sh          # all stages in one job
```

Per-stage launchers (`synthesize.sh`, `rag.sh`, `evaluation.sh`, `detector.sh`) take the
same arguments. `ENV=<conda-env>` overrides the environment name (default `canarytrace`).

Before committing to a real run (the 70B download plus OpenAI calls for 500 canaries),
`configs/smoke.yaml` exercises all five stages on a 1B model with 5 canaries in a few
minutes. Its numbers are meaningless — it only confirms the environment and the
plumbing work. It still needs a GPU and an OpenAI key:

```bash
sbatch --partition=<gpu-partition> --gpus=1 --mem=64gb --time=1:00:00 \
       --export=ALL,CONFIG=smoke scripts/slurm/pipeline.sh
```

## Data

The MS MARCO knowledge base and the NFCorpus IP dataset download automatically from
BEIR into `datasets/` on first use. Canary documents and probing questions are
generated by stages 1 and 1b — no canary files ship with this repository, so every run
produces a fresh canary set under your own watermark key.


## Citation

If you find this repository useful for your research or applications, please cite our paper:
```
@article{liu2025dataset,
  title={Dataset protection via watermarked canaries in retrieval-augmented llms},
  author={Liu, Yepeng and Zhao, Xuandong and Song, Dawn and Bu, Yuheng},
  journal={arXiv preprint arXiv:2502.10673},
  year={2025}
}
```
