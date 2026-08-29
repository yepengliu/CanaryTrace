"""CanaryTrace: dataset protection via watermarked canaries in retrieval-augmented LLMs.

Consolidated, config-driven reimplementation of the four-stage pipeline:

    synthesize  ->  rag  ->  evaluation  ->  detector

Each stage is a module with a ``main(cfg)`` entry point and a ``__main__`` guard
that loads an experiment config from ``configs/``. See the top-level README.
"""

from . import config  # noqa: F401

__all__ = ["config"]
