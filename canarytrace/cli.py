"""Shared command-line front-end for the pipeline stages.

Every stage is invoked the same way::

    python -m canarytrace.rag --config main_nfcorpus
    python -m canarytrace.rag --config main_nfcorpus --set data.base_dataset_num=2000000

``--config`` names a YAML file under ``configs/`` (extension optional). Repeated
``--set section.key=value`` flags override individual values.
"""

import argparse

from .config import load_config


def parse_and_load(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", "-c", required=True,
        help="Experiment config name (under configs/) or path to a YAML file.",
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="section.key=value",
        help="Override a config value (repeatable).",
    )
    args = parser.parse_args()
    return load_config(args.config, args.overrides)
