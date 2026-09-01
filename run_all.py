"""Retired entry point for the pre-audit CAB pipeline."""

import sys


def main() -> int:
    print(
        "run_all.py is retired because it bypassed provenance and protocol "
        "validation. Use `python run_experiments.py` for the corrected CAB v2 "
        "pipeline.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
