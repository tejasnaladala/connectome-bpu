"""Retired entry point for the pre-audit CAB benchmark runner."""

import sys


def main() -> int:
    print(
        "benchmarks/runner.py is retired because it could invent demo data and "
        "bypass provenance checks. Use `python run_experiments.py` for the "
        "corrected CAB v2 pipeline.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
