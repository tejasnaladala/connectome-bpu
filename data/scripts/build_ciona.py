"""Retired synthetic Ciona builder.

The corrected benchmark accepts only measured connectome artifacts with
source provenance and a verified SHA-256 digest.
"""

import sys


def main() -> int:
    print(
        "Synthetic Ciona generation is retired. Supply the measured source "
        "artifact and provenance required by data/scripts/standardize.py.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
