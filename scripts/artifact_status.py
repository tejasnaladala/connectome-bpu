import json
from pathlib import Path
import sys


def require_reportable_artifact(results_dir: Path) -> None:
    results_dir = Path(results_dir)
    status_path = results_dir / "ARTIFACT_STATUS.json"
    with status_path.open(encoding="utf-8") as handle:
        status = json.load(handle)

    if status.get("usable_for_public_claims", False):
        return
    if "--allow-invalidated-artifact" in sys.argv:
        return

    print(
        "The committed result artifact is invalidated for public findings. "
        "See results/ARTIFACT_STATUS.json. Pass --allow-invalidated-artifact "
        "only for forensic inspection.",
        file=sys.stderr,
    )
    raise SystemExit(2)
