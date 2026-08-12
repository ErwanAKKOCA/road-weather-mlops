from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest_path: Path, root: Path, required: set[str] | None = None) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {row["filename"]: row for row in manifest.get("files", [])}
    if required:
        missing_entries = sorted(required - entries.keys())
        if missing_entries:
            raise SystemExit(
                f"Manifest {manifest_path} is missing entries: {', '.join(missing_entries)}"
            )
    failures = []
    filenames = sorted(required) if required else sorted(entries)
    for filename in filenames:
        row = entries[filename]
        path = root / filename
        if not path.is_file():
            failures.append(f"missing file: {path}")
            continue
        actual = sha256(path)
        if actual != row.get("sha256"):
            failures.append(f"sha256 mismatch: {path}")
    if failures:
        raise SystemExit("Release validation failed:\n- " + "\n- ".join(failures))
    print(f"OK: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate locked model artifacts before build/deployment."
    )
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    artifacts = args.artifacts
    validate_manifest(
        artifacts / "release_manifest_v4_1.json",
        artifacts,
        {"dinov2_logistic_semantic_sky_v3_1.joblib", "region_definitions_v3_1.json"},
    )
    seg = artifacts / "mask2former_kitti_best"
    validate_manifest(
        seg / "release_manifest.json",
        seg,
        {"config.json", "preprocessor_config.json", "model.safetensors"},
    )
    print("All deployable model artifacts passed SHA-256 validation.")


if __name__ == "__main__":
    main()
