#!/usr/bin/env bash
set -euo pipefail

# Materialize the nine-video, 42-trajectory SVCBench selection described by
# data/svcbench_real_natural9_v1.json.  The selected media total about 618 MiB;
# this wrapper does not run unless invoked explicitly.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
SELECTION="$REPO_ROOT/data/svcbench_real_natural9_v1.json"
export SVC_ROOT=${SVC_ROOT:-/data/quzitsix/svcbench_real_natural9_v1}
export SVC_REVISION=${SVC_REVISION:-4c9bd87ef3b0f269ca8b4503c081f5f38bc1fc9a}
export SVC_VIDEOS='Ego4D/b71f48ff-db06-42d4-a49b-b58d7f4ba7e1.mp4,Ego4D/e5d94463-f292-451d-8d7c-0926001da4e1.mp4,Ego4D/20f345f7-d592-4d52-851f-3ee4544a847d.mp4,arkitscenes/42898849.mp4,scannet/scene0645_00.mp4,scannetpp/5eb31827b7.mp4,RoomTour3D/B8Y35YnKONk.mp4,CODa_full/6.mp4,youtube/youtube_0019.mp4'

if [[ "${CONDA_DEFAULT_ENV:-}" == "meowbench" ]]; then
  PYTHON=(python)
elif command -v conda >/dev/null 2>&1; then
  PYTHON=(conda run --no-capture-output -n meowbench python)
else
  echo "The meowbench conda environment is required" >&2
  exit 2
fi

bash "$SCRIPT_DIR/download_svcbench_subset.sh"

echo "Verifying downloaded media against the pinned Git-LFS SHA-256 values"
"${PYTHON[@]}" - "$SVC_ROOT" "$SELECTION" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root, selection_path = Path(sys.argv[1]), Path(sys.argv[2])
selection = json.loads(selection_path.read_text(encoding="utf-8"))
for item in selection["videos"]:
    path = root / "videos" / item["video"]
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    expected = item["sha256"]
    if actual != expected:
        raise SystemExit(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    if path.stat().st_size != item["size_bytes"]:
        raise SystemExit(f"size mismatch for {path}")
    print(f"OK {actual}  {path}")
PY
