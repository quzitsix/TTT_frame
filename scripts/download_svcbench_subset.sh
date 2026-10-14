#!/usr/bin/env bash
set -euo pipefail

# Download only SVCBench metadata and a few selected source videos.  The
# default simulated video is ~15 MB and contains two complete count
# trajectories; this script never invokes a repository-wide download.

ROOT_DIR=${SVC_ROOT:-/data/quzitsix/svcbench_subset}
REPO_ID=${SVC_REPO_ID:-buaaplay/SVCBench}
VIDEO_SPEC=${SVC_VIDEOS:-simulated/0030.mp4}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ "${CONDA_DEFAULT_ENV:-}" == "meowbench" ]]; then
  PYTHON=(python)
elif command -v conda >/dev/null 2>&1; then
  PYTHON=(conda run --no-capture-output -n meowbench python)
else
  PYTHON=(python3)
fi

mkdir -p "$ROOT_DIR"

IFS=',' read -r -a VIDEO_LIST <<< "$VIDEO_SPEC"
VIDEO_ARGS=()
for video in "${VIDEO_LIST[@]}"; do
  video="${video#"${video%%[![:space:]]*}"}"
  video="${video%"${video##*[![:space:]]}"}"
  [[ -n "$video" ]] || continue
  VIDEO_ARGS+=(--video "$video")
done
if [[ ${#VIDEO_ARGS[@]} -eq 0 ]]; then
  echo "SVC_VIDEOS did not contain a source video path" >&2
  exit 2
fi

echo "Downloading SVCBench annotations only: $REPO_ID"
"${PYTHON[@]}" - "$REPO_ID" "$ROOT_DIR" <<'PY'
from pathlib import Path
import shutil
import sys
from huggingface_hub import hf_hub_download

repo, root = sys.argv[1], Path(sys.argv[2])
source = Path(hf_hub_download(repo, "vcbench_eval.jsonl", repo_type="dataset",
                               local_dir=str(root)))
target = root / "vcbench_eval.jsonl"
if source.resolve() != target.resolve():
    shutil.copy2(source, target)
print(target)
PY

SUBSET_DIR="$ROOT_DIR/subset"
echo "Preparing grouped trajectories: ${VIDEO_LIST[*]}"
"${PYTHON[@]}" "$SCRIPT_DIR/prepare_svcbench_subset.py" \
  --annotation "$ROOT_DIR/vcbench_eval.jsonl" \
  --output "$SUBSET_DIR" \
  "${VIDEO_ARGS[@]}"

echo "Downloading selected source videos (one file per requested video)"
"${PYTHON[@]}" - "$REPO_ID" "$ROOT_DIR" "$SUBSET_DIR/download_manifest.txt" <<'PY'
from pathlib import Path
import sys
from huggingface_hub import hf_hub_download

repo, root, manifest = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
for line in manifest.read_text(encoding="utf-8").splitlines():
    rel = line.strip()
    if not rel:
        continue
    path = hf_hub_download(repo, rel, repo_type="dataset", local_dir=str(root))
    print(path)
PY

echo "Prepared subset: $SUBSET_DIR/svcbench_subset.json"
echo "Video files:"
while IFS= read -r rel; do
  [[ -n "$rel" ]] || continue
  if [[ -f "$ROOT_DIR/$rel" ]]; then
    du -h "$ROOT_DIR/$rel"
  else
    echo "MISSING $ROOT_DIR/$rel" >&2
    exit 1
  fi
done < "$SUBSET_DIR/download_manifest.txt"
