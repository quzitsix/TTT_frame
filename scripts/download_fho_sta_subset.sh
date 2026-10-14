#!/usr/bin/env bash
# Download only a small FHO-STA validation subset after Ego4D credentials are configured.
#
# Required before running:
#   aws configure --profile ego4d
#   export EGO4D_DIR=/data/quzitsix/ego4d_fho_subset
#
# The Ego4D license/AWS credentials are not bundled with this repository.  This
# script deliberately downloads annotations and selected FHO clips only; it does
# not request the complete clips/full_scale dataset, LMDB frames, or STA model
# checkpoints. Set FHO_MEDIA_DATASET=full_scale only when the clip manifest is
# unavailable and canonical video access is needed.

set -euo pipefail

ROOT_DIR="${EGO4D_DIR:-/data/quzitsix/ego4d_fho_subset}"
VERSION="${EGO4D_VERSION:-v2_1}"
PROFILE="${EGO4D_AWS_PROFILE:-ego4d}"
LIMIT="${FHO_LIMIT:-16}"
SEED="${FHO_SEED:-0}"
MEDIA_DATASET="${FHO_MEDIA_DATASET:-clips}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$MEDIA_DATASET" in
  clips)
    UID_BASENAME="clip_uids.txt"
    FRAME_FIELD="clip_frame"
    ;;
  full_scale)
    UID_BASENAME="video_uids.txt"
    FRAME_FIELD="frame"
    ;;
  *)
    echo "FHO_MEDIA_DATASET must be clips or full_scale (got: $MEDIA_DATASET)" >&2
    exit 2
    ;;
esac

if ! aws configure list --profile "$PROFILE" >/dev/null 2>&1; then
  cat >&2 <<EOF
No usable AWS profile '$PROFILE' was found.
Configure the licensed Ego4D credentials first:

    aws configure --profile $PROFILE

Then rerun this script.  No dataset files were downloaded.
EOF
  exit 2
fi

mkdir -p "$ROOT_DIR"
conda run --no-capture-output -n meowbench python -m ego4d.cli.cli \
  --output_directory="$ROOT_DIR" \
  --datasets annotations --benchmarks FHO --version "$VERSION" \
  --aws_profile_name="$PROFILE" --no-metadata -y

ANNOTATION=""
for candidate in \
  "$ROOT_DIR/$VERSION/annotations/fho_sta_val.json" \
  "$ROOT_DIR/${VERSION}/annotations/fho_sta_val.json"; do
  if [ -f "$candidate" ]; then
    ANNOTATION="$candidate"
    break
  fi
done
if [ -z "$ANNOTATION" ]; then
  ANNOTATION="$(find "$ROOT_DIR" -type f -name fho_sta_val.json -print -quit)"
fi
if [ -z "$ANNOTATION" ] || [ ! -f "$ANNOTATION" ]; then
  echo "Downloaded annotations but could not locate fho_sta_val.json under $ROOT_DIR" >&2
  exit 3
fi

SUBSET_DIR="$ROOT_DIR/fho_sta_subset_${LIMIT}"
conda run --no-capture-output -n meowbench python \
  "$REPO_ROOT/scripts/prepare_fho_sta_subset.py" \
  --annotation "$ANNOTATION" --output "$SUBSET_DIR" --limit "$LIMIT" --seed "$SEED"

# Download only the selected media UIDs; keeping the file explicit makes the
# request auditable and restartable. The default clips dataset is much smaller
# and uses clip_uid/clip_frame, which is the preferred smoke-test path.
UID_FILE="$SUBSET_DIR/$UID_BASENAME"
if [ ! -s "$UID_FILE" ]; then
  echo "Subset has no $UID_BASENAME; refusing to let the Ego4D CLI fall back to all media." >&2
  exit 4
fi
conda run --no-capture-output -n meowbench python -m ego4d.cli.cli \
  --output_directory="$ROOT_DIR" \
  --datasets "$MEDIA_DATASET" --benchmarks FHO --version "$VERSION" \
  --video_uid_file="$UID_FILE" \
  --aws_profile_name="$PROFILE" --no-metadata -y

echo "FHO-STA subset is ready: $SUBSET_DIR"
echo "Use --annotation $SUBSET_DIR/fho_sta_val_subset.json for the smoke evaluator."
echo "Use --video-root $ROOT_DIR/v2/$MEDIA_DATASET --frame-field $FRAME_FIELD."
