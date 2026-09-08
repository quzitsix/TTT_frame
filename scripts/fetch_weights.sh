#!/usr/bin/env bash
# Fetch the encoder weights this repo's experiment needs.
#
#   bash scripts/fetch_weights.sh                    # try the hub, then a mirror
#   HF_ENDPOINT=https://hf-mirror.com bash scripts/fetch_weights.sh
#
# Why this exists: the 8x4090 server has no route to huggingface.co, and
# `from_pretrained` does not fail fast there — it retries five times per file
# with exponential backoff, so a missing weight looks like a five-minute hang
# rather than an error. Measured: `pytest` spent 331 s in one test before
# pytest-timeout killed it.
#
# Nothing here is required for the memory itself. `pytest` covers the fast
# weights with 13 tests that need no weights at all; only the end-to-end
# perception test and scripts/exp_scene_change.py need CLIP.

set -euo pipefail

MODEL="${MODEL:-openai/clip-vit-base-patch32}"

say() { printf '\n==> %s\n' "$*"; }

# Reachability probe before committing to a download, with a short timeout so a
# blocked network is reported in seconds.
probe() {
  curl -sS -m 12 -o /dev/null -w '%{http_code}' \
    "$1/${MODEL}/resolve/main/config.json" 2>/dev/null || echo 000
}

if [ -n "${HF_ENDPOINT:-}" ]; then
  say "using HF_ENDPOINT=$HF_ENDPOINT (as set)"
else
  say "probing huggingface.co"
  # 200 or a 3xx redirect both mean reachable; 000 is curl's "could not connect".
  code="$(probe https://huggingface.co)"
  if [ "$code" != "000" ]; then
    say "huggingface.co reachable (HTTP $code)"
  else
    say "huggingface.co unreachable (curl $code); trying hf-mirror.com"
    mirror="$(probe https://hf-mirror.com)"
    if [ "$mirror" = "000" ]; then
      say "hf-mirror.com is unreachable too."
      cat <<'EOF'

Neither the hub nor the mirror is reachable from this machine. Copy the cache
from a machine that already has it:

    rsync -a ~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32 \
          <this-host>:~/.cache/huggingface/hub/

The directory is about 1.2 GB. Then run with HF_HUB_OFFLINE=1.
EOF
      exit 1
    fi
    export HF_ENDPOINT=https://hf-mirror.com
    say "mirror reachable (HTTP $mirror); exporting HF_ENDPOINT=$HF_ENDPOINT"
  fi
fi

say "downloading $MODEL"
python - "$MODEL" <<'PY'
import sys

from transformers import AutoModel, AutoProcessor

model_id = sys.argv[1]
AutoModel.from_pretrained(model_id)
AutoProcessor.from_pretrained(model_id)
print(f"cached {model_id}")
PY

say "verifying it loads with no network"
HF_HUB_OFFLINE=1 python - "$MODEL" <<'PY'
import sys

from transformers import AutoModel, AutoProcessor

model_id = sys.argv[1]
model = AutoModel.from_pretrained(model_id, local_files_only=True)
AutoProcessor.from_pretrained(model_id, local_files_only=True)
print(f"offline load OK (projection_dim={model.config.projection_dim})")
PY

say "done. now: pytest -q && python scripts/exp_scene_change.py --trials 30"
