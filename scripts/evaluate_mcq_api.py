#!/usr/bin/env python3
"""Evaluate a JSONL MCQ set through an OpenAI-compatible chat endpoint.

This is useful for models that are deployed outside the Qwen3-VL-specific
``ttt_frame`` memory wrappers, such as Qwen3.5.  The evaluator sends only the
question, options, and single-letter instruction; gold answers and evidence
remain local.

Example::

    python scripts/evaluate_mcq_api.py \
      --items-path data/q9_codex_mcq.jsonl \
      --base-url http://127.0.0.1:18000/v1 \
      --model /data/hf_models/Qwen/Qwen3.5-35B-A3B \
      --max-tokens 16 \
      --output runs/q9_codex_mcq_qwen35.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import sys
import time
from typing import Any
import urllib.error
from urllib.parse import urlparse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_spatial_items import (  # noqa: E402
    DEFAULT_ITEMS,
    load_items,
    render_prompt,
    score_reply,
    summarize,
)


DEFAULT_BASE_URL = "http://127.0.0.1:18000/v1"
DEFAULT_MODEL = "/data/hf_models/Qwen/Qwen3.5-35B-A3B"


def _completion_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def _ask(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    api_key: str | None,
) -> tuple[str, dict[str, Any]]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    request = urllib.request.Request(
        _completion_url(base_url),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="POST",
    )
    opener = urllib.request.build_opener()
    # The server commonly has a global HTTP proxy configured.  A local model
    # endpoint must bypass it, otherwise localhost requests can be redirected
    # to an unrelated proxy process.
    if urlparse(request.full_url).hostname in {"127.0.0.1", "localhost", "::1"}:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        detail = ""
        if isinstance(exc, urllib.error.HTTPError):
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)
        raise RuntimeError(f"chat completion request failed: {exc}; {detail[:2000]}") from exc
    try:
        message = payload["choices"][0]["message"]
        content = message.get("content", "") if isinstance(message, dict) else message
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected chat completion response: {payload}") from exc
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return str(content or ""), payload.get("usage", {})


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-path", default=str(DEFAULT_ITEMS))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", default="runs/q9_codex_mcq_api.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.max_tokens <= 0 or args.timeout <= 0 or args.temperature < 0:
        raise SystemExit("max-tokens and timeout must be positive; temperature must be nonnegative")
    items = load_items(args.items_path)
    rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for index, item in enumerate(items, 1):
        started = time.perf_counter()
        raw, usage = _ask(
            base_url=args.base_url,
            model=args.model,
            prompt=render_prompt(item),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            api_key=args.api_key,
        )
        rows.append({
            "item_id": item["item_id"],
            "axis": item.get("axis"),
            "raw_answer": raw,
            **score_reply(raw, item),
            "gold": item["answer"],
            "usage": usage,
            "elapsed_sec": time.perf_counter() - started,
        })
        print(f"[{index:>2}/{len(items)}] {item['item_id']} -> {raw!r}", flush=True)

    payload = {
        "schema": "ttt_frame.api_mcq_eval/1",
        "items_path": str(Path(args.items_path).resolve()),
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "n_items": len(items),
        "rows": rows,
        "summary": summarize(rows),
        "elapsed_sec": time.perf_counter() - started_all,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = payload["summary"]
    print(f"\n{args.model}: {summary['correct']}/{summary['n']} ({summary['accuracy']:.3f})")
    print(f"by axis: {summary['by_axis']}")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
