#!/usr/bin/env python3
"""Cache independent, frozen Qwen candidate embeddings without opening answers.

Only the public inputs manifest is used. Each question/candidate string is
encoded independently without option letters, indices, other choices, video,
or chat-generation suffixes. The final non-padding token's normalized decoder
hidden state is saved; no LM-head logits or model updates are involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ttt_frame.forecast_bench import load_forecast_cases
from ttt_frame.idea1_mcq_readout import candidate_prompt


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()


def case_text_sha256(case) -> str:
    return canonical_sha256({
        "question": case.question,
        "options": [{"label": option.label, "text": option.text} for option in case.options],
    })


class QwenCandidateEncoder:
    """Load only the original checkpoint's frozen text decoder, including norm."""

    def __init__(self, model_path: Path, device: str, dtype: str):
        from safetensors import safe_open
        from transformers import AutoConfig, AutoTokenizer
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextModel, Qwen3VLTextRotaryEmbedding,
        )

        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        model_path = model_path.resolve()
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        if config.model_type != "qwen3_vl":
            raise ValueError("candidate encoder requires a local Qwen3-VL checkpoint")
        text_config = config.text_config
        text_config._attn_implementation = "sdpa"
        with torch.device("meta"):
            model = Qwen3VLTextModel(text_config)
        # Rotary buffers are nonpersistent, and hence absent from the checkpoint.
        model.rotary_emb = Qwen3VLTextRotaryEmbedding(text_config, device="cpu")
        state, digest = {}, hashlib.sha256()
        config_files = {}
        for filename in ("config.json", "tokenizer.json", "tokenizer_config.json",
                         "vocab.json", "merges.txt"):
            path = model_path / filename
            if path.is_file():
                config_files[filename] = file_sha256(path)
                digest.update(filename.encode())
                digest.update(path.read_bytes())
        prefix = "model.language_model."
        expected = set(model.state_dict())
        for path in sorted(model_path.glob("model*.safetensors")):
            with safe_open(str(path), framework="pt", device="cpu") as source:
                for name in sorted(source.keys()):
                    if not name.startswith(prefix):
                        continue
                    local = name[len(prefix):]
                    if local in state:
                        raise ValueError(f"duplicate text decoder tensor: {local}")
                    raw = source.get_tensor(name)
                    digest.update(name.encode())
                    digest.update(str((tuple(raw.shape), raw.dtype)).encode())
                    digest.update(raw.contiguous().view(torch.uint8).numpy())
                    state[local] = raw.to(dtype=self.dtype)
        if set(state) != expected:
            raise ValueError(f"text checkpoint mismatch: missing={expected - set(state)}, "
                             f"extra={set(state) - expected}")
        model.load_state_dict(state, assign=True, strict=True)
        self.model = model.to(self.device).eval().requires_grad_(False)
        del state
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.fingerprint = {
            "kind": "text_decoder_tensor_content_plus_config_tokenizer_sha256",
            "sha256": digest.hexdigest(), "text_tensors": len(expected),
            "config_tokenizer_sha256": config_files,
        }

    @torch.inference_mode()
    def encode(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if not prompts:
            raise ValueError("cannot encode an empty prompt batch")
        batch = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False,
        ).to(self.device)
        lengths = batch["attention_mask"].sum(dim=1)
        if not bool((lengths > 0).all()):
            raise ValueError("empty tokenized candidate")
        if int(lengths.max()) > self.model.config.max_position_embeddings:
            raise ValueError("candidate exceeds the model context; truncation is forbidden")
        hidden = self.model(**batch, use_cache=False, return_dict=True).last_hidden_state
        embeddings = hidden[torch.arange(len(prompts), device=self.device), lengths - 1]
        if not bool(torch.isfinite(embeddings).all()):
            raise ValueError("nonfinite candidate text embeddings")
        return embeddings.float().cpu(), lengths.cpu()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/share/raw/bench_v0"))
    parser.add_argument("--model-path", type=Path,
                        default=Path("/data/quzitsix/models/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/forecast_bench_v0/idea01_mcq_v1/text_features.pt"))
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.torch_threads < 1:
        raise ValueError("batch size and thread count must be positive")
    torch.set_num_threads(args.torch_threads)
    cases = load_forecast_cases(args.root)
    manifest_hash = file_sha256(args.root / "inputs_mcq.jsonl")
    if args.output.exists():
        existing = torch.load(args.output, map_location="cpu", weights_only=True)
        if (existing.get("schema") != "idea01_candidate_text_features_v1"
                or existing.get("input_manifest_sha256") != manifest_hash
                or set(existing.get("records", {})) != {case.id for case in cases}):
            raise ValueError("existing text cache does not match the complete public manifest")
        if existing.get("protocol_sha256") != canonical_sha256(existing["protocol"]):
            raise ValueError("existing text cache protocol checksum mismatch")
        requested = {"dtype": args.dtype, "batch_size": args.batch_size,
                     "model_path": str(args.model_path.resolve())}
        if any(existing["protocol"].get(key) != value for key, value in requested.items()):
            raise ValueError("existing text cache differs from requested execution settings")
        for case in cases:
            record = existing["records"][case.id]
            if (record.get("text_sha256") != case_text_sha256(case)
                    or record.get("labels") != [option.label for option in case.options]
                    or record.get("embedding_sha256") != tensor_sha256(record["embeddings"])):
                raise ValueError(f"existing text cache is corrupt or stale for {case.id}")
        print(json.dumps({"status": "existing_cache_validated", "cases": len(cases),
                          "output": str(args.output)}), flush=True)
        return 0

    started = time.monotonic()
    print(json.dumps({"status": "loading_text_decoder", "device": args.device}), flush=True)
    encoder = QwenCandidateEncoder(args.model_path, args.device, args.dtype)
    import transformers
    protocol = {
        "model_path": str(args.model_path.resolve()), "model_fingerprint": encoder.fingerprint,
        "input_manifest_sha256": manifest_hash, "dtype": args.dtype, "batch_size": args.batch_size,
        "prompt_template": "问题：{question}\\n候选动作：{candidate}",
        "pooling": "last_nonpadding_token_after_final_decoder_norm",
        "add_special_tokens": False, "chat_template": False, "truncation": False,
        "candidate_order_features": False, "answer_keys_used": False, "videos_used": False,
        "model_training": False, "local_files_only": True, "attention_implementation": "sdpa",
        "source_sha256": {
            name: file_sha256(ROOT / name) for name in (
                "scripts/cache_idea1_text.py", "ttt_frame/idea1_mcq_readout.py",
                "ttt_frame/forecast_bench.py",
            )
        },
        "libraries": {"torch": str(torch.__version__), "transformers": transformers.__version__,
                      "cuda": torch.version.cuda},
    }
    jobs = [(case.id, option.label, candidate_prompt(case.question, option.text))
            for case in cases for option in case.options]
    encodings, counts = [], []
    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start:start + args.batch_size]
        embeddings, lengths = encoder.encode([job[2] for job in batch])
        encodings.append(embeddings)
        counts.append(lengths)
        if start % (10 * args.batch_size) == 0 or start + args.batch_size >= len(jobs):
            print(json.dumps({"status": "encoding", "completed": start + len(batch),
                              "total": len(jobs), "elapsed_seconds": time.monotonic() - started}), flush=True)
    embeddings, lengths = torch.cat(encodings), torch.cat(counts)
    records, offset = {}, 0
    for case in cases:
        size = len(case.options)
        features = embeddings[offset:offset + size].clone()
        records[case.id] = {
            "labels": [option.label for option in case.options], "embeddings": features,
            "token_counts": lengths[offset:offset + size].clone(),
            "text_sha256": case_text_sha256(case), "embedding_sha256": tensor_sha256(features),
            "prompt_sha256": [hashlib.sha256(job[2].encode()).hexdigest()
                              for job in jobs[offset:offset + size]],
        }
        offset += size
    payload = {
        "schema": "idea01_candidate_text_features_v1", "input_manifest_sha256": manifest_hash,
        "protocol": protocol, "protocol_sha256": canonical_sha256(protocol), "records": records,
        "elapsed_seconds": time.monotonic() - started,
    }
    if file_sha256(args.root / "inputs_mcq.jsonl") != manifest_hash:
        raise ValueError("public input manifest changed during encoding")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        # Exclusive install avoids overwriting an independently generated cache.
        os.link(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"status": "complete", "cases": len(cases), "shape": list(embeddings.shape),
                      "elapsed_seconds": payload["elapsed_seconds"],
                      "output": str(args.output), "file_sha256": file_sha256(args.output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
