#!/usr/bin/env python3
"""Answer every public bench_v0 MCQ with frozen Qwen and Spatial-TTT controls.

Prepare one protocol before launching workers. Inference never opens answers,
metadata, annotations or future video; score the completed JSONL separately.
The nine arms share a constrained single-token A--F readout and RGB sampling.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ttt_frame.forecast_bench import load_forecast_cases
from ttt_frame.forecast_mcq import (
    ARMS, OFFICIAL_ARMS, QWEN_ARMS, build_mcq_prompt,
    load_frame_bundle, write_chunks,
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_hashes() -> dict:
    names = ['scripts/evaluate_forecast_mcq.py', 'ttt_frame/forecast_mcq.py',
             'ttt_frame/forecast_bench.py', 'ttt_frame/forecast_features.py',
             'ttt_frame/spatial_videoqa.py', 'ttt_frame/spatial_model.py',
             'ttt_frame/spatial_memory.py', 'scripts/evaluate_fho_sta.py']
    if (ROOT / 'ttt_frame/vision_attention.py').exists():
        names.append('ttt_frame/vision_attention.py')
    if (ROOT / 'ttt_frame/vision_patch_embed.py').exists():
        names.append('ttt_frame/vision_patch_embed.py')
    return {name: file_hash(ROOT / name) for name in names}


def prepare_protocol(args) -> dict:
    cases = load_forecast_cases(args.root)
    if args.protocol.exists():
        raise ValueError('protocol already exists; choose a new path')
    model_path = args.model_path.resolve()
    artifacts = {p.name: file_hash(p) for p in sorted(model_path.iterdir())
                 if p.is_file() and p.suffix in {'.json', '.safetensors', '.txt', '.jinja'}}
    if not any(name.endswith('.safetensors') for name in artifacts):
        raise ValueError('no local base model weights')
    # Validate sampling before accepting the protocol, without decoding video.
    load_frame_bundle(cases[0], context_frames=args.context_frames,
                      recent_frames=args.recent_frames, max_side=args.max_side, decode=False)
    write_chunks(list(range(args.context_frames)), args.write_chunk_frames)
    payload = {
        'schema': 'forecast_mcq_protocol_v1',
        'root': str(args.root.resolve()),
        'inputs_sha256': file_hash(args.root / 'inputs_mcq.jsonl'),
        'case_ids': [c.id for c in cases], 'arms': list(ARMS),
        'model_path': str(model_path), 'base_artifact_sha256': artifacts,
        'spatial_checkpoint': str(args.spatial_checkpoint.resolve()),
        'spatial_checkpoint_sha256': file_hash(args.spatial_checkpoint),
        'source_hashes': source_hashes(),
        'sampling': {'context_frames': args.context_frames, 'recent_frames': args.recent_frames,
                     'max_side': args.max_side, 'write_chunk_frames': args.write_chunk_frames,
                     'policy': 'uniform_left_edges_separately_in_context_and_recent',
                     'audio_used': False},
        'dtype': 'bfloat16', 'seed': 0,
        'readout': 'argmax_next_token_logit_over_single_token_labels_A_to_F',
        'unrestricted_top_token_recorded': True,
        'episode_reset': True, 'training_on_benchmark': False,
        'vision_batched_sdpa': args.vision_batched_sdpa,
        'vision_linear_patch': args.vision_linear_patch,
    }
    result = {'payload': payload, 'protocol_fingerprint': fingerprint(payload),
              'created_at': datetime.now(timezone.utc).isoformat(),
              'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    args.protocol.parent.mkdir(parents=True, exist_ok=True)
    args.protocol.write_text(json.dumps(result, indent=2) + '\n')
    return result


def read_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    p = protocol['payload']
    if p.get('schema') != 'forecast_mcq_protocol_v1' or fingerprint(p) != protocol['protocol_fingerprint']:
        raise ValueError('protocol fingerprint mismatch')
    if p['source_hashes'] != source_hashes():
        raise ValueError('source files changed since protocol preparation')
    if p['inputs_sha256'] != file_hash(Path(p['root']) / 'inputs_mcq.jsonl'):
        raise ValueError('input manifest changed since protocol preparation')
    return protocol


class _BaseController:
    def context(self, mode):
        if mode != 'base':
            raise ValueError('original Qwen has no fast memory')
        return nullcontext()


class QwenEngine:
    def __init__(self, model_path, device):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.device, self.dtype = torch.device(device), torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, local_files_only=True, dtype=self.dtype,
            device_map={'': str(self.device)}, attn_implementation='sdpa',
        ).eval().requires_grad_(False)
        self.controller = _BaseController()
        self.position_offset = 0

    def _move(self, batch):
        import torch
        return {k: v.to(self.device, dtype=self.dtype if v.is_floating_point() else v.dtype)
                for k, v in batch.items() if torch.is_tensor(v)}

    def _clear_generation_state(self):
        for module in self.model.modules():
            if 'rope_deltas' in vars(module):
                module.rope_deltas = None

    def reset(self):
        self.position_offset = 0
        self._clear_generation_state()


def label_tokens(tokenizer) -> list[int]:
    tokens = [tokenizer.encode(c, add_special_tokens=False) for c in 'ABCDEF']
    if any(len(ids) != 1 for ids in tokens) or len({ids[0] for ids in tokens}) != 6:
        raise ValueError('constrained readout requires distinct single-token A--F labels')
    return [ids[0] for ids in tokens]


def choose_label(logits, token_ids, tokenizer) -> dict:
    import torch
    scores = logits.float().flatten()
    if not bool(torch.isfinite(scores).all()):
        raise ValueError('nonfinite answer logits')
    selected = scores[token_ids]
    index = int(selected.argmax())
    top_id = int(scores.argmax())
    return {
        'prediction': 'ABCDEF'[index], 'raw_answer': 'ABCDEF'[index], 'status': 'ok',
        'label_logits': selected.tolist(), 'label_probabilities': selected.softmax(-1).tolist(),
        'label_probability_mass': float((selected.logsumexp(-1) - scores.logsumexp(-1)).exp()),
        'unrestricted_top_token_id': top_id,
        'unrestricted_top_token': tokenizer.decode([top_id]),
    }


def answer_mcq(engine, prompt, *, token_ids, images=None, timestamps=None, use_memory=False):
    import torch
    from scripts.evaluate_fho_sta import _video_batch
    engine._clear_generation_state()
    visual = images is not None
    if visual:
        if use_memory:
            raise ValueError('visual ICL is a base-only control')
        batch = _video_batch(engine, images, timestamps, prompt)
    else:
        rendered = engine.processor.apply_chat_template(
            [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}],
            tokenize=False, add_generation_prompt=True,
        )
        batch = engine._move(engine.processor(text=[rendered], return_tensors='pt'))
        length = batch['input_ids'].shape[1]
        positions = torch.arange(length, device=engine.device) + engine.position_offset
        batch['position_ids'] = positions[None, None].expand(3, 1, -1)
    try:
        with torch.no_grad(), engine.controller.context('read' if use_memory else 'base'):
            output = engine.model(**batch, use_cache=False, logits_to_keep=1)
        result = choose_label(output.logits[0, -1], token_ids, engine.processor.tokenizer)
        result['input_tokens'] = int(batch['input_ids'].shape[1])
        result['video_grid_thw'] = batch['video_grid_thw'].tolist() if visual else []
        result['visual_tokens_at_answer'] = int(batch['input_ids'].eq(engine.model.config.video_token_id).sum()) if visual else 0
        result['position_offset'] = engine.position_offset
        return result
    finally:
        engine._clear_generation_state()


def completed_ids(output: Path, expected_arms, selected_ids, protocol_fingerprint) -> set:
    rows = {}
    if not output.exists():
        return set()
    for line in output.read_text().splitlines():
        row = json.loads(line)
        if (row.get('schema') != 'forecast_mcq_prediction_v1'
                or row.get('prediction') not in tuple('ABCDEF')
                or row.get('raw_answer') != row.get('prediction')):
            raise ValueError('invalid prediction row in existing output')
        if row.get('protocol_fingerprint') != protocol_fingerprint or row.get('id') not in selected_ids:
            raise ValueError('existing output belongs to another protocol or shard')
        key = row['id'], row['arm']
        if row['arm'] not in expected_arms or key in rows:
            raise ValueError('duplicate or unexpected arm in existing output')
        rows[key] = row
    ids = {key[0] for key in rows}
    if any({arm for cid, arm in rows if cid == case_id} != set(expected_arms) for case_id in ids):
        raise ValueError('existing output has an incomplete episode; retain it and use a new output')
    if any(row['status'] != 'ok' for row in rows.values()):
        raise ValueError('existing output has errors; retain it and use a new output')
    return ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/share/raw/bench_v0'))
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--model-path', type=Path, default=Path('/data/quzitsix/models/Qwen3-VL-2B-Instruct'))
    parser.add_argument('--spatial-checkpoint', type=Path, default=Path('/data/quzitsix/models/Spatial-TTT-nano/model.safetensors'))
    parser.add_argument('--context-frames', type=int, default=30)
    parser.add_argument('--recent-frames', type=int, default=60)
    parser.add_argument('--max-side', type=int, default=224)
    parser.add_argument('--write-chunk-frames', type=int, default=8)
    parser.add_argument('--vision-batched-sdpa', action='store_true')
    parser.add_argument('--vision-linear-patch', action='store_true')
    parser.add_argument('--family', choices=['qwen', 'spatial'], default='spatial')
    parser.add_argument('--device', default='cuda:4')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards or args.threads < 1:
        parser.error('invalid shard or thread count')
    source_root = args.root.resolve()
    for path in (args.protocol, args.output):
        if path and path.resolve().is_relative_to(source_root):
            parser.error('outputs must be outside source benchmark')
    if args.prepare:
        result = prepare_protocol(args)
        print(json.dumps({'protocol': str(args.protocol), 'fingerprint': result['protocol_fingerprint']}), flush=True)
        return 0
    protocol = read_protocol(args.protocol)
    p = protocol['payload']
    if args.output and args.output.resolve().is_relative_to(Path(p['root']).resolve()):
        parser.error('output must be outside protocol source benchmark')
    cases = load_forecast_cases(p['root'], ids=args.ids)
    cases = [case for i, case in enumerate(cases) if i % args.num_shards == args.shard_index]
    arms = QWEN_ARMS if args.family == 'qwen' else OFFICIAL_ARMS
    if not cases:
        parser.error('empty shard')
    if args.dry_run:
        print(json.dumps({'case_ids': [c.id for c in cases], 'arms': list(arms),
                          'expected_rows': len(cases) * len(arms), 'protocol': protocol}, indent=2))
        return 0
    if args.output is None:
        parser.error('--output is required for inference')
    if args.output.exists() and not args.resume:
        parser.error('output already exists; use --resume for complete successful episodes')
    if args.output.resolve() == args.protocol.resolve():
        parser.error('output cannot replace protocol')
    done = completed_ids(args.output, arms, {c.id for c in cases}, protocol['protocol_fingerprint'])
    cases = [c for c in cases if c.id not in done]
    if not cases:
        print('All selected episodes already complete.', flush=True)
        return 0
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(p['seed'])
    for name, expected in p['base_artifact_sha256'].items():
        if file_hash(Path(p['model_path']) / name) != expected:
            raise ValueError(f'base artifact changed: {name}')
    print(f'Loading {args.family} on {args.device}; {len(cases)} episodes', flush=True)
    if args.family == 'qwen':
        engine = QwenEngine(p['model_path'], args.device)
    else:
        from ttt_frame.spatial_videoqa import SpatialVideoMemory, SpatialVideoConfig
        if file_hash(Path(p['spatial_checkpoint'])) != p['spatial_checkpoint_sha256']:
            raise ValueError('Spatial checkpoint changed')
        engine = SpatialVideoMemory(SpatialVideoConfig(
            model_path=p['model_path'], device=args.device, dtype=p['dtype'],
            spatial_checkpoint=p['spatial_checkpoint'], max_side=p['sampling']['max_side'],
        ))
    if p['vision_batched_sdpa']:
        from ttt_frame.vision_attention import EqualLengthVisionSDPA
        engine.vision_attention_patch = EqualLengthVisionSDPA(engine.model)
    if p['vision_linear_patch']:
        from ttt_frame.vision_patch_embed import LinearVisionPatchEmbed
        engine.vision_patch_embedding = LinearVisionPatchEmbed(engine.model)
    tokens = label_tokens(engine.processor.tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sampling = p['sampling']
    with args.output.open('a' if args.resume else 'x') as destination:
        for count, case in enumerate(cases, 1):
            started = time.perf_counter()
            bundle = load_frame_bundle(case, context_frames=sampling['context_frames'],
                                       recent_frames=sampling['recent_frames'], max_side=sampling['max_side'])
            decode_seconds = time.perf_counter() - started
            prompt = build_mcq_prompt(case)
            print(f'{case.id} decoded {sampling["context_frames"] + sampling["recent_frames"]} frames in {decode_seconds:.2f}s', flush=True)
            results = {}

            def call(arm, **kwargs):
                call_started = time.perf_counter()
                try:
                    result = answer_mcq(engine, prompt, token_ids=tokens, **kwargs)
                except Exception as exc:
                    result = {'prediction': None, 'raw_answer': '', 'status': 'error',
                              'error': f'{type(exc).__name__}: {exc}'}
                result['elapsed_seconds'] = time.perf_counter() - call_started
                results[arm] = result
                print(json.dumps({'id': case.id, 'arm': arm, **result}), flush=True)

            prefix = 'qwen_' if args.family == 'qwen' else ''
            engine.reset()
            call(prefix + 'blind')
            engine.reset()
            call(prefix + 'recent_icl', images=bundle['recent_images'], timestamps=bundle['recent_actual_seconds'])
            engine.reset()
            call(prefix + 'full_icl', images=bundle['context_images'] + bundle['recent_images'],
                 timestamps=bundle['context_actual_seconds'] + bundle['recent_actual_seconds'])
            write_report = None
            if args.family == 'spatial':
                engine.reset()
                write_start = time.perf_counter()
                try:
                    for role in ('context', 'recent'):
                        pairs = list(zip(bundle[f'{role}_images'], bundle[f'{role}_actual_seconds']))
                        for chunk in write_chunks(pairs, sampling['write_chunk_frames']):
                            pictures, times = zip(*chunk)
                            engine.ingest_frames(list(pictures), list(times))
                    write_report = engine.finish_ingest()
                    write_report['elapsed_seconds'] = time.perf_counter() - write_start
                    call('spatial', use_memory=True)
                    call('spatial_base', use_memory=False)
                    engine.discard_fast_state()
                    call('spatial_slow_only', use_memory=True)
                except Exception as exc:
                    for arm in ('spatial', 'spatial_base', 'spatial_slow_only'):
                        results.setdefault(arm, {'prediction': None, 'raw_answer': '', 'status': 'error',
                                                'error': f'{type(exc).__name__}: {exc}',
                                                'elapsed_seconds': time.perf_counter() - write_start})
            audit = {k: v for k, v in bundle.items() if not k.endswith('_images')}
            for arm in arms:
                row = {'schema': 'forecast_mcq_prediction_v1', 'id': case.id, 'arm': arm,
                       'protocol_fingerprint': protocol['protocol_fingerprint'],
                       'protocol_file': str(args.protocol.resolve()), 'device': args.device,
                       'sampling': audit, 'decode_seconds': decode_seconds,
                       'write_report': write_report if arm.startswith('spatial') else None,
                       'episode_elapsed_seconds': time.perf_counter() - started,
                       **results[arm]}
                destination.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            destination.flush()
            print(f'Completed {count}/{len(cases)}: {case.id}, {time.perf_counter()-started:.2f}s', flush=True)
            if any(result['status'] != 'ok' for result in results.values()):
                raise RuntimeError('episode has recorded errors; stopping instead of repeating a failing configuration')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
