#!/usr/bin/env python3
"""Extract native spatial tokens and audit Idea2 on fixed bench_v0 day splits."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from ttt_frame.forecast_bench import load_forecast_cases
from ttt_frame.forecast_features import FeatureConfig, causal_video_frames
from ttt_frame.forecast_mcq import frame_bundle_hash
from ttt_frame.idea2_transport_audit import (
    DenseSpatialFeatures, aggregate_pairs, audit_matched_pair, fit_pca, pair_schedule,
    RANK, COSINE_THRESHOLD, MIN_MATCHES, MIN_IMPROVEMENT, REGULARIZATIONS,
)
from ttt_frame.vision_attention import EqualLengthVisionSDPA
from ttt_frame.vision_patch_embed import LinearVisionPatchEmbed

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["ttt_frame/idea2_transport_audit.py", "scripts/run_idea2_transport.py",
           "ttt_frame/key_transport.py", "ttt_frame/forecast_features.py",
           "ttt_frame/forecast_mcq.py", "ttt_frame/forecast_bench.py",
           "ttt_frame/vision_attention.py", "ttt_frame/vision_patch_embed.py"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def make_protocol(args):
    split = json.loads(args.split_manifest.read_text())
    return {"schema":"idea02_transport_real_video_v1", "base_git":subprocess.check_output(["git","rev-parse","HEAD"],text=True,cwd=ROOT).strip(),
            "root":str(args.root.resolve()),"split":split,"split_sha256":digest(args.split_manifest),
            "public_inputs_sha256":digest(args.root / "inputs_mcq.jsonl"),
            "model_path":str(args.model_path.resolve()),"seed":0,"rank":RANK,
            "schedule":pair_schedule(),"max_side":448,"batch_images":24,
            "feature":"native Qwen per-image merged spatial tokens, fixed orthogonal 2048->256 projection, token LayerNorm; train-only PCA16 and unit-norm keys",
            "matching":"256-D cosine mutual nearest neighbors; no object identity labels",
            "cosine_threshold":COSINE_THRESHOLD,"minimum_matches":MIN_MATCHES,
            "fit_gate_audit":"deterministic random 1/2 fit,1/4 gate,remaining independent audit; fit rank ratio >=1e-4",
            "regularizations":REGULARIZATIONS,"validation_selection":"minimum pooled gate transported RMSE, ties lower regularization; never audit split",
            "acceptance":"perpair gate relative RMSE >=0.15; final test audit separate",
            "go_rule":"test audit candidate and gate-selected RMSE improve >=15%, eligible fraction >=50%, permuted and wrong-video candidate gains <=0; otherwise stop full integration",
            "controls":["identity","matched","permuted_correspondences","wrong_video_mnn"],
            "answer_files_read":False,"future_media_read":False,"audio_used":False,
            "sources":{p:digest(ROOT/p) for p in SOURCES}}


def extract(args):
    protocol_path=args.output_dir / "protocol.json"
    protocol=json.loads(protocol_path.read_text())
    for name, sha in protocol["sources"].items():
        if digest(ROOT/name)!=sha: raise ValueError(f"source changed: {name}")
    cases=load_forecast_cases(args.root)
    cases=[c for i,c in enumerate(cases) if i%args.num_shards==args.shard_index]
    config=FeatureConfig(max_side=448)
    encoder=DenseSpatialFeatures(args.model_path,config,args.device)
    write_json(args.output_dir/f"worker_{args.shard_index}.json",{"model":encoder.model_fingerprint,"device":args.device,"torch":str(torch.__version__),"cases":[c.id for c in cases],"protocol_sha256":digest(protocol_path)})
    start=time.perf_counter()
    with LinearVisionPatchEmbed(encoder.model), EqualLengthVisionSDPA(encoder.model):
        for i,case in enumerate(cases):
            path=args.output_dir/"features"/f"{case.id}.pt"
            if path.exists():
                if not args.resume: raise ValueError(f"refusing existing {path}")
                old=torch.load(path,weights_only=True,map_location="cpu")
                if old["protocol_sha256"]!=digest(protocol_path): raise ValueError("stale cache protocol")
                continue
            before=time.perf_counter(); images=[]; audits=[]
            for role in ("context","recent"):
                pairs=[p for p in pair_schedule() if p["role"]==role]
                times=[p[t] for p in pairs for t in ("old_seconds","new_seconds")]
                media=case.context_path if role=="context" else case.recent_path
                stat=media.stat()
                decoded=list(causal_video_frames(media,times,max_side=448))
                actual=[t for _,t in decoded]; pictures=[x for x,_ in decoded]
                if any(a>t+1e-9 for a,t in zip(actual,times)): raise ValueError("noncausal frame")
                if len(decoded)!=len(times): raise ValueError("missing decoded frame")
                audits.append({"role":role,"path":str(media),"requested_seconds":times,"actual_seconds":actual,
                               "frame_bundle_sha256":frame_bundle_hash(pictures,actual,times),
                               "source_size":stat.st_size,"source_mtime_ns":stat.st_mtime_ns})
                images.extend(pictures)
                if (stat.st_size,stat.st_mtime_ns)!=(media.stat().st_size,media.stat().st_mtime_ns): raise ValueError("media changed")
            decoded_at=time.perf_counter(); features=encoder.encode(images); finished=time.perf_counter()
            cache={"id":case.id,"features":features,"sampling":audits,"protocol_sha256":digest(protocol_path),
                   "decode_seconds":decoded_at-before,"encode_seconds":finished-decoded_at,"elapsed_seconds":finished-before}
            path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(".tmp");torch.save(cache,tmp);tmp.replace(path)
            print(json.dumps({"id":case.id,"case":i+1,"count":len(cases),"tokens":[len(x) for x in features],"seconds":finished-before}),flush=True)
    write_json(args.output_dir/f"worker_{args.shard_index}.completion.json",{"complete":True,"count":len(cases),"seconds":time.perf_counter()-start})


def analyze(args):
    run=args.output_dir;protocol=json.loads((run/"protocol.json").read_text());split=protocol["split"]
    for name,sha in protocol["sources"].items():
        if digest(ROOT/name)!=sha:raise ValueError(f"source changed: {name}")
    allids=[sid for ids in split["splits"].values() for sid in ids]
    caches={sid:torch.load(run/"features"/f"{sid}.pt",map_location="cpu",weights_only=True) for sid in allids}
    if any(c["protocol_sha256"]!=digest(run/"protocol.json") for c in caches.values()):raise ValueError("cache protocol mismatch")
    pca=fit_pca(f for sid in split["splits"]["train"] for f in caches[sid]["features"])
    torch.save(pca,run/"train_pca.pt")
    records=[]
    for reg in REGULARIZATIONS:
        rows=[]
        for sid in split["splits"]["validation"]:
            features=caches[sid]["features"]
            for i in range(12): rows.append(audit_matched_pair(features[2*i],features[2*i+1],pca,key=f"{sid}/{i}",regularization=reg))
        metrics=aggregate_pairs(rows)
        records.append({"regularization":reg,"validation_gate":metrics["gate"],"eligible":metrics["eligible"]})
    eligible=[r for r in records if r["validation_gate"] is not None]
    if not eligible: raise ValueError("no eligible validation pair; cannot select lambda")
    selected=min(eligible,key=lambda r:(r["validation_gate"]["candidate_rmse"],r["regularization"]))["regularization"]
    selection={"regularization":selected,"grid":records,"labels_used":False,"selection_split":"validation gate correspondences only","pca_train_ids":split["splits"]["train"],"pca_sha256":digest(run/"train_pca.pt"),"explained_variance_ratio":pca["explained_variance_ratio"],"token_count":pca["token_count"]}
    write_json(run/"selection.json",selection)
    rows=[];case_rows=[];train_ids=sorted(split["splits"]["train"])
    for research_split,ids in split["splits"].items():
        for sid in ids:
            donor=next(d for d in train_ids if split["id_to_day"][d]!=split["id_to_day"][sid])
            own=caches[sid]["features"];wrong=caches[donor]["features"]; ownrows=[]
            for i,schedule in enumerate(pair_schedule()):
                key=f"{sid}/{i}"
                for arm,new,perm in [("matched",own[2*i+1],False),("permuted",own[2*i+1],True),("wrong_video",wrong[2*i+1],False)]:
                    row={"id":sid,"research_split":research_split,"day":split["id_to_day"][sid],"pair":i,"schedule":schedule,"arm":arm,"donor_id":donor if arm=="wrong_video" else None,
                         **audit_matched_pair(own[2*i],new,pca,key=key,regularization=selected,permuted=perm)}
                    rows.append(row);ownrows.append(row)
            case_rows.append({"id":sid,"research_split":research_split,"day":split["id_to_day"][sid],"sampling":caches[sid]["sampling"],
                              "runtime":{k:caches[sid][k] for k in ["decode_seconds","encode_seconds","elapsed_seconds"]},
                              "arms":{arm:aggregate_pairs([r for r in ownrows if r["arm"]==arm]) for arm in ("matched","permuted","wrong_video")}})
            print(json.dumps({"analyzed":sid,"split":research_split}),flush=True)
    (run/"pairs.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n" for r in rows))
    (run/"per_case.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n" for r in case_rows))
    metrics={s:{arm:aggregate_pairs([r for r in rows if r["research_split"]==s and r["arm"]==arm]) for arm in ("matched","permuted","wrong_video")} for s in split["splits"]}
    test=metrics["test"];matched=test["matched"]
    go=bool(matched["audit"] and matched["eligible_fraction"]>=.5 and matched["audit"]["candidate_relative_improvement"]>=.15 and matched["audit"]["gated_relative_improvement"]>=.15 and all(test[a]["audit"] is not None and test[a]["audit"]["candidate_relative_improvement"]<=0 for a in ("permuted","wrong_video")))
    summary={"schema":"idea02_transport_results_v1","cases":len(caches),"pairs":len(rows),"selection":selection,"metrics":metrics,"go":go,
             "interpretation":"frozen visual feature preflight; no forecast answer produced, no semantic identity labels, no live memory transport",
             "limitations":["overlapping episodes and only two exploratory test days","matching does not verify object identity or separate camera motion from object change","fixed random projection and train PCA are not the proposed learned key encoder","not same input budget as90frame forecasting; this probes12adjacentpairs"],
             "protocol_sha256":digest(run/"protocol.json"),"sources":protocol["sources"],"answer_files_read":False,
             "runtime_seconds":{k:sum(c[k] for c in caches.values()) for k in ("decode_seconds","encode_seconds","elapsed_seconds")}}
    write_json(run/"summary.json",summary)
    print(json.dumps({"go":go,"test":test},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["prepare","extract","analyze"],required=True)
    parser.add_argument("--root",type=Path,default=Path("/data/share/raw/bench_v0"))
    parser.add_argument("--split-manifest",type=Path,default=Path("runs/forecast_bench_v0/research_day_split.json"))
    parser.add_argument("--model-path",type=Path,default=Path("/data/quzitsix/models/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--device",default="cuda:0");parser.add_argument("--num-shards",type=int,default=1);parser.add_argument("--shard-index",type=int,default=0)
    parser.add_argument("--resume",action="store_true");parser.add_argument("--threads",type=int,default=2)
    args=parser.parse_args();torch.set_num_threads(args.threads)
    if args.output_dir.resolve().is_relative_to(args.root.resolve()):parser.error("outputs must be outside benchmark")
    if not 0<=args.shard_index<args.num_shards:parser.error("invalid shard")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.stage=="prepare":
        if (args.output_dir/"protocol.json").exists():raise ValueError("protocol exists")
        write_json(args.output_dir/"protocol.json",make_protocol(args))
    elif args.stage=="extract":extract(args)
    else:analyze(args)


if __name__=="__main__":main()
