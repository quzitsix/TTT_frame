#!/usr/bin/env python3
"""Real-prefix finite-response compression and rollout diagnostic (Idea1 teacher).

Stages are separate: teacher/fit/forecast never read answer files. Score reads
answers only after forecasts and predictions have been frozen and hashed.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time, sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ttt_frame.idea4_response_memory import ResponseMemory, initial_teacher, teacher_read, teacher_write, select_probes
from ttt_frame.forecast_residual import RidgeRecentPredictor
from ttt_frame.idea1_mcq_readout import SharedRidgeReadout, pool_forecasts
from scripts.run_idea1_mcq_forecasts import validated_cache, read_feature_protocol
from scripts.run_forecast_residual import read_split

BASE=ROOT/'runs/forecast_bench_v0/idea01_mcq_v1'
SPLIT=ROOT/'runs/forecast_bench_v0/research_day_split.json'
SOURCE=['scripts/run_idea4_response.py','ttt_frame/idea4_response_memory.py',
        'ttt_frame/idea1_mcq_memory.py','ttt_frame/idea1_mcq_readout.py','ttt_frame/forecast_residual.py']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_json(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)
def save_pt(path,data):
    # torch.save preserves backing storage of views; clone recursively to keep
    # one case from serializing an entire GPU/CPU batch repeatedly.
    def compact(x):
        if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
        if isinstance(x,dict):return {k:compact(v) for k,v in x.items()}
        if isinstance(x,list):return [compact(v) for v in x]
        return x
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(compact(data),tmp);tmp.replace(path)
def load(path):return torch.load(path,map_location='cpu',weights_only=True)
def ids_for(split):return [i for role in ['train','validation','test'] for i in split['splits'][role]]
def role_for(split,i):return next(r for r,x in split['splits'].items() if i in x)
def source_record():return {name:sha(ROOT/name) for name in SOURCE}

def prepare(args):
    if args.output.exists():raise ValueError('prepare requires new output directory')
    args.output.mkdir(parents=True)
    split=read_split(SPLIT);baseline=RidgeRecentPredictor.from_dict(load(BASE/'forecasts/recent_predictor.pt')['baseline'])
    if load(BASE/'forecasts/recent_predictor.pt')['train_ids']!=split['splits']['train']:raise ValueError('baseline train mismatch')
    feature_record=read_feature_protocol(BASE/'features',split);candidates=[];identity={}
    for sid in ids_for(split):
        cache,item=validated_cache(BASE/'features'/f'{sid}.pt',sid,split,feature_record);identity[sid]=item
        if sid in split['splits']['train']:
            selected=torch.linspace(0,3659,16).round().long()
            candidates.append(((cache['features'][selected]-baseline.mean)/baseline.std).flatten(0,1))
    projection=torch.randn(256,32,generator=torch.Generator().manual_seed(4))/math.sqrt(256)
    probes=select_probes(torch.cat(candidates),128,projection)
    save_pt(args.output/'probes.pt',{'probes':probes,'projection':projection,'train_ids':split['splits']['train']})
    save_json(args.output/'protocol.json',{'schema':'idea04_response_diagnostic_v1','split':split,'sources':source_record(),
      'baseline_sha256':sha(BASE/'forecasts/recent_predictor.pt'),'readout_sha256':sha(BASE/'mcq/shared_readout.pt'),
      'text_cache_sha256':sha(BASE/'text_features.pt'),'probes_sha256':sha(args.output/'probes.pt'),
      'feature_protocol':feature_record,'cache_identity':identity,'teacher':'Idea1 residual MLP H128 seed0 lr.05; four horizons and eight tokens',
      'probe_counts':[32,64,128],'rollout_steps':20,'ridge':.05,'train_seed':0,'fit_steps':args.fit_steps,
      'training':'train-only teacher response targets; first half two-step and second half twenty-step rollout; no action labels',
      'query_probe':'final visible spatial tokens held out from response training, used only for measurement and validation selection',
      'control':'response copy, oracle exact probes, fixed RBF, state-dependent bandwidth/gain, recent-only reset',
      'write_approximation':'eight-token mean simultaneous update; original card sequential token updates not implemented',
      'future_media_used':False,'answer_keys_used':False,'full_official_spatial':False,'device':args.device,
      'torch':str(torch.__version__),'git_revision':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    print('prepared',len(identity),'cases',flush=True)

def context(args):
    protocol=json.loads((args.output/'protocol.json').read_text()); split=protocol['split']
    if protocol['sources']!=source_record():raise ValueError('source changed after protocol freeze')
    for p,key in [(BASE/'forecasts/recent_predictor.pt','baseline_sha256'),(BASE/'mcq/shared_readout.pt','readout_sha256'),
                  (BASE/'text_features.pt','text_cache_sha256'),(args.output/'probes.pt','probes_sha256')]:
        if sha(p)!=protocol[key]:raise ValueError('input artifact changed')
    baseline=RidgeRecentPredictor.from_dict(load(BASE/'forecasts/recent_predictor.pt')['baseline'])
    probes=load(args.output/'probes.pt');return protocol,split,baseline,probes

def sequences(ids,protocol,baseline,device):
    values=[]
    for sid in ids:
        cache,item=validated_cache(BASE/'features'/f'{sid}.pt',sid,protocol['split'],protocol['feature_protocol'])
        if item!=protocol['cache_identity'][sid]:raise ValueError('feature cache changed')
        values.append(cache['features'])
    return (torch.stack(values).to(device)-baseline.mean.to(device))/baseline.std.to(device)

def recent_all(z,baseline):
    # Explicit four-frame mean matches the existing teacher's mean reduction.
    means=[]
    for t in range(z.shape[1]):means.append(z[:,max(0,t-baseline.window_steps+1):t+1].mean(1))
    x=torch.cat((z,torch.stack(means,1),torch.ones_like(z[...,:1])),-1)
    return torch.einsum('ntpf,hfd->nthpd',x,baseline.coefficients.to(z.device))

@torch.no_grad()
def teacher(args):
    protocol,split,baseline,pdata=context(args);p=pdata['probes'].to(args.device)
    dest=args.output/'trajectories';dest.mkdir(exist_ok=True);ids=ids_for(split);started=time.perf_counter()
    for offset in range(0,len(ids),args.batch):
        batch_ids=ids[offset:offset+args.batch]
        if all((dest/f'{i}.pt').exists() for i in batch_ids):continue
        z=sequences(batch_ids,protocol,baseline,args.device);b=recent_all(z,baseline); n,t,pn,d=z.shape;h=baseline.horizons
        wi,wo=initial_teacher(n,len(h),d,128,args.device);hs=torch.tensor(h,device=args.device);hi=torch.arange(len(h),device=args.device)
        responses=[];queries=[];tailkeys=[];tailtargets=[];q=z[:,-1].unsqueeze(1)
        for step in range(t):
            origin=(step-hs).clamp_min(0);keys=z[:,origin];targets=z[:,step].unsqueeze(1)-b[:,origin,hi]
            if step==t-20:responses.append(teacher_read(wi,wo,p));queries.append(teacher_read(wi,wo,q))
            teacher_write(wi,wo,keys,targets,(hs<=step).reshape(1,len(h),1,1))
            if step>=t-20:
                responses.append(teacher_read(wi,wo,p));queries.append(teacher_read(wi,wo,q));tailkeys.append(keys);tailtargets.append(targets)
        response=torch.stack(responses,1).cpu();query=torch.stack(queries,1).cpu()
        final=(teacher_read(wi,wo,q)+b[:,-1])*baseline.std.to(args.device)+baseline.mean.to(args.device)
        keys=torch.stack(tailkeys,1).cpu(); targets=torch.stack(tailtargets,1).cpu()
        for j,sid in enumerate(batch_ids):
            save_pt(dest/f'{sid}.pt',{'id':sid,'response':response[j],'query_response':query[j],
              'keys':keys[j],'targets':targets[j],'query':q[j,0].cpu(),'baseline':b[j,-1].cpu(),
              'teacher_forecast':final[j].cpu(),'step_indices':list(range(t-20,t)),
              'warmstart_step':t-21,'protocol_sha256':sha(args.output/'protocol.json')})
        print(json.dumps({'teacher_cases':offset+len(batch_ids),'seconds':time.perf_counter()-started}),flush=True)
    save_json(args.output/'teacher_completion.json',{'complete':True,'cases':len(ids),'seconds':time.perf_counter()-started,
       'files':{sid:sha(dest/f'{sid}.pt') for sid in ids}})

def collect(args,ids,device):
    completion=json.loads((args.output/'teacher_completion.json').read_text())
    if not completion['complete']:raise ValueError('teacher trajectories incomplete')
    rows=[];protocol_sha=sha(args.output/'protocol.json')
    for sid in ids:
        path=args.output/'trajectories'/f'{sid}.pt'
        if sha(path)!=completion['files'][sid]:raise ValueError('teacher trajectory hash changed')
        row=load(path)
        if row['id']!=sid or row['protocol_sha256']!=protocol_sha:raise ValueError('teacher trajectory identity mismatch')
        rows.append(row)
    return {key:torch.stack([r[key] for r in rows]).to(device) for key in ['response','query_response','keys','targets','query','baseline','teacher_forecast']}

@torch.no_grad()
def trajectory_metrics(model,data):
    m=len(model.probes);s=data['response'][:,0,:,:m].clone();out={}
    q=data['query'].unsqueeze(1)
    for step in range(20):
        s=model(s,data['keys'][:,step],data['targets'][:,step])
        if step in [0,1,19]:
            target=data['response'][:,step+1,:,:m];truth=data['query_response'][:,step+1]
            oracle=model.read(target,q);pred=model.read(s,q);copy=model.read(data['response'][:,0,:,:m],q)
            out[str(step+1)]={
             'response_mse':float((s-target).square().mean()),'query_mse':float((pred-truth).square().mean()),
             'oracle_query_mse':float((oracle-truth).square().mean()),'copy_query_mse':float((copy-truth).square().mean()),
             'probe_consistency_mse':float((model.read(s,model.probes)-s).square().mean()),
             'query_signal_mse':float(truth.square().mean())}
    return out

def fit(args):
    protocol,split,baseline,pdata=context(args);train=collect(args,split['splits']['train'],args.device);val=collect(args,split['splits']['validation'],args.device)
    dest=args.output/'models';dest.mkdir(exist_ok=True);records=[];started=time.perf_counter()
    for m in protocol['probe_counts']:
      for conditioned in [False,True]:
        name=f'{"conditioned" if conditioned else "fixed"}_m{m}'
        model=ResponseMemory(pdata['probes'][:m].to(args.device),pdata['projection'].to(args.device),conditioned=conditioned)
        opt=torch.optim.Adam(model.parameters(),lr=.03);g=torch.Generator().manual_seed(0); losses=[]
        for iteration in range(protocol['fit_steps']):
            idx=torch.randint(len(split['splits']['train']),(4,),generator=g).to(args.device)
            state=train['response'][idx,0,:,:m].clone();loss=0.;steps=2 if iteration<protocol['fit_steps']//2 else 20
            for step in range(steps):
                state=model(state,train['keys'][idx,step],train['targets'][idx,step])
                target=train['response'][idx,step+1,:,:m]
                loss=loss+(state-target).square().mean()
            loss=loss/steps;opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
            losses.append(float(loss.detach()))
        metrics=trajectory_metrics(model,val)
        payload={'config':model.config(),'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'losses':losses,'validation':metrics,
                 'train_ids':split['splits']['train'],'protocol_sha256':sha(args.output/'protocol.json')}
        save_pt(dest/f'{name}.pt',payload);records.append({'name':name,'config':model.config(),'validation':metrics,'sha256':sha(dest/f'{name}.pt')})
        print(json.dumps(records[-1]),flush=True)
    selected={kind:min([r for r in records if r['name'].startswith(kind)],key=lambda r:(r['validation']['20']['query_mse'],r['config']['probes']))['name'] for kind in ['fixed','conditioned']}
    save_json(args.output/'selection.json',{'rule':'lowest validation twenty-write unseen visual query MSE; ties smaller probe count; no MCQ labels',
      'selected':selected,'records':records,'fit_seconds':time.perf_counter()-started,'answer_files_read':False})

def get_model(args,name,pdata):
    path=args.output/'models'/f'{name}.pt';selection=json.loads((args.output/'selection.json').read_text())
    expected=next(r['sha256'] for r in selection['records'] if r['name']==name)
    if sha(path)!=expected:raise ValueError('selected model hash changed')
    obj=load(path)
    if obj['protocol_sha256']!=sha(args.output/'protocol.json'):raise ValueError('model protocol mismatch')
    c=obj['config'];model=ResponseMemory(pdata['probes'][:c['probes']].to(args.device),pdata['projection'].to(args.device),conditioned=c['conditioned'],ridge=c['ridge'])
    model.load_state_dict(obj['state_dict']);return model.eval()

@torch.no_grad()
def forecast(args):
    protocol,split,baseline,pdata=context(args);selection=json.loads((args.output/'selection.json').read_text());ids=ids_for(split)
    models={k:get_model(args,name,pdata) for k,name in selection['selected'].items()};readout=SharedRidgeReadout.from_dict(load(BASE/'mcq/shared_readout.pt'));text=load(BASE/'text_features.pt')['records']
    rows=[];dest=args.output/'forecasts';dest.mkdir(exist_ok=True);start=time.perf_counter();diagnostics={}
    for offset in range(0,len(ids),args.batch):
        batch_ids=ids[offset:offset+args.batch];data=collect(args,batch_ids,args.device)
        z=sequences(batch_ids,protocol,baseline,args.device);b=recent_all(z,baseline);n,t,pn,d=z.shape;hs=torch.tensor(baseline.horizons,device=args.device);hi=torch.arange(len(hs),device=args.device)
        states={k:torch.zeros(n,len(hs),len(model.probes),d,device=args.device) for k,model in models.items()}
        # Full-prefix students start from the exact teacher initialization response, zero.
        for step in range(t):
            origin=(step-hs).clamp_min(0);keys=z[:,origin];target=z[:,step].unsqueeze(1)-b[:,origin,hi];active=(hs<=step).reshape(1,len(hs),1,1)
            for k,model in models.items():states[k]=model(states[k],keys,target,active)
        forecasts={'teacher':data['teacher_forecast'],'recent_only':data['baseline']*baseline.std.to(args.device)+baseline.mean.to(args.device)}
        for k,model in models.items():
            m=len(model.probes);warm=data['response'][:,0,:,:m].clone()
            for step in range(20):warm=model(warm,data['keys'][:,step],data['targets'][:,step])
            for arm,state in [(f'{k}_full',states[k]),(f'{k}_warm20',warm),(f'{k}_copy',data['response'][:,0,:,:m]),(f'{k}_oracle',data['response'][:,-1,:,:m])]:
                forecasts[arm]=(model.read(state,data['query'].unsqueeze(1))+data['baseline'])*baseline.std.to(args.device)+baseline.mean.to(args.device)
        if not all(torch.isfinite(v).all() for v in forecasts.values()):raise RuntimeError('nonfinite forecast')
        for j,sid in enumerate(batch_ids):
            single={k:v[j].cpu() for k,v in forecasts.items()};save_pt(dest/f'{sid}.pt',single)
            arms=list(single);features=pool_forecasts(torch.stack(list(single.values()))).unsqueeze(0);scores=readout.predict_scores(features,text[sid]['embeddings'].unsqueeze(0))[0]
            row={'id':sid,'research_split':role_for(split,sid),'predictions':{k:text[sid]['labels'][int(scores[j].argmax())] for j,k in enumerate(arms)},'scores':{k:scores[j].tolist() for j,k in enumerate(arms)}}
            rows.append(row)
        print(json.dumps({'forecast_cases':offset+len(batch_ids),'seconds':time.perf_counter()-start}),flush=True)
    # Report every trained capacity's diagnostic by split, not only selected ones.
    for r in selection['records']:
        model=get_model(args,r['name'],pdata);diagnostics[r['name']]={}
        for role in ['train','validation','test']:
            diagnostics[r['name']][role]=trajectory_metrics(model,collect(args,split['splits'][role],args.device))
    pred=args.output/'predictions.jsonl';pred.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    save_json(args.output/'diagnostics.json',diagnostics)
    save_json(args.output/'forecast_completion.json',{'complete':True,'cases':len(rows),'predictions_sha256':sha(pred),'selection_sha256':sha(args.output/'selection.json'),
      'seconds':time.perf_counter()-start,'answer_files_read':False,'forecast_files':{i:sha(dest/f'{i}.pt') for i in ids}})

def score(args):
    protocol,split,baseline,pdata=context(args);complete=json.loads((args.output/'forecast_completion.json').read_text())
    if not complete['complete'] or complete['predictions_sha256']!=sha(args.output/'predictions.jsonl') or complete['selection_sha256']!=sha(args.output/'selection.json'):raise ValueError('frozen predictions changed')
    rows=[json.loads(x) for x in (args.output/'predictions.jsonl').read_text().splitlines()];labels={x['id']:x['correct_answer'] for x in map(json.loads,(Path(split['root'])/'answers.jsonl').read_text().splitlines())}
    for r in rows:r['correct']={k:v==labels[r['id']] for k,v in r['predictions'].items()}
    metrics={arm:{role:{'n':len(split['splits'][role]),'correct':sum(r['correct'][arm] for r in rows if r['research_split']==role)} for role in ['train','validation','test']} for arm in rows[0]['predictions']}
    pairs={}
    for arm in metrics:
      pairs[arm]={}
      for base in ['teacher','recent_only']:
        pairs[arm][base]={}
        for role in ['train','validation','test']:
          subset=[r for r in rows if r['research_split']==role];g=[r['id'] for r in subset if r['correct'][arm] and not r['correct'][base]];l=[r['id'] for r in subset if not r['correct'][arm] and r['correct'][base]]
          pairs[arm][base][role]={'gains':len(g),'losses':len(l),'gain_ids':g,'loss_ids':l}
    (args.output/'per_case.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    save_json(args.output/'summary.json',{'metrics':metrics,'paired':pairs,'selection':json.loads((args.output/'selection.json').read_text())['selected'],
        'test_status':'exploratory reused silver labels; same day split; no test model selection','predictions_sha256':complete['predictions_sha256']})
    print(json.dumps(metrics,indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['prepare','teacher','fit','forecast','score'],required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cuda:2');parser.add_argument('--batch',type=int,default=16);parser.add_argument('--fit-steps',type=int,default=80)
    args=parser.parse_args();torch.set_num_threads(2);torch.manual_seed(0)
    if args.batch<1 or args.fit_steps<2:parser.error('positive batch and at least two fit steps required')
    globals()[args.stage](args)
if __name__=='__main__':main()
