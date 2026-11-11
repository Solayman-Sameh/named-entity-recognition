"""Fresh isolated process only: scheduler sets CUDA_VISIBLE_DEVICES before imports."""
import os,sys,json,time,platform,random,shutil,subprocess
from pathlib import Path
os.environ.setdefault('TOKENIZERS_PARALLELISM','false');os.environ.setdefault('WANDB_DISABLED','true')
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import *
from models import *

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def device_setup():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    n=torch.cuda.device_count(); visible=os.environ.get('CUDA_VISIBLE_DEVICES','unset')
    assert n<=1, f'Worker isolation failed: sees {n} GPUs; CVD={visible}'
    device=torch.device('cuda:0' if n else 'cpu')
    print(json.dumps({'pid':os.getpid(),'CUDA_VISIBLE_DEVICES':visible,'visible_cuda_devices':n,'device':str(device),
                      'torch':torch.__version__,'cuda_build':torch.version.cuda,'gpu_name':torch.cuda.get_device_name(0) if n else None}),flush=True)
    return device

def move(batch,device):return {k:v.to(device) for k,v in batch.items()}

def recurrent_predictions(model,rows,vocab,device,batch_size=32):
    loader=DataLoader(rows,batch_size=batch_size,shuffle=False,collate_fn=lambda r:collate_recurrent(r,vocab),num_workers=0)
    out=[];losses=[];model.eval();start=time.perf_counter()
    if device.type=='cuda':torch.cuda.synchronize()
    with torch.no_grad():
        for b in loader:
            b=move(b,device);e=model(b['words'],b['chars'],b['mask']);losses.append(float(model.loss(e,b['tags'],b['mask'])))
            out.extend([[vocab['labels'][t] for t in row] for row in model.decode(e,b['mask'])])
    if device.type=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    assert len(out)==len(rows) and all(len(a)==len(r['tokens']) for a,r in zip(out,rows))
    return out,elapsed,float(np.mean(losses))

def base_rows(root,mode):
    train=read_json(root/'prepared/train.json');val=read_json(root/'prepared/validation.json')
    if mode=='smoke':train=train[:32];val=val[:16]
    return train,val

def recurrent(root,model_name,mode,out,device):
    vocab=read_json(root/'prepared/vocab.json');train,val=base_rows(root,mode)
    cfg={'model':model_name,'n_words':len(vocab['words']),'n_chars':len(vocab['chars']),'n_labels':len(vocab['labels'])}
    model=RecurrentNER(cfg,np.load(root/'prepared/embeddings.npy')).to(device)
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    optim=torch.optim.Adam(model.parameters(),lr=.001)
    use_amp=device.type=='cuda';scaler=torch.amp.GradScaler('cuda',enabled=use_amp)
    loader=DataLoader(train,batch_size=32,shuffle=True,num_workers=0,collate_fn=lambda r:collate_recurrent(r,vocab),generator=torch.Generator().manual_seed(42))
    best=-1.;bad=0;history=[];gradient_evidence={};start=time.perf_counter();best_epoch=None
    for epoch in range(1,(1 if mode=='smoke' else 15)+1):
        model.train();loss_sum=0.;count=0
        for step,b in enumerate(loader):
            b=move(b,device);optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp):
                e=model(b['words'],b['chars'],b['mask']);loss=model.loss(e,b['tags'],b['mask'])
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(optim)
            if not gradient_evidence:
                gradient_evidence={name:float(p.grad.detach().float().norm()) for name,p in model.named_parameters() if p.grad is not None}
                assert gradient_evidence['encoder.weight_ih_l0']>0 and gradient_evidence['char.weight']>0
                if model.crf:assert gradient_evidence['crf.transitions']>0
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(optim);scaler.update()
            with torch.no_grad():model.word.weight[0].zero_();model.char.weight[0].zero_()
            loss_sum+=float(loss.detach())*len(b['words']);count+=len(b['words'])
            if step%100==0:print(f'{model_name} epoch {epoch} step {step}/{len(loader)} loss={float(loss.detach()):.4f}',flush=True)
        pred,latency,val_loss=recurrent_predictions(model,val,vocab,device)
        score=metrics([r['tags'] for r in val],pred)
        history.append({'epoch':epoch,'train_loss':loss_sum/count,'val_loss':val_loss,'val_f1':score['f1'],'validation_seconds':latency})
        print(model_name,json.dumps(history[-1]),flush=True)
        if score['f1']>best:
            best=score['f1'];bad=0;best_epoch=epoch
            torch.save({'state':{k:v.cpu() for k,v in model.state_dict().items()},'config':cfg},out/'best.pt')
        else:bad+=1
        if bad>=3:break
    train_seconds=time.perf_counter()-start
    model=load_recurrent(out/'best.pt',device)
    pred,latency,val_loss=recurrent_predictions(model,val,vocab,device)
    reloaded=load_recurrent(out/'best.pt',device)
    pred2,_,_=recurrent_predictions(reloaded,val[:2],vocab,device);assert pred2==pred[:2]
    write_json(out/'validation_predictions.json',pred)
    return {'parameters':params,'trainable_parameters':params,'history':history,'best_epoch':best_epoch,
            'validation':metrics([r['tags'] for r in val],pred),'validation_seconds':latency,
            'train_seconds':train_seconds,'gradient_evidence':gradient_evidence,'reload_exact':True,
            'mixed_precision':'fp16 autocast + scaler; CRF sequence computations float32' if use_amp else 'float32 CPU'}

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self,features):self.features=features
    def __len__(self):return len(self.features)
    def __getitem__(self,i):return self.features[i]

def transformer_predict(model,tok,records,labels,length,device,batch_size=16):
    from transformers import DataCollatorForTokenClassification
    feats,mapping=aligned_features(records,tok,labels,length)
    loader=DataLoader(FeatureDataset(feats),batch_size=batch_size,collate_fn=DataCollatorForTokenClassification(tok),shuffle=False,num_workers=0)
    chunks=[];model.eval();start=time.perf_counter()
    if device.type=='cuda':torch.cuda.synchronize()
    with torch.no_grad():
        for b in loader:
            b={k:v.to(device) for k,v in b.items() if k!='labels'}
            logits=model(**b).logits;chunks.extend(logits.argmax(-1).cpu().tolist())
    if device.type=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    return reconstruct(chunks,mapping,records,labels),elapsed

def transformer(root,mode,out,device):
    from transformers import (AutoTokenizer,AutoModelForTokenClassification,Trainer,TrainingArguments,
        DataCollatorForTokenClassification,EarlyStoppingCallback,TrainerCallback)
    from prepare import fetch
    lock=read_json(Path(__file__).with_name('source_lock.json'))
    info=lock['files']['distilbert/model.safetensors']
    fetch(info['url'],root/'assets/distilbert/model.safetensors',info['sha256'])
    tok=AutoTokenizer.from_pretrained(root/'assets/distilbert',use_fast=True,local_files_only=True)
    vocab=read_json(root/'prepared/vocab.json');labels=vocab['labels'];length=read_json(root/'prepared/length_policy.json')['max_subwords']
    train,val=base_rows(root,mode)
    ft,mt=aligned_features(train,tok,labels,length);fv,mv=aligned_features(val,tok,labels,length)
    model=AutoModelForTokenClassification.from_pretrained(root/'assets/distilbert',local_files_only=True,
        num_labels=len(labels),id2label=dict(enumerate(labels)),label2id={t:i for i,t in enumerate(labels)},attn_implementation='eager')
    params=sum(p.numel() for p in model.parameters());assert all(p.requires_grad for p in model.distilbert.parameters())
    # Observe a real encoder gradient on Trainer's first backward pass; no additional training path.
    gradient={}
    handle=model.distilbert.transformer.layer[0].attention.q_lin.weight.register_hook(lambda g:gradient.update({'encoder_q_gradient_norm':float(g.float().norm())}))
    def compute(p):
        logits=p.predictions[0] if isinstance(p.predictions,tuple) else p.predictions
        pred=reconstruct(np.argmax(logits,axis=-1),mv,val,labels)
        return {'f1':metrics([r['tags'] for r in val],pred)['f1']}
    class FiniteCallback(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            if logs:print('distilbert',json.dumps(logs),flush=True)
    args=TrainingArguments(output_dir=str(out/'trainer'),learning_rate=2e-5,per_device_train_batch_size=16,
        per_device_eval_batch_size=16,gradient_accumulation_steps=1,num_train_epochs=1 if mode=='smoke' else 5,
        weight_decay=.01,warmup_ratio=.1,eval_strategy='epoch',save_strategy='epoch',logging_strategy='steps',logging_steps=50,
        load_best_model_at_end=True,metric_for_best_model='f1',greater_is_better=True,save_total_limit=1,
        save_only_model=True,fp16=device.type=='cuda',bf16=False,max_grad_norm=1.,report_to=[],seed=42,data_seed=42,
        dataloader_num_workers=0,dataloader_pin_memory=device.type=='cuda',use_cpu=device.type=='cpu',disable_tqdm=False)
    assert args.n_gpu<=1,'Trainer must see exactly its assigned one GPU'
    trainer=Trainer(model=model,args=args,train_dataset=FeatureDataset(ft),eval_dataset=FeatureDataset(fv),
        processing_class=tok,data_collator=DataCollatorForTokenClassification(tok),compute_metrics=compute,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2),FiniteCallback()])
    start=time.perf_counter();train_output=trainer.train();elapsed=time.perf_counter()-start;handle.remove()
    assert gradient.get('encoder_q_gradient_norm',0)>0,'No actual transformer encoder gradient'
    assert all(np.isfinite(x) for x in gradient.values())
    bestdir=out/'best';trainer.save_model(bestdir);tok.save_pretrained(bestdir)
    pred,latency=transformer_predict(trainer.model,tok,val,labels,length,device)
    reload_model=AutoModelForTokenClassification.from_pretrained(bestdir,local_files_only=True).to(device).eval()
    pred2,_=transformer_predict(reload_model,tok,val[:2],labels,length,device);assert pred2==pred[:2]
    write_json(out/'validation_predictions.json',pred)
    history=[]
    for row in trainer.state.log_history:
        if 'eval_f1' in row:history.append({'epoch':row.get('epoch'),'val_f1':row['eval_f1'],'val_loss':row.get('eval_loss')})
        elif 'loss' in row:history.append({'epoch':row.get('epoch'),'train_loss':row['loss']})
        elif 'train_loss' in row:history.append({'epoch':row.get('epoch'),'train_loss':row['train_loss']})
    write_json(out/'trainer_state_evidence.json',__import__('dataclasses').asdict(trainer.state))
    best_checkpoint=trainer.state.best_model_checkpoint
    shutil.rmtree(out/'trainer',ignore_errors=True) # completed checkpoints only; no optimizer resume claim
    return {'parameters':params,'trainable_parameters':params,'history':history,'best_checkpoint':best_checkpoint,
        'validation':metrics([r['tags'] for r in val],pred),'validation_seconds':latency,'train_seconds':elapsed,
        'gradient_evidence':gradient,'reload_exact':True,'mixed_precision':'fp16 Trainer' if device.type=='cuda' else 'float32 CPU'}

def run(root,model,mode,identity):
    root=Path(root);out=root/'models'/model;out.mkdir(parents=True,exist_ok=True)
    seed_all(42);device=device_setup()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    result=transformer(root,mode,out,device) if model=='distilbert' else recurrent(root,model,mode,out,device)
    result.update({'model':model,'mode':mode,'experiment_id':identity,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'visible_gpus':torch.cuda.device_count(),'device':str(device),'cuda_max_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
        'cuda_max_reserved_bytes':torch.cuda.max_memory_reserved() if device.type=='cuda' else None,
        'checkpoint_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and (p.suffix=='.pt' or p.suffix=='.safetensors')),
        'cpu_peak_rss_bytes':__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss*1024,
        'timing_note':'Concurrent training; validation timing is synchronized, one pass, batch32 RNN/batch16 transformer; not isolated benchmarks',
        'optimizer_resume':False})
    write_json(out/'result.json',result)
    files={str(p.relative_to(out)):file_hash(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='done.json'}
    write_json(out/'done.json',{'experiment_id':identity,'files':files})
    print('COMPLETED',model,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--model',required=True);p.add_argument('--mode',default='full');p.add_argument('--identity',required=True)
    a=p.parse_args();run(a.root,a.model,a.mode,a.identity)
