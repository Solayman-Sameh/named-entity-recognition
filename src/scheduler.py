"""No torch import in parent. GPU visibility is fixed in each subprocess environment."""
import os,sys,json,time,subprocess,signal,shutil
from pathlib import Path
from common import read_json,write_json,digest,file_hash
MODELS=['distilbert','lstm','bilstm','bilstm_crf']
DISTRIBUTED_KEYS=['LOCAL_RANK','RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','ACCELERATE_USE_CPU','ACCELERATE_MIXED_PRECISION']
def worker_env(device):
    env=os.environ.copy()
    for k in DISTRIBUTED_KEYS:env.pop(k,None)
    env.update({'CUDA_VISIBLE_DEVICES':str(device) if device!='cpu' else '',
        'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
        'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1','WANDB_DISABLED':'true','HF_HUB_DISABLE_TELEMETRY':'1'})
    return env

def discover_devices(python=sys.executable):
    cmd=[python,'-c','import torch,json; print(json.dumps({"count":torch.cuda.device_count(),"torch":torch.__version__,"cuda":torch.version.cuda}))']
    env=os.environ.copy()
    for k in DISTRIBUTED_KEYS:env.pop(k,None)
    info=json.loads(subprocess.check_output(cmd,env=env,text=True).strip().splitlines()[-1])
    n=info['count'];existing=os.environ.get('CUDA_VISIBLE_DEVICES')
    devices=existing.split(',')[:n] if existing else [str(i) for i in range(n)]
    return devices[:2] or ['cpu'],info

def completed(root,model,identity):
    p=Path(root)/'models'/model/'done.json'
    if not p.exists():return False
    done=read_json(p)
    if done['experiment_id']!=identity:raise RuntimeError(f'Incompatible completed model {model}; choose a new RUN_NAME, do not mix results')
    for f,h in done['files'].items():
        q=p.parent/f
        if not q.is_file() or file_hash(q)!=h:raise RuntimeError(f'Corrupt completed artifact: {q}')
    return True

def stop_active(active,events,reason):
    for p,job,dev,f in active:
        if p.poll() is None:
            events.append({'event':'cancel','model':job,'device':dev,'time':time.time(),'reason':reason,'pid':p.pid})
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    deadline=time.time()+5
    for p,job,dev,f in active:
        try:p.wait(timeout=max(.01,deadline-time.time()))
        except subprocess.TimeoutExpired:
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            p.wait()
        f.close()

def run_queue(jobs,devices,command,root,env_factory=worker_env,poll=.5):
    """One job per device; first failure/interrupt cancels every active process group."""
    root=Path(root);logs=root/'logs';logs.mkdir(parents=True,exist_ok=True)
    events=[];queue=list(jobs);active=[];last_echo=0;positions={}
    try:
        while queue or active:
            used={d for p,j,d,f in active}
            for dev in devices:
                if queue and dev not in used:
                    job=queue.pop(0);f=open(logs/(job+'.log'),'w')
                    p=subprocess.Popen(command(job,dev),env=env_factory(dev),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    active.append((p,job,dev,f));used.add(dev)
                    events.append({'event':'start','model':job,'device':dev,'pid':p.pid,'time':time.time()})
                    print(f'Started {job} on {dev}; log: {logs/(job+".log")}',flush=True)
            finished=[]
            for item in active:
                p,job,dev,f=item;code=p.poll()
                if code is not None:
                    events.append({'event':'exit','model':job,'device':dev,'pid':p.pid,'code':code,'time':time.time()})
                    if code:
                        print((logs/(job+'.log')).read_text(errors='replace')[-12000:],flush=True)
                        raise RuntimeError(f'{job} exited {code}; all active jobs cancelled. Completed compatible models remain reusable.')
                    f.close();finished.append(item)
            active=[x for x in active if x not in finished]
            if time.time()-last_echo>15:
                for p,j,d,f in active:
                    data=(logs/(j+'.log')).read_text(errors='replace');old=positions.get(j,0)
                    if len(data)>old:print(f'[{j} / {d}] '+data[old:][-2500:],flush=True);positions[j]=len(data)
                last_echo=time.time()
            if queue or active:time.sleep(poll)
    except BaseException as exc:
        stop_active(active,events,type(exc).__name__);raise
    finally:write_json(root/'schedule_events.json',events)
    return events

def experiment_identity(root,mode):
    import importlib.metadata as im
    code={p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*')) if p.is_file() and p.suffix in ('.py','.json','.txt')}
    versions={name:im.version(name) for name in ['torch','transformers','accelerate','seqeval','numpy','pyarrow','tokenizers','safetensors']}
    config={'seed':42,'rnn_epochs':15,'rnn_patience':3,'rnn_batch':32,'rnn_lr':.001,'hidden':128,'char_dim':30,'char_features':50,
            'dropout':.3,'transformer_epochs':5,'transformer_patience':2,'transformer_batch':16,'accumulation':1,'transformer_lr':2e-5,
            'mode':mode,'data':read_json(Path(root)/'prepared/identity.json'),'code':code,'versions':versions}
    return digest(config),config

def train_all(root,mode='full',force_devices=None):
    root=Path(root);devices,hardware=discover_devices()
    if force_devices is not None:devices=force_devices
    identity,config=experiment_identity(root,mode)
    write_json(root/'experiment.json',{'id':identity,'configuration':config,'hardware':hardware,'devices':devices})
    jobs=[m for m in MODELS if not completed(root,m,identity)]
    write_json(root/'reuse_verification.json',{'experiment_id':identity,'reused':[m for m in MODELS if m not in jobs],'scheduled':jobs,'time':time.time()})
    for m in jobs:
        p=root/'models'/m
        # Incomplete models restart from pretrained/initial state, never optimizer-resume.
        if p.exists():shutil.rmtree(p)
    def command(job,dev):return [sys.executable,str(Path(__file__).with_name('worker.py')),'--root',str(root.resolve()),'--model',job,'--mode',mode,'--identity',identity]
    if jobs:run_queue(jobs,devices,command,root)
    else:print('Reused all four completed models after identity and file-hash verification; no optimizer resume.',flush=True)
    for m in MODELS:assert completed(root,m,identity),f'Missing completion: {m}'
    return identity

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--mode',default='full');a=p.parse_args();train_all(a.root,a.mode)
