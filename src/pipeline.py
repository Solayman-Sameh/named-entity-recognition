"""Run all stages in fresh processes; the notebook kernel imports no ML libraries."""
import os,sys,shutil,subprocess,json
from pathlib import Path
from common import *
from scheduler import train_all,discover_devices,worker_env

def stage(source,name,args,env=None):
    env=dict(os.environ if env is None else env);env['MPLBACKEND']='Agg'
    print('\n=== '+name+' ===',flush=True)
    subprocess.run([sys.executable,str(source/name),*args],env=env,check=True)

def preflight_root(root):
    root=Path(root);check=root/'preflight';(check/'prepared').mkdir(parents=True,exist_ok=True)
    for name in ['train.json','validation.json','vocab.json','embeddings.npy','length_policy.json','identity.json']:
        shutil.copy2(root/'prepared'/name,check/'prepared'/name)
    dest=check/'assets'
    if not dest.exists():dest.symlink_to((root/'assets').resolve(),target_is_directory=True)
    return check

def main(root,mode):
    root=Path(root).resolve();root.mkdir(exist_ok=True,parents=True);source=Path(__file__).parent
    stage(source,'prepare.py',['--root',str(root),'--mode',mode])
    stage(source,'verify.py',['--root',str(root)])
    devices,info=discover_devices()
    if mode=='prep':
        print('Preparation complete; no training requested.',flush=True);return
    if mode=='full':
        check=preflight_root(root);train_all(check,'smoke')
        stage(source,'finalize.py',['--root',str(check),'--mode','smoke'],worker_env(devices[0]))
        shutil.copytree(check/'reports',root/'reports/preflight',dirs_exist_ok=True)
        if devices==['cpu']:
            raise RuntimeError('All bounded preflight checks passed on CPU. Full training requires a Kaggle GPU accelerator; select T4x2 or one GPU and Run All. No factory reset/restart needed.')
    train_all(root,mode)
    stage(source,'finalize.py',['--root',str(root),'--mode',mode],worker_env(devices[0]))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--mode',choices=['full','smoke','prep'],default='full');a=p.parse_args();main(a.root,a.mode)
