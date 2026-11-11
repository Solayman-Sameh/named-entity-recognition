"""Install first, validate in a fresh process. Never restart/kill the notebook kernel."""
import sys,os,subprocess,json,venv,sysconfig,importlib.util,importlib.metadata as im
from pathlib import Path
os.environ['MPLBACKEND']='Agg'

def main(env_path,source):
    env_path=Path(env_path).resolve();source=Path(source).resolve()
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('The host needs working Kaggle CUDA PyTorch. Select a Kaggle GPU accelerator, then Run All. No torch downgrade was attempted.') from exc
    from packaging.version import Version
    if Version(torch.__version__.split('+')[0])<Version('2.4.0'):
        raise RuntimeError(f'Host torch {torch.__version__} is too old; use a current Kaggle image. Host torch was not changed.')
    host={'torch':torch.__version__,'cuda':torch.version.cuda,'python':sys.version,'host_executable':sys.executable}
    marker=env_path/'host_torch.json'
    if marker.exists() and json.loads(marker.read_text())!=host:
        raise RuntimeError('Isolated runtime belongs to a different host Python/PyTorch. Use a new RUNTIME_NAME in the notebook settings.')
    # Avoid ensurepip: host pip seeds an isolated pip-less environment.
    # Also repairs a partially created runtime left by a failed ensurepip attempt.
    if not (env_path/'bin/python').exists():
        venv.EnvBuilder(with_pip=False,system_site_packages=False).create(env_path)
    py=env_path/'bin/python'
    import pip
    if Version(pip.__version__) < Version('22.3'):
        raise RuntimeError('Host pip >=22.3 required for --python targeting. Use a current Kaggle image; host packages were not changed.')
    print('Seeding isolated pip without ensurepip; host packages are preserved.',flush=True)
    subprocess.run([sys.executable,'-m','pip','--python',str(py),'install',
                    '--disable-pip-version-check','pip==25.3','setuptools==80.9.0','wheel==0.45.1','wrapt==1.17.3'],check=True)
    subprocess.run([str(py),'-m','pip','--version'],check=True)
    site=Path(subprocess.check_output([str(py),'-c','import sysconfig;print(sysconfig.get_paths()["purelib"])'],text=True).strip())
    # Expose only host PyTorch/CUDA components, not preloaded vision/audio integrations.
    for name in ['torch','torchgen','functorch','triton','nvidia','cuda']:
        spec=importlib.util.find_spec(name)
        if spec is None:continue
        paths=list(spec.submodule_search_locations or [])
        if paths:
            original=Path(paths[0]);target=site/name
            if not target.exists():target.symlink_to(original,target_is_directory=True)
    selected=[]
    for dist in im.distributions():
        name=dist.metadata['Name'].lower().replace('_','-')
        if name in {'torch','triton'} or name.startswith(('nvidia-','cuda-','pytorch-triton')):
            origin=Path(dist._path);target=site/origin.name
            if not target.exists():target.symlink_to(origin,target_is_directory=True)
            selected.append(name)
    print('Installing isolated text runtime; preserving host CUDA PyTorch',host,flush=True)
    subprocess.run([str(py),'-m','pip','install','--disable-pip-version-check','--quiet','-r',str(source/'requirements.txt')],check=True)
    env=os.environ.copy();env.update({'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'})
    code="""import torch,transformers,accelerate,seqeval,pyarrow,gradio,json
from transformers import Trainer,AutoTokenizer,AutoModelForTokenClassification,TrainingArguments
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'transformers':transformers.__version__,'accelerate':accelerate.__version__,'gradio':gradio.__version__,'cuda_devices':torch.cuda.device_count()}))"""
    result=subprocess.check_output([str(py),'-c',code],env=env,text=True)
    print(result,flush=True);got=json.loads(result.strip().splitlines()[-1]);assert got['torch']==host['torch'] and got['cuda']==host['cuda']
    subprocess.run([str(py),'-m','pip','check'],check=True)
    marker.write_text(json.dumps(host,indent=2));print('Runtime ready. No kernel restart required.',flush=True)
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
