"""Bounded correctness tests. No real test split is loaded in this module."""
import os,sys,time,itertools,tempfile,subprocess,json,math
from pathlib import Path
import numpy as np
import torch
from common import *
from models import *

def run_tests(root):
    from transformers import AutoTokenizer,DataCollatorForTokenClassification,Trainer,TrainingArguments
    from seqeval.metrics import f1_score
    from seqeval.scheme import IOB2
    root=Path(root);torch.set_num_threads(2);torch.manual_seed(7);evidence={}
    labels=read_json(root/'prepared/vocab.json')['labels'];vocab=read_json(root/'prepared/vocab.json')
    assert normalize_iob(['I-PER','I-PER','I-ORG','B-ORG','O','I-LOC'])[0]==['B-PER','I-PER','B-ORG','B-ORG','O','B-LOC']
    assert len(normalize_iob(['I-PER','I-PER','I-ORG','B-ORG','O','I-LOC'])[1])==3
    try:normalize_iob(['B-DATE']);raise AssertionError('Bad label accepted')
    except ValueError:pass
    evidence['iob_normalization']='orphan and type-change conversion; valid B retained; invalid taxonomy rejected'
    raw='Dr. Smith met U.S. officials and Jean-Claude in Baden-Wuerttemberg.'
    parts=text_sentences(raw)
    assert len(parts)==1 and [w[0] for w in parts[0]][:4]==['Dr.','Smith','met','U.S.']
    assert 'Jean-Claude' in [w[0] for w in parts[0]]
    raw='\ufeffAlice\u200bSmith works at Micro\u00adsoft.'
    words=text_sentences(raw)[0]
    assert words[0][0]=='AliceSmith' and words[3][0]=='Microsoft'
    assert highlight_spans(raw,words,['B-PER','O','O','B-ORG','O'])[0]['text']=='Alice\u200bSmith'
    evidence['raw_text_v2']='acronyms, titles, hyphens, invisible characters and exact source offsets'

    g=[['B-PER','I-PER','O','B-LOC','B-LOC']]
    assert metrics(g,g)['f1']==1.
    assert metrics([['B-PER','I-PER']],[['B-PER','O']])['f1']==0
    assert metrics([['B-PER']],[['B-ORG']])['f1']==0
    assert metrics([['B-PER']],[['O']])['f1']==0
    assert metrics([['B-PER','I-PER']],[['I-PER','I-PER']])['f1']==0
    for length in [1,2,3]:
        for tags in itertools.product(['O','B-PER','I-PER','B-LOC','I-LOC'],repeat=length):
            metrics([list(tags)],[list(tags)])
    evidence['strict_metrics']='known exact/partial/type/missing/repeated/illegal examples; decoder cross-checked with strict seqeval'
    crf=LinearCRF(3);e=torch.randn(2,3,3,requires_grad=True);tags=torch.tensor([[0,1,2],[2,1,0]]);mask=torch.tensor([[1,1,1],[1,1,0]],dtype=torch.bool)
    nll=crf.nll(e,tags,mask);paths=crf.decode(e,mask);manual=[]
    for b,n in enumerate([3,2]):
        scores=[];ps=list(itertools.product(range(3),repeat=n))
        for path in ps:
            s=crf.start[path[0]]+crf.end[path[-1]]+sum(e[b,i,t] for i,t in enumerate(path))
            s=s+sum(crf.transitions[path[i-1],path[i]] for i in range(1,n));scores.append(s)
        stacked=torch.stack(scores);truth=tuple(tags[b,:n].tolist());manual.append(torch.logsumexp(stacked,0)-stacked[ps.index(truth)])
        assert paths[b]==list(ps[int(stacked.argmax())])
    assert torch.allclose(nll,torch.stack(manual).mean(),atol=1e-5)
    nll.backward();assert e.grad.abs().sum()>0 and crf.transitions.grad.abs().sum()>0
    padded=e.detach().clone();padded[1,2]=1e4
    assert torch.allclose(crf.nll(padded,tags,mask),nll.detach()) and crf.decode(padded,mask)==paths
    try:crf.nll(e.detach(),tags.masked_fill(~mask,-100),mask);raise AssertionError('-100 allowed')
    except AssertionError as exc:assert '-100' in str(exc)
    assert crf.decode(e[:,:1].detach(),mask[:,:1]) and torch.isfinite(crf.nll(e[:,:1],tags[:,:1],mask[:,:1]))
    evidence['crf']='brute-force partition, exact likelihood/Viterbi, start/end transitions, length1, padding invariance, -100 rejection, finite real gradients'
    rows=read_json(root/'prepared/train.json')[:3]
    for name in ['lstm','bilstm','bilstm_crf']:
        cfg={'model':name,'n_words':len(vocab['words']),'n_chars':len(vocab['chars']),'n_labels':len(labels)}
        model=RecurrentNER(cfg,np.load(root/'prepared/embeddings.npy')).eval()
        a=collate_recurrent([rows[0]],vocab);b=collate_recurrent(rows,vocab)
        with torch.no_grad():
            ea=model(a['words'],a['chars'],a['mask']);eb=model(b['words'],b['chars'],b['mask'])
        assert torch.allclose(ea[0],eb[0,:ea.shape[1]],atol=2e-6),name
        model.train();ee=model(b['words'],b['chars'],b['mask']);loss=model.loss(ee,b['tags'],b['mask']);loss.backward()
        assert model.char.weight.grad.abs().sum()>0 and model.encoder.weight_ih_l0.grad.abs().sum()>0
        assert model.word.weight.grad[0].abs().sum()==0
        unknown=collate_recurrent([{'tokens':['ZzyzxNovelName']}],vocab)
        assert unknown['words'][0,0]==1 and (unknown['chars']!=0).any()
    evidence['recurrent']='all 3: padding-invariant eval emissions, masked loss, character/LSTM gradients, zero pad gradient, unknown-word characters'
    tok=AutoTokenizer.from_pretrained(root/'assets/distilbert',local_files_only=True,use_fast=True)
    sample=[{'tokens':['Alice',',','Supercalifragilisticexpialidocious','New','York','.'],'tags':['B-PER','O','O','B-LOC','I-LOC','O']},
            {'tokens':['London'],'tags':['B-LOC']}]
    feats,mapping=aligned_features(sample,tok,labels,256)
    assert len(tok(sample[0]['tokens'][2],add_special_tokens=False)['input_ids'])>1
    coll=DataCollatorForTokenClassification(tok)(feats)
    assert (coll['labels'][coll['attention_mask']==0]==-100).all()
    ids=[]
    for f in feats:
        assert f['labels'][0]==-100 and f['labels'][-1]==-100
        ids.append([t if t!=-100 else 0 for t in f['labels']])
    assert reconstruct(ids,mapping,sample,labels)==[r['tags'] for r in sample]
    assert sum(t!=-100 for f in feats for t in f['labels'])==7
    long=[{'tokens':['Alice']*700,'tags':['B-PER']+['I-PER']*699}]
    ff,mm=aligned_features(long,tok,labels,256);assert len(ff)>1
    pp=reconstruct([[t if t!=-100 else 0 for t in f['labels']] for f in ff],mm,long,labels)
    assert pp==[long[0]['tags']] and len(strict_spans(pp[0]))==1
    evidence['alignment']='first subword only, special/padded -100, split word, punctuation, one output per word, 700-word entity reconstructed across chunks'
    text='Alice,  New York!\nAlice & <script>'
    sentences=text_sentences(text);words=[w for s in sentences for w in s]
    assert all(text[a:b]==w for w,a,b in words)
    tags=['B-PER' if w=='Alice' else 'B-LOC' if w=='New' else 'I-LOC' if w=='York' else 'O' for w,a,b in words]
    spans=highlight_spans(text,words,tags)
    assert [s['text'] for s in spans]==['Alice','New York','Alice']
    from app import colored_html
    assert '&lt;script&gt;' in colored_html(text,spans)
    evidence['offsets']='original punctuation, repeated entities, multiword whitespace, linebreak offsets and HTML escaping'
    # Ensure the BiLSTM and CRF encoder structures match exactly.
    a=RecurrentNER({'model':'bilstm','n_words':len(vocab['words']),'n_chars':len(vocab['chars']),'n_labels':len(labels)})
    b=RecurrentNER({**a.cfg,'model':'bilstm_crf'})
    assert {k:tuple(v.shape) for k,v in a.state_dict().items()}=={k:tuple(v.shape) for k,v in b.state_dict().items() if not k.startswith('crf.')}
    evidence['matched_encoders']=True
    write_json(root/'reports/unit_verification.json',evidence);print(json.dumps(evidence,indent=2),flush=True)
    return evidence

def scheduler_tests(root):
    from scheduler import run_queue,worker_env,completed
    root=Path(root)/'scheduler_checks'
    if root.exists():__import__('shutil').rmtree(root)
    root.mkdir(parents=True,exist_ok=True)
    script=root/'fake.py'
    script.write_text("import os,sys,time\nfrom pathlib import Path\nPath(sys.argv[1]).write_text(str(os.getpid()))\nassert os.environ['CUDA_VISIBLE_DEVICES']==sys.argv[2]\ntime.sleep(float(sys.argv[3]))\nsys.exit(int(sys.argv[4]))\n")
    def cmd(job,dev):return [sys.executable,str(script.resolve()),str((root/(job+'.pid')).resolve()),dev,'0.3' if job=='fail' else '1.0','7' if job=='fail' else '0']
    two=run_queue(['a','b','c'],['0','1'],cmd,root/'two',poll=.05)
    starts={e['model']:e['time'] for e in two if e['event']=='start'};ends={e['model']:e['time'] for e in two if e['event']=='exit'}
    assert starts['a']<ends['b'] and starts['b']<ends['a']
    single=run_queue(['a','b'],['0'],cmd,root/'single',poll=.05)
    assert [e['event'] for e in single]==['start','exit','start','exit']
    try:run_queue(['fail','peer','never'],['0','1'],cmd,root/'failure',poll=.05);raise AssertionError('Expected failure')
    except RuntimeError:pass
    failed=read_json(root/'failure/schedule_events.json');assert any(e['event']=='cancel' and e['model']=='peer' for e in failed)
    assert not any(e['event']=='start' and e['model']=='never' for e in failed)
    for p in ['fail','peer']:
        pid=int((root/(p+'.pid')).read_text())
        try:os.kill(pid,0);raise AssertionError('Leaked process')
        except ProcessLookupError:pass
    # Explicit SIGINT cancellation exercises the same parent cleanup used by notebook interruption.
    wrapper=root/'interrupt_wrapper.py'
    wrapper.write_text('import sys\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(Path(__file__).parent.resolve()))+')\nfrom scheduler import run_queue\nr=Path('+repr(str(root.resolve()))+')\nscript=r/"fake.py"\nrun_queue(["interrupt_a","interrupt_b"],["0","1"],lambda job,dev:[sys.executable,str(script),str(r/(job+".pid")),dev,"20","0"],r/"interrupt",poll=.05)\n')
    with open(root/'interrupt_parent.log','w') as log:
        parent=subprocess.Popen([sys.executable,str(wrapper)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.time()+8
        while not all((root/(n+'.pid')).exists() for n in ['interrupt_a','interrupt_b']):
            if time.time()>deadline:parent.kill();raise RuntimeError('Interrupt simulation failed to start')
            time.sleep(.05)
        import signal
        parent.send_signal(signal.SIGINT);parent.wait(timeout=8)
    interrupt=read_json(root/'interrupt/schedule_events.json')
    assert sum(e['event']=='cancel' for e in interrupt)==2
    for name in ['interrupt_a','interrupt_b']:
        pid=int((root/(name+'.pid')).read_text())
        try:os.kill(pid,0);raise AssertionError('Interrupted child leaked')
        except ProcessLookupError:pass
    env=worker_env('1');assert env['CUDA_VISIBLE_DEVICES']=='1' and 'RANK' not in env
    cache=root/'cache';folder=cache/'models'/'fixture';folder.mkdir(parents=True)
    item=folder/'weights.bin';item.write_bytes(b'known fixture')
    write_json(folder/'done.json',{'experiment_id':'known','files':{'weights.bin':file_hash(item)}})
    assert completed(cache,'fixture','known')
    try:completed(cache,'fixture','changed');raise AssertionError('Identity mismatch accepted')
    except RuntimeError:pass
    item.write_bytes(b'corrupt')
    try:completed(cache,'fixture','known');raise AssertionError('Corruption accepted')
    except RuntimeError:pass
    out={'type':'real CPU subprocess scheduler with simulated GPU identifiers; not real CUDA',
         'overlap':True,'single_device_serial':True,'failure_cancels_peer':True,'queue_stops_on_failure':True,'no_leaked_processes':True,'environment_isolation':True,'explicit_sigint_cancellation':True,'cache_identity_and_corruption_rejection':True}
    write_json(root.parent/'reports/scheduler_verification.json',out);return out

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run_tests(a.root);scheduler_tests(a.root)
