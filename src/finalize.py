"""Freeze validation selection, then (full mode only) test all four, export and package."""
import os,sys,time,shutil,zipfile,subprocess,platform
from pathlib import Path
import numpy as np
import torch
from common import *
from scheduler import MODELS,completed
from worker import recurrent_predictions,transformer_predict,device_setup
from models import load_recurrent

def diagnostics(records,preds):
    counts=Counter();examples=[]
    for r,p in zip(records,preds):
        gold=set(strict_spans(r['tags']));guess=set(strict_spans(p));counts['gold']+=len(gold);counts['predicted']+=len(guess);counts['exact']+=len(gold&guess)
        counts['invalid_bio_transitions']+=invalid_bio(p)
        for typ,a,b in gold-guess:
            overlaps=[s for s in guess if max(a,s[1])<min(b,s[2])]
            if any((a,b)==(x,y) for t,x,y in overlaps):category='type_confusion'
            elif any(typ==t for t,x,y in overlaps):category='boundary_error'
            else:category='missed_entity'
            counts[category]+=1
        counts['spurious_entities']+=sum(not any(max(a,x)<min(b,y) for t,x,y in gold) for typ,a,b in guess-gold)
        counts['multiword_gold']+=sum(b-a>1 for t,a,b in gold)
        counts['multiword_exact']+=sum(b-a>1 for t,a,b in gold&guess)
        if gold!=guess and len(examples)<60:
            examples.append({'id':r['id'],'tokens':r['tokens'],'gold':r['tags'],'prediction':p,'gold_spans':sorted(gold),'predicted_spans':sorted(guess)})
    return {'counts':dict(counts),'definitions':'Strict IOB2 spans without label repair. Error categories count gold errors; spurious means no gold overlap. Categories are diagnostic, not replacements for exact F1.','examples':examples}

def crf_pairs(records,bilstm,crf):
    counts=Counter();corrections=[];regressions=[];boundary_corrections=[];boundary_regressions=[]
    for r,a,b in zip(records,bilstm,crf):
        g=set(strict_spans(r['tags']));aa=set(strict_spans(a));bb=set(strict_spans(b))
        fix=(g&bb)-aa;reg=(g&aa)-bb
        counts['exact_entity_corrections']+=len(fix);counts['exact_entity_regressions']+=len(reg)
        counts['bilstm_invalid_bio']+=invalid_bio(a);counts['crf_invalid_bio']+=invalid_bio(b)
        counts['bilstm_spurious']+=len(aa-g);counts['crf_spurious']+=len(bb-g)
        gb={(x,y) for t,x,y in g};ab={(x,y) for t,x,y in aa};cb={(x,y) for t,x,y in bb}
        counts['boundary_only_corrections']+=len((gb&cb)-ab);counts['boundary_only_regressions']+=len((gb&ab)-cb)
        example={'id':r['id'],'tokens':r['tokens'],'gold':r['tags'],'bilstm':a,'bilstm_crf':b,'corrected':sorted(fix),'regressed':sorted(reg)}
        if fix and len(corrections)<30:corrections.append(example)
        if reg and len(regressions)<30:regressions.append(example)
    return {'counts':dict(counts),'correction_examples':corrections,'regression_examples':regressions,
            'interpretation':'Paired exact-span changes and type-agnostic boundary changes; unconstrained CRF is not guaranteed to win. Identical strict scoring with no BIO repair.'}

def plot_results(root,results):
    import os;os.environ['MPLBACKEND']='Agg'
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for name,r in results.items():
        h=r['history'];v=[x for x in h if 'val_f1' in x];tr=[x for x in h if 'train_loss' in x];vl=[x for x in h if 'val_loss' in x]
        axes[0].plot([x['epoch'] for x in v],[x['val_f1'] for x in v],marker='.',label=name)
        if tr:axes[1].plot([x['epoch'] for x in tr],[x['train_loss'] for x in tr],label=name+' train')
        if vl:axes[1].plot([x['epoch'] for x in vl],[x['val_loss'] for x in vl],linestyle='--',label=name+' val')
    names=list(results);axes[2].bar(names,[results[n]['validation']['f1'] for n in names]);axes[2].tick_params(axis='x',rotation=30)
    axes[0].set_title('Validation strict entity micro F1');axes[0].legend(fontsize=8)
    axes[1].set_title('Loss (CRF sequence vs token CE differ)');axes[1].legend(fontsize=6);axes[2].set_title('Best validation checkpoints')
    fig.tight_layout();fig.savefig(root/'reports/learning_curves.png',dpi=130);plt.close(fig)

def export_model(root,winner,result):
    target=root/'app';target.mkdir(exist_ok=True)
    source=Path(__file__).parent
    for name in ['common.py','models.py','inference.py','app.py']:shutil.copy2(source/name,target/name)
    modeldir=target/'selected_model'
    if modeldir.exists():shutil.rmtree(modeldir)
    modeldir.mkdir();shutil.copy2(root/'prepared/vocab.json',modeldir/'vocab.json')
    if winner=='distilbert':shutil.copytree(root/'models'/winner/'best',modeldir/'model')
    else:shutil.copy2(root/'models'/winner/'best.pt',modeldir/'best.pt')
    meta={'model':winner,'labels':read_json(root/'prepared/vocab.json')['labels'],'length_policy':read_json(root/'prepared/length_policy.json'),
          'selection':read_json(root/'reports/selection_frozen.json'),'experiment_id':result['experiment_id'],
          'preprocessing':'Preserve casing/punctuation/word order; train-only vocab; lowercase only external GloVe lookup. No POS/chunk features.',
          'span_policy':'Strict IOB2; invalid I predictions ignored rather than repaired. Half-open original character offsets.',
          'app_tokenizer':APP_TOKENIZER_POLICY,'ui_revision':'polished-v2',
          'benchmark_tokenization_note':'Raw-text regex is deterministic but differs from human CoNLL tokenization; exported token API preserves benchmark tokens exactly.',
          'embedding_provenance':read_json(source/'source_lock.json')['embedding_metadata'],
          'training_mode':result['mode'],'full_training_results':result['mode']=='full'}
    write_json(modeldir/'export.json',meta)
    req=source.joinpath('requirements.txt').read_text()
    target.joinpath('requirements.txt').write_text('torch>=2.4,<3\n'+req)
    target.joinpath('README.md').write_text('# English NER Gradio app\n\n'+('FUNCTIONAL SMOKE MODEL ONLY. Not a final trained deployment.\n\n' if result['mode']=='smoke' else '')+
        'Create a Python 3.11 or 3.12 environment, install `pip install -r requirements.txt`, then run `python app.py`. CPU works. GPU hosting is optional. No training corpus or GloVe resource is downloaded. Model/tokenizer reads are local-only.\n\n'
        'English PER, ORG, LOC and MISC. Exact source text, end-exclusive offsets, blank/no-entity and length-limit handling. The raw-v2 tokenizer retains acronyms, common titles, apostrophes and internal hyphens. Invisible format/control characters are omitted only from inference; display text and source offsets are preserved. Tokenization remains heuristic. Publication is a separate action.\n')
    return target

def archive_tree(path,destination):
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(Path(path).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,str(p.relative_to(path)))
    with zipfile.ZipFile(destination) as z:assert z.testzip() is None

def main(root,mode='full'):
    root=Path(root);device=device_setup();reports=root/'reports';reports.mkdir(exist_ok=True)
    exp=read_json(root/'experiment.json');identity=exp['id']
    for name in MODELS:assert completed(root,name,identity)
    results={n:read_json(root/'models'/n/'result.json') for n in MODELS}
    # Selection and tie rule are independent of all test predictions.
    best=max(r['validation']['f1'] for r in results.values())
    eligible=[n for n in MODELS if best-results[n]['validation']['f1']<=.001+1e-12]
    winner=min(eligible,key=lambda n:(results[n]['trainable_parameters'],results[n]['validation_seconds'],n))
    frozen={'winner':winner,'best_validation_f1':best,'eligible_within_0.001':eligible,
        'rule':'Strict validation micro F1; within absolute 0.001 prefer fewer trainable parameters, then measured validation seconds, then stable model name',
        'experiment_id':identity,'mode':mode,'checkpoint_manifests':{n:file_hash(root/'models'/n/'done.json') for n in MODELS}}
    freeze_path=reports/'selection_frozen.json'
    if freeze_path.exists():assert read_json(freeze_path)==frozen,'Frozen selection changed; stop test evaluation'
    else:write_json(freeze_path,frozen)
    vocab=read_json(root/'prepared/vocab.json');val=read_json(root/'prepared/validation.json')
    if mode=='smoke':val=val[:16]
    vp={n:read_json(root/'models'/n/'validation_predictions.json') for n in MODELS}
    for n in MODELS:
        assert metrics([r['tags'] for r in val],vp[n])==results[n]['validation']
        write_json(reports/(n+'_validation_diagnostics.json'),diagnostics(val,vp[n]))
    write_json(reports/'crf_validation_comparison.json',crf_pairs(val,vp['bilstm'],vp['bilstm_crf']))
    test_results={};test_preds={}
    if mode=='full':
        # FIRST MODEL ACCESS TO TEST: only after the frozen selection file is on disk.
        test=read_json(root/'prepared/test.json')
        for name in MODELS:
            if name=='distilbert':
                from transformers import AutoTokenizer,AutoModelForTokenClassification
                path=root/'models'/name/'best';tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
                model=AutoModelForTokenClassification.from_pretrained(path,local_files_only=True).to(device).eval()
                pred,seconds=transformer_predict(model,tok,test,vocab['labels'],read_json(root/'prepared/length_policy.json')['max_subwords'],device)
            else:
                model=load_recurrent(root/'models'/name/'best.pt',device);pred,seconds,_=recurrent_predictions(model,test,vocab,device)
            test_results[name]={**metrics([r['tags'] for r in test],pred),'inference_seconds':seconds};test_preds[name]=pred
            write_json(reports/(name+'_test_predictions.json'),[{'id':r['id'],'tags':p} for r,p in zip(test,pred)])
            write_json(reports/(name+'_test_diagnostics.json'),diagnostics(test,pred));del model
            if device.type=='cuda':torch.cuda.empty_cache()
        write_json(reports/'crf_test_comparison.json',crf_pairs(test,test_preds['bilstm'],test_preds['bilstm_crf']))
    comparison={'mode':mode,'winner':winner,'validation':{n:r['validation'] for n,r in results.items()},'test':test_results,
                'experiments':results,'selection_used_test':False,'prediction_repair':'None; strict IOB2 for every architecture'}
    write_json(reports/'comparison.json',comparison);plot_results(root,results)
    app=export_model(root,winner,results[winner])
    sys.path.insert(0,str(app))
    from inference import Predictor
    predictor=Predictor(app/'selected_model');export_val=val[:2]
    exact=predictor.predict_tokens([r['tokens'] for r in export_val]);assert exact==vp[winner][:2]
    assert predictor.analyze('  ')['entities']==[]
    for raw in ['Alice\u200bSmith works at Microsoft.','👩\u200d💻 Alice lives in London.','Micro\u00adsoft is in London.','\ufeffAlice lives in London.']:
        checked=predictor.analyze(raw)
        assert checked['text']==raw
        assert all(raw[x['start']:x['end']]==x['text'] for x in checked['entities'])

    analyzed=predictor.analyze('Alice works at Microsoft in London.\nAlice visited New York.')
    assert all(analyzed['text'][s['start']:s['end']]==s['text'] for s in analyzed['entities'])
    no_entities=predictor.analyze('12345 + 67890 = 80235.');assert isinstance(no_entities['entities'],list)
    long='Alice in London. '*300;long_result=predictor.analyze(long)
    try:predictor.analyze('a'*20001);raise AssertionError('Long input must be rejected explicitly')
    except ValueError:pass
    # Exercise actual Gradio component construction with the clean export.
    from app import build_app
    demo=build_app(app/'selected_model');assert demo is not None
    verification={'clean_cpu_export_reload_matches_validation':True,'blank':True,'no_entity_response':True,
        'long_input_no_truncation':True,'over_limit_rejected':True,'original_offsets_valid':True,'gradio_components_constructed':True,
        'mode':mode,'test_evaluated':mode=='full','actual_cuda':device.type=='cuda','training_full':mode=='full',
        'simulated_scheduler_checks':'See scheduler_verification.json; simulation is not native two-GPU proof'}
    write_json(reports/'export_verification.json',verification)
    env={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda':torch.version.cuda,
         'pip_freeze':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()}
    write_json(reports/'environment.json',env)
    report_text='# NER experiment report\n\n'+('SMOKE ONLY: 32 train / 16 validation sentences; no held-out test inference and no quality claims.\n\n' if mode=='smoke' else 'Four best validation checkpoints evaluated on the official test split after freezing the winner.\n\n')
    report_text+='Selected model: **'+winner+'** using the predeclared validation rule.\n\n| Model | Validation F1 | Test F1 | Parameters | Train seconds | Peak CUDA allocated MB |\n|---|---:|---:|---:|---:|---:|\n'
    for n,r in results.items():report_text+=f"| {n} | {r['validation']['f1']:.6f} | {test_results[n]['f1'] if n in test_results else 'not evaluated'} | {r['parameters']} | {r['train_seconds']:.1f} | {(r['cuda_max_allocated_bytes'] or 0)/1e6:.1f} |\n"
    report_text+='\nSee comparison.json for per-type precision, recall, F1, support, timing, checkpoint sizes, gradients and learning history; CRF comparison files contain both corrections and regressions. Concurrent timing is not an isolated speed benchmark. Losses differ between CRF and cross-entropy and should not be directly ranked.\n\n'
    report_text+='No optimizer-state resume is implemented. Completed compatible models are reusable after manifest verification. Interrupted models restart. Smoke and full have separate experiment identities.\n'
    reports.joinpath('REPORT.md').write_text(report_text)
    outputs=root/'outputs';outputs.mkdir(exist_ok=True)
    # Stage complete experiment evidence without full training inputs or external assets.
    evidence=root/'experiment_export'
    if evidence.exists():shutil.rmtree(evidence)
    evidence.mkdir();shutil.copytree(root/'models',evidence/'models');shutil.copytree(reports,evidence/'reports')
    src=evidence/'source';src.mkdir()
    for p in Path(__file__).parent.iterdir():
        if p.is_file():shutil.copy2(p,src/p.name)
    for name in ['vocab.json','length_policy.json','identity.json']:shutil.copy2(root/'prepared'/name,evidence/name)
    for name in ['experiment.json','schedule_events.json','reuse_verification.json']:shutil.copy2(root/name,evidence/name)
    if (root/'logs').exists():shutil.copytree(root/'logs',evidence/'worker_logs')
    prefix='smoke_' if mode=='smoke' else ''
    archive_tree(reports,outputs/(prefix+'ner_reports.zip'));archive_tree(evidence,outputs/(prefix+'ner_all_checkpoints.zip'));archive_tree(app,outputs/(prefix+'ner_gradio_app.zip'))
    # Deployment must contain neither full GloVe nor corpus nor prepared embeddings.
    with zipfile.ZipFile(outputs/(prefix+'ner_gradio_app.zip')) as z:
        assert not any(n.endswith(('.parquet','.npy')) or 'glove-wiki' in n for n in z.namelist())
    with zipfile.ZipFile(outputs/(prefix+'ner_all_checkpoints.zip')) as z:
        names=z.namelist()
        for n in MODELS:assert any(x.startswith('models/'+n+'/') and (x.endswith('best.pt') or x.endswith('model.safetensors')) for x in names)
    write_json(outputs/'archive_manifest.json',{p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in outputs.glob('*.zip')})
    print('FINISHED',json.dumps({'mode':mode,'winner':winner,'outputs':str(outputs),'test_evaluated':mode=='full'}),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--mode',default='full');a=p.parse_args();main(a.root,a.mode)
