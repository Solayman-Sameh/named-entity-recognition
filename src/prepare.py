"""Data-only, hash-checked acquisition and full audit. No dataset loading script runs."""
import os,json,urllib.request,gzip,shutil,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import pyarrow.parquet as pq
from common import *

def fetch(url,path,sha256=None,md5=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists() or (sha256 and file_hash(path)!=sha256):
        for attempt in range(3):
            try:
                print('Downloading',url,flush=True)
                with urllib.request.urlopen(url,timeout=120) as r,open(str(path)+'.part','wb') as f:
                    shutil.copyfileobj(r,f,1024*1024)
                Path(str(path)+'.part').replace(path);break
            except Exception:
                if attempt==2:raise
                time.sleep(1+attempt)
    if sha256:assert file_hash(path)==sha256, f'Checksum mismatch: {path}'
    if md5:assert file_hash(path,'md5')==md5, f'Published MD5 mismatch: {path}'
    return path

def summary(values):
    a=np.asarray(values)
    return {'min':int(a.min()),'median':float(np.median(a)),'p95':float(np.percentile(a,95)),
            'p99':float(np.percentile(a,99)),'max':int(a.max())}

def metadata_labels(schema):
    metadata=json.loads(schema.metadata[b'huggingface'])
    f=metadata['info']['features']['ner_tags']
    while 'names' not in f:
        f=f.get('feature',f.get('list',f.get('sequence')))
        if f is None:raise ValueError('Missing NER ClassLabel metadata')
    names=f['names']
    return [names[str(i)] for i in range(len(names))] if isinstance(names,dict) else names

def canonical_rows(path,split,labels):
    table=pq.read_table(path); assert set(table.column_names)=={'id','tokens','pos_tags','chunk_tags','ner_tags'}
    assert metadata_labels(table.schema)==labels
    result=[];ledger=[]
    for index,row in enumerate(table.to_pylist()):
        tokens=row['tokens'];ids=row['ner_tags']
        assert tokens and len(tokens)==len(ids)==len(row['pos_tags'])==len(row['chunk_tags']), (split,index,'length')
        assert all(isinstance(t,str) and t for t in tokens),(split,index,'empty token')
        assert all(isinstance(t,int) and 0<=t<len(labels) for t in ids),(split,index,'label ID')
        original=[labels[t] for t in ids];tags,changes=normalize_iob(original)
        result.append({'id':str(row['id']),'split':split,'index':index,'tokens':tokens,'original_tags':original,'tags':tags})
        for c in changes:ledger.append({'split':split,'id':row['id'],**c})
    assert len({r['id'] for r in result})==len(result),'Duplicate sentence IDs'
    return result,ledger,str(table.schema)

def build(root,mode='full'):
    from transformers import AutoTokenizer
    root=Path(root);src=root/'assets';prep=root/'prepared';reports=root/'reports'
    for p in [src,prep,reports]:p.mkdir(parents=True,exist_ok=True)
    lock=read_json(Path(__file__).with_name('source_lock.json'))
    downloaded={}
    for name,info in lock['files'].items():
        # Test audit only in full/prep. Smoke never downloads or opens test data.
        if mode=='smoke' and name in ('test.parquet','original_corpus.zip'):continue
        if name.startswith('distilbert/') and name.endswith('model.safetensors'):continue # worker fetches pinned weights
        p=fetch(info['url'],src/name,info['sha256'],info.get('published_md5'))
        downloaded[name]={'sha256':file_hash(p),'bytes':p.stat().st_size}
    tok=AutoTokenizer.from_pretrained(src/'distilbert',use_fast=True,local_files_only=True)
    assert tok.is_fast
    labels=metadata_labels(pq.read_schema(src/'train.parquet'))
    assert set(labels)=={'O'}|{p+'-'+t for p in ['B','I'] for t in TYPES}
    rows={};conversions=[];audit={'source':lock,'downloaded':downloaded,'labels_by_numeric_id':labels,
        'split_policy':'Official boundaries; all rows retained; overlap reported without deduplication.',
        'normalization_rule':'Orphan/type-changing I-X -> B-X only; B-X preserved.',
        'predictive_features':'words, characters, subwords only; no POS/chunk features',
        'mode':mode,'splits':{},'exclusion_ledger':[]}
    splits=['train','validation']+(['test'] if mode!='smoke' else [])
    for split in splits:
        rs,ledger,schema=canonical_rows(src/(split+'.parquet'),split,labels);rows[split]=rs;conversions+=ledger
        tags=Counter(t for r in rs for t in r['tags']);ents=Counter();ent_lengths=[]
        sub=[];unk=0
        for r in rs:
            spans=strict_spans(r['tags']);ents.update(t for t,a,b in spans);ent_lengths += [b-a for t,a,b in spans]
            ids=tok(r['tokens'],is_split_into_words=True,add_special_tokens=True,truncation=False)['input_ids']
            sub.append(len(ids));unk+=ids.count(tok.unk_token_id)
        audit['splits'][split]={'sentences':len(rs),'tokens':sum(len(r['tokens']) for r in rs),
            'fingerprint':digest(rs),'schema':schema,'empty_or_malformed':0,'tag_counts':dict(tags),
            'o_prevalence':tags['O']/sum(tags.values()),'entities':dict(ents),'entity_lengths':summary(ent_lengths),
            'word_lengths':summary([len(r['tokens']) for r in rs]),'subword_lengths':summary(sub),'unk_subwords':unk,
            'normalization_conversions':len(ledger)}
        write_json(prep/(split+'.json'),rs)
    # Independently compare every token and raw NER label to the upstream loader's ZIP as data.
    if mode!='smoke':
        import zipfile
        comparison={}
        with zipfile.ZipFile(src/'original_corpus.zip') as z:
            for split,name in [('train','train.txt'),('validation','valid.txt'),('test','test.txt')]:
                original=[];words=[];tags=[]
                for line in z.read(name).decode('utf8').splitlines()+['']:
                    if not line or line.startswith('-DOCSTART-'):
                        if words:original.append((words,tags));words=[];tags=[]
                    else:
                        fields=line.split(' ');assert len(fields)==4
                        words.append(fields[0]);tags.append(fields[3])
                assert original==[(r['tokens'],r['original_tags']) for r in rows[split]], 'Mirror split differs from original source'
                comparison[split]={'matched_all_tokens_and_raw_ner_labels':True,'sentences':len(original)}
        write_json(reports/'provenance_comparison.json',comparison)
    train_words=sorted({t for r in rows['train'] for t in r['tokens']})
    train_chars=sorted({c for t in train_words for c in t})
    vocab={'words':{'<PAD>':0,'<UNK>':1,**{w:i+2 for i,w in enumerate(train_words)}},
           'chars':{'<PAD>':0,'<UNK>':1,**{c:i+2 for i,c in enumerate(train_chars)}},'labels':labels,
           'char_max':max(map(len,train_words)), 'train_only':True}
    assert len(vocab['words'])==len(train_words)+2
    # Word and character lengths never use validation to change vocab or char cap.
    maximum=audit['splits']['train']['subword_lengths']['max']
    max_length=256 if maximum<=256 else min(512,maximum)
    length={'max_subwords':int(max_length),'word_policy':'No recurrent word truncation; packed dynamic batches',
            'char_max':vocab['char_max'],'char_policy':'Train max; held-out/app longer chars clipped, count reported',
            'transformer_policy':'Greedy whole-word chunks; no overlap; label-independent; reconstruct before scoring',
            'app_recurrent_chunk_words':256,'app_max_characters':20000,'app_sentence_policy':APP_TOKENIZER_POLICY}
    for split,rs in rows.items():
        feats,mapping=aligned_features(rs,tok,labels,max_length)
        crossed=0
        by_sentence=defaultdict(list)
        for m in mapping:by_sentence[m['sentence']].append(m['b'])
        for sid,r in enumerate(rs):
            cuts=set(by_sentence[sid][:-1]);crossed+=sum(any(a<c<b for c in cuts) for typ,a,b in strict_spans(r['tags']))
        oov=sum(w not in vocab['words'] for r in rs for w in r['tokens'])
        chars=[c for r in rs for w in r['tokens'] for c in w]
        audit['splits'][split].update({'word_oov_tokens':oov,'word_oov_rate':oov/audit['splits'][split]['tokens'],
            'char_oov':sum(c not in vocab['chars'] for c in chars),'char_oov_rate':sum(c not in vocab['chars'] for c in chars)/len(chars),
            'words_over_char_cap':sum(len(w)>vocab['char_max'] for r in rs for w in r['tokens']),
            'transformer_chunks':len(feats),'chunk_crossed_gold_entities':crossed,'aligned_words':sum(len(m['first']) for m in mapping)})
        assert audit['splits'][split]['aligned_words']==audit['splits'][split]['tokens']
    counts={s:Counter(tuple(r['tokens']) for r in rs) for s,rs in rows.items()}
    duplicate={'within':{},'across':{}}
    for s,c in counts.items():
        variants=defaultdict(set)
        for r in rows[s]:variants[tuple(r['tokens'])].add(tuple(r['tags']))
        differing=[{'tokens':list(k),'tag_variants':sorted(v)} for k,v in variants.items() if len(v)>1]
        duplicate['within'][s]={'repeated_keys':sum(n>1 for n in c.values()),'extra_rows':sum(n-1 for n in c.values()),'annotation_differences':differing}
    for i,a in enumerate(splits):
        for b in splits[i+1:]:
            common=set(counts[a])&set(counts[b]); diffs=[]
            for tokens in sorted(common):
                aa={tuple(r['tags']) for r in rows[a] if tuple(r['tokens'])==tokens}
                bb={tuple(r['tags']) for r in rows[b] if tuple(r['tokens'])==tokens}
                if aa!=bb:diffs.append({'tokens':tokens,a:sorted(aa),b:sorted(bb)})
            duplicate['across'][a+'__'+b]={'sentence_keys':len(common),'rows_a':sum(counts[a][x] for x in common),
                'rows_b':sum(counts[b][x] for x in common),'annotation_differences':diffs}
    # Retain only training word vectors. A set of vocabulary strings supports descriptive held-out coverage.
    matrix=np.random.default_rng(42).normal(0,.05,(len(vocab['words']),50)).astype('float32');matrix[0]=0
    needed=defaultdict(list)
    for w,i in vocab['words'].items():
        if i>=2:needed[w.lower()].append(i)
    found=set();all_glove=set();bad=0
    with gzip.open(src/'glove-wiki-gigaword-50.gz','rt',encoding='utf8') as f:
        total,dim=map(int,f.readline().split());assert (total,dim)==(400000,50)
        count=0
        for line in f:
            bits=line.rstrip().split(' ');word=bits[0]; count+=1
            if len(bits)!=51:raise ValueError('Malformed GloVe row')
            all_glove.add(word)
            if word in needed:
                vec=np.asarray(bits[1:],dtype=np.float32);assert np.isfinite(vec).all()
                for i in needed[word]:matrix[i]=vec;found.add(i)
        assert count==total
    assert found,'Pretrained embedding failed: refusing all-random word layer'
    np.save(prep/'embeddings.npy',matrix)
    for split,rs in rows.items():
        words=[w for r in rs for w in r['tokens']]
        audit['splits'][split]['external_glove_token_coverage']=sum(w.lower() in all_glove for w in words)/len(words)
        audit['splits'][split]['usable_task_glove_token_coverage']=sum(vocab['words'].get(w,1) in found for w in words)/len(words)
    audit['embedding']={'source':lock['embedding_metadata'],'train_vocab_rows':len(train_words),
        'pretrained_rows':len(found),'type_coverage':len(found)/len(train_words),
        'matrix_shape':list(matrix.shape),'matrix_sha256':file_hash(prep/'embeddings.npy'),
        'oov_policy':'One reproducibly initialized UNK row plus learned characters; known train misses initialized by seed42',
        'fine_tunable':True,'pad_fixed_zero':True}
    audit['duplicates']=duplicate;audit['length_policy']=length
    audit['observed_iob']='IOB2' if not conversions else 'IOB1-compatible orphan I observed; normalized with ledger'
    audit['conversion_count']=len(conversions)
    write_json(reports/'normalization_ledger.json',conversions)
    # Corpus provides tokens rather than original article character offsets; visualization uses explicit space-joined display only.
    import html
    colors={'PER':'#c7d2fe','ORG':'#fde68a','LOC':'#a7f3d0','MISC':'#fecdd3'}
    snippets=[]
    examples=[]
    for r in rows['train'][:8]:
        chunks=[];last=0
        for typ,a,b in strict_spans(r['tags']):
            chunks.append(html.escape(' '.join(r['tokens'][last:a])))
            chunks.append('<mark style="background:'+colors[typ]+'">'+html.escape(' '.join(r['tokens'][a:b]))+' ['+typ+']</mark>');last=b
        chunks.append(html.escape(' '.join(r['tokens'][last:])))
        snippets.append('<p>'+str(r['id'])+': '+' '.join(chunks)+'</p>');examples.append(r)
    write_json(reports/'annotated_train_examples.json',examples)
    reports.joinpath('annotated_train_examples.html').write_text('<!doctype html><meta charset="utf-8"><h1>Gold train examples</h1><p>Space-joined corpus tokens for display; original article offsets are not supplied by this dataset.</p>'+''.join(snippets))
    write_json(prep/'vocab.json',vocab);write_json(prep/'length_policy.json',length);write_json(reports/'data_audit.json',audit)
    identity={'split_fingerprints':{s:audit['splits'][s]['fingerprint'] for s in splits},
              'vocab':digest(vocab),'embedding':audit['embedding']['matrix_sha256'],'length_policy':length}
    write_json(prep/'identity.json',identity)
    reference=read_json(Path(__file__).with_name('reference.json'))
    checks={}
    if reference.get('observed',False):
        for s in splits:
            expected=reference['splits'][s]
            for key in ['sentences','tokens','fingerprint','normalization_conversions','entities','tag_counts','subword_lengths','aligned_words']:
                assert audit['splits'][s][key]==expected[key],f'Reference mismatch: {s}/{key}'
            checks[s]='exact counts, tags, spans, lengths and canonical fingerprint matched'
        assert identity['vocab']==reference['vocab_fingerprint']
        assert identity['embedding']==reference['embedding_matrix_sha256']
    write_json(reports/'reference_comparison.json',{'cloud_observed_reference':reference.get('observed',False),'checks':checks})
    make_figures(audit,reports)
    print(json.dumps({s:{k:v for k,v in info.items() if k in ['sentences','tokens','entities','subword_lengths','word_oov_rate']} for s,info in audit['splits'].items()},indent=2),flush=True)
    return audit

def make_figures(audit,reports):
    import os;os.environ['MPLBACKEND']='Agg'
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(14,4))
    ss=list(audit['splits'])
    ax[0].bar(ss,[audit['splits'][s]['tokens'] for s in ss]);ax[0].set_title('Official split token counts')
    for j,t in enumerate(TYPES):ax[1].bar(np.arange(len(ss))+.18*j,[audit['splits'][s]['entities'].get(t,0) for s in ss],.18,label=t)
    ax[1].set_xticks(np.arange(len(ss))+.27,ss);ax[1].legend();ax[1].set_title('Gold exact entities')
    ax[2].bar(ss,[audit['splits'][s]['word_oov_rate'] for s in ss]);ax[2].set_title('Train-vocabulary word OOV rate')
    fig.tight_layout();fig.savefig(reports/'eda.png',dpi=130);plt.close(fig)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--mode',choices=['full','smoke','prep'],default='full');a=p.parse_args();build(a.root,a.mode)
