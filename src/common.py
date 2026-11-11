"""Shared strict scoring and deterministic text policies. No prediction BIO repair."""
import json, hashlib, re, os, unicodedata
from pathlib import Path
from collections import Counter
TYPES=('PER','ORG','LOC','MISC')
def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))
def write_json(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8"); tmp.replace(p)
def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def file_hash(p,method='sha256'):
    h=hashlib.new(method)
    with open(p,'rb') as f:
        while b:=f.read(1024*1024): h.update(b)
    return h.hexdigest()
def normalize_iob(tags):
    """IOB1 -> IOB2: orphan/type-changing I-X becomes B-X; retain every valid B-X."""
    result=[]; conversions=[]; previous='O'
    for i,t in enumerate(tags):
        if t!='O' and not re.fullmatch(r'[BI]-(PER|ORG|LOC|MISC)',t): raise ValueError(f'Malformed tag {t}')
        new=t
        if t.startswith('I-') and (previous=='O' or previous[2:]!=t[2:]):
            new='B-'+t[2:]; conversions.append({'word':i,'from':t,'to':new})
        result.append(new); previous=new
    return result,conversions

def strict_spans(tags):
    """Match seqeval strict IOB2: orphan I is ignored; never synthesize B predictions."""
    out=[]; start=None; typ=None
    for i,t in enumerate(list(tags)+['O']):
        if start is not None and t!='I-'+typ:
            out.append((typ,start,i)); start=None;typ=None
        if t.startswith('B-'): typ=t[2:];start=i
    return out

def metrics(gold,pred):
    from seqeval.metrics import precision_score,recall_score,f1_score,classification_report,accuracy_score
    from seqeval.scheme import IOB2
    assert len(gold)==len(pred) and all(len(a)==len(b) for a,b in zip(gold,pred))
    kw={'mode':'strict','scheme':IOB2,'zero_division':0}
    report=classification_report(gold,pred,output_dict=True,**kw)
    per={t:report.get(t,{'precision':0.,'recall':0.,'f1-score':0.,'support':0}) for t in TYPES}
    per={t:{k:int(v) if k=='support' else float(v) for k,v in row.items()} for t,row in per.items()}
    # Independent exact-span cross-check, including illegal I transitions.
    gs={(i,*s) for i,row in enumerate(gold) for s in strict_spans(row)}
    ps={(i,*s) for i,row in enumerate(pred) for s in strict_spans(row)}
    exact=2*len(gs&ps)/(len(gs)+len(ps)) if gs or ps else 0.
    f=float(f1_score(gold,pred,**kw)); assert abs(exact-f)<1e-10,(exact,f)
    return {'precision':float(precision_score(gold,pred,**kw)),'recall':float(recall_score(gold,pred,**kw)),
            'f1':f,'macro_f1':sum(per[t]['f1-score'] for t in TYPES)/4,
            'token_accuracy':float(accuracy_score(gold,pred)), 'per_type':per,
            'sentences':len(gold),'tokens':sum(map(len,gold))}

def invalid_bio(tags):
    prev='O';n=0
    for t in tags:
        n+=int(t.startswith('I-') and (prev=='O' or prev[2:]!=t[2:]));prev=t
    return n

# Raw-text policy only; benchmark token lists are never retokenized.
TOKEN_PATTERN=r"(?:[^\W\d_]\.){2,}|(?i:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc)\.|\w+(?:['\u2019\-\u2010\u2011]\w+)*|[^\w\s]"
APP_TOKENIZER_POLICY='raw-v2: acronyms/titles and hyphenated words retained; format/control characters omitted from inference with original offsets preserved; standalone .!? and newlines split sentences'

def text_sentences(text):
    # Map the inference view back to the untouched input, including internal gaps.
    positions=[i for i,c in enumerate(text) if unicodedata.category(c) not in ('Cf','Cc') or c in '\t\n\r']
    view=''.join(text[i] for i in positions)
    result=[];current=[]
    matches=list(re.finditer(TOKEN_PATTERN,view,re.UNICODE))
    for i,m in enumerate(matches):
        current.append((m.group(),positions[m.start()],positions[m.end()-1]+1))
        nxt=matches[i+1].start() if i+1<len(matches) else len(view)
        gap=view[m.end():nxt]
        if '\n' in gap or (m.group() in '.!?' and (gap or i+1==len(matches))):
            result.append(current);current=[]
    if current:result.append(current)
    return result

def highlight_spans(text,words,tags):
    assert len(words)==len(tags)
    spans=[]
    for typ,a,b in strict_spans(tags):
        start,end=words[a][1],words[b-1][2]
        spans.append({'start':start,'end':end,'label':typ,'text':text[start:end]})
    return spans

def word_chunks(tokens,tokenizer,max_length):
    """Greedy, label-independent whole-word chunks. No dropped words or subword targets.
    Each word must fit; BERT's single-word [UNK] behavior is allowed and audited.
    Chunking never consults gold labels. Reconstruction precedes strict scoring.
    """
    lengths=[len(tokenizer(t,add_special_tokens=False)['input_ids']) for t in tokens]
    assert all(n>0 for n in lengths),'Tokenizer produced zero subwords'
    budget=max_length-tokenizer.num_special_tokens_to_add(pair=False)
    out=[];start=0;used=0
    for i,n in enumerate(lengths):
        if n>budget:raise ValueError(f'One word exceeds transformer capacity at {i}; no truncation permitted')
        if used+n>budget:out.append((start,i));start=i;used=0
        used+=n
    if tokens:out.append((start,len(tokens)))
    assert sum(b-a for a,b in out)==len(tokens)
    return out,lengths

def aligned_features(records,tokenizer,labels,max_length):
    label2id={x:i for i,x in enumerate(labels)};features=[];mapping=[]
    for sid,row in enumerate(records):
        chunks,_=word_chunks(row['tokens'],tokenizer,max_length)
        for a,b in chunks:
            enc=tokenizer(row['tokens'][a:b],is_split_into_words=True,add_special_tokens=True,truncation=False)
            assert len(enc['input_ids'])<=max_length
            ids=enc.word_ids(); prev=None; targets=[];first=[]
            for pos,w in enumerate(ids):
                if w is not None and w!=prev:
                    targets.append(label2id[row['tags'][a+w]]);first.append((pos,a+w))
                else:targets.append(-100)
                prev=w
            assert [w for p,w in first]==list(range(a,b))
            feature={k:v for k,v in enc.items() if k in ('input_ids','attention_mask')};feature['labels']=targets
            features.append(feature);mapping.append({'sentence':sid,'a':a,'b':b,'first':first})
    return features,mapping

def reconstruct(chunk_predictions,mapping,records,labels):
    result=[[None]*len(r['tokens']) for r in records]
    assert len(chunk_predictions)==len(mapping)
    for ids,mp in zip(chunk_predictions,mapping):
        for pos,w in mp['first']:
            assert result[mp['sentence']][w] is None
            result[mp['sentence']][w]=labels[int(ids[pos])]
    assert all(all(t is not None for t in row) for row in result)
    return result
