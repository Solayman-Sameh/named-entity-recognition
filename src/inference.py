"""Clean CPU inference; only exported assets, no dataset or embedding downloads."""
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
from pathlib import Path
import torch
from common import read_json,text_sentences,highlight_spans,word_chunks,strict_spans
from models import load_recurrent,collate_recurrent

class Predictor:
    def __init__(self,path,device='cpu'):
        self.path=Path(path);self.meta=read_json(self.path/'export.json');self.vocab=read_json(self.path/'vocab.json')
        self.length=self.meta['length_policy'];self.kind=self.meta['model'];self.device=torch.device(device)
        torch.set_num_threads(2)
        if self.kind=='distilbert':
            from transformers import AutoModelForTokenClassification,AutoTokenizer
            self.tokenizer=AutoTokenizer.from_pretrained(self.path/'model',use_fast=True,local_files_only=True)
            self.model=AutoModelForTokenClassification.from_pretrained(self.path/'model',local_files_only=True).to(self.device).eval()
        else:self.model=load_recurrent(self.path/'best.pt',self.device)
    @torch.inference_mode()
    def predict_tokens(self,sentences):
        out=[]
        for words in sentences:
            if not words:out.append([]);continue
            if self.kind=='distilbert':
                chunks,_=word_chunks(words,self.tokenizer,self.length['max_subwords']);tags=[]
                for a,b in chunks:
                    enc=self.tokenizer(words[a:b],is_split_into_words=True,return_tensors='pt',truncation=False)
                    ids=enc.word_ids();values=self.model(**{k:v.to(self.device) for k,v in enc.items() if k in ('input_ids','attention_mask')}).logits.argmax(-1)[0].tolist()
                    previous=None
                    for i,w in enumerate(ids):
                        if w is not None and w!=previous:tags.append(self.vocab['labels'][values[i]])
                        previous=w
            else:
                tags=[];cap=self.length['app_recurrent_chunk_words']
                for a in range(0,len(words),cap):
                    row={'tokens':words[a:a+cap]};batch=collate_recurrent([row],self.vocab)
                    batch={k:v.to(self.device) for k,v in batch.items()};e=self.model(batch['words'],batch['chars'],batch['mask'])
                    tags.extend(self.vocab['labels'][i] for i in self.model.decode(e,batch['mask'])[0])
            assert len(tags)==len(words);out.append(tags)
        return out
    def analyze(self,text):
        if not isinstance(text,str):raise ValueError('Please enter English text.')
        if not text.strip():return {'text':text,'entities':[],'status':'Enter English text to analyze.'}
        if len(text)>self.length['app_max_characters']:
            raise ValueError(f"Input has {len(text):,} characters; maximum {self.length['app_max_characters']:,}. Divide it into smaller passages; nothing was truncated.")
        sentences=text_sentences(text)
        if self.kind=='distilbert':
            # Some standalone Unicode marks normalize to zero BERT subwords.
            # Omit these only in raw-text inference, retaining original display offsets.
            sentences=[[w for w in s if self.tokenizer(w[0],add_special_tokens=False)['input_ids']] for s in sentences]
            sentences=[s for s in sentences if s]
        preds=self.predict_tokens([[x[0] for x in s] for s in sentences])
        spans=[]
        for words,tags in zip(sentences,preds):spans.extend(highlight_spans(text,words,tags))
        for s in spans:assert text[s['start']:s['end']]==s['text']
        return {'text':text,'entities':spans,'status':f'{len(spans)} entities found.' if spans else 'No supported entities detected.'}
