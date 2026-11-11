"""GloVe + character CNN recurrent NER and a genuine unconstrained linear-chain CRF."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

class LinearCRF(nn.Module):
    """P(y|x), learned start/end and transitions[previous,next]. No hard BIO constraints."""
    def __init__(self,n):
        super().__init__();self.start=nn.Parameter(torch.empty(n));self.end=nn.Parameter(torch.empty(n));self.transitions=nn.Parameter(torch.empty(n,n))
        for p in self.parameters():nn.init.uniform_(p,-.1,.1)
    def validate(self,emissions,tags,mask):
        assert emissions.ndim==3 and mask.dtype==torch.bool and mask.shape==emissions.shape[:2]
        assert mask[:,0].all() and not ((~mask[:,:-1])&mask[:,1:]).any(),'Masks must be nonempty right padded'
        if tags is not None:
            assert tags.shape==mask.shape and tags.dtype==torch.long
            assert (tags>=0).all() and (tags<emissions.shape[-1]).all(),'CRF requires valid tags even on padding; never -100'
    def nll(self,emissions,tags,mask):
        self.validate(emissions,tags,mask)
        with torch.autocast(device_type=emissions.device.type,enabled=False):
            e=emissions.float();b=e.shape[0];ix=torch.arange(b,device=e.device)
            score=self.start[tags[:,0]]+e[ix,0,tags[:,0]]
            alpha=self.start+e[:,0]
            for t in range(1,e.shape[1]):
                score=score+(self.transitions[tags[:,t-1],tags[:,t]]+e[ix,t,tags[:,t]])*mask[:,t]
                nxt=torch.logsumexp(alpha.unsqueeze(2)+self.transitions.unsqueeze(0),dim=1)+e[:,t]
                alpha=torch.where(mask[:,t,None],nxt,alpha)
            last=mask.long().sum(1)-1;score=score+self.end[tags[ix,last]]
            logz=torch.logsumexp(alpha+self.end,dim=1);loss=(logz-score).mean()
            assert loss.dtype==torch.float32 and torch.isfinite(loss)
            return loss
    @torch.no_grad()
    def decode(self,emissions,mask):
        self.validate(emissions,None,mask)
        with torch.autocast(device_type=emissions.device.type,enabled=False):
            e=emissions.float();score=self.start+e[:,0];history=[]
            for t in range(1,e.shape[1]):
                best,back=(score.unsqueeze(2)+self.transitions.unsqueeze(0)).max(1)
                score=torch.where(mask[:,t,None],best+e[:,t],score);history.append(back)
            ends=(score+self.end).argmax(1);paths=[]
            for b,length in enumerate(mask.sum(1).tolist()):
                tag=int(ends[b]);path=[tag]
                for t in range(length-2,-1,-1):tag=int(history[t][b,tag]);path.append(tag)
                paths.append(path[::-1])
            return paths

class RecurrentNER(nn.Module):
    def __init__(self,cfg,embeddings=None):
        super().__init__();self.cfg=cfg
        self.word=nn.Embedding(cfg['n_words'],50,padding_idx=0)
        if embeddings is not None:self.word.weight.data.copy_(torch.as_tensor(embeddings))
        self.char=nn.Embedding(cfg['n_chars'],30,padding_idx=0)
        self.char_conv=nn.Conv1d(30,50,3,padding=1)
        self.dropout=nn.Dropout(.3)
        self.encoder=nn.LSTM(100,128,batch_first=True,bidirectional=cfg['model']!='lstm',num_layers=1)
        self.head=nn.Linear(128*(1 if cfg['model']=='lstm' else 2),cfg['n_labels'])
        self.crf=LinearCRF(cfg['n_labels']) if cfg['model']=='bilstm_crf' else None
    def forward(self,words,chars,mask):
        b,t,c=chars.shape;flat=chars.reshape(b*t,c);valid=flat!=0
        encoded=self.char_conv(self.char(flat).transpose(1,2)).relu()
        encoded=encoded.masked_fill(~valid[:,None,:],-1e4).max(2).values
        encoded=torch.where(valid.any(1,keepdim=True),encoded,torch.zeros_like(encoded)).reshape(b,t,50)
        x=self.dropout(torch.cat((self.word(words),encoded),dim=-1))
        packed=pack_padded_sequence(x,mask.sum(1).cpu(),batch_first=True,enforce_sorted=False)
        encoded,_=self.encoder(packed);encoded,_=pad_packed_sequence(encoded,batch_first=True,total_length=t)
        return self.head(self.dropout(encoded))
    def loss(self,emissions,tags,mask):
        return self.crf.nll(emissions,tags,mask) if self.crf else F.cross_entropy(emissions[mask].float(),tags[mask])
    def decode(self,emissions,mask):
        return self.crf.decode(emissions,mask) if self.crf else [r[:int(n)].tolist() for r,n in zip(emissions.argmax(-1),mask.sum(1))]

def collate_recurrent(rows,vocab):
    b=len(rows);t=max(len(r['tokens']) for r in rows)
    cap=vocab['char_max'];c=max(1,min(cap,max(len(w) for r in rows for w in r['tokens'])))
    words=torch.zeros(b,t,dtype=torch.long);chars=torch.zeros(b,t,c,dtype=torch.long)
    mask=torch.zeros(b,t,dtype=torch.bool);tags=torch.full((b,t),vocab['labels'].index('O'),dtype=torch.long)
    lid={x:i for i,x in enumerate(vocab['labels'])}
    for i,r in enumerate(rows):
        for j,w in enumerate(r['tokens']):
            words[i,j]=vocab['words'].get(w,1);mask[i,j]=True
            for k,ch in enumerate(w[:c]):chars[i,j,k]=vocab['chars'].get(ch,1)
            if 'tags' in r:tags[i,j]=lid[r['tags'][j]]
    return {'words':words,'chars':chars,'mask':mask,'tags':tags}

def load_recurrent(checkpoint,device='cpu'):
    obj=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=RecurrentNER(obj['config']);model.load_state_dict(obj['state']);model.to(device).eval()
    return model
